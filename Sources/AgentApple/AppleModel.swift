import AgentCore
import Foundation
import FoundationModels

/// The on-device Apple Intelligence model, wrapping `LanguageModelSession`.
///
/// This is the schema substrate's home turf: structured output uses the model's
/// native guided generation via `respond(to:schema:)`, with the same
/// `GenerationSchema` the rest of the SDK already produces.
///
/// Phase 1 scope: text and native structured output, single-turn. Tool calling
/// on-device is deferred (see plan risk #1) and requests with tools throw.
public struct AppleModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile
    let systemModel: SystemLanguageModel

    public init(model: SystemLanguageModel = .default) {
        self.modelName = "apple-on-device"
        self.systemModel = model
        self.profile = ModelProfile(
            supportsNativeStructuredOutput: true,
            supportsParallelToolCalls: false,
            defaultOutputMode: .native,
            jsonSchemaTransform: { $0 })
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        guard tools.isEmpty else {
            throw AgentError.unsupportedOutputMode(
                "AgentApple (Phase 1) does not support tool calling; use a remote provider.")
        }
        switch systemModel.availability {
        case .available:
            break
        case let .unavailable(reason):
            throw AgentError.provider("on-device model unavailable: \(reason)")
        }

        // On-device is single-turn and text-only in Phase 1: gather system + user
        // text from request messages. Any `.userMedia` attachments are ignored here
        // (the on-device model takes text prompts only).
        let requestParts = messages.compactMap(\.asRequest).flatMap(\.parts)
        let system = requestParts.compactMap { part -> String? in
            if case let .system(s) = part { return s } else { return nil }
        }.joined(separator: "\n\n")
        let prompt = requestParts.compactMap { part -> String? in
            if case let .userText(t) = part { return t } else { return nil }
        }.joined(separator: "\n")

        let session = system.isEmpty
            ? LanguageModelSession(model: systemModel)
            : LanguageModelSession(model: systemModel, instructions: system)

        let usage = Usage(requests: 1)

        switch output.mode {
        case .text:
            let response = try await session.respond(to: prompt)
            return ModelResponse(parts: [.text(response.content)], usage: usage,
                                 modelName: modelName, finishReason: .stop)

        case .native, .prompted, .tool:
            guard let schemaJSON = output.schema else {
                throw AgentError.noOutput("structured output requested without a schema")
            }
            let schema = try decodeSchema(schemaJSON)
            let response = try await session.respond(
                to: prompt, schema: schema, includeSchemaInPrompt: true)
            let json = try response.content.toJSONValue()

            if output.mode == .tool {
                // Surface as a forced-output tool call so the run loop finalizes.
                return ModelResponse(parts: [.toolCall(ToolCall(
                    id: "apple-output", name: output.name, arguments: json))], usage: usage,
                    modelName: modelName, finishReason: .toolCall)
            }
            return ModelResponse(parts: [.text(try json.jsonString())], usage: usage,
                                 modelName: modelName, finishReason: .stop)
        }
    }

    /// Reconstruct a `GenerationSchema` from the vendor-neutral JSON we emitted.
    private func decodeSchema(_ json: JSONValue) throws -> GenerationSchema {
        let data = Data(try json.jsonString().utf8)
        return try JSONDecoder().decode(GenerationSchema.self, from: data)
    }
}

/// Registers the `"apple:<anything>"` selector with `ModelRegistry`.
public enum AgentApple {
    public static func register() {
        ModelRegistry.shared.register(provider: "apple") { _ in
            AppleModel()
        }
    }

    /// Whether the on-device model is currently usable.
    public static var isAvailable: Bool {
        if case .available = SystemLanguageModel.default.availability { return true }
        return false
    }
}
