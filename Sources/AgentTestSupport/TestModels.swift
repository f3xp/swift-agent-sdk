import AgentCore
import Foundation

/// A fully scriptable model: you supply a handler that decides each turn's
/// response from the conversation so far. The flexible primitive for run-loop
/// tests (tool cycles, retries, multi-turn).
public struct FunctionModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile
    let handler: @Sendable (_ messages: [ModelMessage], _ tools: [ToolDefinition], _ output: OutputSpec) async throws -> ModelResponse

    public init(
        modelName: String = "function",
        profile: ModelProfile = .default,
        handler: @escaping @Sendable (_ messages: [ModelMessage], _ tools: [ToolDefinition], _ output: OutputSpec) async throws -> ModelResponse
    ) {
        self.modelName = modelName
        self.profile = profile
        self.handler = handler
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        try await handler(messages, tools, output)
    }
}

/// A deterministic, no-network model for simple cases.
///
/// - `.text` output: returns `text` (default `"success"`).
/// - `.tool` output: calls the output tool with `structuredOutput`.
/// - `.native`/`.prompted`: returns `structuredOutput` as a JSON text part.
public struct TestModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile
    let text: String
    let structuredOutput: JSONValue?

    public init(
        modelName: String = "test",
        text: String = "success",
        structuredOutput: JSONValue? = nil,
        profile: ModelProfile = .default
    ) {
        self.modelName = modelName
        self.text = text
        self.structuredOutput = structuredOutput
        self.profile = profile
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        let usage = Usage(requests: 1, inputTokens: 10, outputTokens: 5)
        switch output.mode {
        case .text:
            return ModelResponse(parts: [.text(text)], usage: usage, finishReason: .stop)
        case .tool:
            guard let value = structuredOutput else {
                throw AgentError.noOutput("TestModel needs `structuredOutput` for tool-mode output")
            }
            let call = ToolCall(id: "test-output-1", name: output.name, arguments: value)
            return ModelResponse(parts: [.toolCall(call)], usage: usage, finishReason: .toolCall)
        case .native, .prompted:
            guard let value = structuredOutput else {
                throw AgentError.noOutput("TestModel needs `structuredOutput` for structured output")
            }
            return ModelResponse(parts: [.text(try value.jsonString())], usage: usage, finishReason: .stop)
        }
    }
}

/// A model that replays a fixed script of streaming events. Drives run-loop
/// streaming tests (partials, tool-arg deltas) that the default `stream()` shim
/// can't exercise. `request()` returns the script's final `.completed` response.
public struct ScriptedStreamModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile
    let events: [ModelStreamEvent]

    public init(
        modelName: String = "scripted",
        profile: ModelProfile = .default,
        events: [ModelStreamEvent]
    ) {
        self.modelName = modelName
        self.profile = profile
        self.events = events
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        for event in events {
            if case let .completed(response) = event { return response }
        }
        throw AgentError.noOutput("ScriptedStreamModel script has no .completed event")
    }

    public func stream(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> AsyncThrowingStream<ModelStreamEvent, any Error> {
        let events = self.events
        return AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }
}

/// Tries each model in order, advancing to the next on a thrown error.
/// Mirrors pydantic-ai's `FallbackModel`.
public struct FallbackModel: ModelProtocol {
    public let modelName: String
    let models: [any ModelProtocol]

    public init(_ models: [any ModelProtocol]) {
        precondition(!models.isEmpty, "FallbackModel requires at least one model")
        self.models = models
        self.modelName = "fallback(" + models.map(\.modelName).joined(separator: ",") + ")"
    }

    public var profile: ModelProfile { models[0].profile }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        var lastError: (any Error)?
        for model in models {
            do {
                return try await model.request(
                    messages: messages, settings: settings, tools: tools, output: output)
            } catch {
                lastError = error
            }
        }
        throw lastError ?? AgentError.provider("FallbackModel: all models failed")
    }
}
