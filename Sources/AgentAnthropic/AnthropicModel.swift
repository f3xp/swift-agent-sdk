import AgentCore
import AgentHTTP
import Foundation

/// An Anthropic Claude model speaking the Messages API.
public struct AnthropicModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile

    let apiKey: String
    let baseURL: URL
    let apiVersion: String
    let client: HTTPClient

    public init(
        model: String,
        apiKey: String? = nil,
        baseURL: URL = URL(string: "https://api.anthropic.com")!,
        apiVersion: String = "2023-06-01",
        client: HTTPClient = HTTPClient()
    ) {
        self.modelName = model
        self.apiKey = apiKey ?? ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? ""
        self.baseURL = baseURL
        self.apiVersion = apiVersion
        self.client = client
        // Anthropic has no native JSON-schema response mode; we use the
        // forced-output-tool strategy. Its `input_schema` accepts standard
        // JSON Schema, so the transform is identity in Phase 1.
        self.profile = ModelProfile(
            supportsNativeStructuredOutput: false,
            supportsParallelToolCalls: true,
            defaultOutputMode: .tool,
            jsonSchemaTransform: { $0 })
    }

    private var messagesURL: URL { baseURL.appendingPathComponent("v1/messages") }

    private var headers: [String: String] {
        [
            "x-api-key": apiKey,
            "anthropic-version": apiVersion,
            "content-type": "application/json",
        ]
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        let body = AnthropicWire.encodeRequest(
            model: modelName, messages: messages, settings: settings, tools: tools, output: output)
        let json = try await client.postJSON(url: messagesURL, headers: headers, body: body)
        return try AnthropicWire.decodeResponse(json, modelName: modelName)
    }

    public func stream(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> AsyncThrowingStream<ModelStreamEvent, any Error> {
        let body = AnthropicWire.encodeRequest(
            model: modelName, messages: messages, settings: settings,
            tools: tools, output: output, stream: true)
        let streamHeaders = headers.merging(["accept": "text/event-stream"]) { _, new in new }
        let url = messagesURL
        let client = self.client
        let modelName = self.modelName
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var decoder = AnthropicStreamDecoder(modelName: modelName)
                    for try await sse in client.streamSSE(url: url, headers: streamHeaders, body: body) {
                        for event in decoder.ingest(sse.data) { continuation.yield(event) }
                    }
                    for event in decoder.finish() { continuation.yield(event) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Registers the `"anthropic:<model>"` selector with `ModelRegistry`.
public enum AgentAnthropic {
    public static func register() {
        ModelRegistry.shared.register(provider: "anthropic") { model in
            AnthropicModel(model: model)
        }
    }
}
