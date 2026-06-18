import AgentCore
import AgentHTTP
import Foundation

/// An OpenAI Chat Completions model.
///
/// Also the gateway to OpenAI-compatible providers (Ollama, OpenRouter, Together,
/// LiteLLM, …): point `baseURL` at their endpoint.
public struct OpenAIModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile

    let apiKey: String
    let baseURL: URL
    let organization: String?
    let client: HTTPClient

    public init(
        model: String,
        apiKey: String? = nil,
        baseURL: URL = URL(string: "https://api.openai.com/v1")!,
        organization: String? = nil,
        client: HTTPClient = HTTPClient()
    ) {
        self.modelName = model
        self.apiKey = apiKey ?? ProcessInfo.processInfo.environment["OPENAI_API_KEY"] ?? ""
        self.baseURL = baseURL
        self.organization = organization
        self.client = client
        // OpenAI supports native JSON-schema structured output (strict mode).
        self.profile = ModelProfile(
            supportsNativeStructuredOutput: true,
            supportsParallelToolCalls: true,
            defaultOutputMode: .native,
            jsonSchemaTransform: OpenAISchema.strict)
    }

    private var completionsURL: URL { baseURL.appendingPathComponent("chat/completions") }

    private var headers: [String: String] {
        var h = [
            "Authorization": "Bearer \(apiKey)",
            "content-type": "application/json",
        ]
        if let organization { h["OpenAI-Organization"] = organization }
        return h
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        let body = OpenAIWire.encodeRequest(
            model: modelName, messages: messages, settings: settings, tools: tools, output: output)
        let json = try await client.postJSON(url: completionsURL, headers: headers, body: body)
        return try OpenAIWire.decodeResponse(json, modelName: modelName)
    }

    public func stream(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> AsyncThrowingStream<ModelStreamEvent, any Error> {
        let body = OpenAIWire.encodeRequest(
            model: modelName, messages: messages, settings: settings,
            tools: tools, output: output, stream: true)
        let url = completionsURL
        let client = self.client
        let headers = self.headers
        let modelName = self.modelName
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var decoder = OpenAIStreamDecoder(modelName: modelName)
                    for try await sse in client.streamSSE(url: url, headers: headers, body: body) {
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

/// Registers the `"openai:<model>"` selector with `ModelRegistry`.
public enum AgentOpenAI {
    public static func register() {
        ModelRegistry.shared.register(provider: "openai") { model in
            OpenAIModel(model: model)
        }
    }
}
