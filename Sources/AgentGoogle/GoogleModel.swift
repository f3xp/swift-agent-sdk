import AgentCore
import AgentHTTP
import Foundation

/// A Google Gemini model speaking the Generative Language `generateContent` API.
public struct GoogleModel: ModelProtocol {
    public let modelName: String
    public let profile: ModelProfile

    let apiKey: String
    let baseURL: URL
    let client: HTTPClient

    public init(
        model: String,
        apiKey: String? = nil,
        baseURL: URL = URL(string: "https://generativelanguage.googleapis.com/v1beta")!,
        client: HTTPClient = HTTPClient()
    ) {
        self.modelName = model
        self.apiKey = apiKey
            ?? ProcessInfo.processInfo.environment["GEMINI_API_KEY"]
            ?? ProcessInfo.processInfo.environment["GOOGLE_API_KEY"]
            ?? ""
        self.baseURL = baseURL
        self.client = client
        self.profile = ModelProfile(
            supportsNativeStructuredOutput: true,
            supportsParallelToolCalls: true,
            defaultOutputMode: .native,
            jsonSchemaTransform: GeminiSchema.normalize)
    }

    private var generateURL: URL {
        baseURL.appendingPathComponent("models/\(modelName):generateContent")
    }

    private var streamURL: URL {
        let path = baseURL.appendingPathComponent("models/\(modelName):streamGenerateContent")
        var comps = URLComponents(url: path, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "alt", value: "sse")]
        return comps.url!
    }

    private var headers: [String: String] {
        [
            "x-goog-api-key": apiKey,
            "content-type": "application/json",
        ]
    }

    public func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse {
        let body = GeminiWire.encodeRequest(
            model: modelName, messages: messages, settings: settings, tools: tools, output: output)
        let json = try await client.postJSON(url: generateURL, headers: headers, body: body)
        return try GeminiWire.decodeResponse(json, modelName: modelName)
    }

    public func stream(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> AsyncThrowingStream<ModelStreamEvent, any Error> {
        let body = GeminiWire.encodeRequest(
            model: modelName, messages: messages, settings: settings, tools: tools, output: output)
        let url = streamURL
        let client = self.client
        let headers = self.headers
        let modelName = self.modelName
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var decoder = GeminiStreamDecoder(modelName: modelName)
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

/// Registers the `"google:<model>"` selector with `ModelRegistry`.
public enum AgentGoogle {
    public static func register() {
        ModelRegistry.shared.register(provider: "google") { model in
            GoogleModel(model: model)
        }
    }
}
