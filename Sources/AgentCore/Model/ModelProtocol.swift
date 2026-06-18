import Foundation

/// Opt-in extended-thinking / reasoning configuration. Mapped per provider
/// (Anthropic `thinking`, Gemini `thinkingConfig`); ignored where unsupported.
public struct ThinkingConfig: Sendable, Equatable {
    /// Token budget for thinking. `nil` lets the provider pick a default.
    public var budgetTokens: Int?
    /// Whether the provider should surface thought summaries in the response.
    public var includeThoughts: Bool

    public init(budgetTokens: Int? = nil, includeThoughts: Bool = true) {
        self.budgetTokens = budgetTokens
        self.includeThoughts = includeThoughts
    }
}

/// Per-request generation settings, mapped onto each provider's parameters.
public struct ModelSettings: Sendable, Equatable {
    public var temperature: Double?
    public var maxTokens: Int?
    public var topP: Double?
    public var stopSequences: [String]?
    /// Opt-in extended thinking / reasoning. `nil` leaves it off.
    public var thinking: ThinkingConfig?

    public init(
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        stopSequences: [String]? = nil,
        thinking: ThinkingConfig? = nil
    ) {
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.stopSequences = stopSequences
        self.thinking = thinking
    }
}

/// The wire-ready, type-erased description of a tool. The model only ever sees
/// definitions — it never executes them.
public struct ToolDefinition: Sendable, Equatable, Codable {
    public var name: String
    public var description: String
    /// JSON Schema for the tool's arguments (an object schema).
    public var parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// How the agent extracts structured output from a model.
public enum OutputMode: String, Sendable, Equatable, Codable {
    /// Plain text (`Output == String`). No schema.
    case text
    /// A forced "output tool" whose arguments are the structured output.
    case tool
    /// The provider's native structured-output / JSON-schema response feature.
    case native
    /// The schema is injected into the prompt and the model returns JSON text.
    case prompted
}

/// Describes the desired output for a single model request.
public struct OutputSpec: Sendable, Equatable {
    public var mode: OutputMode
    public var name: String
    public var description: String
    /// JSON Schema for the output. `nil` only for `.text`.
    public var schema: JSONValue?

    public init(mode: OutputMode, name: String = "final_result", description: String = "The final result", schema: JSONValue? = nil) {
        self.mode = mode
        self.name = name
        self.description = description
        self.schema = schema
    }

    public static let text = OutputSpec(mode: .text)
}

/// Vendor-specific quirks that shape how requests/schemas are constructed.
public struct ModelProfile: Sendable {
    public var supportsNativeStructuredOutput: Bool
    public var supportsParallelToolCalls: Bool
    public var defaultOutputMode: OutputMode
    /// Normalizes a JSON Schema into this provider's accepted dialect
    /// (e.g. OpenAI strict mode → `additionalProperties:false` + all keys required).
    public var jsonSchemaTransform: @Sendable (JSONValue) -> JSONValue

    public init(
        supportsNativeStructuredOutput: Bool = false,
        supportsParallelToolCalls: Bool = true,
        defaultOutputMode: OutputMode = .tool,
        jsonSchemaTransform: @escaping @Sendable (JSONValue) -> JSONValue = { $0 }
    ) {
        self.supportsNativeStructuredOutput = supportsNativeStructuredOutput
        self.supportsParallelToolCalls = supportsParallelToolCalls
        self.defaultOutputMode = defaultOutputMode
        self.jsonSchemaTransform = jsonSchemaTransform
    }

    public static let `default` = ModelProfile()
}

/// An incremental fragment of a streamed tool call. Providers emit these as the
/// arguments arrive; the call is also surfaced whole via `.toolCall` once complete.
public struct ToolCallDelta: Sendable, Equatable {
    /// Position of this call within the response (correlates fragments).
    public var index: Int
    /// The call id, usually present only on the first fragment.
    public var id: String?
    /// The tool name, usually present only on the first fragment.
    public var name: String?
    /// A fragment of the JSON arguments string.
    public var argumentsFragment: String

    public init(index: Int, id: String? = nil, name: String? = nil, argumentsFragment: String = "") {
        self.index = index
        self.id = id
        self.name = name
        self.argumentsFragment = argumentsFragment
    }
}

/// An incremental event emitted while streaming a single model request.
public enum ModelStreamEvent: Sendable, Equatable {
    case textDelta(String)
    case thinkingDelta(String)
    /// An incremental fragment of a tool call's arguments.
    case toolCallDelta(ToolCallDelta)
    case toolCall(ToolCall)
    /// Terminal event carrying the fully-assembled response.
    case completed(ModelResponse)
}

/// The vendor-agnostic model interface. The agent loop calls `request` (or
/// `stream`) repeatedly: request → tool calls → tool results → re-request.
public protocol ModelProtocol: Sendable {
    var modelName: String { get }
    var profile: ModelProfile { get }

    /// Perform one round trip and return the assembled response.
    func request(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) async throws -> ModelResponse

    /// Streaming variant; the run loop folds deltas and ends on `.completed`.
    func stream(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> AsyncThrowingStream<ModelStreamEvent, any Error>
}

extension ModelProtocol {
    public var profile: ModelProfile { .default }

    /// Default streaming shim: run the non-streaming request and emit one
    /// `.completed`. Providers with native streaming override this.
    public func stream(
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> AsyncThrowingStream<ModelStreamEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let response = try await request(
                        messages: messages, settings: settings, tools: tools, output: output)
                    for part in response.parts {
                        switch part {
                        case let .text(t): continuation.yield(.textDelta(t))
                        case let .thinking(t): continuation.yield(.thinkingDelta(t))
                        case let .toolCall(c): continuation.yield(.toolCall(c))
                        }
                    }
                    continuation.yield(.completed(response))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
