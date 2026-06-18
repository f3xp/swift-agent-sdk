import AgentCore
import FoundationModels

/// What to do when the model emits the final-result (output) tool call alongside
/// other tool calls in the same response (mirrors pydantic-ai's `EndStrategy`).
public enum EndStrategy: String, Sendable, Equatable {
    /// Finalize immediately, ignoring the sibling tool calls (default).
    case early
    /// Run the sibling tool calls (for their side effects) before finalizing.
    case exhaustive
}

/// The central agent type, generic over its dependency type and its output type
/// — the Swift analogue of pydantic-ai's `Agent[DepsType, OutputType]`.
///
/// `Agent` is an immutable, `Sendable` value: configuration builders
/// (`tool`, `instructions`, `outputValidator`, …) return a configured copy, so
/// they chain naturally. Mutable per-run state lives only inside `run`.
public struct Agent<Deps: Sendable, Output: AgentSchema & Sendable>: Sendable {
    enum InstructionsSource: Sendable {
        case `static`(String)
        case dynamic(@Sendable (RunContext<Deps>) async throws -> String)
    }

    var model: any ModelProtocol
    var instructionSources: [InstructionsSource]
    var tools: [AnyAgentTool<Deps>]
    var outputValidators: [@Sendable (Output, RunContext<Deps>) async throws -> Output]
    var outputModeOverride: OutputMode?
    var retries: Int
    var settings: ModelSettings
    var endStrategy: EndStrategy

    // MARK: Init

    public init(
        _ model: any ModelProtocol,
        deps: Deps.Type = Deps.self,
        output: Output.Type = Output.self,
        instructions: String? = nil,
        outputMode: OutputMode? = nil,
        retries: Int = 1,
        settings: ModelSettings = .init(),
        endStrategy: EndStrategy = .early
    ) {
        self.model = model
        self.instructionSources = instructions.map { [.static($0)] } ?? []
        self.tools = []
        self.outputValidators = []
        self.outputModeOverride = outputMode
        self.retries = retries
        self.settings = settings
        self.endStrategy = endStrategy
    }

    /// Resolve a `"provider:model"` selector via `ModelRegistry`.
    public init(
        _ selector: ModelSelector,
        deps: Deps.Type = Deps.self,
        output: Output.Type = Output.self,
        instructions: String? = nil,
        outputMode: OutputMode? = nil,
        retries: Int = 1,
        settings: ModelSettings = .init(),
        endStrategy: EndStrategy = .early
    ) throws {
        let model = try ModelRegistry.shared.resolve(selector)
        self.init(
            model, deps: deps, output: output, instructions: instructions,
            outputMode: outputMode, retries: retries, settings: settings,
            endStrategy: endStrategy)
    }

    // MARK: Builders

    /// Register a tool from a typed closure that receives the `RunContext`.
    public func tool<A: AgentSchema>(
        _ name: String,
        _ description: String,
        _ body: @escaping @Sendable (A, RunContext<Deps>) async throws -> ToolResult
    ) -> Self {
        var copy = self
        copy.tools.append(.closure(name: name, description: description, body: body))
        return copy
    }

    /// Register a concrete `AgentTool`.
    public func tool<T: AgentTool>(_ tool: T) -> Self where T.Deps == Deps {
        var copy = self
        copy.tools.append(.erasing(tool))
        return copy
    }

    /// Append a static instruction.
    public func instructions(_ text: String) -> Self {
        var copy = self
        copy.instructionSources.append(.static(text))
        return copy
    }

    /// Append a dynamic instruction computed from the run context.
    public func instructions(
        _ make: @escaping @Sendable (RunContext<Deps>) async throws -> String
    ) -> Self {
        var copy = self
        copy.instructionSources.append(.dynamic(make))
        return copy
    }

    /// Add an output validator; throw `ModelRetry` to request self-correction.
    public func outputValidator(
        _ validator: @escaping @Sendable (Output, RunContext<Deps>) async throws -> Output
    ) -> Self {
        var copy = self
        copy.outputValidators.append(validator)
        return copy
    }

    // MARK: Run

    public func run(
        _ prompt: String,
        attachments: [MediaContent] = [],
        deps: Deps,
        usageLimits: UsageLimits = .none,
        messageHistory: [ModelMessage] = []
    ) async throws -> AgentRunResult<Output> {
        try await runToCompletion(
            prompt: prompt, attachments: attachments, deps: deps,
            usageLimits: usageLimits, messageHistory: messageHistory)
    }

    public func runStream(
        _ prompt: String,
        attachments: [MediaContent] = [],
        deps: Deps,
        usageLimits: UsageLimits = .none,
        messageHistory: [ModelMessage] = []
    ) -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error> {
        runStreamImpl(
            prompt: prompt, attachments: attachments, deps: deps,
            usageLimits: usageLimits, messageHistory: messageHistory)
    }
}

// MARK: - Void-deps conveniences

extension Agent where Deps == Void {
    public init(
        _ model: any ModelProtocol,
        output: Output.Type = Output.self,
        instructions: String? = nil,
        outputMode: OutputMode? = nil,
        retries: Int = 1,
        settings: ModelSettings = .init(),
        endStrategy: EndStrategy = .early
    ) {
        self.init(
            model, deps: Void.self, output: output, instructions: instructions,
            outputMode: outputMode, retries: retries, settings: settings,
            endStrategy: endStrategy)
    }

    public func run(
        _ prompt: String,
        attachments: [MediaContent] = [],
        usageLimits: UsageLimits = .none,
        messageHistory: [ModelMessage] = []
    ) async throws -> AgentRunResult<Output> {
        try await run(
            prompt, attachments: attachments, deps: (), usageLimits: usageLimits,
            messageHistory: messageHistory)
    }

    public func runStream(
        _ prompt: String,
        attachments: [MediaContent] = [],
        usageLimits: UsageLimits = .none
    ) -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error> {
        runStream(prompt, attachments: attachments, deps: (), usageLimits: usageLimits)
    }
}
