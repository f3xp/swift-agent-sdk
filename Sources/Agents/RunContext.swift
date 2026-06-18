import AgentCore

/// The context injected into every tool call, dynamic instruction, and output
/// validator. Carries the agent's dependencies plus run state.
///
/// This is the concept Apple's `FoundationModels.Tool` lacks — dependency
/// injection — which is why we define our own tool abstraction on top.
public struct RunContext<Deps: Sendable>: Sendable {
    /// User-supplied dependencies, passed at `run(deps:)`.
    public let deps: Deps
    /// Usage accumulated so far this run.
    public let usage: Usage
    /// The current retry index (0 on the first attempt).
    public let retry: Int
    /// The model executing this run.
    public let model: any ModelProtocol
    /// Conversation history so far.
    public let messages: [ModelMessage]
    /// The current step of the run — incremented once per model request
    /// (pydantic-ai's `run_step`). `0` before the first request completes.
    public let runStep: Int
    /// The id of the tool call currently being serviced (when in a tool).
    public let toolCallID: String?

    public init(
        deps: Deps,
        usage: Usage,
        retry: Int,
        model: any ModelProtocol,
        messages: [ModelMessage],
        runStep: Int = 0,
        toolCallID: String? = nil
    ) {
        self.deps = deps
        self.usage = usage
        self.retry = retry
        self.model = model
        self.messages = messages
        self.runStep = runStep
        self.toolCallID = toolCallID
    }
}
