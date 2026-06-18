import AgentCore
import AgentGraph
import FoundationModels

/// A handle to a streamable graph node — call `events()` to drive the node's
/// model/tool work and observe its stream. The node only executes once (whether
/// via `events()` here or when the run advances past it), so streaming a node and
/// then advancing the run does not double-run it.
public struct NodeStream<Output: Sendable>: Sendable {
    let make: @Sendable () -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error>

    /// The node's event stream (deltas/partials for a model request; tool returns
    /// for a tool call).
    public func events() -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error> { make() }
}

/// One node of an agent run, surfaced by `Agent.iter()`. Mirrors pydantic-ai's
/// `UserPromptNode` / `ModelRequestNode` / `CallToolsNode` / `End`.
public enum AgentNode<Output: Sendable>: Sendable {
    /// The initial node that builds the first request from the prompt.
    case userPrompt
    /// A request to the model. Stream it via the attached `NodeStream`.
    case modelRequest(NodeStream<Output>)
    /// Handling of the model's response (output finalization or tool calls).
    case callTools(NodeStream<Output>)
    /// The terminal node, carrying the completed result.
    case end(AgentRunResult<Output>)
}

/// An observable, steppable agent run — the value returned by `Agent.iter()`.
///
/// Iterate it as an `AsyncSequence` to walk the run node-by-node; each node is
/// yielded *before* it executes, so a `.modelRequest` / `.callTools` node can be
/// streamed (via its `NodeStream`) before the run advances past it. After
/// iteration completes, `result` holds the final result. Drive within a single
/// task (the run is not `Sendable`).
public final class AgentRun<Deps: Sendable, Output: AgentSchema & Sendable>: AsyncSequence, AsyncIteratorProtocol {
    public typealias Element = AgentNode<Output>

    private let graphRun: GraphRun<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>>
    private var endEmitted = false

    init(graphRun: GraphRun<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>>) {
        self.graphRun = graphRun
    }

    /// The completed result, available once the run reaches its end.
    public var result: AgentRunResult<Output>? { graphRun.result }
    /// Usage accumulated so far.
    public var usage: Usage { graphRun.ctx.state.usage }
    /// Conversation history so far.
    public var messages: [ModelMessage] { graphRun.ctx.state.messages }
    /// The current run step (model requests completed so far).
    public var runStep: Int { graphRun.ctx.state.runStep }

    public func makeAsyncIterator() -> AgentRun { self }

    public func next() async throws -> AgentNode<Output>? {
        if let node = try await graphRun.next() {
            return wrap(node)
        }
        if let result = graphRun.result, !endEmitted {
            endEmitted = true
            return .end(result)
        }
        return nil
    }

    private func wrap(_ node: any GraphNode<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>>) -> AgentNode<Output> {
        let ctx = graphRun.ctx
        if let modelRequest = node as? ModelRequestNode<Deps, Output> {
            return .modelRequest(NodeStream { modelRequest.stream(ctx) })
        } else if let callTools = node as? CallToolsNode<Deps, Output> {
            return .callTools(NodeStream { callTools.stream(ctx) })
        } else {
            return .userPrompt
        }
    }
}

extension Agent {
    /// Begin an observable, steppable run — the third run entry point alongside
    /// `run` and `runStream` (pydantic-ai's `agent.iter()`). Walk the returned
    /// `AgentRun` as an `AsyncSequence` to observe each node; stream `.modelRequest`
    /// / `.callTools` nodes for fine-grained events. Read `result` once it ends.
    public func iter(
        _ prompt: String,
        attachments: [MediaContent] = [],
        deps: Deps,
        usageLimits: UsageLimits = .none,
        messageHistory: [ModelMessage] = []
    ) throws -> AgentRun<Deps, Output> {
        let run = try makeGraphRun(
            prompt: prompt, attachments: attachments, deps: deps,
            usageLimits: usageLimits, messageHistory: messageHistory)
        return AgentRun(graphRun: run)
    }
}

extension Agent where Deps == Void {
    public func iter(
        _ prompt: String,
        attachments: [MediaContent] = [],
        usageLimits: UsageLimits = .none,
        messageHistory: [ModelMessage] = []
    ) throws -> AgentRun<Void, Output> {
        try iter(
            prompt, attachments: attachments, deps: (),
            usageLimits: usageLimits, messageHistory: messageHistory)
    }
}
