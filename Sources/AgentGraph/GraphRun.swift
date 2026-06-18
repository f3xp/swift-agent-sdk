/// A single, observable execution of a `Graph` — the Swift analogue of
/// `pydantic_graph`'s `GraphRun` (and what powers pydantic-ai's `agent.iter()`).
///
/// `GraphRun` is an `AsyncSequence` of the nodes it executes, in order. Each
/// element is yielded *before* it runs, so the consumer can inspect or stream a
/// node and only then advance — advancing (the next `next()` call) runs the
/// previously-yielded node and yields whatever it transitions to. The terminal
/// `.end` value is never yielded as a node; it lands in `result` once iteration
/// completes.
///
/// One run iterates once. It is not `Sendable`: create and drive it within a
/// single concurrency domain (nodes run sequentially — never concurrently).
public final class GraphRun<State: Sendable, Deps: Sendable, RunEnd: Sendable>: AsyncSequence, AsyncIteratorProtocol {
    public typealias Element = any GraphNode<State, Deps, RunEnd>

    /// The run's shared context (state + deps).
    public let ctx: GraphRunContext<State, Deps>
    /// The terminal value, set once the run reaches `.end`.
    public private(set) var result: RunEnd?

    private let persistence: any GraphPersistence<State>
    private var current: (any GraphNode<State, Deps, RunEnd>)?
    private var started = false
    private var finished = false

    init(
        start: any GraphNode<State, Deps, RunEnd>,
        ctx: GraphRunContext<State, Deps>,
        persistence: any GraphPersistence<State>
    ) {
        self.current = start
        self.ctx = ctx
        self.persistence = persistence
    }

    public func makeAsyncIterator() -> GraphRun { self }

    public func next() async throws -> Element? {
        if finished { return nil }
        if !started {
            // Yield the start node without running it yet.
            started = true
            return current
        }
        guard let node = current else {
            finished = true
            return nil
        }
        let step = try await execute(node)
        switch step {
        case let .end(value):
            result = value
            current = nil
            finished = true
            return nil
        case let .next(node):
            current = node
            return node
        }
    }

    /// Run a specific node and advance the run to its result. Mirrors
    /// `pydantic_graph`'s manual `next(node)` stepping. The returned step is also
    /// reflected in `result` (on `.end`) and the run's internal cursor.
    @discardableResult
    public func next(_ node: any GraphNode<State, Deps, RunEnd>) async throws -> NextStep<State, Deps, RunEnd> {
        let step = try await execute(node)
        switch step {
        case let .end(value):
            result = value
            current = nil
            finished = true
        case let .next(node):
            current = node
        }
        return step
    }

    private func execute(_ node: any GraphNode<State, Deps, RunEnd>) async throws -> NextStep<State, Deps, RunEnd> {
        await persistence.beforeNode(id: node.nodeID, state: ctx.state)
        let step = try await node.run(ctx)
        if case let .end(value) = step {
            await persistence.recordResult(value)
        }
        return step
    }
}
