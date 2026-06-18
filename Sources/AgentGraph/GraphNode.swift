/// A node in a `Graph` — one step of an asynchronous state machine.
///
/// The Swift analogue of `pydantic_graph`'s `BaseNode`. A node's `run` performs
/// its work against the shared `GraphRunContext` and returns the next step: either
/// another node to execute (`.next`) or the run's terminal value (`.end`).
///
/// Nodes sharing the same `State`, `Deps`, and `RunEnd` form one graph and can be
/// held as a single `any GraphNode<State, Deps, RunEnd>` existential (Swift 6
/// primary associated types).
///
/// Unlike `pydantic_graph`, Swift can't infer the graph's edges from `run`'s
/// return-type annotation — declare them via `GraphNodeRegistration` for the
/// mermaid/diagnostics view. Those declarations are documentation only; the
/// runtime always follows the node a `run` call actually returns.
public protocol GraphNode<State, Deps, RunEnd>: Sendable {
    associatedtype State: Sendable
    associatedtype Deps: Sendable
    associatedtype RunEnd: Sendable

    /// A stable identifier for this node, used for persistence snapshots and
    /// run observation. Defaults to the concrete type name.
    var nodeID: String { get }

    /// Perform this node's work and return the next step.
    func run(_ ctx: GraphRunContext<State, Deps>) async throws -> NextStep<State, Deps, RunEnd>
}

extension GraphNode {
    public var nodeID: String { String(describing: type(of: self)) }
}

/// The outcome of running a `GraphNode`: continue to another node, or end the
/// run with a terminal value (mirrors `pydantic_graph`'s `BaseNode | End`).
public enum NextStep<State: Sendable, Deps: Sendable, RunEnd: Sendable>: Sendable {
    /// Continue the run at another node.
    case next(any GraphNode<State, Deps, RunEnd>)
    /// End the run, carrying its final value.
    case end(RunEnd)
}
