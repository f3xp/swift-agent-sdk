/// The persistence seam for a graph run, called at each node boundary.
///
/// This is the hook `pydantic_graph` builds durable snapshot persistence and
/// resume on. W2 ships only the seam plus a no-op default: `beforeNode` fires
/// before every node runs, and `recordResult` fires when the run reaches `.end`.
///
/// Durable, `Codable`-snapshot-based persistence and resume-from-snapshot are
/// intentionally deferred to a later workstream — implement them as a concrete
/// `GraphPersistence` once the run state's snapshot format is settled.
public protocol GraphPersistence<State>: Sendable {
    associatedtype State: Sendable

    /// Called immediately before a node runs. A durable implementation would
    /// snapshot `state` keyed by `id` here.
    func beforeNode(id: String, state: State) async

    /// Called once when the run reaches its end, with the terminal value.
    func recordResult<R: Sendable>(_ result: R) async
}

/// The default persistence: records nothing. Node boundaries still fire through
/// it, so swapping in a durable implementation is a one-line change.
public struct NoOpPersistence<State: Sendable>: GraphPersistence {
    public init() {}
    public func beforeNode(id: String, state: State) async {}
    public func recordResult<R: Sendable>(_ result: R) async {}
}
