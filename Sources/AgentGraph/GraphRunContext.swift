/// The context handed to every `GraphNode.run` — the run's shared `state` plus
/// its immutable `deps` (the Swift analogue of `pydantic_graph`'s
/// `GraphRunContext`).
///
/// `state` is typically a *reference* type (a `final class`), so mutations made
/// by one node are visible to the next — matching `pydantic_graph`, which mutates
/// its state object in place. If `State` is a value type, the caller is
/// responsible for threading updates between nodes.
public struct GraphRunContext<State: Sendable, Deps: Sendable>: Sendable {
    /// The run's shared, mutable state.
    public let state: State
    /// The run's immutable dependencies.
    public let deps: Deps

    public init(state: State, deps: Deps) {
        self.state = state
        self.deps = deps
    }
}
