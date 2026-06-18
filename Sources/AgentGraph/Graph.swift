/// A reusable, asynchronous state machine over a set of `GraphNode`s — the Swift
/// analogue of `pydantic_graph`'s `Graph`, generic over its shared `State`, its
/// `Deps`, and the `RunEnd` value a run produces.
///
/// A graph is constructed from its node *types* (for the diagram and node-set
/// validation) and executed from a `start` node. Runtime transitions follow the
/// node each `run` returns; the registered `edges` are used only by
/// `mermaidCode()`. See `GraphNode` for why edges are declared rather than
/// inferred.
public struct Graph<State: Sendable, Deps: Sendable, RunEnd: Sendable>: Sendable {
    /// A registered node's id and declared outgoing edges (for diagram/diagnostics).
    public struct NodeDescriptor: Sendable {
        public let id: String
        public let edges: [String]
    }

    /// The nodes making up this graph (for diagram/diagnostics).
    public let nodes: [NodeDescriptor]
    /// An optional name, used as the diagram title.
    public let name: String?

    public init(name: String? = nil, nodes: [any GraphNodeRegistration.Type]) {
        self.name = name
        self.nodes = nodes.map { NodeDescriptor(id: $0.graphNodeID, edges: $0.edges) }
    }

    /// Drive the graph from `start` to its end and return the terminal value.
    public func run(
        start: any GraphNode<State, Deps, RunEnd>,
        state: State,
        deps: Deps,
        persistence: any GraphPersistence<State> = NoOpPersistence<State>()
    ) async throws -> RunEnd {
        let run = iter(start: start, state: state, deps: deps, persistence: persistence)
        while try await run.next() != nil {}
        guard let result = run.result else { throw GraphError.didNotReachEnd }
        return result
    }

    /// Begin an observable run, stepping node-by-node. The caller drives it as an
    /// `AsyncSequence` (or via `GraphRun.next(_:)`).
    public func iter(
        start: any GraphNode<State, Deps, RunEnd>,
        state: State,
        deps: Deps,
        persistence: any GraphPersistence<State> = NoOpPersistence<State>()
    ) -> GraphRun<State, Deps, RunEnd> {
        GraphRun(start: start, ctx: GraphRunContext(state: state, deps: deps), persistence: persistence)
    }

    /// A mermaid `flowchart` of the declared nodes and edges.
    public func mermaidCode() -> String {
        var lines = ["flowchart TD"]
        if let name { lines.insert("%% \(name)", at: 0) }
        for node in nodes {
            if node.edges.isEmpty {
                lines.append("    \(node.id)")
            } else {
                for target in node.edges {
                    lines.append("    \(node.id) --> \(target)")
                }
            }
        }
        return lines.joined(separator: "\n")
    }
}

/// Errors raised by the graph engine itself.
public enum GraphError: Error, Sendable, Equatable {
    /// A `run` finished iterating without any node returning `.end`.
    case didNotReachEnd
}
