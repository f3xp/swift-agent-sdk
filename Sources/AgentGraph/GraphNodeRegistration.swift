/// Type-level metadata a `Graph` uses to render its diagram and validate its
/// node set. Kept separate from `GraphNode` (which has associated types) so a
/// graph's nodes can be registered as a homogeneous `[any GraphNodeRegistration.Type]`.
///
/// `edges` is declared by hand because Swift can't infer them from `run`'s return
/// type. They feed `Graph.mermaidCode()` and diagnostics only — never runtime
/// transitions, which always follow the node a `run` call returns.
public protocol GraphNodeRegistration {
    /// The node's stable identifier. Defaults to the concrete type name and must
    /// match the corresponding `GraphNode.nodeID`.
    static var graphNodeID: String { get }
    /// The ids of nodes this node may transition to. Use `GraphEnd` for a
    /// terminal edge (a `run` that can return `.end`).
    static var edges: [String] { get }
}

extension GraphNodeRegistration {
    public static var graphNodeID: String { String(describing: Self.self) }
}

/// The sentinel edge target denoting a transition to the run's end (`.end`).
public let GraphEnd = "End"
