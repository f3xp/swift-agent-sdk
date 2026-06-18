import Testing
import AgentGraph

// MARK: - A toy graph: count up to a target, ping-ponging through two nodes.

final class CounterState: @unchecked Sendable {
    var count: Int = 0
    var visits: [String] = []
    init() {}
}

struct CounterDeps: Sendable {
    let target: Int
}

/// Increments the counter; ends when it reaches the target, else hands off to `LogNode`.
struct IncrementNode: GraphNode, GraphNodeRegistration {
    typealias State = CounterState
    typealias Deps = CounterDeps
    typealias RunEnd = Int

    static var edges: [String] { ["LogNode", GraphEnd] }

    func run(_ ctx: GraphRunContext<CounterState, CounterDeps>) async throws -> NextStep<CounterState, CounterDeps, Int> {
        ctx.state.count += 1
        ctx.state.visits.append("IncrementNode")
        if ctx.state.count >= ctx.deps.target {
            return .end(ctx.state.count)
        }
        return .next(LogNode())
    }
}

struct LogNode: GraphNode, GraphNodeRegistration {
    typealias State = CounterState
    typealias Deps = CounterDeps
    typealias RunEnd = Int

    static var edges: [String] { ["IncrementNode"] }

    func run(_ ctx: GraphRunContext<CounterState, CounterDeps>) async throws -> NextStep<CounterState, CounterDeps, Int> {
        ctx.state.visits.append("LogNode")
        return .next(IncrementNode())
    }
}

func counterGraph() -> Graph<CounterState, CounterDeps, Int> {
    Graph(name: "Counter", nodes: [IncrementNode.self, LogNode.self])
}

// MARK: - A persistence spy.

actor PersistenceSpy: GraphPersistence {
    typealias State = CounterState
    private(set) var nodeIDs: [String] = []
    private(set) var recordedResult: Int?

    func beforeNode(id: String, state: CounterState) async { nodeIDs.append(id) }
    func recordResult<R: Sendable>(_ result: R) async { recordedResult = result as? Int }
}

@Suite("Graph engine")
struct GraphEngineTests {

    @Test("run drives to End and returns the terminal value")
    func runReachesEnd() async throws {
        let result = try await counterGraph().run(
            start: IncrementNode(), state: CounterState(), deps: CounterDeps(target: 3))
        #expect(result == 3)
    }

    @Test("iter yields each node in execution order before running it")
    func iterYieldsNodes() async throws {
        let run = counterGraph().iter(
            start: IncrementNode(), state: CounterState(), deps: CounterDeps(target: 3))
        var seen: [String] = []
        for try await node in run {
            seen.append(node.nodeID)
        }
        // 3 increments interleaved with 2 logs: Inc, Log, Inc, Log, Inc(end).
        #expect(seen == ["IncrementNode", "LogNode", "IncrementNode", "LogNode", "IncrementNode"])
        #expect(run.result == 3)
        #expect(run.ctx.state.count == 3)
    }

    @Test("manual next(_:) stepping reaches the same result")
    func manualStepping() async throws {
        let run = counterGraph().iter(
            start: IncrementNode(), state: CounterState(), deps: CounterDeps(target: 2))
        var node: (any GraphNode<CounterState, CounterDeps, Int>)? = IncrementNode()
        var ended: Int?
        while let current = node {
            switch try await run.next(current) {
            case let .next(n): node = n
            case let .end(value): ended = value; node = nil
            }
        }
        #expect(ended == 2)
    }

    @Test("mermaidCode renders declared nodes and edges")
    func mermaid() {
        let code = counterGraph().mermaidCode()
        #expect(code.contains("flowchart TD"))
        #expect(code.contains("IncrementNode --> LogNode"))
        #expect(code.contains("IncrementNode --> End"))
        #expect(code.contains("LogNode --> IncrementNode"))
    }

    @Test("persistence seam fires beforeNode per node and records the result")
    func persistenceSeam() async throws {
        let spy = PersistenceSpy()
        let result = try await counterGraph().run(
            start: IncrementNode(), state: CounterState(), deps: CounterDeps(target: 2),
            persistence: spy)
        #expect(result == 2)
        let ids = await spy.nodeIDs
        #expect(ids == ["IncrementNode", "LogNode", "IncrementNode"])
        let recorded = await spy.recordedResult
        #expect(recorded == 2)
    }
}
