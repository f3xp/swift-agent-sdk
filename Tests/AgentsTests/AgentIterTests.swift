import Testing
import FoundationModels
import AgentCore
import Agents
import AgentTestSupport

/// Records the `runStep` observed inside a tool call.
actor StepRecorder {
    private(set) var step = -1
    func set(_ s: Int) { step = s }
}

@Suite("Agent iter()")
struct AgentIterTests {

    private func label<O>(_ node: AgentNode<O>) -> String {
        switch node {
        case .userPrompt: return "userPrompt"
        case .modelRequest: return "modelRequest"
        case .callTools: return "callTools"
        case .end: return "end"
        }
    }

    @Test("iter yields UserPrompt → ModelRequest → CallTools → End")
    func nodeSequence() async throws {
        let agent = Agent<Void, String>(TestModel(text: "Hello"))
        let run = try agent.iter("hi")
        var labels: [String] = []
        for try await node in run { labels.append(label(node)) }
        #expect(labels == ["userPrompt", "modelRequest", "callTools", "end"])
        #expect(run.result?.output == "Hello")
        #expect(run.runStep == 1)
    }

    @Test("a tool round-trip produces two model requests")
    func toolRoundTrip() async throws {
        let model = FunctionModel { messages, _, _ in
            let hasToolReturn = messages.contains { msg in
                msg.asRequest?.parts.contains { if case .toolReturn = $0 { return true } else { return false } } ?? false
            }
            if hasToolReturn {
                return ModelResponse(parts: [.text("The weather in Paris is sunny.")], usage: Usage())
            }
            return ModelResponse(parts: [.toolCall(ToolCall(
                id: "c1", name: "get_weather", arguments: .object(["city": "Paris"])))], usage: Usage())
        }
        let agent = Agent<Void, String>(model)
            .tool("get_weather", "Get the weather for a city") { (_: WeatherArgs, _) in ToolResult("sunny") }

        let run = try agent.iter("What's the weather in Paris?")
        var labels: [String] = []
        for try await node in run { labels.append(label(node)) }

        #expect(labels == ["userPrompt", "modelRequest", "callTools", "modelRequest", "callTools", "end"])
        #expect(run.result?.output == "The weather in Paris is sunny.")
        #expect(run.runStep == 2)
    }

    @Test("streaming a ModelRequest node emits its deltas and does not double-run")
    func perNodeStreaming() async throws {
        let model = ScriptedStreamModel(events: [
            .textDelta("Hello, "),
            .textDelta("world!"),
            .completed(ModelResponse(parts: [.text("Hello, world!")], finishReason: .stop)),
        ])
        let agent = Agent<Void, String>(model)

        let run = try agent.iter("hi")
        var deltas: [String] = []
        for try await node in run {
            if case let .modelRequest(stream) = node {
                for try await event in stream.events() {
                    if case let .textDelta(t) = event { deltas.append(t) }
                }
            }
        }
        #expect(deltas == ["Hello, ", "world!"])
        // Advancing past a streamed node consumes the cached response — counted once.
        #expect(run.result?.output == "Hello, world!")
        #expect(run.usage.requests == 1)
    }

    @Test("runStep is visible to a tool and reflects the current model request")
    func runStepInTool() async throws {
        let recorder = StepRecorder()
        let model = FunctionModel { messages, _, _ in
            let hasToolReturn = messages.contains { msg in
                msg.asRequest?.parts.contains { if case .toolReturn = $0 { return true } else { return false } } ?? false
            }
            if hasToolReturn {
                return ModelResponse(parts: [.text("done")], usage: Usage())
            }
            return ModelResponse(parts: [.toolCall(ToolCall(
                id: "c1", name: "probe", arguments: .object([:])))], usage: Usage())
        }
        let agent = Agent<StepRecorder, String>(model)
            .tool("probe", "records the run step") { (_: EmptyArgs, ctx) in
                await ctx.deps.set(ctx.runStep)
                return ToolResult("ok")
            }
        let result = try await agent.run("go", deps: recorder)
        #expect(result.output == "done")
        // The tool runs after the first model response, so runStep is 1.
        #expect(await recorder.step == 1)
    }

    @Test("graphDiagram renders the agent's node graph")
    func diagram() {
        let agent = Agent<Void, String>(TestModel())
        let mermaid = agent.graphDiagram()
        #expect(mermaid.contains("UserPromptNode --> ModelRequestNode"))
        #expect(mermaid.contains("ModelRequestNode --> CallToolsNode"))
        #expect(mermaid.contains("CallToolsNode --> ModelRequestNode"))
        #expect(mermaid.contains("CallToolsNode --> End"))
    }
}
