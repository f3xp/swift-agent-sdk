import Testing
import FoundationModels
import AgentCore
import Agents
import AgentTestSupport

@Suite("Agent streaming partials")
struct AgentStreamingTests {

    private let london = CityLocation(city: "London", country: "United Kingdom")

    @Test("native text channel emits progressive partials then final")
    func nativeTextPartials() async throws {
        let full = #"{"city":"London","country":"United Kingdom"}"#
        let model = ScriptedStreamModel(
            profile: ModelProfile(supportsNativeStructuredOutput: true, defaultOutputMode: .native),
            events: [
                .textDelta(#"{"city":"Lon"#),
                .textDelta(#"don","country":"United Kingdom"}"#),
                .completed(ModelResponse(parts: [.text(full)], usage: Usage(inputTokens: 5, outputTokens: 9),
                                         finishReason: .stop)),
            ])
        let agent = Agent<Void, CityLocation>(model)

        var partials: [CityLocation.PartiallyGenerated] = []
        var final: CityLocation?
        for try await event in agent.runStream("Where were the 2012 Olympics held?") {
            if let p = try event.partialOutput() { partials.append(p) }
            if case let .final(result) = event { final = result.output }
        }

        #expect(!partials.isEmpty)
        #expect(partials.last?.city == "London")
        #expect(final == london)
    }

    @Test("tool-args channel emits partials for the output tool")
    func toolArgsPartials() async throws {
        let full: JSONValue = ["city": "London", "country": "United Kingdom"]
        let model = ScriptedStreamModel(
            profile: .default,  // defaultOutputMode == .tool
            events: [
                .toolCallDelta(ToolCallDelta(index: 0, id: "out", name: "final_result",
                                             argumentsFragment: #"{"city":"Lon"#)),
                .toolCallDelta(ToolCallDelta(index: 0, argumentsFragment: #"don","country":"United Kingdom"}"#)),
                .toolCall(ToolCall(id: "out", name: "final_result", arguments: full)),
                .completed(ModelResponse(parts: [.toolCall(ToolCall(id: "out", name: "final_result", arguments: full))],
                                         usage: Usage(inputTokens: 5, outputTokens: 9), finishReason: .toolCall)),
            ])
        let agent = Agent<Void, CityLocation>(model)

        var partials: [CityLocation.PartiallyGenerated] = []
        var final: CityLocation?
        for try await event in agent.runStream("Where were the 2012 Olympics held?") {
            if let p = try event.partialOutput() { partials.append(p) }
            if case let .final(result) = event { final = result.output }
        }

        #expect(!partials.isEmpty)
        #expect(partials.last?.city == "London")
        #expect(final == london)
    }

    @Test("plain-text output emits no partials")
    func noPartialsForString() async throws {
        let model = ScriptedStreamModel(events: [
            .textDelta("Hello, "),
            .textDelta("world!"),
            .completed(ModelResponse(parts: [.text("Hello, world!")], finishReason: .stop)),
        ])
        let agent = Agent<Void, String>(model)

        var sawPartial = false
        var final: String?
        for try await event in agent.runStream("hi") {
            if case .partial = event { sawPartial = true }
            if case let .final(result) = event { final = result.output }
        }

        #expect(sawPartial == false)
        #expect(final == "Hello, world!")
    }
}
