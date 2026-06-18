import Testing
import FoundationModels
import AgentCore
import Agents
import AgentTestSupport

@Generable
struct CityLocation: Equatable {
    @Guide(description: "The city name")
    var city: String
    @Guide(description: "The country name")
    var country: String
}

@Generable
struct WeatherArgs {
    @Guide(description: "The city to look up")
    var city: String
}

@Generable
struct EmptyArgs {}

/// Records whether a tool actually ran (Sendable side-effect sink for tests).
actor SideEffectRecorder {
    private(set) var ran = false
    func mark() { ran = true }
}

@Suite("Agent run loop")
struct AgentRunLoopTests {

    @Test("hello world — plain text output")
    func helloWorld() async throws {
        let agent = Agent<Void, String>(TestModel(text: "Hello, world!"))
        let result = try await agent.run("hi")
        #expect(result.output == "Hello, world!")
        #expect(result.usage.requests == 1)
    }

    @Test("structured output via forced output tool")
    func structuredOutput() async throws {
        let model = TestModel(
            structuredOutput: .object(["city": "London", "country": "United Kingdom"]))
        let agent = Agent<Void, CityLocation>(model)
        let result = try await agent.run("Where were the 2012 Olympics held?")
        #expect(result.output == CityLocation(city: "London", country: "United Kingdom"))
    }

    @Test("tool call cycle — model calls a tool, then answers")
    func toolCallCycle() async throws {
        let model = FunctionModel { messages, _, _ in
            let hasToolReturn = messages.contains { msg in
                msg.asRequest?.parts.contains { if case .toolReturn = $0 { return true } else { return false } } ?? false
            }
            if hasToolReturn {
                return ModelResponse(parts: [.text("The weather in Paris is sunny.")],
                                     usage: Usage(inputTokens: 5, outputTokens: 3))
            } else {
                return ModelResponse(parts: [.toolCall(ToolCall(
                    id: "c1", name: "get_weather", arguments: .object(["city": "Paris"])))],
                    usage: Usage(inputTokens: 5, outputTokens: 3))
            }
        }

        let agent = Agent<Void, String>(model)
            .tool("get_weather", "Get the weather for a city") { (args: WeatherArgs, _) in
                #expect(args.city == "Paris")
                return ToolResult("sunny")
            }

        let result = try await agent.run("What's the weather in Paris?")
        #expect(result.output == "The weather in Paris is sunny.")
        #expect(result.usage.toolCalls == 1)
        #expect(result.usage.requests == 2)
    }

    @Test("output validator can trigger a retry")
    func outputValidatorRetry() async throws {
        let agent = Agent<Void, String>(TestModel(text: "answer"))
            .outputValidator { output, ctx in
                if ctx.retry == 0 { throw ModelRetry("please reconsider") }
                return output
            }
        let result = try await agent.run("go")
        #expect(result.output == "answer")
        // One initial request + one after the retry prompt.
        #expect(result.usage.requests == 2)
    }

    @Test("usage limit is enforced")
    func usageLimit() async throws {
        // A model that never finalizes — always calls the noop tool.
        let model = FunctionModel { _, _, _ in
            ModelResponse(parts: [.toolCall(ToolCall(id: "c", name: "noop", arguments: .object([:])))],
                          usage: Usage(inputTokens: 1, outputTokens: 1))
        }
        let agent = Agent<Void, String>(model)
            .tool("noop", "does nothing") { (_: EmptyArgs, _) in ToolResult("ok") }

        await #expect(throws: AgentError.self) {
            _ = try await agent.run("loop", usageLimits: UsageLimits(requestLimit: 3))
        }
    }

    @Test("fallback model switches on error")
    func fallback() async throws {
        let failing = FunctionModel { _, _, _ in throw AgentError.provider("boom") }
        let working = TestModel(text: "recovered")
        let agent = Agent<Void, String>(FallbackModel([failing, working]))
        let result = try await agent.run("hi")
        #expect(result.output == "recovered")
    }

    /// A model that emits the output tool call AND a sibling tool call in the
    /// same response — the case `EndStrategy` governs.
    private func outputPlusSiblingModel() -> FunctionModel {
        FunctionModel { _, _, output in
            ModelResponse(parts: [
                .toolCall(ToolCall(id: "out", name: output.name,
                    arguments: .object(["city": "London", "country": "United Kingdom"]))),
                .toolCall(ToolCall(id: "se", name: "side_effect", arguments: .object([:]))),
            ], usage: Usage(inputTokens: 1, outputTokens: 1))
        }
    }

    @Test("end strategy .early skips sibling tool calls")
    func endStrategyEarly() async throws {
        let recorder = SideEffectRecorder()
        let agent = Agent<SideEffectRecorder, CityLocation>(outputPlusSiblingModel())
            .tool("side_effect", "records that it ran") { (_: EmptyArgs, ctx) in
                await ctx.deps.mark(); return ToolResult("ok")
            }
        let result = try await agent.run("go", deps: recorder)
        #expect(result.output == CityLocation(city: "London", country: "United Kingdom"))
        #expect(await recorder.ran == false)
        #expect(result.usage.toolCalls == 0)
    }

    @Test("end strategy .exhaustive runs sibling tool calls before finalizing")
    func endStrategyExhaustive() async throws {
        let recorder = SideEffectRecorder()
        let agent = Agent<SideEffectRecorder, CityLocation>(
            outputPlusSiblingModel(), endStrategy: .exhaustive)
            .tool("side_effect", "records that it ran") { (_: EmptyArgs, ctx) in
                await ctx.deps.mark(); return ToolResult("ok")
            }
        let result = try await agent.run("go", deps: recorder)
        #expect(result.output == CityLocation(city: "London", country: "United Kingdom"))
        #expect(await recorder.ran == true)
        #expect(result.usage.toolCalls == 1)
    }

    @Test("tool retry budget is bounded")
    func toolRetryExhaustion() async throws {
        let model = FunctionModel { _, _, _ in
            ModelResponse(parts: [.toolCall(ToolCall(id: "c", name: "flaky", arguments: .object([:])))],
                          usage: Usage())
        }
        let agent = Agent<Void, String>(model, retries: 2)
            .tool("flaky", "always retries") { (_: EmptyArgs, _) in
                throw ModelRetry("not yet")
            }
        await #expect(throws: AgentError.self) {
            _ = try await agent.run("go")
        }
    }
}
