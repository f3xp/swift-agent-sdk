import Testing
import FoundationModels
import AgentCore
import Agents
import AgentApple

@Generable
struct Capital: Equatable {
    @Guide(description: "The capital city")
    var city: String
}

@Suite("Apple on-device provider")
struct AppleModelTests {

    @Test("registers the apple selector")
    func registration() throws {
        AgentApple.register()
        let model = try ModelRegistry.shared.resolve(ModelSelector("apple:on-device"))
        #expect(model.modelName == "apple-on-device")
        #expect(model.profile.supportsNativeStructuredOutput)
    }

    @Test("tool calling is rejected in Phase 1")
    func toolsRejected() async throws {
        let agent = Agent<Void, String>(AppleModel())
            .tool("noop", "x") { (_: Capital, _) in ToolResult("ok") }
        // The output tool / user tools make `tools` non-empty → unsupported.
        await #expect(throws: AgentError.self) {
            _ = try await agent.run("hello")
        }
    }

    // Live test — only runs when Apple Intelligence is available on this machine.
    @Test("text generation on-device", .enabled(if: AgentApple.isAvailable))
    func liveText() async throws {
        let agent = Agent<Void, String>(
            AppleModel(), instructions: "Reply with a single word.")
        let result = try await agent.run("Say hello.")
        #expect(!result.output.isEmpty)
    }

    @Test("structured output on-device", .enabled(if: AgentApple.isAvailable))
    func liveStructured() async throws {
        let agent = Agent<Void, Capital>(AppleModel())
        let result = try await agent.run("What is the capital of France?")
        #expect(!result.output.city.isEmpty)
    }
}
