// Ports of canonical pydantic-ai documentation examples, as living proof of
// API parity. These run live (Anthropic needs ANTHROPIC_API_KEY; Apple needs
// on-device availability), so they're a demo target, not part of the test suite.
//
// Usage:  swift run Examples [hello|city|support]

import AgentKit
import FoundationModels
import Foundation

// MARK: hello world

func helloWorld() async throws {
    // pydantic-ai: Agent('anthropic:claude-sonnet-4-6', instructions='Be concise…')
    let agent = Agent<Void, String>(
        AnthropicModel(model: "claude-sonnet-4-6"),
        instructions: "Be concise, reply with one sentence.")
    let result = try await agent.run(#"Where does "hello world" come from?"#)
    print(result.output)
}

// MARK: structured output

@Generable
struct CityLocation {
    @Guide(description: "The city") var city: String
    @Guide(description: "The country") var country: String
}

func structuredOutput() async throws {
    let agent = Agent<Void, CityLocation>(AnthropicModel(model: "claude-sonnet-4-6"))
    let result = try await agent.run("Where were the 2012 Olympics held?")
    print("city=\(result.output.city) country=\(result.output.country)")
}

// OpenAI with native structured output (response_format: json_schema, strict).
func openAIStructured() async throws {
    let agent = Agent<Void, CityLocation>(OpenAIModel(model: "gpt-5.2"))
    let result = try await agent.run("Where were the 2012 Olympics held?")
    print("city=\(result.output.city) country=\(result.output.country)")
}

// Gemini with native structured output (generationConfig.responseSchema).
func geminiStructured() async throws {
    let agent = Agent<Void, CityLocation>(GoogleModel(model: "gemini-3-flash"))
    let result = try await agent.run("Where were the 2012 Olympics held?")
    print("city=\(result.output.city) country=\(result.output.country)")
}

// MARK: dependency injection + tool + structured output (bank support)

struct DatabaseConn: Sendable {
    func customerBalance(id: Int, includePending: Bool) async -> Double { 123.45 }
}

struct SupportDependencies: Sendable {
    let customerID: Int
    let db: DatabaseConn
}

@Generable
struct SupportOutput {
    @Guide(description: "Advice returned to the customer") var supportAdvice: String
    @Guide(description: "Whether to block their card") var blockCard: Bool
    @Guide(description: "Risk level, 0-10") var risk: Int
}

@Generable
struct BalanceArgs {
    @Guide(description: "Whether to include pending transactions") var includePending: Bool
}

func bankSupport() async throws {
    let agent = Agent<SupportDependencies, SupportOutput>(
        AnthropicModel(model: "claude-sonnet-4-6"),
        instructions: "You are a support agent in our bank. Judge the risk and advise the customer.")
        .tool("customer_balance", "Return the customer's current account balance.") {
            (args: BalanceArgs, ctx: RunContext<SupportDependencies>) in
            let balance = await ctx.deps.db.customerBalance(
                id: ctx.deps.customerID, includePending: args.includePending)
            return ToolResult("\(balance)")
        }

    let deps = SupportDependencies(customerID: 123, db: DatabaseConn())
    let result = try await agent.run("What is my balance?", deps: deps)
    print("advice=\(result.output.supportAdvice)")
    print("blockCard=\(result.output.blockCard) risk=\(result.output.risk)")
}

// MARK: iter() — observe and stream a run node-by-node

func iterDemo() async throws {
    let agent = Agent<Void, String>(
        AnthropicModel(model: "claude-sonnet-4-6"),
        instructions: "Be concise, reply with one sentence.")

    // The run is a graph — print its shape (no network needed).
    print("graph:\n\(agent.graphDiagram())\n")

    let run = try agent.iter(#"Where does "hello world" come from?"#)
    for try await node in run {
        switch node {
        case .userPrompt:
            print("• user prompt")
        case let .modelRequest(stream):
            print("• model request")
            for try await event in stream.events() {
                if case let .textDelta(t) = event { print(t, terminator: "") }
            }
            print()
        case .callTools:
            print("• handling response")
        case let .end(result):
            print("• end: \(result.output)")
        }
    }
}

// MARK: dispatch

let which = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "hello"
do {
    switch which {
    case "hello": try await helloWorld()
    case "city": try await structuredOutput()
    case "openai": try await openAIStructured()
    case "gemini": try await geminiStructured()
    case "support": try await bankSupport()
    case "iter": try await iterDemo()
    default: print("unknown example: \(which) (try: hello | city | openai | gemini | support | iter)")
    }
} catch {
    print("error: \(error)")
    exit(1)
}
