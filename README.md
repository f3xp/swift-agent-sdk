# SwiftAgentSDK

A Swift port of Python's [pydantic-ai](https://github.com/pydantic/pydantic-ai) — a
typed, model-agnostic LLM **agent framework**. macOS 26+, Swift 6.

Instead of reinventing Pydantic's runtime schema magic, the structured-output and
tool-schema substrate is Apple's **FoundationModels** (`@Generable` /
`GenerationSchema` / `GeneratedContent`) — used to drive *both* the on-device
Apple model *and* remote providers.

## Status — Phase 1 (vertical slice, complete)

| Area | Status |
|------|--------|
| `Agent<Deps, Output>`, `RunContext`, run loop | ✅ |
| Tools (closure + protocol), parallel execution, type erasure | ✅ |
| Output modes: text, forced-output-tool, native, prompted | ✅ |
| Retries + `ModelRetry` self-correction (tool & output validator) | ✅ |
| Usage limits, message history, streaming (`runStream`) | ✅ |
| Schema bridge: `GenerationSchema ⇆ JSONValue ⇆ GeneratedContent` | ✅ |
| Providers: **Apple on-device**, **Anthropic** | ✅ |
| `TestModel` / `FunctionModel` / `FallbackModel` | ✅ |

36 tests pass offline (live on-device / network tests are opt-in and auto-skip).

### Phase 2 (in progress)

| Area | Status |
|------|--------|
| **OpenAI** provider (Chat Completions; gateway to OpenAI-compatible endpoints) | ✅ |
| **Google Gemini** provider (`generateContent`) | ✅ |
| Native structured output + per-provider schema normalization (OpenAI strict mode, Gemini OpenAPI subset) | ✅ |
| Multimodal parts, reasoning parts, parallel tool calls, typed streamed partials | ⏳ |

**Phase 1 limitations:** Apple provider is single-turn, text + native structured
output only (on-device tool calling deferred). Streaming surfaces text deltas +
final output; typed partials deferred. See `Plan` for Phases 2–4 (OpenAI, Gemini,
graph engine, MCP, evals, observability).

## Quick start

```swift
import AgentKit

// Plain text
let agent = Agent<Void, String>(
    AnthropicModel(model: "claude-sonnet-4-6"),
    instructions: "Be concise.")
let result = try await agent.run("Where does \"hello world\" come from?")
print(result.output)
```

```swift
// Structured output — any @Generable type
@Generable struct CityLocation {
    @Guide(description: "The city") var city: String
    @Guide(description: "The country") var country: String
}

let agent = Agent<Void, CityLocation>(AnthropicModel(model: "claude-sonnet-4-6"))
let r = try await agent.run("Where were the 2012 Olympics held?")
print(r.output.city, r.output.country)   // London United Kingdom
```

```swift
// On-device (no keys, no network)
let agent = Agent<Void, CityLocation>(AppleModel())
let r = try await agent.run("What is the capital of France?")
```

```swift
// Dependencies + tools + structured output
struct Deps: Sendable { let customerID: Int; let db: DatabaseConn }
@Generable struct Support { var advice: String; var blockCard: Bool; var risk: Int }

let agent = Agent<Deps, Support>(AnthropicModel(model: "claude-sonnet-4-6"))
    .instructions("You are a bank support agent.")
    .tool("balance", "Get the customer balance.") { (args: BalanceArgs, ctx) in
        ToolResult("\(await ctx.deps.db.balance(ctx.deps.customerID))")
    }
let r = try await agent.run("What's my balance?", deps: Deps(customerID: 1, db: db))
```

`"provider:model"` selectors work after registering providers:

```swift
registerBundledProviders()
let agent = try Agent<Void, String>(ModelSelector("anthropic:claude-sonnet-4-6"))
```

## Architecture

```
AgentCore   schema bridge, ModelMessage/Part, Usage, errors, ModelProtocol/Provider/ModelProfile
Agents      Agent<Deps,Output> (Sendable struct, value-semantics builders), RunContext, run loop
AgentHTTP   shared SSE parser + retry/backoff
AgentApple  FoundationModels on-device  (only target importing the session APIs)
AgentAnthropic  Messages API (+ offline-tested wire translation)
AgentOpenAI  Chat Completions + native structured output + strict-schema normalization
AgentGoogle  Gemini generateContent + native structured output + OpenAPI-subset schema normalization
AgentTestSupport  TestModel / FunctionModel / FallbackModel
AgentKit    umbrella re-export + registerBundledProviders()
```

The run loop is a direct `async` loop (not built on a graph yet); its node
boundaries match a future graph engine so the public API won't change when the
graph lands in Phase 3.

## Build & test

```sh
swift build
swift test
swift run Examples hello     # live: needs ANTHROPIC_API_KEY
swift run Examples city
swift run Examples support
```
