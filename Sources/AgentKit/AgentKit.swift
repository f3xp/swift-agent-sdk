// AgentKit — umbrella convenience module.
//
// Re-exports the core API and the bundled providers so a typical app needs a
// single `import AgentKit`.
@_exported import AgentCore
@_exported import AgentGraph
@_exported import Agents
@_exported import AgentApple
@_exported import AgentAnthropic
@_exported import AgentOpenAI
@_exported import AgentGoogle

/// Registers all bundled providers with `ModelRegistry`, enabling
/// `"provider:model"` selector resolution (e.g. `"anthropic:claude-sonnet-4-6"`,
/// `"apple:on-device"`). Call once at startup.
public func registerBundledProviders() {
    AgentApple.register()
    AgentAnthropic.register()
    AgentOpenAI.register()
    AgentGoogle.register()
}
