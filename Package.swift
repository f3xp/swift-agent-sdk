// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwiftAgentSDK",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        // Umbrella convenience — re-exports core + the common providers.
        .library(name: "AgentKit", targets: ["AgentKit"]),

        // Slim core (no providers, no network SDKs).
        .library(name: "Agents", targets: ["Agents"]),
        .library(name: "AgentCore", targets: ["AgentCore"]),

        // Generic async state-machine engine (no agent coupling).
        .library(name: "AgentGraph", targets: ["AgentGraph"]),

        // Providers — each optional.
        .library(name: "AgentApple", targets: ["AgentApple"]),
        .library(name: "AgentAnthropic", targets: ["AgentAnthropic"]),
        .library(name: "AgentOpenAI", targets: ["AgentOpenAI"]),
        .library(name: "AgentGoogle", targets: ["AgentGoogle"]),

        // Test support.
        .library(name: "AgentTestSupport", targets: ["AgentTestSupport"]),
    ],
    targets: [
        // ---- core ----
        .target(name: "AgentCore"),
        .target(name: "AgentGraph"),
        .target(name: "Agents", dependencies: ["AgentCore", "AgentGraph"]),

        // ---- shared transport ----
        .target(name: "AgentHTTP", dependencies: ["AgentCore"]),

        // ---- providers ----
        .target(name: "AgentApple", dependencies: ["Agents"]),
        .target(name: "AgentAnthropic", dependencies: ["Agents", "AgentHTTP"]),
        .target(name: "AgentOpenAI", dependencies: ["Agents", "AgentHTTP"]),
        .target(name: "AgentGoogle", dependencies: ["Agents", "AgentHTTP"]),

        // ---- umbrella ----
        .target(name: "AgentKit", dependencies: [
            "Agents", "AgentGraph", "AgentApple", "AgentAnthropic", "AgentOpenAI", "AgentGoogle",
        ]),

        // ---- examples (live demo; compile-checked) ----
        .executableTarget(name: "Examples", dependencies: ["AgentKit"]),

        // ---- test support ----
        .target(name: "AgentTestSupport", dependencies: ["AgentCore", "Agents"]),

        // ---- tests ----
        .testTarget(name: "AgentCoreTests", dependencies: ["AgentCore", "AgentTestSupport"]),
        .testTarget(name: "AgentGraphTests", dependencies: ["AgentGraph"]),
        .testTarget(name: "AgentsTests", dependencies: ["Agents", "AgentTestSupport"]),
        .testTarget(name: "AgentAnthropicTests", dependencies: ["AgentAnthropic", "AgentTestSupport"]),
        .testTarget(name: "AgentOpenAITests", dependencies: ["AgentOpenAI", "AgentTestSupport"]),
        .testTarget(name: "AgentGoogleTests", dependencies: ["AgentGoogle", "AgentTestSupport"]),
        .testTarget(name: "AgentAppleTests", dependencies: ["AgentApple", "Agents"]),
    ]
)
