import AgentCore
import FoundationModels

/// The result of a completed agent run.
public struct AgentRunResult<Output: Sendable>: Sendable {
    /// The validated, typed output.
    public let output: Output
    /// Full conversation history (prior history + this run's new messages).
    public let messages: [ModelMessage]
    /// Total usage for this run.
    public let usage: Usage

    public init(output: Output, messages: [ModelMessage], usage: Usage) {
        self.output = output
        self.messages = messages
        self.usage = usage
    }
}

/// An incremental event emitted while streaming an agent run.
public enum AgentStreamEvent<Output: Sendable>: Sendable {
    case textDelta(String)
    case thinkingDelta(String)
    case toolCall(ToolCall)
    case toolReturn(ToolReturn)
    /// A progressively-filled snapshot of the structured output as it streams.
    /// Carried as `GeneratedContent` (Sendable); read it typed via `partialOutput()`.
    case partial(GeneratedContent)
    /// Terminal event carrying the completed result.
    case final(AgentRunResult<Output>)
}

extension AgentStreamEvent where Output: Generable {
    /// The latest partial structured snapshot, reconstructed as the typed value.
    /// Returns `nil` for non-`.partial` events.
    public func partialOutput() throws -> Output.PartiallyGenerated? {
        guard case let .partial(content) = self else { return nil }
        return try Output.PartiallyGenerated(content)
    }
}
