import Foundation

/// Token / request accounting accumulated across an agent run.
public struct Usage: Sendable, Equatable, Codable {
    public var requests: Int
    public var inputTokens: Int
    public var outputTokens: Int
    public var toolCalls: Int

    public init(requests: Int = 0, inputTokens: Int = 0, outputTokens: Int = 0, toolCalls: Int = 0) {
        self.requests = requests
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.toolCalls = toolCalls
    }

    public var totalTokens: Int { inputTokens + outputTokens }

    public static let zero = Usage()

    public static func + (lhs: Usage, rhs: Usage) -> Usage {
        Usage(
            requests: lhs.requests + rhs.requests,
            inputTokens: lhs.inputTokens + rhs.inputTokens,
            outputTokens: lhs.outputTokens + rhs.outputTokens,
            toolCalls: lhs.toolCalls + rhs.toolCalls
        )
    }

    public static func += (lhs: inout Usage, rhs: Usage) {
        lhs = lhs + rhs
    }
}

/// Caps enforced over the course of a run; exceeding any throws `AgentError.usageLimitExceeded`.
public struct UsageLimits: Sendable, Equatable {
    public var requestLimit: Int?
    public var totalTokensLimit: Int?
    public var toolCallsLimit: Int?

    public init(requestLimit: Int? = nil, totalTokensLimit: Int? = nil, toolCallsLimit: Int? = nil) {
        self.requestLimit = requestLimit
        self.totalTokensLimit = totalTokensLimit
        self.toolCallsLimit = toolCallsLimit
    }

    public static let none = UsageLimits()

    /// Throws if the supplied usage already exceeds any configured limit.
    public func check(_ usage: Usage) throws {
        if let l = requestLimit, usage.requests > l {
            throw AgentError.usageLimitExceeded("request limit of \(l) exceeded")
        }
        if let l = totalTokensLimit, usage.totalTokens > l {
            throw AgentError.usageLimitExceeded("total token limit of \(l) exceeded")
        }
        if let l = toolCallsLimit, usage.toolCalls > l {
            throw AgentError.usageLimitExceeded("tool-call limit of \(l) exceeded")
        }
    }
}
