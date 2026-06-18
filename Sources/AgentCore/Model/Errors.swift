import Foundation

/// Thrown by a tool or an output validator to ask the model to try again.
///
/// The agent run loop catches this, appends the message back to the model as a
/// retry prompt, and re-requests — bounded by the agent's `retries` budget.
/// This mirrors pydantic-ai's `ModelRetry`.
public struct ModelRetry: Error, Sendable, Equatable {
    public let message: String
    public init(_ message: String) { self.message = message }
}

/// Errors surfaced by the agent / model layer.
public enum AgentError: Error, Sendable, Equatable {
    /// A usage limit was exceeded.
    case usageLimitExceeded(String)
    /// The retry budget was exhausted (last underlying message included).
    case retriesExhausted(String)
    /// Output could not be decoded into the requested type.
    case outputDecodingFailed(String)
    /// The model called a tool the agent does not know about.
    case toolNotFound(String)
    /// The requested output mode is not supported by the model / output type.
    case unsupportedOutputMode(String)
    /// The model produced no usable output.
    case noOutput(String)
    /// A provider-level / transport error.
    case provider(String)
    /// Could not resolve a `"provider:model"` selector.
    case unknownModel(String)
}

extension AgentError: CustomStringConvertible {
    public var description: String {
        switch self {
        case let .usageLimitExceeded(m): return "Usage limit exceeded: \(m)"
        case let .retriesExhausted(m): return "Retries exhausted: \(m)"
        case let .outputDecodingFailed(m): return "Output decoding failed: \(m)"
        case let .toolNotFound(m): return "Tool not found: \(m)"
        case let .unsupportedOutputMode(m): return "Unsupported output mode: \(m)"
        case let .noOutput(m): return "No output: \(m)"
        case let .provider(m): return "Provider error: \(m)"
        case let .unknownModel(m): return "Unknown model: \(m)"
        }
    }
}
