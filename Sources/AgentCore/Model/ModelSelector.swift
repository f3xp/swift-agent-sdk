import Foundation

/// A `"provider:model"` reference, e.g. `"anthropic:claude-sonnet-4-6"`.
public struct ModelSelector: Sendable, Equatable, ExpressibleByStringLiteral {
    public let provider: String
    public let model: String

    public init(provider: String, model: String) {
        self.provider = provider
        self.model = model
    }

    /// Parses `"provider:model"`. Everything after the first colon is the model
    /// id (model ids themselves may contain colons).
    public init(_ selector: String) {
        if let idx = selector.firstIndex(of: ":") {
            provider = String(selector[selector.startIndex..<idx])
            model = String(selector[selector.index(after: idx)...])
        } else {
            provider = ""
            model = selector
        }
    }

    public init(stringLiteral value: String) {
        self.init(value)
    }
}

/// A registry mapping provider names to model factories, so a `"provider:model"`
/// string can be resolved to a concrete `ModelProtocol`. Each provider module
/// registers itself (e.g. `AgentAnthropic.register()`).
public final class ModelRegistry: @unchecked Sendable {
    public static let shared = ModelRegistry()

    public typealias Factory = @Sendable (_ model: String) throws -> any ModelProtocol

    private let lock = NSLock()
    private var factories: [String: Factory] = [:]

    public func register(provider: String, factory: @escaping Factory) {
        lock.lock(); defer { lock.unlock() }
        factories[provider] = factory
    }

    public func resolve(_ selector: ModelSelector) throws -> any ModelProtocol {
        lock.lock()
        let factory = factories[selector.provider]
        lock.unlock()
        guard let factory else {
            throw AgentError.unknownModel(
                "no provider registered for \"\(selector.provider)\" (selector: \(selector.provider):\(selector.model))")
        }
        return try factory(selector.model)
    }
}
