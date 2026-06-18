import Foundation

/// A model tool invocation requested in a response.
public struct ToolCall: Sendable, Equatable, Codable {
    /// Provider-assigned id used to correlate the matching `ToolReturn`.
    public var id: String
    public var name: String
    /// Arguments as a JSON value (round-trips to `GeneratedContent` for decoding).
    public var arguments: JSONValue

    public init(id: String, name: String, arguments: JSONValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// The result of executing a tool, fed back to the model.
public struct ToolReturn: Sendable, Equatable, Codable {
    public var callID: String
    public var name: String
    public var content: JSONValue

    public init(callID: String, name: String, content: JSONValue) {
        self.callID = callID
        self.name = name
        self.content = content
    }
}

/// A retry prompt surfaced back to the model after a `ModelRetry`.
public struct RetryPrompt: Sendable, Equatable, Codable {
    public var message: String
    /// Set when the retry is tied to a specific failed tool call.
    public var toolCallID: String?
    public var toolName: String?

    public init(message: String, toolCallID: String? = nil, toolName: String? = nil) {
        self.message = message
        self.toolCallID = toolCallID
        self.toolName = toolName
    }
}

/// Non-text user content (images, audio, files). Minimal in Phase 1.
public struct MediaContent: Sendable, Equatable, Codable {
    public enum Kind: String, Sendable, Codable { case image, audio, file }
    public var kind: Kind
    public var mediaType: String     // e.g. "image/png"
    /// Either inline base64 data or a URL string.
    public var base64Data: String?
    public var url: String?

    public init(kind: Kind, mediaType: String, base64Data: String? = nil, url: String? = nil) {
        self.kind = kind
        self.mediaType = mediaType
        self.base64Data = base64Data
        self.url = url
    }

    // MARK: Convenience factories

    /// An inline image, e.g. `mediaType: "image/png"`.
    public static func image(base64: String, mediaType: String) -> MediaContent {
        MediaContent(kind: .image, mediaType: mediaType, base64Data: base64)
    }
    /// A remote image referenced by URL.
    public static func image(url: String, mediaType: String = "image/jpeg") -> MediaContent {
        MediaContent(kind: .image, mediaType: mediaType, url: url)
    }
    /// Inline audio, e.g. `mediaType: "audio/wav"`.
    public static func audio(base64: String, mediaType: String) -> MediaContent {
        MediaContent(kind: .audio, mediaType: mediaType, base64Data: base64)
    }
    /// An inline file/document, e.g. a PDF (`mediaType: "application/pdf"`).
    public static func file(base64: String, mediaType: String) -> MediaContent {
        MediaContent(kind: .file, mediaType: mediaType, base64Data: base64)
    }
    /// A remote file/document referenced by URL.
    public static func file(url: String, mediaType: String) -> MediaContent {
        MediaContent(kind: .file, mediaType: mediaType, url: url)
    }
}

/// A normalized reason the model stopped generating, mapped from each provider's
/// raw stop/finish value (mirrors pydantic-ai's `FinishReason`).
public enum FinishReason: String, Sendable, Equatable, Codable {
    case stop
    case length
    case contentFilter
    case toolCall
    case error
}

// MARK: - Parts

/// A part of a request *to* the model. Distinct from `ModelResponsePart` so the
/// type system forbids mixing request- and response-side content (e.g. a request
/// can never hold a `toolCall`). Mirrors pydantic-ai's `ModelRequestPart`.
public enum ModelRequestPart: Sendable, Equatable, Codable {
    case system(String)
    case userText(String)
    case userMedia(MediaContent)
    case toolReturn(ToolReturn)
    case retryPrompt(RetryPrompt)
}

/// A part of a response *from* the model. Mirrors pydantic-ai's `ModelResponsePart`.
public enum ModelResponsePart: Sendable, Equatable, Codable {
    case text(String)
    case thinking(String)
    case toolCall(ToolCall)
}

extension ModelResponsePart {
    public var asToolCall: ToolCall? {
        if case let .toolCall(c) = self { return c }
        return nil
    }
    public var asText: String? {
        if case let .text(t) = self { return t }
        return nil
    }
}

// MARK: - Messages

/// A request sent to the model. Carries request-side parts only.
public struct ModelRequest: Sendable, Equatable, Codable {
    public var parts: [ModelRequestPart]

    public init(parts: [ModelRequestPart]) {
        self.parts = parts
    }
}

extension ModelRequest: ExpressibleByArrayLiteral {
    /// Lets a request be written as `.request([.system(...), .userText(...)])`.
    public init(arrayLiteral elements: ModelRequestPart...) {
        self.init(parts: elements)
    }
}

/// A response assembled from a single model round trip. Carries response-side
/// parts plus usage and the (normalized) provider metadata that used to live in
/// a separate `ProviderMetadata` struct.
public struct ModelResponse: Sendable, Equatable, Codable {
    public var parts: [ModelResponsePart]
    public var usage: Usage?
    public var modelName: String?
    public var responseID: String?
    public var finishReason: FinishReason?

    public init(
        parts: [ModelResponsePart],
        usage: Usage? = nil,
        modelName: String? = nil,
        responseID: String? = nil,
        finishReason: FinishReason? = nil
    ) {
        self.parts = parts
        self.usage = usage
        self.modelName = modelName
        self.responseID = responseID
        self.finishReason = finishReason
    }

    /// All tool calls requested in this response.
    public var toolCalls: [ToolCall] {
        parts.compactMap(\.asToolCall)
    }

    /// Concatenated text parts.
    public var text: String {
        parts.compactMap(\.asText).joined()
    }
}

extension ModelResponse: ExpressibleByArrayLiteral {
    /// Lets a response be written as `.response([.toolCall(...)])` (no usage).
    public init(arrayLiteral elements: ModelResponsePart...) {
        self.init(parts: elements)
    }
}

/// One message in the conversation: either a request (to the model) or a
/// response (from the model) — pydantic-ai's `ModelMessage = ModelRequest |
/// ModelResponse`. Codable so message history can be persisted and replayed into
/// a later run (`message_history`).
public enum ModelMessage: Sendable, Equatable, Codable {
    case request(ModelRequest)
    case response(ModelResponse)

    /// The request payload, if this is a request message.
    public var asRequest: ModelRequest? {
        if case let .request(r) = self { return r }
        return nil
    }

    /// The response payload, if this is a response message.
    public var asResponse: ModelResponse? {
        if case let .response(r) = self { return r }
        return nil
    }
}
