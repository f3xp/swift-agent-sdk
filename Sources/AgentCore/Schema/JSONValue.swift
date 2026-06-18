import Foundation

/// A minimal JSON abstract-syntax tree.
///
/// Used as the vendor-neutral representation of JSON Schema documents and of
/// tool-call arguments / tool results as they cross the wire. Keeping a single
/// value type (rather than `[String: Any]`) lets the whole package stay
/// `Sendable` and `Equatable` under Swift 6 strict concurrency.
public enum JSONValue: Sendable, Equatable, Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

extension JSONValue {
    /// The value as a Swift dictionary if this is an object, else `nil`.
    public var objectValue: [String: JSONValue]? {
        if case let .object(o) = self { return o }
        return nil
    }

    public var stringValue: String? {
        if case let .string(s) = self { return s }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case let .array(a) = self { return a }
        return nil
    }

    public var boolValue: Bool? {
        if case let .bool(b) = self { return b }
        return nil
    }

    public var intValue: Int? {
        if case let .int(i) = self { return i }
        return nil
    }

    public subscript(_ key: String) -> JSONValue? {
        objectValue?[key]
    }

    public subscript(_ index: Int) -> JSONValue? {
        guard case let .array(a) = self, a.indices.contains(index) else { return nil }
        return a[index]
    }

    /// Returns a copy with `value` set at `key` (object values only; no-op otherwise).
    public func setting(_ key: String, _ value: JSONValue) -> JSONValue {
        guard case var .object(o) = self else { return self }
        o[key] = value
        return .object(o)
    }
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([JSONValue].self) {
            self = .array(a)
        } else if let o = try? container.decode([String: JSONValue].self) {
            self = .object(o)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case let .bool(b): try container.encode(b)
        case let .int(i): try container.encode(i)
        case let .double(d): try container.encode(d)
        case let .string(s): try container.encode(s)
        case let .array(a): try container.encode(a)
        case let .object(o): try container.encode(o)
        }
    }
}

// MARK: - Raw JSON bridging

extension JSONValue {
    /// Parse a JSON string into a `JSONValue`.
    public init(jsonString: String) throws {
        let data = Data(jsonString.utf8)
        self = try JSONDecoder().decode(JSONValue.self, from: data)
    }

    /// Serialize to a JSON string. Pass `sortedKeys: true` for stable golden-file output.
    public func jsonString(sortedKeys: Bool = false, prettyPrinted: Bool = false) throws -> String {
        let encoder = JSONEncoder()
        var opts: JSONEncoder.OutputFormatting = []
        if sortedKeys { opts.insert(.sortedKeys) }
        if prettyPrinted { opts.insert(.prettyPrinted) }
        encoder.outputFormatting = opts
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}

// MARK: - Literals

extension JSONValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}
extension JSONValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) { self = .int(value) }
}
extension JSONValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}
extension JSONValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) { self = .double(value) }
}
extension JSONValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
}
extension JSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(Dictionary(uniqueKeysWithValues: elements))
    }
}
