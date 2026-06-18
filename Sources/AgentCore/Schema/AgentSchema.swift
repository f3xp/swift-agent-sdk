import Foundation
import FoundationModels

/// A type that can be used as structured agent output or as tool arguments.
///
/// We reuse Apple's `Generable` as the substrate, so any `@Generable` struct/enum
/// (and the standard scalars `String`, `Int`, `Bool`, `Double`, `Float`, `Decimal`,
/// and `Array` of `Generable`) satisfies this for free — no hand-written conformance.
///
/// `Generable` already provides everything we need:
///   - `static var generationSchema: GenerationSchema`  (schema source of truth)
///   - `init(_ content: GeneratedContent) throws`        (decode remote / on-device JSON)
///   - `var generatedContent: GeneratedContent`          (encode back out)
public typealias AgentSchema = Generable

extension GenerationSchema {
    /// Project this schema into the vendor-neutral `JSONValue` AST.
    ///
    /// `GenerationSchema` is `Codable`; we round-trip it through JSON to obtain a
    /// standard `{"type":"object","properties":…,"required":[…]}` document that
    /// each remote provider then normalizes into its own dialect
    /// (see `ModelProfile.jsonSchemaTransform`).
    public func toJSONValue() throws -> JSONValue {
        let data = try JSONEncoder().encode(self)
        return try JSONDecoder().decode(JSONValue.self, from: data)
    }
}

extension JSONValue {
    /// Build a `GeneratedContent` from this value (via its JSON string form).
    public func toGeneratedContent() throws -> GeneratedContent {
        try GeneratedContent(json: jsonString())
    }
}

extension GeneratedContent {
    /// Project a `GeneratedContent` into the `JSONValue` AST.
    public func toJSONValue() throws -> JSONValue {
        try JSONValue(jsonString: jsonString)
    }
}

extension Generable {
    /// The JSON Schema for this type as a `JSONValue`.
    public static func schemaJSON() throws -> JSONValue {
        try generationSchema.toJSONValue()
    }

    /// Decode an instance from a `JSONValue` (e.g. a remote model's structured response).
    public init(jsonValue: JSONValue) throws {
        try self.init(jsonValue.toGeneratedContent())
    }
}
