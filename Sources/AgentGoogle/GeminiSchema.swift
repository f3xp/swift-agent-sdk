import AgentCore

/// Gemini JSON Schema normalization.
///
/// Gemini's `responseSchema` / function `parameters` accept an **OpenAPI 3.0
/// subset**, not standard JSON Schema. Differences we must reconcile from
/// Apple's `GenerationSchema` output:
///   - `type` values are UPPERCASE enum names (`STRING`, `OBJECT`, …).
///   - `additionalProperties` and `$`-prefixed keywords are rejected.
/// Supported keywords (kept): `type`, `description`, `format`, `nullable`,
/// `enum`, `items`, `properties`, `required`, `anyOf`.
enum GeminiSchema {
    private static let typeMap: [String: String] = [
        "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
        "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT", "null": "NULL",
    ]

    private static let unsupportedKeys: Set<String> = [
        "additionalProperties", "$schema", "$id", "$defs", "$ref", "definitions",
    ]

    static func normalize(_ value: JSONValue) -> JSONValue {
        switch value {
        case var .object(object):
            for key in unsupportedKeys { object.removeValue(forKey: key) }

            if let type = object["type"]?.stringValue {
                object["type"] = .string(typeMap[type.lowercased()] ?? type.uppercased())
            }
            if let properties = object["properties"]?.objectValue {
                var rewritten: [String: JSONValue] = [:]
                for (key, propSchema) in properties { rewritten[key] = normalize(propSchema) }
                object["properties"] = .object(rewritten)
            }
            if let items = object["items"] {
                object["items"] = normalize(items)
            }
            if let anyOf = object["anyOf"]?.arrayValue {
                object["anyOf"] = .array(anyOf.map(normalize))
            }
            return .object(object)
        case let .array(array):
            return .array(array.map(normalize))
        default:
            return value
        }
    }
}
