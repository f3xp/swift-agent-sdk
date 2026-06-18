import AgentCore

/// OpenAI strict-mode JSON Schema normalization.
///
/// OpenAI's structured outputs (and strict function calling) require every
/// object to set `additionalProperties: false` and to list *all* of its
/// properties in `required`. Apple's `GenerationSchema` emits a standard schema
/// that omits these, so we transform it here. This is the per-provider
/// normalization the design flags as where subtle bugs live, so it's covered by
/// golden-style tests.
enum OpenAISchema {
    static func strict(_ value: JSONValue) -> JSONValue {
        switch value {
        case var .object(object):
            // Recurse into nested schema-bearing keywords first.
            if let properties = object["properties"]?.objectValue {
                var rewritten: [String: JSONValue] = [:]
                for (key, propSchema) in properties {
                    rewritten[key] = strict(propSchema)
                }
                object["properties"] = .object(rewritten)
                // Strict mode: every declared property must be required.
                object["required"] = .array(rewritten.keys.sorted().map(JSONValue.string))
                object["additionalProperties"] = .bool(false)
            }
            if let items = object["items"] {
                object["items"] = strict(items)
            }
            for keyword in ["anyOf", "allOf", "oneOf"] {
                if let arr = object[keyword]?.arrayValue {
                    object[keyword] = .array(arr.map(strict))
                }
            }
            if let defs = object["$defs"]?.objectValue {
                var rewritten: [String: JSONValue] = [:]
                for (key, def) in defs { rewritten[key] = strict(def) }
                object["$defs"] = .object(rewritten)
            }
            return .object(object)
        case let .array(array):
            return .array(array.map(strict))
        default:
            return value
        }
    }
}
