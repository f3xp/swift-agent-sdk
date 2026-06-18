import AgentCore
import Foundation

/// Pure translation between the SDK's vendor-neutral types and the Gemini
/// `generateContent` wire format. Networking-free for offline unit testing.
///
/// Note: Gemini correlates function calls/responses by **name**, not by id (the
/// classic API has no call ids), so we synthesize `ToolCall.id` from the name.
enum GeminiWire {

    // MARK: Request

    static func encodeRequest(
        model: String,
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec
    ) -> JSONValue {
        var systemParts: [String] = []
        var contents: [JSONValue] = []

        for message in messages {
            switch message {
            case let .request(req):
                var parts: [JSONValue] = []
                for part in req.parts {
                    switch part {
                    case let .system(text):
                        systemParts.append(text)
                    case let .userText(text):
                        parts.append(.object(["text": .string(text)]))
                    case let .userMedia(media):
                        if let part = encodeMedia(media) { parts.append(part) }
                    case let .toolReturn(ret):
                        parts.append(.object(["functionResponse": .object([
                            "name": .string(ret.name),
                            "response": wrapResponse(ret.content),
                        ])]))
                    case let .retryPrompt(retry):
                        parts.append(.object(["text": .string("Error: \(retry.message)")]))
                    }
                }
                if !parts.isEmpty {
                    contents.append(.object(["role": "user", "parts": .array(parts)]))
                }

            case let .response(resp):
                var parts: [JSONValue] = []
                for part in resp.parts {
                    switch part {
                    case let .text(text):
                        parts.append(.object(["text": .string(text)]))
                    case let .toolCall(call):
                        parts.append(.object(["functionCall": .object([
                            "name": .string(call.name),
                            "args": call.arguments,
                        ])]))
                    default:
                        break
                    }
                }
                if !parts.isEmpty {
                    contents.append(.object(["role": "model", "parts": .array(parts)]))
                }
            }
        }

        var body: [String: JSONValue] = ["contents": .array(contents)]

        if !systemParts.isEmpty {
            body["systemInstruction"] = .object([
                "parts": .array([.object(["text": .string(systemParts.joined(separator: "\n\n"))])]),
            ])
        }

        var genConfig: [String: JSONValue] = [:]
        if let t = settings.temperature { genConfig["temperature"] = .double(t) }
        if let m = settings.maxTokens { genConfig["maxOutputTokens"] = .int(m) }
        if let p = settings.topP { genConfig["topP"] = .double(p) }
        if let stop = settings.stopSequences, !stop.isEmpty {
            genConfig["stopSequences"] = .array(stop.map(JSONValue.string))
        }
        if output.mode == .native, let schema = output.schema {
            genConfig["responseMimeType"] = "application/json"
            genConfig["responseSchema"] = GeminiSchema.normalize(schema)
        }
        if let thinking = settings.thinking {
            var thinkingConfig: [String: JSONValue] = ["includeThoughts": .bool(thinking.includeThoughts)]
            if let budget = thinking.budgetTokens { thinkingConfig["thinkingBudget"] = .int(budget) }
            genConfig["thinkingConfig"] = .object(thinkingConfig)
        }
        if !genConfig.isEmpty { body["generationConfig"] = .object(genConfig) }

        if !tools.isEmpty {
            let declarations = tools.map { tool -> JSONValue in
                .object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": GeminiSchema.normalize(tool.parameters),
                ])
            }
            body["tools"] = .array([.object(["functionDeclarations": .array(declarations)])])

            if output.mode == .tool, tools.count == 1, tools[0].name == output.name {
                body["toolConfig"] = .object([
                    "functionCallingConfig": .object([
                        "mode": "ANY",
                        "allowedFunctionNames": .array([.string(output.name)]),
                    ]),
                ])
            }
        }

        return .object(body)
    }

    /// Encode a media attachment as a Gemini content part: inline base64 via
    /// `inlineData`, or a remote reference via `fileData`.
    private static func encodeMedia(_ media: MediaContent) -> JSONValue? {
        if let data = media.base64Data {
            return .object(["inlineData": .object([
                "mimeType": .string(media.mediaType),
                "data": .string(data),
            ])])
        } else if let url = media.url {
            return .object(["fileData": .object([
                "mimeType": .string(media.mediaType),
                "fileUri": .string(url),
            ])])
        }
        return nil
    }

    /// Gemini's `functionResponse.response` must be a JSON object.
    private static func wrapResponse(_ content: JSONValue) -> JSONValue {
        if case .object = content { return content }
        return .object(["result": content])
    }

    // MARK: Response

    static func decodeResponse(_ json: JSONValue, modelName: String) throws -> ModelResponse {
        if let err = json["error"]?["message"]?.stringValue {
            throw AgentError.provider("Gemini: \(err)")
        }
        guard let candidate = json["candidates"]?[0] else {
            throw AgentError.provider("Gemini: no candidates in response")
        }

        var parts: [ModelResponsePart] = []
        var callIndex = 0
        for part in candidate["content"]?["parts"]?.arrayValue ?? [] {
            if let text = part["text"]?.stringValue {
                // Parts flagged `thought: true` are reasoning summaries.
                if part["thought"]?.boolValue == true {
                    parts.append(.thinking(text))
                } else {
                    parts.append(.text(text))
                }
            } else if let fn = part["functionCall"] {
                guard let name = fn["name"]?.stringValue else { continue }
                let args = fn["args"] ?? .object([:])
                // Synthesize a stable id since Gemini omits call ids.
                let id = "\(name)-\(callIndex)"
                callIndex += 1
                parts.append(.toolCall(ToolCall(id: id, name: name, arguments: args)))
            }
        }

        var usage = Usage()
        if let meta = json["usageMetadata"] {
            if case let .int(i) = meta["promptTokenCount"] ?? .null { usage.inputTokens = i }
            if case let .int(o) = meta["candidatesTokenCount"] ?? .null { usage.outputTokens = o }
        }

        return ModelResponse(
            parts: parts,
            usage: usage,
            modelName: json["modelVersion"]?.stringValue ?? modelName,
            responseID: json["responseId"]?.stringValue,
            finishReason: mapFinishReason(candidate["finishReason"]?.stringValue))
    }

    /// Map Gemini's `finishReason` onto the normalized `FinishReason`.
    static func mapFinishReason(_ raw: String?) -> FinishReason? {
        switch raw {
        case "STOP": return .stop
        case "MAX_TOKENS": return .length
        case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT": return .contentFilter
        case nil: return nil
        default: return .error
        }
    }
}
