import AgentCore
import Foundation

/// Pure translation between the SDK's vendor-neutral types and the Anthropic
/// Messages API wire format. Kept free of networking so it can be unit-tested
/// offline (golden-style).
enum AnthropicWire {

    // MARK: Request

    static func encodeRequest(
        model: String,
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec,
        stream: Bool = false
    ) -> JSONValue {
        var systemParts: [String] = []
        var apiMessages: [JSONValue] = []

        for message in messages {
            switch message {
            case let .request(req):
                var blocks: [JSONValue] = []
                for part in req.parts {
                    switch part {
                    case let .system(text):
                        systemParts.append(text)
                    case let .userText(text):
                        blocks.append(.object(["type": "text", "text": .string(text)]))
                    case let .userMedia(media):
                        if let block = encodeMedia(media) { blocks.append(block) }
                    case let .toolReturn(ret):
                        blocks.append(.object([
                            "type": "tool_result",
                            "tool_use_id": .string(ret.callID),
                            "content": .string(stringify(ret.content)),
                        ]))
                    case let .retryPrompt(retry):
                        if let id = retry.toolCallID {
                            blocks.append(.object([
                                "type": "tool_result",
                                "tool_use_id": .string(id),
                                "is_error": true,
                                "content": .string(retry.message),
                            ]))
                        } else {
                            blocks.append(.object(["type": "text", "text": .string(retry.message)]))
                        }
                    }
                }
                if !blocks.isEmpty {
                    apiMessages.append(.object(["role": "user", "content": .array(blocks)]))
                }

            case let .response(resp):
                var blocks: [JSONValue] = []
                for part in resp.parts {
                    switch part {
                    case let .text(text):
                        blocks.append(.object(["type": "text", "text": .string(text)]))
                    case let .toolCall(call):
                        blocks.append(.object([
                            "type": "tool_use",
                            "id": .string(call.id),
                            "name": .string(call.name),
                            "input": call.arguments,
                        ]))
                    default:
                        break
                    }
                }
                if !blocks.isEmpty {
                    apiMessages.append(.object(["role": "assistant", "content": .array(blocks)]))
                }
            }
        }

        // Extended thinking requires `max_tokens` to exceed the thinking budget.
        let budget = settings.thinking?.budgetTokens ?? 1024
        var maxTokens = settings.maxTokens ?? 4096
        if settings.thinking != nil, maxTokens <= budget { maxTokens = budget + 4096 }

        var body: [String: JSONValue] = [
            "model": .string(model),
            "max_tokens": .int(maxTokens),
            "messages": .array(apiMessages),
        ]
        if !systemParts.isEmpty {
            body["system"] = .string(systemParts.joined(separator: "\n\n"))
        }
        if let t = settings.temperature { body["temperature"] = .double(t) }
        if let p = settings.topP { body["top_p"] = .double(p) }
        if let stop = settings.stopSequences, !stop.isEmpty {
            body["stop_sequences"] = .array(stop.map(JSONValue.string))
        }
        if settings.thinking != nil {
            body["thinking"] = .object([
                "type": "enabled",
                "budget_tokens": .int(budget),
            ])
        }
        if stream { body["stream"] = .bool(true) }
        if !tools.isEmpty {
            body["tools"] = .array(tools.map { tool in
                .object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "input_schema": tool.parameters,
                ])
            })
            // If a structured output tool is the only tool, force its use.
            if output.mode == .tool, tools.count == 1, tools[0].name == output.name {
                body["tool_choice"] = .object(["type": "tool", "name": .string(output.name)])
            }
        }
        return .object(body)
    }

    private static func encodeMedia(_ media: MediaContent) -> JSONValue? {
        // Anthropic supports `image` and `document` blocks, each with a base64 or
        // url source. Audio input is unsupported by the Messages API → skip.
        let blockType: String
        switch media.kind {
        case .image: blockType = "image"
        case .file: blockType = "document"
        case .audio: return nil
        }
        let source: JSONValue
        if let data = media.base64Data {
            source = .object([
                "type": "base64",
                "media_type": .string(media.mediaType),
                "data": .string(data),
            ])
        } else if let url = media.url {
            source = .object(["type": "url", "url": .string(url)])
        } else {
            return nil
        }
        return .object(["type": .string(blockType), "source": source])
    }

    /// Anthropic `tool_result` content is a string; serialize structured content.
    private static func stringify(_ value: JSONValue) -> String {
        if case let .string(s) = value { return s }
        return (try? value.jsonString()) ?? ""
    }

    // MARK: Response

    static func decodeResponse(_ json: JSONValue, modelName: String) throws -> ModelResponse {
        guard let content = json["content"]?.objectValueArray else {
            // Surface API error payloads.
            if let err = json["error"]?["message"]?.stringValue {
                throw AgentError.provider("Anthropic: \(err)")
            }
            throw AgentError.provider("Anthropic: unexpected response shape")
        }

        var parts: [ModelResponsePart] = []
        for block in content {
            switch block["type"]?.stringValue {
            case "text":
                if let text = block["text"]?.stringValue { parts.append(.text(text)) }
            case "tool_use":
                if let id = block["id"]?.stringValue,
                   let name = block["name"]?.stringValue {
                    let input = block["input"] ?? .object([:])
                    parts.append(.toolCall(ToolCall(id: id, name: name, arguments: input)))
                }
            case "thinking":
                if let text = block["thinking"]?.stringValue { parts.append(.thinking(text)) }
            case "redacted_thinking":
                if let data = block["data"]?.stringValue { parts.append(.thinking(data)) }
            default:
                break
            }
        }

        var usage = Usage()
        if let u = json["usage"] {
            if case let .int(i) = u["input_tokens"] ?? .null { usage.inputTokens = i }
            if case let .int(o) = u["output_tokens"] ?? .null { usage.outputTokens = o }
        }

        return ModelResponse(
            parts: parts,
            usage: usage,
            modelName: json["model"]?.stringValue ?? modelName,
            responseID: json["id"]?.stringValue,
            finishReason: mapFinishReason(json["stop_reason"]?.stringValue))
    }

    /// Map Anthropic's `stop_reason` onto the normalized `FinishReason`.
    static func mapFinishReason(_ raw: String?) -> FinishReason? {
        switch raw {
        case "end_turn", "stop_sequence": return .stop
        case "max_tokens": return .length
        case "tool_use": return .toolCall
        case "refusal": return .contentFilter
        case nil: return nil
        default: return .error
        }
    }
}

private extension JSONValue {
    /// The value as `[JSONValue]` if this is an array, else `nil`.
    var objectValueArray: [JSONValue]? {
        if case let .array(a) = self { return a }
        return nil
    }
}
