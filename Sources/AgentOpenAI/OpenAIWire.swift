import AgentCore
import Foundation

/// Pure translation between the SDK's vendor-neutral types and the OpenAI
/// Chat Completions wire format. Networking-free so it can be unit-tested offline.
enum OpenAIWire {

    // MARK: Request

    static func encodeRequest(
        model: String,
        messages: [ModelMessage],
        settings: ModelSettings,
        tools: [ToolDefinition],
        output: OutputSpec,
        stream: Bool = false
    ) -> JSONValue {
        var apiMessages: [JSONValue] = []

        for message in messages {
            switch message {
            case let .request(req):
                // A request message may carry system text, user text/media, tool
                // results, and retry prompts. User text and media are coalesced into
                // a single user message (array `content` when media is present);
                // other parts flush that pending message first to preserve order.
                var userBlocks: [JSONValue] = []
                func flushUser() {
                    guard !userBlocks.isEmpty else { return }
                    // Plain string content when it's a single text block (keeps the
                    // common text-only path byte-for-byte identical to before).
                    if userBlocks.count == 1, let only = userBlocks.first,
                       only["type"]?.stringValue == "text", let text = only["text"]?.stringValue {
                        apiMessages.append(.object(["role": "user", "content": .string(text)]))
                    } else {
                        apiMessages.append(.object(["role": "user", "content": .array(userBlocks)]))
                    }
                    userBlocks = []
                }
                for part in req.parts {
                    switch part {
                    case let .system(text):
                        flushUser()
                        apiMessages.append(.object(["role": "system", "content": .string(text)]))
                    case let .userText(text):
                        userBlocks.append(.object(["type": "text", "text": .string(text)]))
                    case let .userMedia(media):
                        if let block = encodeMedia(media) { userBlocks.append(block) }
                    case let .toolReturn(ret):
                        flushUser()
                        apiMessages.append(.object([
                            "role": "tool",
                            "tool_call_id": .string(ret.callID),
                            "content": .string(stringify(ret.content)),
                        ]))
                    case let .retryPrompt(retry):
                        flushUser()
                        if let id = retry.toolCallID {
                            apiMessages.append(.object([
                                "role": "tool",
                                "tool_call_id": .string(id),
                                "content": .string("Error: \(retry.message)"),
                            ]))
                        } else {
                            apiMessages.append(.object(["role": "user", "content": .string(retry.message)]))
                        }
                    }
                }
                flushUser()

            case let .response(resp):
                // Collapse a response into one assistant message with optional
                // text content plus a `tool_calls` array.
                let text = resp.text
                let toolCalls = resp.toolCalls
                var assistant: [String: JSONValue] = ["role": "assistant"]
                assistant["content"] = text.isEmpty ? .null : .string(text)
                if !toolCalls.isEmpty {
                    assistant["tool_calls"] = .array(toolCalls.map { call in
                        .object([
                            "id": .string(call.id),
                            "type": "function",
                            "function": .object([
                                "name": .string(call.name),
                                "arguments": .string((try? call.arguments.jsonString()) ?? "{}"),
                            ]),
                        ])
                    })
                }
                apiMessages.append(.object(assistant))
            }
        }

        var body: [String: JSONValue] = [
            "model": .string(model),
            "messages": .array(apiMessages),
        ]
        if let t = settings.temperature { body["temperature"] = .double(t) }
        if let m = settings.maxTokens { body["max_tokens"] = .int(m) }
        if let p = settings.topP { body["top_p"] = .double(p) }
        if let stop = settings.stopSequences, !stop.isEmpty {
            body["stop"] = .array(stop.map(JSONValue.string))
        }

        // Tools — function calling. Parameters are strict-normalized.
        if !tools.isEmpty {
            body["tools"] = .array(tools.map { tool in
                .object([
                    "type": "function",
                    "function": .object([
                        "name": .string(tool.name),
                        "description": .string(tool.description),
                        "parameters": OpenAISchema.strict(tool.parameters),
                        "strict": true,
                    ]),
                ])
            })
            if output.mode == .tool, tools.count == 1, tools[0].name == output.name {
                body["tool_choice"] = .object([
                    "type": "function",
                    "function": .object(["name": .string(output.name)]),
                ])
            }
        }

        // Native structured output via response_format.
        if output.mode == .native, let schema = output.schema {
            body["response_format"] = .object([
                "type": "json_schema",
                "json_schema": .object([
                    "name": .string(output.name),
                    "strict": true,
                    "schema": OpenAISchema.strict(schema),
                ]),
            ])
        }
        if stream {
            body["stream"] = .bool(true)
            // Ask for a trailing usage chunk (otherwise streaming omits usage).
            body["stream_options"] = .object(["include_usage": .bool(true)])
        }
        return .object(body)
    }

    private static func stringify(_ value: JSONValue) -> String {
        if case let .string(s) = value { return s }
        return (try? value.jsonString()) ?? ""
    }

    /// Encode a media attachment as an OpenAI content block. Returns `nil` for
    /// shapes OpenAI can't take (e.g. a url-only file).
    private static func encodeMedia(_ media: MediaContent) -> JSONValue? {
        switch media.kind {
        case .image:
            let url: String
            if let data = media.base64Data {
                url = "data:\(media.mediaType);base64,\(data)"
            } else if let remote = media.url {
                url = remote
            } else {
                return nil
            }
            return .object(["type": "image_url", "image_url": .object(["url": .string(url)])])
        case .audio:
            guard let data = media.base64Data else { return nil }
            // mediaType like "audio/wav" → format "wav".
            let format = media.mediaType.split(separator: "/").last.map(String.init) ?? media.mediaType
            return .object([
                "type": "input_audio",
                "input_audio": .object(["data": .string(data), "format": .string(format)]),
            ])
        case .file:
            // Inline files only (base64); url-only files aren't representable here.
            guard let data = media.base64Data else { return nil }
            return .object([
                "type": "file",
                "file": .object(["file_data": .string("data:\(media.mediaType);base64,\(data)")]),
            ])
        }
    }

    // MARK: Response

    static func decodeResponse(_ json: JSONValue, modelName: String) throws -> ModelResponse {
        if let err = json["error"]?["message"]?.stringValue {
            throw AgentError.provider("OpenAI: \(err)")
        }
        guard let choice = json["choices"]?[0] else {
            throw AgentError.provider("OpenAI: no choices in response")
        }
        let messageObj = choice["message"]

        var parts: [ModelResponsePart] = []
        // Reasoning models exposed over OpenAI-compatible APIs surface chain-of-thought
        // in `reasoning_content` (DeepSeek-style) or `reasoning` (OpenRouter-style).
        if let reasoning = (messageObj?["reasoning_content"] ?? messageObj?["reasoning"])?.stringValue,
           !reasoning.isEmpty {
            parts.append(.thinking(reasoning))
        }
        if let content = messageObj?["content"]?.stringValue, !content.isEmpty {
            parts.append(.text(content))
        }
        if let toolCalls = messageObj?["tool_calls"]?.arrayValue {
            for call in toolCalls {
                guard let id = call["id"]?.stringValue,
                      let name = call["function"]?["name"]?.stringValue else { continue }
                let argsString = call["function"]?["arguments"]?.stringValue ?? "{}"
                let arguments = (try? JSONValue(jsonString: argsString)) ?? .object([:])
                parts.append(.toolCall(ToolCall(id: id, name: name, arguments: arguments)))
            }
        }

        var usage = Usage()
        if let u = json["usage"] {
            if case let .int(i) = u["prompt_tokens"] ?? .null { usage.inputTokens = i }
            if case let .int(o) = u["completion_tokens"] ?? .null { usage.outputTokens = o }
        }

        return ModelResponse(
            parts: parts,
            usage: usage,
            modelName: json["model"]?.stringValue ?? modelName,
            responseID: json["id"]?.stringValue,
            finishReason: mapFinishReason(choice["finish_reason"]?.stringValue))
    }

    /// Map OpenAI's `finish_reason` onto the normalized `FinishReason`.
    static func mapFinishReason(_ raw: String?) -> FinishReason? {
        switch raw {
        case "stop": return .stop
        case "length": return .length
        case "tool_calls", "function_call": return .toolCall
        case "content_filter": return .contentFilter
        case nil: return nil
        default: return .error
        }
    }
}
