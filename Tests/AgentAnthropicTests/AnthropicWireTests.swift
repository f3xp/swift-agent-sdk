import Testing
import FoundationModels
import AgentCore
@testable import AgentAnthropic

@Suite("Anthropic wire translation")
struct AnthropicWireTests {

    @Test("encodes system + user prompt and tools")
    func encodeRequest() throws {
        let messages: [ModelMessage] = [
            .request([.system("Be terse."), .userText("Hello")])
        ]
        let tools = [ToolDefinition(
            name: "get_weather", description: "weather",
            parameters: .object(["type": "object", "properties": .object(["city": .object(["type": "string"])])]))]
        let body = AnthropicWire.encodeRequest(
            model: "claude-sonnet-4-6", messages: messages,
            settings: ModelSettings(maxTokens: 512), tools: tools, output: .text)

        #expect(body["model"]?.stringValue == "claude-sonnet-4-6")
        #expect(body["max_tokens"] == .int(512))
        #expect(body["system"]?.stringValue == "Be terse.")

        // One user message with a single text block.
        guard case let .array(msgs) = body["messages"]! else { Issue.record("messages not array"); return }
        #expect(msgs.count == 1)
        #expect(msgs[0]["role"]?.stringValue == "user")

        // Tool surfaced with input_schema.
        guard case let .array(apiTools) = body["tools"]! else { Issue.record("tools not array"); return }
        #expect(apiTools[0]["name"]?.stringValue == "get_weather")
        #expect(apiTools[0]["input_schema"]?["type"]?.stringValue == "object")
    }

    @Test("forces the output tool when it's the only tool")
    func forcesOutputTool() throws {
        let outputTool = ToolDefinition(
            name: "final_result", description: "out",
            parameters: .object(["type": "object"]))
        let body = AnthropicWire.encodeRequest(
            model: "claude", messages: [.request([.userText("hi")])],
            settings: ModelSettings(), tools: [outputTool],
            output: OutputSpec(mode: .tool, name: "final_result", schema: .object(["type": "object"])))
        #expect(body["tool_choice"]?["type"]?.stringValue == "tool")
        #expect(body["tool_choice"]?["name"]?.stringValue == "final_result")
    }

    @Test("encodes tool results from prior history")
    func encodesToolResult() throws {
        let messages: [ModelMessage] = [
            .request([.userText("weather?")]),
            .response([.toolCall(ToolCall(id: "c1", name: "get_weather", arguments: .object(["city": "Paris"])))]),
            .request([.toolReturn(ToolReturn(callID: "c1", name: "get_weather", content: .string("sunny")))]),
        ]
        let body = AnthropicWire.encodeRequest(
            model: "claude", messages: messages, settings: ModelSettings(), tools: [], output: .text)
        guard case let .array(msgs) = body["messages"]! else { Issue.record("no messages"); return }
        #expect(msgs.count == 3)
        #expect(msgs[1]["role"]?.stringValue == "assistant")
        // tool_result block in the third (user) message.
        #expect(msgs[2]["content"]?[0]?["type"]?.stringValue == "tool_result")
        #expect(msgs[2]["content"]?[0]?["tool_use_id"]?.stringValue == "c1")
    }

    @Test("decodes text + tool_use + usage")
    func decodeResponse() throws {
        let payload: JSONValue = .object([
            "id": "msg_1",
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "content": .array([
                .object(["type": "text", "text": "Let me check."]),
                .object([
                    "type": "tool_use", "id": "tu_1", "name": "get_weather",
                    "input": .object(["city": "Paris"]),
                ]),
            ]),
            "usage": .object(["input_tokens": 12, "output_tokens": 7]),
        ])
        let message = try AnthropicWire.decodeResponse(payload, modelName: "claude")
        #expect(message.text == "Let me check.")
        #expect(message.toolCalls.count == 1)
        #expect(message.toolCalls[0].name == "get_weather")
        #expect(message.toolCalls[0].arguments["city"]?.stringValue == "Paris")
        #expect(message.usage?.inputTokens == 12)
        #expect(message.usage?.outputTokens == 7)
        #expect(message.finishReason == .toolCall)
    }

    @Test("encodes image (inline + url) and document media blocks")
    func encodesMedia() throws {
        let messages: [ModelMessage] = [
            .request([
                .userText("describe these"),
                .userMedia(.image(base64: "aGVsbG8=", mediaType: "image/png")),
                .userMedia(.image(url: "https://example.com/cat.jpg")),
                .userMedia(.file(base64: "JVBERi0=", mediaType: "application/pdf")),
            ])
        ]
        let body = AnthropicWire.encodeRequest(
            model: "claude", messages: messages, settings: ModelSettings(), tools: [], output: .text)
        guard case let .array(msgs) = body["messages"]! else { Issue.record("no messages"); return }
        let blocks = msgs[0]["content"]?.arrayValue ?? []
        #expect(blocks.count == 4)
        #expect(blocks[0]["type"]?.stringValue == "text")
        // Inline image → base64 source.
        #expect(blocks[1]["type"]?.stringValue == "image")
        #expect(blocks[1]["source"]?["type"]?.stringValue == "base64")
        #expect(blocks[1]["source"]?["media_type"]?.stringValue == "image/png")
        // URL image → url source.
        #expect(blocks[2]["source"]?["type"]?.stringValue == "url")
        #expect(blocks[2]["source"]?["url"]?.stringValue == "https://example.com/cat.jpg")
        // File → document block.
        #expect(blocks[3]["type"]?.stringValue == "document")
        #expect(blocks[3]["source"]?["data"]?.stringValue == "JVBERi0=")
    }

    @Test("emits thinking config and bumps max_tokens above the budget")
    func encodesThinking() throws {
        let body = AnthropicWire.encodeRequest(
            model: "claude", messages: [.request([.userText("think")])],
            settings: ModelSettings(maxTokens: 1000, thinking: ThinkingConfig(budgetTokens: 2048)),
            tools: [], output: .text)
        #expect(body["thinking"]?["type"]?.stringValue == "enabled")
        #expect(body["thinking"]?["budget_tokens"] == .int(2048))
        // max_tokens (1000) was <= budget, so it must have been raised above it.
        if case let .int(maxTokens) = body["max_tokens"]! {
            #expect(maxTokens > 2048)
        } else {
            Issue.record("max_tokens not an int")
        }
    }

    @Test("no thinking key when unset")
    func noThinkingByDefault() throws {
        let body = AnthropicWire.encodeRequest(
            model: "claude", messages: [.request([.userText("hi")])],
            settings: ModelSettings(), tools: [], output: .text)
        #expect(body["thinking"] == nil)
    }

    @Test("decodes thinking and redacted_thinking blocks")
    func decodesThinking() throws {
        let payload: JSONValue = .object([
            "id": "msg_2", "model": "claude", "stop_reason": "end_turn",
            "content": .array([
                .object(["type": "thinking", "thinking": "step one"]),
                .object(["type": "redacted_thinking", "data": "encrypted"]),
                .object(["type": "text", "text": "answer"]),
            ]),
        ])
        let message = try AnthropicWire.decodeResponse(payload, modelName: "claude")
        let thinking = message.parts.compactMap { part -> String? in
            if case let .thinking(t) = part { return t } else { return nil }
        }
        #expect(thinking == ["step one", "encrypted"])
        #expect(message.text == "answer")
    }

    @Test("surfaces API errors")
    func decodeError() throws {
        let payload: JSONValue = .object([
            "error": .object(["type": "invalid_request_error", "message": "bad key"])
        ])
        #expect(throws: AgentError.self) {
            _ = try AnthropicWire.decodeResponse(payload, modelName: "claude")
        }
    }

    @Test("stream decoder assembles text + tool_use across SSE events")
    func streamDecode() throws {
        let payloads: [JSONValue] = [
            ["type": "message_start", "message": ["id": "msg_1", "model": "claude-x",
                "usage": ["input_tokens": 10, "output_tokens": 0]]],
            ["type": "content_block_start", "index": 0, "content_block": ["type": "text"]],
            ["type": "content_block_delta", "index": 0, "delta": ["type": "text_delta", "text": "Hello"]],
            ["type": "content_block_delta", "index": 0, "delta": ["type": "text_delta", "text": " there"]],
            ["type": "content_block_stop", "index": 0],
            ["type": "content_block_start", "index": 1,
                "content_block": ["type": "tool_use", "id": "tu_1", "name": "get_weather"]],
            ["type": "content_block_delta", "index": 1,
                "delta": ["type": "input_json_delta", "partial_json": #"{"city":"#]],
            ["type": "content_block_delta", "index": 1,
                "delta": ["type": "input_json_delta", "partial_json": #""Paris"}"#]],
            ["type": "content_block_stop", "index": 1],
            ["type": "message_delta", "delta": ["stop_reason": "tool_use"], "usage": ["output_tokens": 7]],
            ["type": "message_stop"],
        ]
        var decoder = AnthropicStreamDecoder(modelName: "claude")
        var events: [ModelStreamEvent] = []
        for p in payloads { events.append(contentsOf: decoder.ingest(try p.jsonString())) }

        let texts = events.compactMap { if case let .textDelta(t) = $0 { return t } else { return nil } }
        #expect(texts == ["Hello", " there"])

        let calls = events.compactMap { if case let .toolCall(c) = $0 { return c } else { return nil } }
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["city"]?.stringValue == "Paris")

        guard case let .completed(resp)? = events.last else { Issue.record("no completed event"); return }
        #expect(resp.text == "Hello there")
        #expect(resp.toolCalls.count == 1)
        #expect(resp.usage?.inputTokens == 10)
        #expect(resp.usage?.outputTokens == 7)
        #expect(resp.finishReason == .toolCall)
    }
}
