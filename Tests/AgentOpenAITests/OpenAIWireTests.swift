import Testing
import FoundationModels
import AgentCore
import Agents
import AgentTestSupport
@testable import AgentOpenAI

@Suite("OpenAI strict schema")
struct OpenAISchemaTests {

    @Test("adds additionalProperties:false and marks all properties required")
    func strictObject() throws {
        let schema: JSONValue = .object([
            "type": "object",
            "properties": .object([
                "city": .object(["type": "string"]),
                "country": .object(["type": "string"]),
            ]),
        ])
        let strict = OpenAISchema.strict(schema)
        #expect(strict["additionalProperties"] == .bool(false))
        let required = strict["required"]?.arrayValue?.compactMap(\.stringValue).sorted()
        #expect(required == ["city", "country"])
    }

    @Test("recurses into nested objects and arrays")
    func strictNested() throws {
        let schema: JSONValue = .object([
            "type": "object",
            "properties": .object([
                "items": .object([
                    "type": "array",
                    "items": .object([
                        "type": "object",
                        "properties": .object(["id": .object(["type": "integer"])]),
                    ]),
                ]),
            ]),
        ])
        let strict = OpenAISchema.strict(schema)
        let inner = strict["properties"]?["items"]?["items"]
        #expect(inner?["additionalProperties"] == .bool(false))
        #expect(inner?["required"]?.arrayValue?.first?.stringValue == "id")
    }
}

@Suite("OpenAI wire translation")
struct OpenAIWireTests {

    @Test("encodes system/user messages and strict tools")
    func encodeRequest() throws {
        let messages: [ModelMessage] = [.request([.system("Be terse."), .userText("Hi")])]
        let tools = [ToolDefinition(
            name: "get_weather", description: "weather",
            parameters: .object(["type": "object", "properties": .object(["city": .object(["type": "string"])])]))]
        let body = OpenAIWire.encodeRequest(
            model: "gpt-5.2", messages: messages,
            settings: ModelSettings(temperature: 0.2, maxTokens: 256), tools: tools, output: .text)

        #expect(body["model"]?.stringValue == "gpt-5.2")
        #expect(body["temperature"] == .double(0.2))
        #expect(body["max_tokens"] == .int(256))

        let msgs = body["messages"]?.arrayValue
        #expect(msgs?.count == 2)
        #expect(msgs?[0]["role"]?.stringValue == "system")
        #expect(msgs?[1]["role"]?.stringValue == "user")

        let fn = body["tools"]?[0]?["function"]
        #expect(fn?["name"]?.stringValue == "get_weather")
        #expect(fn?["strict"] == .bool(true))
        #expect(fn?["parameters"]?["additionalProperties"] == .bool(false))
    }

    @Test("native structured output sets response_format json_schema")
    func nativeOutput() throws {
        let schema: JSONValue = .object([
            "type": "object",
            "properties": .object(["city": .object(["type": "string"])]),
        ])
        let body = OpenAIWire.encodeRequest(
            model: "gpt-5.2", messages: [.request([.userText("where?")])],
            settings: ModelSettings(), tools: [],
            output: OutputSpec(mode: .native, name: "location", schema: schema))
        let jsonSchema = body["response_format"]?["json_schema"]
        #expect(body["response_format"]?["type"]?.stringValue == "json_schema")
        #expect(jsonSchema?["name"]?.stringValue == "location")
        #expect(jsonSchema?["strict"] == .bool(true))
        #expect(jsonSchema?["schema"]?["additionalProperties"] == .bool(false))
    }

    @Test("encodes assistant tool_calls and tool results from history")
    func encodesToolCycle() throws {
        let messages: [ModelMessage] = [
            .request([.userText("weather?")]),
            .response([.toolCall(ToolCall(id: "c1", name: "get_weather", arguments: .object(["city": "Paris"])))]),
            .request([.toolReturn(ToolReturn(callID: "c1", name: "get_weather", content: .string("sunny")))]),
        ]
        let body = OpenAIWire.encodeRequest(
            model: "gpt-5.2", messages: messages, settings: ModelSettings(), tools: [], output: .text)
        let msgs = body["messages"]?.arrayValue
        #expect(msgs?.count == 3)
        // assistant message with a tool_calls array
        #expect(msgs?[1]["role"]?.stringValue == "assistant")
        let toolCall = msgs?[1]["tool_calls"]?[0]
        #expect(toolCall?["function"]?["name"]?.stringValue == "get_weather")
        // arguments are serialized as a JSON string
        #expect(toolCall?["function"]?["arguments"]?.stringValue == #"{"city":"Paris"}"#)
        // tool result message
        #expect(msgs?[2]["role"]?.stringValue == "tool")
        #expect(msgs?[2]["tool_call_id"]?.stringValue == "c1")
    }

    @Test("decodes content + tool_calls + usage")
    func decodeResponse() throws {
        let payload: JSONValue = .object([
            "id": "chatcmpl_1",
            "model": "gpt-5.2",
            "choices": .array([.object([
                "finish_reason": "tool_calls",
                "message": .object([
                    "role": "assistant",
                    "content": .null,
                    "tool_calls": .array([.object([
                        "id": "call_1",
                        "type": "function",
                        "function": .object([
                            "name": "get_weather",
                            "arguments": .string(#"{"city":"Paris"}"#),
                        ]),
                    ])]),
                ]),
            ])]),
            "usage": .object(["prompt_tokens": 20, "completion_tokens": 8]),
        ])
        let message = try OpenAIWire.decodeResponse(payload, modelName: "gpt-5.2")
        #expect(message.toolCalls.count == 1)
        #expect(message.toolCalls[0].name == "get_weather")
        #expect(message.toolCalls[0].arguments["city"]?.stringValue == "Paris")
        #expect(message.usage?.inputTokens == 20)
        #expect(message.usage?.outputTokens == 8)
        #expect(message.finishReason == .toolCall)
    }

    @Test("text-only user message stays a plain string")
    func textOnlyStaysString() throws {
        let body = OpenAIWire.encodeRequest(
            model: "gpt-5.2", messages: [.request([.userText("hi")])],
            settings: ModelSettings(), tools: [], output: .text)
        let user = body["messages"]?[0]
        #expect(user?["role"]?.stringValue == "user")
        #expect(user?["content"]?.stringValue == "hi")
    }

    @Test("coalesces text + image + audio into one array-content user message")
    func multimodalContentArray() throws {
        let messages: [ModelMessage] = [
            .request([
                .userText("transcribe and describe"),
                .userMedia(.image(base64: "aGVsbG8=", mediaType: "image/png")),
                .userMedia(.image(url: "https://example.com/x.jpg")),
                .userMedia(.audio(base64: "YXVkaW8=", mediaType: "audio/wav")),
            ])
        ]
        let body = OpenAIWire.encodeRequest(
            model: "gpt-5.2", messages: messages, settings: ModelSettings(), tools: [], output: .text)
        let msgs = body["messages"]?.arrayValue
        #expect(msgs?.count == 1)
        let content = msgs?[0]["content"]?.arrayValue
        #expect(content?.count == 4)
        #expect(content?[0]["type"]?.stringValue == "text")
        // Inline image → data URL.
        #expect(content?[1]["type"]?.stringValue == "image_url")
        #expect(content?[1]["image_url"]?["url"]?.stringValue == "data:image/png;base64,aGVsbG8=")
        // Remote image → passthrough URL.
        #expect(content?[2]["image_url"]?["url"]?.stringValue == "https://example.com/x.jpg")
        // Audio → input_audio with derived format.
        #expect(content?[3]["type"]?.stringValue == "input_audio")
        #expect(content?[3]["input_audio"]?["format"]?.stringValue == "wav")
        #expect(content?[3]["input_audio"]?["data"]?.stringValue == "YXVkaW8=")
    }

    @Test("decodes reasoning_content into a thinking part")
    func decodesReasoning() throws {
        let payload: JSONValue = .object([
            "id": "chatcmpl_2", "model": "deepseek-reasoner",
            "choices": .array([.object([
                "finish_reason": "stop",
                "message": .object([
                    "role": "assistant",
                    "reasoning_content": "let me think...",
                    "content": "42",
                ]),
            ])]),
        ])
        let message = try OpenAIWire.decodeResponse(payload, modelName: "deepseek-reasoner")
        let thinking = message.parts.compactMap { part -> String? in
            if case let .thinking(t) = part { return t } else { return nil }
        }
        #expect(thinking == ["let me think..."])
        #expect(message.text == "42")
    }

    @Test("surfaces API errors")
    func decodeError() throws {
        let payload: JSONValue = .object(["error": .object(["message": "invalid api key"])])
        #expect(throws: AgentError.self) {
            _ = try OpenAIWire.decodeResponse(payload, modelName: "gpt-5.2")
        }
    }

    @Test("stream decoder reassembles tool-call args split across chunks")
    func streamToolCall() throws {
        let chunks: [JSONValue] = [
            ["id": "chatcmpl_1", "model": "gpt-5.2", "choices": [["index": 0,
                "delta": ["role": "assistant", "tool_calls": [[
                    "index": 0, "id": "call_1", "function": ["name": "get_weather", "arguments": #"{"ci"#]]]]]]],
            ["choices": [["index": 0, "delta": ["tool_calls": [[
                "index": 0, "function": ["arguments": #"ty":"Paris"}"#]]]]]]],
            ["choices": [["index": 0, "delta": [:], "finish_reason": "tool_calls"]]],
            ["choices": [], "usage": ["prompt_tokens": 20, "completion_tokens": 8]],
        ]
        var decoder = OpenAIStreamDecoder(modelName: "gpt-5.2")
        var events: [ModelStreamEvent] = []
        for c in chunks { events.append(contentsOf: decoder.ingest(try c.jsonString())) }
        events.append(contentsOf: decoder.ingest("[DONE]"))

        let calls = events.compactMap { if case let .toolCall(c) = $0 { return c } else { return nil } }
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].arguments["city"]?.stringValue == "Paris")

        guard case let .completed(resp)? = events.last else { Issue.record("no completed event"); return }
        #expect(resp.toolCalls.count == 1)
        #expect(resp.usage?.inputTokens == 20)
        #expect(resp.usage?.outputTokens == 8)
        #expect(resp.finishReason == .toolCall)
    }

    @Test("stream decoder surfaces reasoning + text deltas")
    func streamReasoning() throws {
        let chunks: [JSONValue] = [
            ["choices": [["index": 0, "delta": ["reasoning_content": "thinking"]]]],
            ["choices": [["index": 0, "delta": ["content": "42"]]]],
            ["choices": [["index": 0, "delta": [:], "finish_reason": "stop"]]],
        ]
        var decoder = OpenAIStreamDecoder(modelName: "deepseek")
        var events: [ModelStreamEvent] = []
        for c in chunks { events.append(contentsOf: decoder.ingest(try c.jsonString())) }
        events.append(contentsOf: decoder.finish())

        let thinking = events.compactMap { if case let .thinkingDelta(t) = $0 { return t } else { return nil } }
        let texts = events.compactMap { if case let .textDelta(t) = $0 { return t } else { return nil } }
        #expect(thinking == ["thinking"])
        #expect(texts == ["42"])

        guard case let .completed(resp)? = events.last else { Issue.record("no completed event"); return }
        #expect(resp.text == "42")
        #expect(resp.finishReason == .stop)
    }

    @Test("registers the openai selector")
    func registration() throws {
        AgentOpenAI.register()
        let model = try ModelRegistry.shared.resolve(ModelSelector("openai:gpt-5.2"))
        #expect(model.modelName == "gpt-5.2")
        #expect(model.profile.supportsNativeStructuredOutput)
    }
}

// MARK: - End-to-end run loop against a mocked OpenAI wire

@Generable
struct OAICity: Equatable {
    @Guide(description: "City") var city: String
    @Guide(description: "Country") var country: String
}

@Suite("OpenAI via run loop")
struct OpenAIRunLoopTests {

    /// Simulates OpenAI native structured output: the model returns JSON content
    /// matching the schema, which the run loop decodes in `.native` mode.
    @Test("native structured output decodes through the loop")
    func nativeStructured() async throws {
        let model = FunctionModel(
            profile: ModelProfile(
                supportsNativeStructuredOutput: true,
                defaultOutputMode: .native,
                jsonSchemaTransform: OpenAISchema.strict)
        ) { _, _, output in
            #expect(output.mode == .native)
            // The profile transform must have been applied to the output schema.
            #expect(output.schema?["additionalProperties"] == .bool(false))
            return ModelResponse(parts: [.text(#"{"city":"London","country":"United Kingdom"}"#)],
                                 usage: Usage(inputTokens: 5, outputTokens: 5))
        }
        let agent = Agent<Void, OAICity>(model)
        let result = try await agent.run("Where were the 2012 Olympics held?")
        #expect(result.output == OAICity(city: "London", country: "United Kingdom"))
    }
}
