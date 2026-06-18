import Testing
import FoundationModels
import AgentCore
import Agents
import AgentTestSupport
@testable import AgentGoogle

@Suite("Gemini schema normalization")
struct GeminiSchemaTests {

    @Test("uppercases types and strips additionalProperties")
    func normalize() throws {
        let schema: JSONValue = .object([
            "type": "object",
            "additionalProperties": .bool(false),
            "properties": .object([
                "city": .object(["type": "string"]),
                "age": .object(["type": "integer"]),
            ]),
        ])
        let g = GeminiSchema.normalize(schema)
        #expect(g["type"]?.stringValue == "OBJECT")
        #expect(g["additionalProperties"] == nil)
        #expect(g["properties"]?["city"]?["type"]?.stringValue == "STRING")
        #expect(g["properties"]?["age"]?["type"]?.stringValue == "INTEGER")
    }

    @Test("recurses into array items")
    func normalizeArray() throws {
        let schema: JSONValue = .object([
            "type": "array",
            "items": .object(["type": "object", "$schema": "x",
                              "properties": .object(["id": .object(["type": "number"])])]),
        ])
        let g = GeminiSchema.normalize(schema)
        #expect(g["type"]?.stringValue == "ARRAY")
        #expect(g["items"]?["type"]?.stringValue == "OBJECT")
        #expect(g["items"]?["$schema"] == nil)
        #expect(g["items"]?["properties"]?["id"]?["type"]?.stringValue == "NUMBER")
    }
}

@Suite("Gemini wire translation")
struct GeminiWireTests {

    @Test("encodes systemInstruction, contents, and native responseSchema")
    func encodeRequest() throws {
        let messages: [ModelMessage] = [.request([.system("Be terse."), .userText("Hi")])]
        let schema: JSONValue = .object([
            "type": "object", "properties": .object(["city": .object(["type": "string"])])])
        let body = GeminiWire.encodeRequest(
            model: "gemini-3-flash", messages: messages,
            settings: ModelSettings(temperature: 0.1, maxTokens: 128), tools: [],
            output: OutputSpec(mode: .native, name: "loc", schema: schema))

        #expect(body["systemInstruction"]?["parts"]?[0]?["text"]?.stringValue == "Be terse.")
        let contents = body["contents"]?.arrayValue
        #expect(contents?.count == 1)
        #expect(contents?[0]["role"]?.stringValue == "user")
        #expect(contents?[0]["parts"]?[0]?["text"]?.stringValue == "Hi")

        let cfg = body["generationConfig"]
        #expect(cfg?["temperature"] == .double(0.1))
        #expect(cfg?["maxOutputTokens"] == .int(128))
        #expect(cfg?["responseMimeType"]?.stringValue == "application/json")
        #expect(cfg?["responseSchema"]?["type"]?.stringValue == "OBJECT")
    }

    @Test("encodes tools as functionDeclarations and tool cycle by name")
    func encodeToolCycle() throws {
        let messages: [ModelMessage] = [
            .request([.userText("weather?")]),
            .response([.toolCall(ToolCall(id: "get_weather-0", name: "get_weather", arguments: .object(["city": "Paris"])))]),
            .request([.toolReturn(ToolReturn(callID: "get_weather-0", name: "get_weather", content: .string("sunny")))]),
        ]
        let tools = [ToolDefinition(
            name: "get_weather", description: "weather",
            parameters: .object(["type": "object", "properties": .object(["city": .object(["type": "string"])])]))]
        let body = GeminiWire.encodeRequest(
            model: "gemini-3-flash", messages: messages, settings: ModelSettings(), tools: tools, output: .text)

        let decl = body["tools"]?[0]?["functionDeclarations"]?[0]
        #expect(decl?["name"]?.stringValue == "get_weather")
        #expect(decl?["parameters"]?["type"]?.stringValue == "OBJECT")

        let contents = body["contents"]?.arrayValue
        #expect(contents?[1]["role"]?.stringValue == "model")
        #expect(contents?[1]["parts"]?[0]?["functionCall"]?["name"]?.stringValue == "get_weather")
        // functionResponse wraps a non-object result and is correlated by name.
        let fnResp = contents?[2]["parts"]?[0]?["functionResponse"]
        #expect(fnResp?["name"]?.stringValue == "get_weather")
        #expect(fnResp?["response"]?["result"]?.stringValue == "sunny")
    }

    @Test("decodes candidates text + functionCall + usage")
    func decodeResponse() throws {
        let payload: JSONValue = .object([
            "modelVersion": "gemini-3-flash",
            "candidates": .array([.object([
                "finishReason": "STOP",
                "content": .object(["role": "model", "parts": .array([
                    .object(["text": "Checking."]),
                    .object(["functionCall": .object([
                        "name": "get_weather", "args": .object(["city": "Paris"])])]),
                ])]),
            ])]),
            "usageMetadata": .object(["promptTokenCount": 11, "candidatesTokenCount": 4]),
        ])
        let message = try GeminiWire.decodeResponse(payload, modelName: "gemini-3-flash")
        #expect(message.text == "Checking.")
        #expect(message.toolCalls.count == 1)
        #expect(message.toolCalls[0].name == "get_weather")
        #expect(message.toolCalls[0].id == "get_weather-0")
        #expect(message.toolCalls[0].arguments["city"]?.stringValue == "Paris")
        #expect(message.usage?.inputTokens == 11)
        #expect(message.usage?.outputTokens == 4)
    }

    @Test("encodes media as inlineData and fileData parts")
    func encodesMedia() throws {
        let messages: [ModelMessage] = [
            .request([
                .userText("describe"),
                .userMedia(.image(base64: "aGVsbG8=", mediaType: "image/png")),
                .userMedia(.file(url: "gs://bucket/doc.pdf", mediaType: "application/pdf")),
            ])
        ]
        let body = GeminiWire.encodeRequest(
            model: "gemini-3-flash", messages: messages, settings: ModelSettings(), tools: [], output: .text)
        let parts = body["contents"]?[0]?["parts"]?.arrayValue
        #expect(parts?.count == 3)
        #expect(parts?[0]["text"]?.stringValue == "describe")
        #expect(parts?[1]["inlineData"]?["mimeType"]?.stringValue == "image/png")
        #expect(parts?[1]["inlineData"]?["data"]?.stringValue == "aGVsbG8=")
        #expect(parts?[2]["fileData"]?["mimeType"]?.stringValue == "application/pdf")
        #expect(parts?[2]["fileData"]?["fileUri"]?.stringValue == "gs://bucket/doc.pdf")
    }

    @Test("emits thinkingConfig in generationConfig")
    func encodesThinking() throws {
        let body = GeminiWire.encodeRequest(
            model: "gemini-3-flash", messages: [.request([.userText("think")])],
            settings: ModelSettings(thinking: ThinkingConfig(budgetTokens: 1024)),
            tools: [], output: .text)
        let cfg = body["generationConfig"]?["thinkingConfig"]
        #expect(cfg?["thinkingBudget"] == .int(1024))
        #expect(cfg?["includeThoughts"] == .bool(true))
    }

    @Test("decodes a thought:true part as thinking")
    func decodesThought() throws {
        let payload: JSONValue = .object([
            "modelVersion": "gemini-3-flash",
            "candidates": .array([.object([
                "finishReason": "STOP",
                "content": .object(["role": "model", "parts": .array([
                    .object(["text": "reasoning here", "thought": .bool(true)]),
                    .object(["text": "final answer"]),
                ])]),
            ])]),
        ])
        let message = try GeminiWire.decodeResponse(payload, modelName: "gemini-3-flash")
        let thinking = message.parts.compactMap { part -> String? in
            if case let .thinking(t) = part { return t } else { return nil }
        }
        #expect(thinking == ["reasoning here"])
        #expect(message.text == "final answer")
    }

    @Test("surfaces API errors")
    func decodeError() throws {
        let payload: JSONValue = .object(["error": .object(["message": "API key not valid"])])
        #expect(throws: AgentError.self) {
            _ = try GeminiWire.decodeResponse(payload, modelName: "gemini-3-flash")
        }
    }

    @Test("stream decoder assembles thought + text + functionCall chunks")
    func streamDecode() throws {
        let chunks: [JSONValue] = [
            ["modelVersion": "gemini-3-flash", "candidates": [["content": ["role": "model",
                "parts": [["text": "reasoning", "thought": true]]]]]],
            ["candidates": [["content": ["role": "model", "parts": [["text": "Checking."]]]]]],
            ["candidates": [["content": ["role": "model",
                "parts": [["functionCall": ["name": "get_weather", "args": ["city": "Paris"]]]]],
                "finishReason": "STOP"]],
                "usageMetadata": ["promptTokenCount": 11, "candidatesTokenCount": 4]],
        ]
        var decoder = GeminiStreamDecoder(modelName: "gemini-3-flash")
        var events: [ModelStreamEvent] = []
        for c in chunks { events.append(contentsOf: decoder.ingest(try c.jsonString())) }
        events.append(contentsOf: decoder.finish())

        let thinking = events.compactMap { if case let .thinkingDelta(t) = $0 { return t } else { return nil } }
        let texts = events.compactMap { if case let .textDelta(t) = $0 { return t } else { return nil } }
        #expect(thinking == ["reasoning"])
        #expect(texts == ["Checking."])

        let calls = events.compactMap { if case let .toolCall(c) = $0 { return c } else { return nil } }
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].id == "get_weather-0")
        #expect(calls[0].arguments["city"]?.stringValue == "Paris")

        guard case let .completed(resp)? = events.last else { Issue.record("no completed event"); return }
        #expect(resp.text == "Checking.")
        #expect(resp.toolCalls.count == 1)
        #expect(resp.usage?.inputTokens == 11)
        #expect(resp.usage?.outputTokens == 4)
        #expect(resp.finishReason == .stop)
    }

    @Test("registers the google selector")
    func registration() throws {
        AgentGoogle.register()
        let model = try ModelRegistry.shared.resolve(ModelSelector("google:gemini-3-flash"))
        #expect(model.modelName == "gemini-3-flash")
        #expect(model.profile.supportsNativeStructuredOutput)
    }
}

@Generable
struct GeoCity: Equatable {
    @Guide(description: "City") var city: String
    @Guide(description: "Country") var country: String
}

@Suite("Gemini via run loop")
struct GeminiRunLoopTests {

    @Test("native structured output decodes through the loop")
    func nativeStructured() async throws {
        let model = FunctionModel(
            profile: ModelProfile(
                supportsNativeStructuredOutput: true,
                defaultOutputMode: .native,
                jsonSchemaTransform: GeminiSchema.normalize)
        ) { _, _, output in
            #expect(output.mode == .native)
            #expect(output.schema?["type"]?.stringValue == "OBJECT")
            return ModelResponse(parts: [.text(#"{"city":"London","country":"United Kingdom"}"#)],
                                 usage: Usage(inputTokens: 5, outputTokens: 5))
        }
        let agent = Agent<Void, GeoCity>(model)
        let result = try await agent.run("Where were the 2012 Olympics held?")
        #expect(result.output == GeoCity(city: "London", country: "United Kingdom"))
    }
}
