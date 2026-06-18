import AgentCore
import Foundation

/// Incremental decoder for Gemini `streamGenerateContent?alt=sse` streams. Pure
/// (no networking); feed it each `data:` payload (a partial GenerateContentResponse).
///
/// Gemini omits tool-call ids, so we synthesize a stable `name-index` id to
/// match the non-streaming `GeminiWire.decodeResponse` behavior.
struct GeminiStreamDecoder {
    private struct TC { var id: String; var name: String; var args: JSONValue }

    private var text = ""
    private var thinking = ""
    private var toolCalls: [TC] = []
    private var callIndex = 0
    private var usage: Usage?
    private var finishReason: FinishReason?
    private var responseID: String?
    private var modelName: String
    private var finished = false

    init(modelName: String) { self.modelName = modelName }

    mutating func ingest(_ data: String) -> [ModelStreamEvent] {
        guard let json = try? JSONValue(jsonString: data) else { return [] }
        modelName = json["modelVersion"]?.stringValue ?? modelName
        responseID = json["responseId"]?.stringValue ?? responseID
        if let meta = json["usageMetadata"] {
            var us = Usage()
            if let i = meta["promptTokenCount"]?.intValue { us.inputTokens = i }
            if let o = meta["candidatesTokenCount"]?.intValue { us.outputTokens = o }
            usage = us
        }

        guard let candidate = json["candidates"]?[0] else { return [] }
        var events: [ModelStreamEvent] = []
        for part in candidate["content"]?["parts"]?.arrayValue ?? [] {
            if let t = part["text"]?.stringValue {
                if part["thought"]?.boolValue == true {
                    thinking += t; events.append(.thinkingDelta(t))
                } else {
                    text += t; events.append(.textDelta(t))
                }
            } else if let fn = part["functionCall"], let name = fn["name"]?.stringValue {
                let args = fn["args"] ?? .object([:])
                let id = "\(name)-\(callIndex)"
                let position = toolCalls.count
                callIndex += 1
                toolCalls.append(TC(id: id, name: name, args: args))
                let frag = (try? args.jsonString()) ?? "{}"
                events.append(.toolCallDelta(ToolCallDelta(
                    index: position, id: id, name: name, argumentsFragment: frag)))
                events.append(.toolCall(ToolCall(id: id, name: name, arguments: args)))
            }
        }
        if let fr = candidate["finishReason"]?.stringValue {
            finishReason = GeminiWire.mapFinishReason(fr)
        }
        return events
    }

    mutating func finish() -> [ModelStreamEvent] {
        if finished { return [] }
        finished = true
        var parts: [ModelResponsePart] = []
        if !thinking.isEmpty { parts.append(.thinking(thinking)) }
        if !text.isEmpty { parts.append(.text(text)) }
        for tc in toolCalls {
            parts.append(.toolCall(ToolCall(id: tc.id, name: tc.name, arguments: tc.args)))
        }
        return [.completed(ModelResponse(
            parts: parts, usage: usage, modelName: modelName,
            responseID: responseID, finishReason: finishReason))]
    }
}
