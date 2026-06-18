import AgentCore
import Foundation

/// Incremental decoder for OpenAI Chat Completions SSE streams. Pure (no
/// networking); feed it each `data:` payload string in order.
struct OpenAIStreamDecoder {
    private struct TC { var id: String?; var name: String?; var args: String = "" }

    private var text = ""
    private var thinking = ""
    private var toolCalls: [Int: TC] = [:]
    private var toolOrder: [Int] = []
    private var usage: Usage?
    private var finishReason: FinishReason?
    private var responseID: String?
    private var modelName: String
    private var emittedComplete = false
    private var finished = false

    init(modelName: String) { self.modelName = modelName }

    mutating func ingest(_ data: String) -> [ModelStreamEvent] {
        if data.trimmingCharacters(in: .whitespaces) == "[DONE]" { return finish() }
        guard let json = try? JSONValue(jsonString: data) else { return [] }

        responseID = json["id"]?.stringValue ?? responseID
        modelName = json["model"]?.stringValue ?? modelName
        if let u = json["usage"], u != .null {
            var us = Usage()
            if let i = u["prompt_tokens"]?.intValue { us.inputTokens = i }
            if let o = u["completion_tokens"]?.intValue { us.outputTokens = o }
            usage = us
        }

        guard let choice = json["choices"]?[0] else { return [] }
        let delta = choice["delta"]
        var events: [ModelStreamEvent] = []

        if let content = delta?["content"]?.stringValue, !content.isEmpty {
            text += content
            events.append(.textDelta(content))
        }
        if let r = (delta?["reasoning_content"] ?? delta?["reasoning"])?.stringValue, !r.isEmpty {
            thinking += r
            events.append(.thinkingDelta(r))
        }
        if let tcs = delta?["tool_calls"]?.arrayValue {
            for tc in tcs {
                let idx = tc["index"]?.intValue ?? 0
                if toolCalls[idx] == nil { toolCalls[idx] = TC(); toolOrder.append(idx) }
                if let id = tc["id"]?.stringValue { toolCalls[idx]?.id = id }
                if let name = tc["function"]?["name"]?.stringValue { toolCalls[idx]?.name = name }
                let frag = tc["function"]?["arguments"]?.stringValue ?? ""
                if !frag.isEmpty { toolCalls[idx]?.args += frag }
                events.append(.toolCallDelta(ToolCallDelta(
                    index: idx, id: toolCalls[idx]?.id, name: toolCalls[idx]?.name, argumentsFragment: frag)))
            }
        }
        if let fr = choice["finish_reason"]?.stringValue {
            finishReason = OpenAIWire.mapFinishReason(fr)
            events.append(contentsOf: emitCompletedToolCalls())
        }
        return events
    }

    mutating func finish() -> [ModelStreamEvent] {
        if finished { return [] }
        finished = true
        var events = emitCompletedToolCalls()
        events.append(completedEvent())
        return events
    }

    private mutating func emitCompletedToolCalls() -> [ModelStreamEvent] {
        guard !emittedComplete, !toolOrder.isEmpty else { return [] }
        emittedComplete = true
        return toolOrder.compactMap { idx -> ModelStreamEvent? in
            guard let tc = toolCalls[idx], let id = tc.id, let name = tc.name else { return nil }
            return .toolCall(ToolCall(id: id, name: name, arguments: parseArgs(tc.args)))
        }
    }

    private func completedEvent() -> ModelStreamEvent {
        var parts: [ModelResponsePart] = []
        if !thinking.isEmpty { parts.append(.thinking(thinking)) }
        if !text.isEmpty { parts.append(.text(text)) }
        for idx in toolOrder {
            if let tc = toolCalls[idx], let id = tc.id, let name = tc.name {
                parts.append(.toolCall(ToolCall(id: id, name: name, arguments: parseArgs(tc.args))))
            }
        }
        return .completed(ModelResponse(
            parts: parts, usage: usage, modelName: modelName,
            responseID: responseID, finishReason: finishReason))
    }

    private func parseArgs(_ s: String) -> JSONValue {
        (try? JSONValue(jsonString: s.isEmpty ? "{}" : s)) ?? .object([:])
    }
}
