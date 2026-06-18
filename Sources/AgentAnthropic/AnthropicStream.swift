import AgentCore
import Foundation

/// Incremental decoder for Anthropic Messages API SSE streams. Pure (no
/// networking) so it can be golden-tested by feeding it the `data` payloads.
///
/// Dispatch is driven by each payload's `type` field rather than the SSE event
/// name, which is more robust to transport quirks.
struct AnthropicStreamDecoder {
    private struct Block {
        var type: String
        var id: String?
        var name: String?
        var text: String = ""
        var json: String = ""
    }

    private var blocks: [Int: Block] = [:]
    private var order: [Int] = []
    private var usage = Usage()
    private var finishReason: FinishReason?
    private var responseID: String?
    private var modelName: String
    private var finished = false

    init(modelName: String) { self.modelName = modelName }

    /// Consume one SSE `data` payload, returning any events it produces.
    mutating func ingest(_ data: String) -> [ModelStreamEvent] {
        guard let json = try? JSONValue(jsonString: data) else { return [] }
        switch json["type"]?.stringValue {
        case "message_start":
            if let m = json["message"] {
                responseID = m["id"]?.stringValue ?? responseID
                modelName = m["model"]?.stringValue ?? modelName
                if let i = m["usage"]?["input_tokens"]?.intValue { usage.inputTokens = i }
            }
            return []

        case "content_block_start":
            guard let index = json["index"]?.intValue else { return [] }
            let cb = json["content_block"]
            var block = Block(type: cb?["type"]?.stringValue ?? "text")
            block.id = cb?["id"]?.stringValue
            block.name = cb?["name"]?.stringValue
            blocks[index] = block
            order.append(index)
            return []

        case "content_block_delta":
            guard let index = json["index"]?.intValue, var block = blocks[index] else { return [] }
            let delta = json["delta"]
            switch delta?["type"]?.stringValue {
            case "text_delta":
                let t = delta?["text"]?.stringValue ?? ""
                block.text += t; blocks[index] = block
                return [.textDelta(t)]
            case "thinking_delta":
                let t = delta?["thinking"]?.stringValue ?? ""
                block.text += t; blocks[index] = block
                return [.thinkingDelta(t)]
            case "input_json_delta":
                let frag = delta?["partial_json"]?.stringValue ?? ""
                block.json += frag; blocks[index] = block
                return [.toolCallDelta(ToolCallDelta(
                    index: index, id: block.id, name: block.name, argumentsFragment: frag))]
            default:
                return []
            }

        case "content_block_stop":
            guard let index = json["index"]?.intValue, let block = blocks[index],
                  block.type == "tool_use", let id = block.id, let name = block.name else { return [] }
            return [.toolCall(ToolCall(id: id, name: name, arguments: parseArgs(block.json)))]

        case "message_delta":
            if let o = json["usage"]?["output_tokens"]?.intValue { usage.outputTokens = o }
            if let stop = json["delta"]?["stop_reason"]?.stringValue {
                finishReason = AnthropicWire.mapFinishReason(stop)
            }
            return []

        case "message_stop":
            return [completedEvent()]

        default:
            return []
        }
    }

    /// Emit the terminal `.completed` if the stream ended without `message_stop`.
    mutating func finish() -> [ModelStreamEvent] {
        finished ? [] : [completedEvent()]
    }

    private mutating func completedEvent() -> ModelStreamEvent {
        finished = true
        var parts: [ModelResponsePart] = []
        for index in order {
            guard let block = blocks[index] else { continue }
            switch block.type {
            case "text": if !block.text.isEmpty { parts.append(.text(block.text)) }
            case "thinking": if !block.text.isEmpty { parts.append(.thinking(block.text)) }
            case "tool_use":
                if let id = block.id, let name = block.name {
                    parts.append(.toolCall(ToolCall(id: id, name: name, arguments: parseArgs(block.json))))
                }
            default: break
            }
        }
        return .completed(ModelResponse(
            parts: parts, usage: usage, modelName: modelName,
            responseID: responseID, finishReason: finishReason))
    }

    private func parseArgs(_ json: String) -> JSONValue {
        (try? JSONValue(jsonString: json.isEmpty ? "{}" : json)) ?? .object([:])
    }
}
