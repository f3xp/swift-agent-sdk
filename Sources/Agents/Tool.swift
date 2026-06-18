import AgentCore
import FoundationModels

/// The value a tool returns to the model.
public struct ToolResult: Sendable {
    public let content: JSONValue

    public init(content: JSONValue) { self.content = content }

    /// A plain-text result.
    public init(_ text: String) { self.content = .string(text) }

    /// A structured result from any `@Generable` value.
    public init(value: some Generable) throws {
        self.content = try value.generatedContent.toJSONValue()
    }
}

/// A user-facing tool, generic over the agent's dependencies and its arguments.
/// Throw `ModelRetry` from `call` to ask the model to correct itself.
public protocol AgentTool<Deps>: Sendable {
    associatedtype Deps: Sendable
    associatedtype Arguments: AgentSchema
    var name: String { get }
    var description: String { get }
    func call(_ args: Arguments, context: RunContext<Deps>) async throws -> ToolResult
}

/// Type-erased tool used for heterogeneous storage on the agent.
///
/// All tools on one agent share `Deps` (they share the run context), so only
/// `Arguments` is erased — and it collapses cleanly through `JSONValue`.
struct AnyAgentTool<Deps: Sendable>: Sendable {
    let definition: ToolDefinition
    let invoke: @Sendable (_ arguments: JSONValue, _ context: RunContext<Deps>) async throws -> JSONValue

    /// Build from a typed closure.
    static func closure<A: AgentSchema>(
        name: String,
        description: String,
        body: @escaping @Sendable (A, RunContext<Deps>) async throws -> ToolResult
    ) -> AnyAgentTool<Deps> {
        let params = (try? A.schemaJSON()) ?? .object(["type": "object"])
        let def = ToolDefinition(name: name, description: description, parameters: params)
        return AnyAgentTool(definition: def) { json, ctx in
            let args = try decodeArguments(A.self, from: json, toolName: name)
            return try await body(args, ctx).content
        }
    }

    /// Build from a concrete `AgentTool`.
    static func erasing<T: AgentTool>(_ tool: T) -> AnyAgentTool<Deps> where T.Deps == Deps {
        let params = (try? T.Arguments.schemaJSON()) ?? .object(["type": "object"])
        let def = ToolDefinition(name: tool.name, description: tool.description, parameters: params)
        return AnyAgentTool(definition: def) { json, ctx in
            let args = try decodeArguments(T.Arguments.self, from: json, toolName: tool.name)
            return try await tool.call(args, context: ctx).content
        }
    }
}

/// Decode tool arguments, converting a decode failure into a `ModelRetry` so the
/// model can correct malformed arguments instead of aborting the run.
private func decodeArguments<A: AgentSchema>(_ type: A.Type, from json: JSONValue, toolName: String) throws -> A {
    do {
        return try A(jsonValue: json)
    } catch {
        throw ModelRetry("Invalid arguments for tool `\(toolName)`: \(error)")
    }
}
