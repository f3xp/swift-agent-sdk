import AgentCore
import AgentGraph
import FoundationModels

// The agent run loop, expressed on the generic `AgentGraph` engine:
//
//   UserPromptNode → ModelRequestNode → CallToolsNode → (ModelRequestNode | End)
//
// `run()` / `runStream()` / `iter()` all drive this same graph (see
// `Agent+RunLoop.swift` and `Agent+Iter.swift`). Behavior is identical to the
// previous hand-written `while` loop — the nodes just make its boundaries
// first-class so a run can be observed and stepped.

// MARK: - Shared run state (the graph's `State`)

/// The mutable state threaded through one agent run. A reference type so each
/// node's mutations are visible to the next (the engine never runs nodes
/// concurrently, so the unchecked `Sendable` conformance is safe).
final class AgentRunState: @unchecked Sendable {
    var messages: [ModelMessage]
    var usage: Usage
    var retry: Int
    /// The graph step counter — incremented once per model request. Surfaced as
    /// `RunContext.runStep` (pydantic-ai's `run_step`).
    var runStep: Int
    let outputSpec: OutputSpec
    let toolDefs: [ToolDefinition]

    init(messages: [ModelMessage], outputSpec: OutputSpec, toolDefs: [ToolDefinition]) {
        self.messages = messages
        self.usage = .zero
        self.retry = 0
        self.runStep = 0
        self.outputSpec = outputSpec
        self.toolDefs = toolDefs
    }
}

// MARK: - Run configuration (the graph's `Deps`)

/// The immutable per-run configuration the nodes need: the agent's model,
/// settings, tools, validators, instructions, and the user's dependencies.
struct AgentGraphDeps<Deps: Sendable, Output: AgentSchema & Sendable>: Sendable {
    let model: any ModelProtocol
    let settings: ModelSettings
    let tools: [AnyAgentTool<Deps>]
    let outputValidators: [@Sendable (Output, RunContext<Deps>) async throws -> Output]
    let instructionSources: [Agent<Deps, Output>.InstructionsSource]
    let endStrategy: EndStrategy
    let retries: Int
    let usageLimits: UsageLimits
    let userDeps: Deps
}

extension AgentGraphDeps {
    /// Concatenate static + dynamic instructions.
    func buildInstructions(context: RunContext<Deps>) async throws -> String {
        var parts: [String] = []
        for source in instructionSources {
            switch source {
            case let .static(s): parts.append(s)
            case let .dynamic(make): parts.append(try await make(context))
            }
        }
        return parts.joined(separator: "\n\n")
    }

    /// Run the output validators in order.
    func validate(_ output: Output, context: RunContext<Deps>) async throws -> Output {
        var current = output
        for validator in outputValidators {
            current = try await validator(current, context)
        }
        return current
    }

    func decodeOutput(fromJSON json: JSONValue) throws -> Output {
        do {
            return try Output(jsonValue: json)
        } catch {
            throw AgentError.outputDecodingFailed("\(error)")
        }
    }

    func decodeOutput(fromText text: String) throws -> Output {
        // Plain-text output.
        if Output.self == String.self, let s = text as? Output {
            return s
        }
        // Structured output returned as JSON text (native / prompted modes).
        let stripped = Self.stripCodeFences(text)
        let json = try JSONValue(jsonString: stripped)
        return try Output(jsonValue: json)
    }

    /// Invoke the given tool calls in parallel (a task group), returning their
    /// results in call order. A tool that throws `ModelRetry` yields a retry part.
    func executeTools(
        _ calls: [ToolCall],
        context: (String?) -> RunContext<Deps>
    ) async throws -> [ModelRequestPart] {
        let toolMap = Dictionary(tools.map { ($0.definition.name, $0) }, uniquingKeysWith: { a, _ in a })

        return try await withThrowingTaskGroup(of: (Int, ModelRequestPart).self) { group in
            for (index, call) in calls.enumerated() {
                guard let tool = toolMap[call.name] else {
                    throw AgentError.toolNotFound(call.name)
                }
                let ctx = context(call.id)
                group.addTask {
                    do {
                        let result = try await tool.invoke(call.arguments, ctx)
                        return (index, .toolReturn(ToolReturn(
                            callID: call.id, name: call.name, content: result)))
                    } catch let retryErr as ModelRetry {
                        return (index, .retryPrompt(RetryPrompt(
                            message: retryErr.message, toolCallID: call.id, toolName: call.name)))
                    }
                }
            }
            var results: [(Int, ModelRequestPart)] = []
            for try await result in group { results.append(result) }
            return results.sorted { $0.0 < $1.0 }.map(\.1)
        }
    }

    /// One streaming model turn: folds the model's deltas, emits agent stream
    /// events (including progressively-decoded `.partial` snapshots), and returns
    /// the assembled response. Ported from the previous `Agent.performTurn`.
    func streamTurn(
        messages: [ModelMessage],
        output: OutputSpec,
        emit: @Sendable (AgentStreamEvent<Output>) async -> Void
    ) async throws -> ModelResponse {
        var completed: ModelResponse?
        let tracksPartials = Output.self != String.self
        var outputBuffer = ""
        var outputToolIndex: Int?
        var lastPartial: GeneratedContent?

        func emitPartial() async {
            guard tracksPartials,
                  let json = PartialJSON.complete(outputBuffer),
                  let content = try? GeneratedContent(json: json),
                  content != lastPartial else { return }
            lastPartial = content
            await emit(.partial(content))
        }

        for try await event in model.stream(
            messages: messages, settings: settings, tools: toolDefs(for: output), output: output
        ) {
            switch event {
            case let .textDelta(t):
                await emit(.textDelta(t))
                if output.mode == .native || output.mode == .prompted {
                    outputBuffer += t
                    await emitPartial()
                }
            case let .thinkingDelta(t):
                await emit(.thinkingDelta(t))
            case let .toolCallDelta(d):
                if output.mode == .tool {
                    if d.name == output.name { outputToolIndex = d.index }
                    if let idx = outputToolIndex, d.index == idx {
                        outputBuffer += d.argumentsFragment
                        await emitPartial()
                    }
                }
            case let .toolCall(c):
                await emit(.toolCall(c))
            case let .completed(m):
                completed = m
            }
        }
        guard let completed else {
            throw AgentError.noOutput("model produced no completion event")
        }
        return completed
    }

    /// The tool definitions presented to the model (agent tools + the output tool
    /// in `.tool` mode). Recomputed from the spec so `streamTurn` matches `run`.
    private func toolDefs(for output: OutputSpec) -> [ToolDefinition] {
        var defs = tools.map(\.definition)
        if output.mode == .tool, let schema = output.schema {
            defs.append(ToolDefinition(
                name: output.name, description: output.description, parameters: schema))
        }
        return defs
    }

    private static func stripCodeFences(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            if let firstNewline = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: firstNewline)...])
            }
            if s.hasSuffix("```") {
                s = String(s.dropLast(3))
            }
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Retry helper

/// Append a retry prompt and consume one unit of the retry budget; throw when the
/// budget is exhausted. Mutates `state`.
func beginRetry(
    _ error: ModelRetry,
    state: AgentRunState,
    retries: Int,
    toolCallID: String?,
    toolName: String?
) throws {
    state.retry += 1
    if state.retry > retries {
        throw AgentError.retriesExhausted(error.message)
    }
    state.messages.append(.request(ModelRequest(parts: [.retryPrompt(
        RetryPrompt(message: error.message, toolCallID: toolCallID, toolName: toolName))])))
}

// MARK: - Nodes

/// Builds the initial request (instructions + user prompt + attachments) and
/// hands off to the first model request.
final class UserPromptNode<Deps: Sendable, Output: AgentSchema & Sendable>: GraphNode, GraphNodeRegistration, @unchecked Sendable {
    typealias State = AgentRunState
    typealias RunEnd = AgentRunResult<Output>

    static var graphNodeID: String { "UserPromptNode" }
    static var edges: [String] { ["ModelRequestNode"] }
    var nodeID: String { "UserPromptNode" }

    let prompt: String
    let attachments: [MediaContent]

    init(prompt: String, attachments: [MediaContent]) {
        self.prompt = prompt
        self.attachments = attachments
    }

    func run(
        _ ctx: GraphRunContext<AgentRunState, AgentGraphDeps<Deps, Output>>
    ) async throws -> NextStep<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>> {
        let context = makeRunContext(state: ctx.state, deps: ctx.deps)
        let instructionText = try await ctx.deps.buildInstructions(context: context(nil))
        var initialParts: [ModelRequestPart] = []
        if !instructionText.isEmpty { initialParts.append(.system(instructionText)) }
        initialParts.append(.userText(prompt))
        for media in attachments { initialParts.append(.userMedia(media)) }
        ctx.state.messages.append(.request(ModelRequest(parts: initialParts)))
        return .next(ModelRequestNode<Deps, Output>())
    }
}

/// Performs one model request. `run` issues a non-streaming request (or consumes
/// a response already produced by `stream`); `stream` issues a streaming request,
/// emitting deltas/partials and caching the result for the subsequent `run`.
final class ModelRequestNode<Deps: Sendable, Output: AgentSchema & Sendable>: GraphNode, GraphNodeRegistration, @unchecked Sendable {
    typealias State = AgentRunState
    typealias RunEnd = AgentRunResult<Output>

    static var graphNodeID: String { "ModelRequestNode" }
    static var edges: [String] { ["CallToolsNode"] }
    var nodeID: String { "ModelRequestNode" }

    private var cachedResponse: ModelResponse?

    /// Stream this turn's model deltas. Caches the assembled response so the
    /// following `run` consumes it instead of re-requesting.
    func stream(
        _ ctx: GraphRunContext<AgentRunState, AgentGraphDeps<Deps, Output>>
    ) -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try ctx.deps.usageLimits.check(ctx.state.usage)
                    let response = try await ctx.deps.streamTurn(
                        messages: ctx.state.messages, output: ctx.state.outputSpec
                    ) { event in continuation.yield(event) }
                    self.cachedResponse = response
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func run(
        _ ctx: GraphRunContext<AgentRunState, AgentGraphDeps<Deps, Output>>
    ) async throws -> NextStep<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>> {
        try ctx.deps.usageLimits.check(ctx.state.usage)
        let response: ModelResponse
        if let cached = cachedResponse {
            response = cached
        } else {
            response = try await ctx.deps.model.request(
                messages: ctx.state.messages, settings: ctx.deps.settings,
                tools: ctx.state.toolDefs, output: ctx.state.outputSpec)
        }
        ctx.state.usage.requests += 1
        if let u = response.usage {
            ctx.state.usage.inputTokens += u.inputTokens
            ctx.state.usage.outputTokens += u.outputTokens
        }
        ctx.state.messages.append(.response(response))
        ctx.state.runStep += 1
        try ctx.deps.usageLimits.check(ctx.state.usage)
        return .next(CallToolsNode<Deps, Output>(response: response))
    }
}

/// Decides what the model's response means: finalize the output (ending the run),
/// or execute the requested tools and loop back for another model request.
/// Handles `EndStrategy`, output validation, and the retry budget.
final class CallToolsNode<Deps: Sendable, Output: AgentSchema & Sendable>: GraphNode, GraphNodeRegistration, @unchecked Sendable {
    typealias State = AgentRunState
    typealias RunEnd = AgentRunResult<Output>

    static var graphNodeID: String { "CallToolsNode" }
    static var edges: [String] { ["ModelRequestNode", GraphEnd] }
    var nodeID: String { "CallToolsNode" }

    let response: ModelResponse
    private var cachedStep: NextStep<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>>?

    init(response: ModelResponse) { self.response = response }

    /// Process tool calls while emitting `.toolReturn` events. Caches the
    /// resulting step so the following `run` reuses it.
    func stream(
        _ ctx: GraphRunContext<AgentRunState, AgentGraphDeps<Deps, Output>>
    ) -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let step = try await self.process(ctx) { event in continuation.yield(event) }
                    self.cachedStep = step
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func run(
        _ ctx: GraphRunContext<AgentRunState, AgentGraphDeps<Deps, Output>>
    ) async throws -> NextStep<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>> {
        if let cachedStep { return cachedStep }
        return try await process(ctx) { _ in }
    }

    private func process(
        _ ctx: GraphRunContext<AgentRunState, AgentGraphDeps<Deps, Output>>,
        emit: @Sendable (AgentStreamEvent<Output>) async -> Void
    ) async throws -> NextStep<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>> {
        let state = ctx.state
        let deps = ctx.deps
        let outputSpec = state.outputSpec
        let allCalls = response.toolCalls
        let context = makeRunContext(state: state, deps: deps)

        // ---- Try to finalize output ----
        if outputSpec.mode == .tool {
            if let outCall = allCalls.first(where: { $0.name == outputSpec.name }) {
                do {
                    let decoded = try deps.decodeOutput(fromJSON: outCall.arguments)
                    // Exhaustive: run the sibling (non-output) tool calls for their
                    // side effects before finalizing. Early skips them.
                    if deps.endStrategy == .exhaustive {
                        let siblings = allCalls.filter { $0.name != outputSpec.name }
                        if !siblings.isEmpty {
                            let parts = try await deps.executeTools(siblings, context: context)
                            state.usage.toolCalls += siblings.count
                            for part in parts {
                                if case let .toolReturn(r) = part { await emit(.toolReturn(r)) }
                            }
                            state.messages.append(.request(ModelRequest(parts: parts)))
                        }
                    }
                    let validated = try await deps.validate(decoded, context: context(outCall.id))
                    return .end(AgentRunResult(output: validated, messages: state.messages, usage: state.usage))
                } catch let retryErr as ModelRetry {
                    try beginRetry(retryErr, state: state, retries: deps.retries, toolCallID: outCall.id, toolName: outCall.name)
                    return .next(ModelRequestNode<Deps, Output>())
                }
            }
        } else if allCalls.isEmpty {
            do {
                let decoded = try deps.decodeOutput(fromText: response.text)
                let validated = try await deps.validate(decoded, context: context(nil))
                return .end(AgentRunResult(output: validated, messages: state.messages, usage: state.usage))
            } catch let retryErr as ModelRetry {
                try beginRetry(retryErr, state: state, retries: deps.retries, toolCallID: nil, toolName: nil)
                return .next(ModelRequestNode<Deps, Output>())
            } catch {
                try beginRetry(
                    ModelRetry("Could not parse the output: \(error). Please try again."),
                    state: state, retries: deps.retries, toolCallID: nil, toolName: nil)
                return .next(ModelRequestNode<Deps, Output>())
            }
        }

        // ---- Execute (non-output) tool calls ----
        let toolCalls = allCalls.filter { $0.name != outputSpec.name }
        if toolCalls.isEmpty {
            try beginRetry(
                ModelRetry("Please call the `\(outputSpec.name)` tool to produce the final result."),
                state: state, retries: deps.retries, toolCallID: nil, toolName: nil)
            return .next(ModelRequestNode<Deps, Output>())
        }

        let parts = try await deps.executeTools(toolCalls, context: context)
        state.usage.toolCalls += toolCalls.count
        for part in parts {
            if case let .toolReturn(r) = part { await emit(.toolReturn(r)) }
        }
        state.messages.append(.request(ModelRequest(parts: parts)))

        // A tool that threw ModelRetry consumes a retry from the budget.
        if parts.contains(where: { if case .retryPrompt = $0 { return true } else { return false } }) {
            state.retry += 1
            if state.retry > deps.retries {
                throw AgentError.retriesExhausted("tool retry budget exhausted")
            }
        }
        return .next(ModelRequestNode<Deps, Output>())
    }
}

// MARK: - RunContext construction

/// A closure that builds a `RunContext` from the current run state, optionally
/// tagged with the tool-call id being serviced.
private func makeRunContext<Deps: Sendable, Output: AgentSchema & Sendable>(
    state: AgentRunState,
    deps: AgentGraphDeps<Deps, Output>
) -> (String?) -> RunContext<Deps> {
    { toolCallID in
        RunContext(
            deps: deps.userDeps, usage: state.usage, retry: state.retry, model: deps.model,
            messages: state.messages, runStep: state.runStep, toolCallID: toolCallID)
    }
}
