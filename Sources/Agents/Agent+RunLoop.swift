import AgentCore
import AgentGraph
import FoundationModels

extension Agent {
    /// The agent's graph type: `UserPromptNode → ModelRequestNode → CallToolsNode`.
    typealias RunGraph = Graph<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>>

    /// Resolve the desired output spec from the `Output` type, the agent's
    /// override, and the model's profile.
    func resolveOutputSpec() throws -> OutputSpec {
        if Output.self == String.self {
            return .text
        }
        let schema = try Output.schemaJSON()
        var mode = outputModeOverride ?? model.profile.defaultOutputMode
        // A structured output can't use .text; native requires support.
        if mode == .text { mode = .tool }
        if mode == .native && !model.profile.supportsNativeStructuredOutput { mode = .tool }
        let transformed = model.profile.jsonSchemaTransform(schema)
        return OutputSpec(
            mode: mode, name: "final_result",
            description: "The final result of the run.", schema: transformed)
    }

    /// The static graph definition (node set) — used for diagrams and to drive runs.
    var runGraph: RunGraph {
        RunGraph(name: "Agent", nodes: [
            UserPromptNode<Deps, Output>.self,
            ModelRequestNode<Deps, Output>.self,
            CallToolsNode<Deps, Output>.self,
        ])
    }

    /// A mermaid `flowchart` of the agent's run graph.
    public func graphDiagram() -> String { runGraph.mermaidCode() }

    /// Build a fresh graph run for the given inputs (shared by `run`/`runStream`/`iter`).
    func makeGraphRun(
        prompt: String,
        attachments: [MediaContent],
        deps: Deps,
        usageLimits: UsageLimits,
        messageHistory: [ModelMessage]
    ) throws -> GraphRun<AgentRunState, AgentGraphDeps<Deps, Output>, AgentRunResult<Output>> {
        let outputSpec = try resolveOutputSpec()
        var toolDefs = tools.map(\.definition)
        if outputSpec.mode == .tool, let schema = outputSpec.schema {
            toolDefs.append(ToolDefinition(
                name: outputSpec.name, description: outputSpec.description, parameters: schema))
        }
        let state = AgentRunState(messages: messageHistory, outputSpec: outputSpec, toolDefs: toolDefs)
        let graphDeps = AgentGraphDeps(
            model: model, settings: settings, tools: tools, outputValidators: outputValidators,
            instructionSources: instructionSources, endStrategy: endStrategy, retries: retries,
            usageLimits: usageLimits, userDeps: deps)
        return runGraph.iter(
            start: UserPromptNode<Deps, Output>(prompt: prompt, attachments: attachments),
            state: state, deps: graphDeps)
    }

    /// Drive a run to completion (no streaming) and return its result.
    func runToCompletion(
        prompt: String,
        attachments: [MediaContent],
        deps: Deps,
        usageLimits: UsageLimits,
        messageHistory: [ModelMessage]
    ) async throws -> AgentRunResult<Output> {
        let run = try makeGraphRun(
            prompt: prompt, attachments: attachments, deps: deps,
            usageLimits: usageLimits, messageHistory: messageHistory)
        while try await run.next() != nil {}
        guard let result = run.result else {
            throw AgentError.noOutput("run finished without producing an output")
        }
        return result
    }

    /// Drive a run, streaming each node's events, ending with `.final`.
    func runStreamImpl(
        prompt: String,
        attachments: [MediaContent],
        deps: Deps,
        usageLimits: UsageLimits,
        messageHistory: [ModelMessage]
    ) -> AsyncThrowingStream<AgentStreamEvent<Output>, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let run = try makeGraphRun(
                        prompt: prompt, attachments: attachments, deps: deps,
                        usageLimits: usageLimits, messageHistory: messageHistory)
                    // Each node is yielded before it runs; stream it, then the next
                    // `next()` executes it (consuming the streamed result).
                    while let node = try await run.next() {
                        if let modelRequest = node as? ModelRequestNode<Deps, Output> {
                            for try await event in modelRequest.stream(run.ctx) {
                                continuation.yield(event)
                            }
                        } else if let callTools = node as? CallToolsNode<Deps, Output> {
                            for try await event in callTools.stream(run.ctx) {
                                continuation.yield(event)
                            }
                        }
                    }
                    guard let result = run.result else {
                        throw AgentError.noOutput("run finished without producing an output")
                    }
                    continuation.yield(.final(result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
