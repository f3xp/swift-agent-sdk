import AgentCore
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// A single Server-Sent Event.
public struct SSEEvent: Sendable, Equatable {
    public var event: String?
    public var data: String
    public init(event: String? = nil, data: String) {
        self.event = event
        self.data = data
    }
}

/// Thin async wrapper over `URLSession` shared by HTTP providers: a JSON POST
/// and an SSE stream. Retries/backoff for transient failures live here too.
public struct HTTPClient: Sendable {
    public var session: URLSession
    public var maxRetries: Int

    public init(session: URLSession = .shared, maxRetries: Int = 2) {
        self.session = session
        self.maxRetries = maxRetries
    }

    public struct HTTPError: Error, Sendable, CustomStringConvertible {
        public let statusCode: Int
        public let body: String
        public var description: String { "HTTP \(statusCode): \(body)" }
    }

    private func makeRequest(url: URL, headers: [String: String], body: Data) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = body
        for (k, v) in headers { request.setValue(v, forHTTPHeaderField: k) }
        return request
    }

    /// POST JSON and return the decoded response body as a `JSONValue`.
    public func postJSON(url: URL, headers: [String: String], body: JSONValue) async throws -> JSONValue {
        let data = Data(try body.jsonString().utf8)
        let request = makeRequest(url: url, headers: headers, body: data)

        var attempt = 0
        while true {
            do {
                let (respData, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw AgentError.provider("non-HTTP response")
                }
                guard (200..<300).contains(http.statusCode) else {
                    let bodyText = String(decoding: respData, as: UTF8.self)
                    // Retry on 429 / 5xx.
                    if (http.statusCode == 429 || http.statusCode >= 500), attempt < maxRetries {
                        attempt += 1
                        try await Task.sleep(for: .milliseconds(200 * (1 << (attempt - 1))))
                        continue
                    }
                    throw HTTPError(statusCode: http.statusCode, body: bodyText)
                }
                return try JSONDecoder().decode(JSONValue.self, from: respData)
            } catch let error as HTTPError {
                throw error
            } catch {
                if attempt < maxRetries {
                    attempt += 1
                    try await Task.sleep(for: .milliseconds(200 * (1 << (attempt - 1))))
                    continue
                }
                throw error
            }
        }
    }

    /// POST and stream Server-Sent Events from the response body.
    public func streamSSE(url: URL, headers: [String: String], body: JSONValue) -> AsyncThrowingStream<SSEEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let data = Data(try body.jsonString().utf8)
                    let request = makeRequest(url: url, headers: headers, body: data)
                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                        let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                        throw HTTPError(statusCode: code, body: "stream request failed")
                    }
                    var eventName: String?
                    var dataLines: [String] = []
                    for try await line in bytes.lines {
                        if line.isEmpty {
                            if !dataLines.isEmpty {
                                continuation.yield(SSEEvent(event: eventName, data: dataLines.joined(separator: "\n")))
                            }
                            eventName = nil
                            dataLines = []
                        } else if line.hasPrefix("event:") {
                            eventName = String(line.dropFirst("event:".count)).trimmingCharacters(in: .whitespaces)
                        } else if line.hasPrefix("data:") {
                            dataLines.append(String(line.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces))
                        }
                        // lines starting with ":" are comments; ignore.
                    }
                    if !dataLines.isEmpty {
                        continuation.yield(SSEEvent(event: eventName, data: dataLines.joined(separator: "\n")))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
