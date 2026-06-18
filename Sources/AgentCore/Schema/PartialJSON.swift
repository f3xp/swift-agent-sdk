import Foundation

/// Best-effort repair of an incomplete JSON string into a parseable document.
///
/// Streamed structured output arrives a few characters at a time, so at any
/// instant the accumulated buffer is usually a *prefix* of valid JSON (an open
/// object, a half-written string, a dangling comma). `complete` closes the open
/// structures and trims the trailing incomplete token so the prefix can be
/// decoded into the latest available snapshot. Returns `nil` until something
/// parses (mirrors pydantic-ai's partial-validation behavior).
public enum PartialJSON {

    /// Return a parseable completion of `s`, or `nil` if no prefix parses yet.
    public static func complete(_ s: String) -> String? {
        var chars = Array(s)
        // Trim trailing whitespace before we begin.
        trimTrailingWhitespace(&chars)
        // Shrink from the end to the previous structural boundary until a
        // closed candidate parses. Bounded by the buffer length.
        while !chars.isEmpty {
            if let candidate = closeStructures(chars),
               (try? JSONValue(jsonString: candidate)) != nil {
                return candidate
            }
            chars.removeLast()
            trimTrailingWhitespace(&chars)
        }
        return nil
    }

    /// Close any open string/containers in `chars` and strip a dangling
    /// `,`/`:` so the result is a syntactically-complete candidate.
    private static func closeStructures(_ chars: [Character]) -> String? {
        var stack: [Character] = []
        var inString = false
        var escaped = false
        for c in chars {
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else {
                switch c {
                case "\"": inString = true
                case "{", "[": stack.append(c)
                case "}": if stack.last == "{" { stack.removeLast() }
                case "]": if stack.last == "[" { stack.removeLast() }
                default: break
                }
            }
        }

        var result = chars
        // A buffer ending mid-string (and not on a trailing escape) can be
        // closed with a quote; a trailing backslash is left for the shrink pass.
        if inString && !escaped { result.append("\"") }
        trimTrailingWhitespace(&result)
        // Drop dangling separators that can't be followed by a closer.
        while let last = result.last, last == "," || last == ":" {
            result.removeLast()
            trimTrailingWhitespace(&result)
        }
        if result.isEmpty { return nil }
        for opener in stack.reversed() {
            result.append(opener == "{" ? "}" : "]")
        }
        return String(result)
    }

    private static func trimTrailingWhitespace(_ chars: inout [Character]) {
        while let last = chars.last, last == " " || last == "\n" || last == "\t" || last == "\r" {
            chars.removeLast()
        }
    }
}
