import Testing
import AgentCore

@Suite("Partial JSON completer")
struct PartialJSONTests {

    /// Completing then parsing should round-trip to the given expected value.
    private func expectValue(_ partial: String, _ expected: JSONValue) throws {
        let completed = try #require(PartialJSON.complete(partial))
        #expect(try JSONValue(jsonString: completed) == expected)
    }

    @Test("complete JSON is returned unchanged in meaning")
    func alreadyComplete() throws {
        try expectValue(#"{"a":1,"b":"x"}"#, .object(["a": .int(1), "b": .string("x")]))
    }

    @Test("closes an open object")
    func openObject() throws {
        try expectValue(#"{"city":"London""#, .object(["city": .string("London")]))
    }

    @Test("drops a dangling comma")
    func danglingComma() throws {
        try expectValue(#"{"a":1,"#, .object(["a": .int(1)]))
    }

    @Test("drops a dangling key with colon")
    func danglingKey() throws {
        try expectValue(#"{"a":1,"b":"#, .object(["a": .int(1)]))
    }

    @Test("closes an open string mid-value")
    func openString() throws {
        try expectValue(#"{"name":"Ada Lovel"#, .object(["name": .string("Ada Lovel")]))
    }

    @Test("closes nested object and array")
    func nested() throws {
        try expectValue(
            #"{"items":[{"id":1},{"id":2"#,
            .object(["items": .array([.object(["id": .int(1)]), .object(["id": .int(2)])])]))
    }

    @Test("trims a partial keyword token")
    func partialKeyword() throws {
        // `tru` is incomplete; the key is dropped, leaving an empty object.
        try expectValue(#"{"ok":tru"#, .object([:]))
    }

    @Test("trims a partial number with trailing dot to its valid prefix")
    func partialNumber() throws {
        // `1.` is incomplete; the shrink pass falls back to the valid integer `1`.
        try expectValue(#"{"x":1."#, .object(["x": .int(1)]))
    }

    @Test("completes an open array")
    func openArray() throws {
        try expectValue(#"["a","b""#, .array([.string("a"), .string("b")]))
    }

    @Test("un-parseable prefix returns nil")
    func unparseable() {
        #expect(PartialJSON.complete("") == nil)
        #expect(PartialJSON.complete("   ") == nil)
    }
}
