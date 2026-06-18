import Testing
import FoundationModels
import AgentCore

@Generable
struct Person: Equatable {
    @Guide(description: "Full name")
    var name: String
    @Guide(description: "Age in years")
    var age: Int
}

@Suite("Schema bridge")
struct SchemaBridgeTests {

    @Test("GenerationSchema projects to a JSON object schema")
    func schemaShape() throws {
        let schema = try Person.schemaJSON()
        // Top-level must be an object schema with the declared properties.
        #expect(schema["type"]?.stringValue == "object")
        let props = schema["properties"]?.objectValue
        #expect(props?["name"] != nil)
        #expect(props?["age"] != nil)
        // Print for visibility into Apple's emitted dialect.
        print("Person schema:", try schema.jsonString(sortedKeys: true, prettyPrinted: true))
    }

    @Test("decode a Generable from a JSONValue and back")
    func roundTrip() throws {
        let value: JSONValue = .object(["name": "Ada", "age": 36])
        let person = try Person(jsonValue: value)
        #expect(person == Person(name: "Ada", age: 36))

        let back = try person.generatedContent.toJSONValue()
        #expect(back["name"]?.stringValue == "Ada")
        #expect(back["age"] == .int(36))
    }

    @Test("JSONValue parses and re-serializes JSON stably")
    func jsonValueRoundTrip() throws {
        let json = #"{"a":1,"b":["x",true,null],"c":{"d":2.5}}"#
        let value = try JSONValue(jsonString: json)
        #expect(value["a"] == .int(1))
        #expect(value["c"]?["d"] == .double(2.5))
        let reserialized = try value.jsonString(sortedKeys: true)
        #expect(reserialized == #"{"a":1,"b":["x",true,null],"c":{"d":2.5}}"#)
    }
}
