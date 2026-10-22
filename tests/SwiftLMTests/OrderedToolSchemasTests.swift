import XCTest
import Foundation
import Jinja
@testable import SwiftLM

/// The tool schemas of a request reach the chat template in the order the client wrote them
/// (eejd/swift-lm#5, #15): the template renders them with `tojson`, and swift-jinja sorts the
/// keys of every Swift dictionary it converts.
final class OrderedToolSchemasTests: XCTestCase {

    /// A schema written the way quantbench / OpenAI clients write it, and its text as Python's
    /// `json.dumps(tool, ensure_ascii=False)` (what mlx-lm's template rendering gives).
    private static let weather =
        #"{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}"#

    private static let search =
        #"{"type": "function", "function": {"name": "search_docs", "description": "Search.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}}"#

    private func body(tools: String, prefix: String = #""model": "m", "messages": [{"role": "user", "content": "hi"}],"#) -> Data {
        Data("{\(prefix) \"tools\": \(tools)}".utf8)
    }

    private func dumps(_ value: Value) throws -> String {
        try JSON.dumps(value, options: .init(ensureASCII: false))
    }

    private func items(_ value: Value?) throws -> [Value] {
        guard case .array(let items)? = value else {
            XCTFail("not an array: \(String(describing: value))")
            return []
        }
        return items
    }

    // MARK: - Order

    func testToolSchemasKeepTheRequestKeyOrder() throws {
        let tools = OrderedToolSchemas.tools(
            inRequestBody: body(tools: "[\(Self.weather), \(Self.search)]"))
        let rendered = try items(tools).map(dumps)
        XCTAssertEqual(rendered, [Self.weather, Self.search])
    }

    func testTemplateRendersTheRequestOrder() throws {
        // The shape of MiniCPM5's chat template.
        let template = try Template(
            #"{% for tool in tools %}{{ "\n" }}{{ tool | tojson(ensure_ascii=False) }}{% endfor %}"#)
        let tools = try XCTUnwrap(
            OrderedToolSchemas.tools(inRequestBody: body(tools: "[\(Self.weather), \(Self.search)]")))
        let rendered = try template.render(["tools": tools])
        XCTAssertEqual(rendered, "\n\(Self.weather)\n\(Self.search)")
    }

    func testTemplateRendersSortedKeysWithoutIt() throws {
        // What the decoded `[String: Any]` path did, and what OrderedToolSchemas replaces.
        let template = try Template(#"{{ tool | tojson(ensure_ascii=False) }}"#)
        let decoded = try JSONSerialization.jsonObject(with: Data(Self.search.utf8))
        let rendered = try template.render(["tool": try Value(any: decoded)])
        XCTAssertNotEqual(rendered, Self.search)
        XCTAssertTrue(rendered.hasPrefix(#"{"function": {"description""#), rendered)
    }

    func testRepeatedKeyKeepsFirstValueLikeTheDecoder() throws {
        let value = try OrderedToolSchemas.parse(Data(#"{"b": 1, "a": 2, "b": 3}"#.utf8))
        XCTAssertEqual(try dumps(value), #"{"b": 1, "a": 2}"#)
        // The decoder that builds the parser's argument schemas reads the same value.
        let decoded = try JSONDecoder().decode([String: Int].self, from: Data(#"{"b": 1, "a": 2, "b": 3}"#.utf8))
        XCTAssertEqual(decoded["b"], 1)
    }

    func testDuplicateToolsMemberAndDuplicateSchemaKeyAgreeWithTheDecoder() throws {
        let data = Data(#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "first", "parameters": {"type": "object", "type": "array", "properties": {}}}}], "tools": [{"type": "function", "function": {"name": "second"}}]}"#.utf8)
        let decoded = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        let ordered = try items(OrderedToolSchemas.tools(inRequestBody: data))
        XCTAssertEqual(decoded.tools?.map(\.function.name), ["first"])
        XCTAssertEqual(ordered.count, 1)
        let text = try dumps(ordered[0])
        XCTAssertTrue(text.contains(#""name": "first""#), text)
        XCTAssertTrue(text.contains(#""type": "object""#) && !text.contains(#""type": "array""#), text)
        XCTAssertEqual(decoded.tools?.first?.function.parameters?["type"]?.value as? String, "object")
    }

    // MARK: - The request body

    func testOnlyTheTopLevelToolsMemberIsRead() throws {
        // `tools` inside a message, braces and escaped quotes inside strings, nested arrays.
        let prefix = #""messages": [{"role": "user", "content": "{\"tools\": [1]} } ] \" "}, {"role": "tool", "tools": [2]}], "metadata": {"tools": [3], "n": [1, [2, {"x": "}"}]], "v": -1.5e2, "t": true, "z": null},"#
        let tools = OrderedToolSchemas.tools(
            inRequestBody: body(tools: "[\(Self.search)]", prefix: prefix))
        XCTAssertEqual(try items(tools).map(dumps), [Self.search])
    }

    func testNoToolsOrNotAnArrayIsNil() {
        XCTAssertNil(OrderedToolSchemas.tools(inRequestBody: Data(#"{"model": "m"}"#.utf8)))
        XCTAssertNil(OrderedToolSchemas.tools(inRequestBody: Data(#"{"tools": null}"#.utf8)))
        XCTAssertNil(OrderedToolSchemas.tools(inRequestBody: Data(#"{"tools": {"a": 1}}"#.utf8)))
        XCTAssertNil(OrderedToolSchemas.tools(inRequestBody: Data("{}".utf8)))
    }

    func testMalformedBodyIsNilNotACrash() throws {
        for text in [#"{"tools": [{"a": }]}"#, #"{"tools": [1, 2"#, #"{"tools": "#, "", "[]",
            #"{"tools": [{"a": "\u12"}]}"#, #"{"tools": ["\ud83d"]}"#]
        {
            XCTAssertNil(OrderedToolSchemas.tools(inRequestBody: Data(text.utf8)), text)
        }
        let deep = String(repeating: "[", count: 400) + String(repeating: "]", count: 400)
        XCTAssertNil(OrderedToolSchemas.tools(inRequestBody: Data("{\"tools\": \(deep)}".utf8)))
        XCTAssertEqual(try items(OrderedToolSchemas.tools(inRequestBody: Data("{\"x\": \(deep), \"tools\": []}".utf8))).count, 0)
    }

    func testRequestToolsAgreeWithTheDecodedTools() throws {
        let data = body(tools: "[\(Self.weather), \(Self.search)]")
        let decoded = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        let ordered = try items(OrderedToolSchemas.tools(inRequestBody: data))
        XCTAssertEqual(ordered.count, decoded.tools?.count)
        for (value, tool) in zip(ordered, decoded.tools ?? []) {
            guard case .object(let object) = value, case .object(let function)? = object[.string("function")],
                case .string(let name)? = function[.string("name")]
            else { return XCTFail("no function name in \(value)") }
            XCTAssertEqual(name, tool.function.name)
        }
    }

    // MARK: - Values

    func testScalarsAndStrings() throws {
        let value = try OrderedToolSchemas.parse(
            Data(#"{"i": -7, "d": 1.5, "e": 1e3, "t": true, "f": false, "n": null, "s": "a\"b\\c\/d\n\t\u00e9\ud83d\ude00 \u2028", "u": "é😀"}"#.utf8))
        guard case .object(let object) = value else { return XCTFail() }
        XCTAssertEqual(object[.string("i")], .int(-7))
        XCTAssertEqual(object[.string("d")], .double(1.5))
        XCTAssertEqual(object[.string("e")], .double(1000))
        XCTAssertEqual(object[.string("t")], .boolean(true))
        XCTAssertEqual(object[.string("f")], .boolean(false))
        XCTAssertEqual(object[.string("n")], .null)
        XCTAssertEqual(object[.string("s")], .string("a\"b\\c/d\n\té😀 \u{2028}"))
        XCTAssertEqual(object[.string("u")], .string("é😀"))
        XCTAssertEqual(
            Array(object.keys), ["i", "d", "e", "t", "f", "n", "s", "u"].map { ObjectKey.string($0) })
    }

    func testBodiesTheDecoderAcceptsInOtherEncodingsAreRead() throws {
        let json = "{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"h\u{e9}\"}], \"tools\": [\(Self.search)]}"
        let utf8 = Data(json.utf8)
        let bom = Data([0xEF, 0xBB, 0xBF]) + utf8
        let utf16 = json.data(using: .utf16)!                       // BOM, native order
        let utf16BE = json.data(using: .utf16BigEndian)!            // no BOM
        let utf16LE = json.data(using: .utf16LittleEndian)!         // no BOM
        let utf32BE = json.data(using: .utf32BigEndian)!
        let utf32LE = json.data(using: .utf32LittleEndian)!         // no BOM
        for (name, body) in [("utf8", utf8), ("utf8 BOM", bom), ("utf16 BOM", utf16), ("utf16 BE", utf16BE),
            ("utf16 LE", utf16LE), ("utf32 BE", utf32BE), ("utf32 LE", utf32LE),
            ("utf16 BE BOM", Data([0xFE, 0xFF]) + utf16BE), ("utf16 LE BOM", Data([0xFF, 0xFE]) + utf16LE)]
        {
            // The decoder accepts it (it does not accept a UTF-32 BOM, which is why none is tested) ...
            XCTAssertNoThrow(try JSONDecoder().decode(ChatCompletionRequest.self, from: body), name)
            // ... so the reader must too, or the model would get the sorted schema.
            guard case .ordered(let tools) = OrderedToolSchemas.read(inRequestBody: body) else {
                XCTFail("\(name): not read: \(OrderedToolSchemas.read(inRequestBody: body))")
                continue
            }
            XCTAssertEqual(try items(tools).map(dumps), [Self.search], name)
        }
    }

    func testDeeplyNestedOtherMembersDoNotHideTheTools() throws {
        // The decoder allows 512 levels; a member the reader only skips has no limit of its own.
        let deep = String(repeating: "[", count: 500) + String(repeating: "]", count: 500)
        let data = Data("{\"x\": \(deep), \"tools\": [\(Self.search)]}".utf8)
        XCTAssertEqual(try items(OrderedToolSchemas.tools(inRequestBody: data)).map(dumps), [Self.search])
    }

    func testUnreadableToolsAreReportedNotHidden() {
        // A tool schema nested deeper than the value reader's bound: reported, not silently sorted.
        let deep = String(repeating: "[", count: 200) + String(repeating: "]", count: 200)
        let data = Data("{\"tools\": [{\"x\": \(deep)}]}".utf8)
        guard case .unreadable = OrderedToolSchemas.read(inRequestBody: data) else {
            return XCTFail("expected .unreadable, got \(OrderedToolSchemas.read(inRequestBody: data))")
        }
        guard case .absent = OrderedToolSchemas.read(inRequestBody: Data(#"{"model": "m"}"#.utf8)) else {
            return XCTFail("expected .absent")
        }
        guard case .unreadable = OrderedToolSchemas.read(inRequestBody: Data(#"{"tools": [1,"#.utf8)) else {
            return XCTFail("expected .unreadable for a truncated body")
        }
    }

    func testNullsOfTheSchemaAreKept() throws {
        // swift-jinja has `.null`; the reference engines render `"default": null` too.
        let tools = try items(
            OrderedToolSchemas.tools(
                inRequestBody: body(tools: #"[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "string", "default": null}}}}}]"#)))
        XCTAssertTrue(try dumps(tools[0]).contains(#""default": null"#))
    }
}
