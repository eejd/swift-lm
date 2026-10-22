import XCTest
import Foundation
import MLXLMCommon
@testable import SwiftLM

/// The tools a generation is given decide what the tool-call parser may return:
/// a call is accepted only for a declared name, and its arguments are typed by the
/// declared schema. `nil` declares nothing to check against, so it accepts any name and
/// reads every value as text (eejd/swift-lm#7, #8, #9).
final class DeclaredToolsTests: XCTestCase {

    private func decodeTools(_ json: String) throws -> [ChatCompletionRequest.ToolDef]? {
        let request = """
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]\(json)}
            """
        return try JSONDecoder().decode(
            ChatCompletionRequest.self, from: Data(request.utf8)
        ).tools
    }

    private static let echoTool = """
        , "tools": [{"type": "function", "function": {
            "name": "echo_args", "description": "Echo.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "default": null},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}},
                "nested": {"type": "object", "properties": {"depth": {"type": "integer"}}}
            }, "required": ["text", "count"]}}}]
        """

    /// Runs `text` through the processor a generation builds for `format` and `tools`.
    private func process(
        _ text: String, format: ToolCallFormat, tools: [[String: any Sendable]]
    ) -> (response: String, calls: [ToolCall], rejected: [RejectedToolCall]) {
        let processor = ToolCallProcessor(format: format, tools: tools)
        var response = ""
        var rest = Substring(text)
        while !rest.isEmpty {
            response += processor.processChunk(String(rest.prefix(7))) ?? ""
            rest = rest.dropFirst(7)
        }
        response += processor.processEOS(returnBufferedText: true) ?? ""
        return (response, processor.toolCalls, processor.rejectedToolCalls)
    }

    // MARK: - The declared list

    func testRequestWithoutToolsDeclaresNone() throws {
        XCTAssertNil(requestToolSpecs(of: try decodeTools("")))
        let none = declaredToolSpecs(of: try decodeTools(""))
        XCTAssertTrue(none.isEmpty)
        XCTAssertTrue(declaredToolSpecs(of: try decodeTools(#", "tools": []"#)).isEmpty)
    }

    func testRequestToolsAreDeclaredWithTheirSchema() throws {
        let specs = declaredToolSpecs(of: try decodeTools(Self.echoTool))
        XCTAssertEqual(specs.count, 1)
        let function = try XCTUnwrap(specs[0]["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "echo_args")
        let parameters = try XCTUnwrap(function["parameters"] as? [String: any Sendable])
        let properties = try XCTUnwrap(parameters["properties"] as? [String: any Sendable])
        let count = try XCTUnwrap(properties["count"] as? [String: any Sendable])
        XCTAssertEqual(count["type"] as? String, "integer")
    }

    // MARK: - Arguments are typed by the request's schema (eejd/swift-lm#7)

    func testQwenXMLArgumentsAreTypedByTheRequestSchema() throws {
        let specs = declaredToolSpecs(of: try decodeTools(Self.echoTool))
        let call = """
            <tool_call>
            <function=echo_args>
            <parameter=text>
            17
            </parameter>
            <parameter=count>
            17
            </parameter>
            <parameter=ratio>
            0.5
            </parameter>
            <parameter=flag>
            True
            </parameter>
            <parameter=items>
            ["age", "computer"]
            </parameter>
            <parameter=nested>
            {"depth": "2"}
            </parameter>
            </function>
            </tool_call>
            """
        let result = process(call, format: .qwen35, tools: specs)
        XCTAssertTrue(result.rejected.isEmpty)
        XCTAssertEqual(
            result.calls.first?.function.arguments,
            [
                "text": .string("17"), "count": .int(17), "ratio": .double(0.5),
                "flag": .bool(true), "items": .array([.string("age"), .string("computer")]),
                "nested": .object(["depth": .int(2)]),
            ])
        // And as the response carries them.
        let json = serializeToolCallArgs(result.calls[0].function.arguments)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object["count"] as? Int, 17)
        XCTAssertEqual(object["flag"] as? Bool, true)
        XCTAssertEqual(object["text"] as? String, "17")
    }

    func testMiniCPM5ArgumentsAreTypedByTheRequestSchema() throws {
        let specs = declaredToolSpecs(of: try decodeTools(Self.echoTool))
        let result = process(
            #"<function name="echo_args"><param name="text">hand summer</param><param name="count">17</param><param name="flag">True</param></function>"#,
            format: .minicpm5, tools: specs)
        XCTAssertEqual(
            result.calls.first?.function.arguments,
            ["text": .string("hand summer"), "count": .int(17), "flag": .bool(true)])
    }

    // MARK: - Nothing is a tool call without a declared tool (eejd/swift-lm#9)

    func testJSONAnswerWithANameKeyIsContentWhenTheRequestHasNoTools() throws {
        let none = declaredToolSpecs(of: try decodeTools(""))
        let answer = #"{"name": "Jane Doe", "age": 31, "email": "jane@example.com"}"#
        for format in [ToolCallFormat.llama3, .json, .minicpm5, .qwen35, .mistral] {
            let result = process(answer, format: format, tools: none)
            XCTAssertEqual(result.response, answer, "\(format)")
            XCTAssertTrue(result.calls.isEmpty, "\(format)")
            XCTAssertTrue(result.rejected.isEmpty, "\(format)")
        }
    }

    func testDeclaredToolIsStillCalled() throws {
        let specs = declaredToolSpecs(of: try decodeTools(Self.echoTool))
        let result = process(
            #"{"name": "echo_args", "parameters": {"text": "a", "count": "3"}}"#,
            format: .llama3, tools: specs)
        XCTAssertEqual(result.response, "")
        XCTAssertEqual(result.calls.first?.function.name, "echo_args")
    }
}
