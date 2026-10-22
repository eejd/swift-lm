import Foundation
import Jinja

/// The `tools` of a chat request as Jinja values that keep the request's key order.
///
/// A chat template renders each tool schema with `tool | tojson`. Python engines (mlx-lm,
/// vLLM, ...) hand the template the request's dicts, so the schema reaches the model as the
/// client wrote it. SwiftLM decoded `tools[].function.parameters` into `[String: AnyCodable]`
/// and swift-jinja's `Value(any:)` sorts the keys of every `[String: Any]` it converts, so the
/// model saw an alphabetized schema (`{"function": {"description": ..., "name": ...}, "type":
/// "function"}`, `"enum"` before `"type"`) and a different prompt than the reference engines
/// (eejd/swift-lm#5). Foundation's decoders do not expose the order of an object's keys either,
/// so the order is read from the request bytes here.
///
/// The result is handed to the template as `additionalContext["tools"]`: swift-transformers
/// converts every additional-context value with `Value(any:)`, which passes a `Value` through
/// unchanged, and applies the additional context after `tools`, so it replaces the sorted copy.
enum OrderedToolSchemas {

    /// The `tools` array of a chat completion request body, or `nil` when the body has none
    /// (or is not JSON this reader understands: the caller then keeps the decoded, sorted tools).
    static func tools(inRequestBody body: Data) -> Value? {
        var reader = Reader(bytes: [UInt8](body))
        guard let value = try? reader.member("tools"), case .array = value else { return nil }
        return value
    }

    /// Reads one JSON document into `Value`, objects as insertion-ordered dictionaries.
    static func parse(_ body: Data) throws -> Value {
        var reader = Reader(bytes: [UInt8](body))
        let value = try reader.value(depth: 0)
        reader.skipWhitespace()
        guard reader.index == reader.bytes.count else { throw ParseError(offset: reader.index) }
        return value
    }

    struct ParseError: Error, Equatable {
        let offset: Int
    }

    /// A request nests a tool schema a few levels; this bounds recursion on hostile input.
    private static let maximumDepth = 128

    private struct Reader {
        let bytes: [UInt8]
        var index = 0

        init(bytes: [UInt8]) { self.bytes = bytes }

        // MARK: Top level

        /// The value of top-level member `name` of the document's object; the other members
        /// are skipped without building values.
        mutating func member(_ name: String) throws -> Value? {
            skipWhitespace()
            try expect(UInt8(ascii: "{"))
            skipWhitespace()
            if peek() == UInt8(ascii: "}") { return nil }
            var found: Value?
            while true {
                skipWhitespace()
                let key = try string()
                skipWhitespace()
                try expect(UInt8(ascii: ":"))
                skipWhitespace()
                if key == name, found == nil {
                    found = try value(depth: 1)
                } else {
                    try skipValue(depth: 1)
                }
                skipWhitespace()
                switch next() {
                case UInt8(ascii: ","): continue
                case UInt8(ascii: "}"): return found
                default: throw ParseError(offset: index)
                }
            }
        }

        // MARK: Values

        mutating func value(depth: Int) throws -> Value {
            guard depth <= OrderedToolSchemas.maximumDepth else { throw ParseError(offset: index) }
            skipWhitespace()
            guard let byte = peek() else { throw ParseError(offset: index) }
            switch byte {
            case UInt8(ascii: "{"):
                index += 1
                var object = OrderedDictionary<ObjectKey, Value>()
                skipWhitespace()
                if peek() == UInt8(ascii: "}") {
                    index += 1
                    return .object(object)
                }
                while true {
                    skipWhitespace()
                    let key = try string()
                    skipWhitespace()
                    try expect(UInt8(ascii: ":"))
                    // A repeated key keeps its first position and takes the last value,
                    // as a Python dict does.
                    object[.string(key)] = try value(depth: depth + 1)
                    skipWhitespace()
                    switch next() {
                    case UInt8(ascii: ","): continue
                    case UInt8(ascii: "}"): return .object(object)
                    default: throw ParseError(offset: index)
                    }
                }
            case UInt8(ascii: "["):
                index += 1
                var items: [Value] = []
                skipWhitespace()
                if peek() == UInt8(ascii: "]") {
                    index += 1
                    return .array(items)
                }
                while true {
                    items.append(try value(depth: depth + 1))
                    skipWhitespace()
                    switch next() {
                    case UInt8(ascii: ","): continue
                    case UInt8(ascii: "]"): return .array(items)
                    default: throw ParseError(offset: index)
                    }
                }
            case UInt8(ascii: "\""):
                return .string(try string())
            case UInt8(ascii: "t"):
                try literal("true")
                return .boolean(true)
            case UInt8(ascii: "f"):
                try literal("false")
                return .boolean(false)
            case UInt8(ascii: "n"):
                try literal("null")
                return .null
            default:
                return try number()
            }
        }

        /// Steps over one value without building it. The document was already accepted by
        /// `JSONDecoder`, so this only has to find where the value ends.
        mutating func skipValue(depth: Int) throws {
            guard depth <= OrderedToolSchemas.maximumDepth else { throw ParseError(offset: index) }
            skipWhitespace()
            guard let byte = peek() else { throw ParseError(offset: index) }
            switch byte {
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                var nesting = 0
                while let current = peek() {
                    switch current {
                    case UInt8(ascii: "\""):
                        try skipString()
                        continue
                    case UInt8(ascii: "{"), UInt8(ascii: "["):
                        nesting += 1
                    case UInt8(ascii: "}"), UInt8(ascii: "]"):
                        nesting -= 1
                    default:
                        break
                    }
                    index += 1
                    if nesting == 0 { return }
                    if nesting > OrderedToolSchemas.maximumDepth { throw ParseError(offset: index) }
                }
                throw ParseError(offset: index)
            case UInt8(ascii: "\""):
                try skipString()
            default:
                while let current = peek(),
                    ![UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]"), 0x20, 0x09, 0x0A, 0x0D]
                        .contains(current)
                {
                    index += 1
                }
            }
        }

        // MARK: Strings

        mutating func skipString() throws {
            try expect(UInt8(ascii: "\""))
            while let byte = next() {
                if byte == UInt8(ascii: "\\") { index += 1 }
                else if byte == UInt8(ascii: "\"") { return }
            }
            throw ParseError(offset: index)
        }

        mutating func string() throws -> String {
            try expect(UInt8(ascii: "\""))
            var out = [UInt8]()
            while true {
                guard let byte = next() else { throw ParseError(offset: index) }
                switch byte {
                case UInt8(ascii: "\""):
                    return String(decoding: out, as: UTF8.self)
                case UInt8(ascii: "\\"):
                    guard let escape = next() else { throw ParseError(offset: index) }
                    switch escape {
                    case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(escape)
                    case UInt8(ascii: "b"): out.append(0x08)
                    case UInt8(ascii: "f"): out.append(0x0C)
                    case UInt8(ascii: "n"): out.append(0x0A)
                    case UInt8(ascii: "r"): out.append(0x0D)
                    case UInt8(ascii: "t"): out.append(0x09)
                    case UInt8(ascii: "u"):
                        var scalar = try hex4()
                        if (0xD800...0xDBFF).contains(scalar) {
                            // A high surrogate pairs with the escape that follows it.
                            guard next() == UInt8(ascii: "\\"), next() == UInt8(ascii: "u") else {
                                throw ParseError(offset: index)
                            }
                            let low = try hex4()
                            guard (0xDC00...0xDFFF).contains(low) else { throw ParseError(offset: index) }
                            scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                        }
                        guard let unicode = Unicode.Scalar(scalar) else { throw ParseError(offset: index) }
                        out.append(contentsOf: Array(String(Character(unicode)).utf8))
                    default:
                        throw ParseError(offset: index)
                    }
                default:
                    out.append(byte)
                }
            }
        }

        private mutating func hex4() throws -> UInt32 {
            var scalar: UInt32 = 0
            for _ in 0..<4 {
                guard let digit = next(), let value = Self.hexValue(digit) else {
                    throw ParseError(offset: index)
                }
                scalar = scalar << 4 | value
            }
            return scalar
        }

        private static func hexValue(_ byte: UInt8) -> UInt32? {
            switch byte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): return UInt32(byte - UInt8(ascii: "0"))
            case UInt8(ascii: "a")...UInt8(ascii: "f"): return UInt32(byte - UInt8(ascii: "a") + 10)
            case UInt8(ascii: "A")...UInt8(ascii: "F"): return UInt32(byte - UInt8(ascii: "A") + 10)
            default: return nil
            }
        }

        // MARK: Numbers and literals

        /// An integer literal is an `Int` (a Python `int`), one with a fraction or an exponent
        /// a `Double`; an integer outside `Int` becomes a `Double`.
        mutating func number() throws -> Value {
            let start = index
            while let byte = peek(),
                (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
                    || [UInt8(ascii: "-"), UInt8(ascii: "+"), UInt8(ascii: "."), UInt8(ascii: "e"),
                        UInt8(ascii: "E")].contains(byte)
            {
                index += 1
            }
            let text = String(decoding: bytes[start..<index], as: UTF8.self)
            if let int = Int(text) { return .int(int) }
            if let double = Double(text), double.isFinite { return .double(double) }
            index = start
            throw ParseError(offset: start)
        }

        private mutating func literal(_ word: String) throws {
            for expected in word.utf8 {
                guard next() == expected else { throw ParseError(offset: index) }
            }
        }

        // MARK: Characters

        func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

        mutating func next() -> UInt8? {
            guard index < bytes.count else { return nil }
            defer { index += 1 }
            return bytes[index]
        }

        mutating func expect(_ byte: UInt8) throws {
            guard next() == byte else { throw ParseError(offset: max(index - 1, 0)) }
        }

        mutating func skipWhitespace() {
            while let byte = peek(), byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D {
                index += 1
            }
        }
    }
}
