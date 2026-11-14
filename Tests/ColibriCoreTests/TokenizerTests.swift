import Foundation
import Testing
@testable import ColibriCore

// Tokenizer tests: byte map, pre-tokenizers, BPE on tiny vocabularies, and the real
// Qwen3-0.6B tokenizer against ids from upstream colibri's own tokenizer
// (Fixtures/tokenizer/qwen3_encode.json, made by Fixtures/tokenizer/oracle/).

// MARK: - Helpers

enum TokTestSupport {
    static let qwen3Path = NSString(string: "~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf").expandingTildeInPath
    static var hasQwen3: Bool { FileManager.default.fileExists(atPath: qwen3Path) }

    /// The real tokenizer, loaded once.
    static let qwen3: BPETokenizer? = {
        guard hasQwen3, let md = try? readMetadata(qwen3Path) else { return nil }
        return try? BPETokenizer(metadata: md)
    }()

    static func fixtureURL(_ name: String) -> URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures/tokenizer/\(name)")
    }

    /// A metadata-only GGUF reader (header and key/values, no tensors), so these tests do
    /// not depend on the GGUF workstream's reader.
    static func readMetadata(_ path: String) throws -> GGUFMetadata {
        let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
        var p = 0
        func take<T: FixedWidthInteger>(_: T.Type) throws -> T {
            guard p + MemoryLayout<T>.size <= data.count else { throw ColibriError.invalidGGUF("truncated") }
            var v: T = 0
            withUnsafeMutableBytes(of: &v) { dst in
                data.copyBytes(to: dst.bindMemory(to: UInt8.self), from: p..<(p + MemoryLayout<T>.size))
            }
            p += MemoryLayout<T>.size
            return T(littleEndian: v)
        }
        func string() throws -> String {
            let n = Int(try take(UInt64.self))
            guard p + n <= data.count else { throw ColibriError.invalidGGUF("truncated string") }
            let s = String(decoding: data[p..<(p + n)], as: UTF8.self)
            p += n
            return s
        }
        func value(_ t: GGUFValueType) throws -> GGUFValue {
            switch t {
            case .u8: return .u8(try take(UInt8.self))
            case .i8: return .i8(try take(Int8.self))
            case .u16: return .u16(try take(UInt16.self))
            case .i16: return .i16(try take(Int16.self))
            case .u32: return .u32(try take(UInt32.self))
            case .i32: return .i32(try take(Int32.self))
            case .u64: return .u64(try take(UInt64.self))
            case .i64: return .i64(try take(Int64.self))
            case .f32: return .f32(Float(bitPattern: try take(UInt32.self)))
            case .f64: return .f64(Double(bitPattern: try take(UInt64.self)))
            case .bool: return .bool(try take(UInt8.self) != 0)
            case .string: return .string(try string())
            case .array:
                guard let et = GGUFValueType(rawValue: try take(UInt32.self)), et != .array else {
                    throw ColibriError.invalidGGUF("bad array type")
                }
                let n = Int(try take(UInt64.self))
                var a: [GGUFValue] = []
                a.reserveCapacity(n)
                for _ in 0..<n { a.append(try value(et)) }
                return .array(et, a)
            }
        }
        guard try take(UInt32.self) == 0x4655_4747 else { throw ColibriError.invalidGGUF("bad magic") }
        _ = try take(UInt32.self)  // version
        _ = try take(UInt64.self)  // tensors
        let nkv = Int(try take(UInt64.self))
        var md = GGUFMetadata()
        for _ in 0..<nkv {
            let key = try string()
            guard let t = GGUFValueType(rawValue: try take(UInt32.self)) else { throw ColibriError.invalidGGUF("bad type") }
            md.set(key, try value(t))
        }
        return md
    }

    struct EncodeFixture: Decodable {
        struct Case: Decodable {
            let name: String
            let text: String
            let ids: [Int32]
        }
        let model: String
        let oracle: String
        let cases: [Case]
    }

    static func encodeFixture(_ name: String = "qwen3_encode.json") throws -> EncodeFixture {
        try JSONDecoder().decode(EncodeFixture.self, from: Data(contentsOf: fixtureURL(name)))
    }

    /// A tiny byte-level vocabulary: all 256 byte pieces, then the merges' results, then
    /// the added tokens.
    static func tinyTokenizer(merges: [(String, String)], added: [(String, TokenType)] = [],
                              pre: PreTokenizer = Qwen2PreTokenizer()) throws -> BPETokenizer {
        var tokens = (0..<256).map { String(ByteLevel.scalar(for: UInt8($0))) }
        var types = [TokenType](repeating: .normal, count: 256)
        var lines: [String] = []
        for (a, b) in merges {
            let ea = ByteLevel.encode(Array(a.utf8)), eb = ByteLevel.encode(Array(b.utf8))
            lines.append("\(ea) \(eb)")
            if !tokens.contains(ea + eb) {
                tokens.append(ea + eb)
                types.append(.normal)
            }
        }
        for (t, ty) in added {
            tokens.append(t)
            types.append(ty)
        }
        return try BPETokenizer(tokens: tokens, types: types, merges: lines, preTokenizer: pre)
    }

    /// Reference BPE: the straightforward rescan loop of c/qwen36.c bpe_piece.
    static func naiveBPE(_ tok: BPETokenizer, _ bytes: [UInt8], merges: [String]) -> [Int32] {
        var rank: [String: Int] = [:]
        for (i, m) in merges.enumerated() where rank[m] == nil { rank[m] = i }
        var syms = bytes.map { String(ByteLevel.scalar(for: $0)) }
        while syms.count > 1 {
            var best = -1, at = -1
            for k in 0..<(syms.count - 1) {
                if let r = rank["\(syms[k]) \(syms[k + 1])"], best < 0 || r < best { best = r; at = k }
            }
            if at < 0 { break }
            syms[at] += syms[at + 1]
            syms.remove(at: at + 1)
        }
        return syms.map { tok.id(of: $0) ?? 0 }
    }
}

// MARK: - Byte map

@Suite struct TokenizerByteLevelTests {
    @Test func knownMappings() {
        #expect(ByteLevel.scalar(for: 0x20) == "\u{120}")  // Ġ
        #expect(ByteLevel.scalar(for: 0x0A) == "\u{10A}")  // Ċ
        #expect(ByteLevel.scalar(for: 0x41) == "A")
        #expect(ByteLevel.scalar(for: 0x00) == "\u{100}")
        #expect(ByteLevel.scalar(for: 0xAD) == "\u{143}")  // the last unmapped byte -> 256 + 67
        #expect(ByteLevel.byte(for: "\u{144}") == nil)
        #expect(ByteLevel.byte(for: "€") == nil)
    }

    @Test func roundTripsEveryByte() {
        var seen = Set<UInt32>()
        for b in 0...255 {
            let s = ByteLevel.scalar(for: UInt8(b))
            #expect(ByteLevel.byte(for: s) == UInt8(b))
            seen.insert(s.value)
        }
        #expect(seen.count == 256)
        let all = (0...255).map { UInt8($0) }
        #expect(ByteLevel.decode(ByteLevel.encode(all)) == all)
    }
}

// MARK: - Pre-tokenizers

@Suite struct TokenizerPreTokenizerTests {
    let qwen = Qwen2PreTokenizer()
    let gpt2 = GPT2PreTokenizer()

    @Test func qwen2Pieces() {
        #expect(qwen.pieces("Hello world") == ["Hello", " world"])
        #expect(qwen.pieces("  x") == [" ", " x"])
        #expect(qwen.pieces("\n  x") == ["\n", " ", " x"])
        #expect(qwen.pieces("a  \tb") == ["a", "  ", "\tb"])
        #expect(qwen.pieces("x\u{303}\u{304}y") == ["x", "\u{303}\u{304}", "y"])  // marks are not letters
        #expect(qwen.pieces("2024") == ["2", "0", "2", "4"])
        #expect(qwen.pieces("I'M here") == ["I", "'M", " here"])
        #expect(qwen.pieces("we'll") == ["we", "'ll"])
        #expect(qwen.pieces("(foo) .bar") == ["(foo", ")", " .", "bar"])
        #expect(qwen.pieces("x.\n\ny") == ["x", ".\n\n", "y"])
        #expect(qwen.pieces("  \n\n  y") == ["  \n\n", " ", " y"])
        #expect(qwen.pieces("end  ") == ["end", "  "])
        #expect(qwen.pieces("") == [])
    }

    @Test func gpt2Pieces() {
        #expect(gpt2.pieces("Hello  world 123 4567's") == ["Hello", " ", " world", " 123", " 4567", "'s"])
        #expect(gpt2.pieces("I'M") == ["I", "'", "M"])  // case-sensitive contractions
        #expect(gpt2.pieces("(foo)") == ["(", "foo", ")"])
        #expect(gpt2.pieces("x.\n\ny") == ["x", ".", "\n", "\n", "y"])  // \n then \n: \s+(?!\S) gives one back
        #expect(gpt2.pieces("a\t b") == ["a", "\t", " b"])
        #expect(gpt2.pieces("end  ") == ["end", "  "])
    }

    /// OLMoE's splitter against upstream c/tok.h pretok_chunk_gpt2 (Fixtures/tokenizer/gpt2_split.json).
    @Test func gpt2MatchesUpstreamSplit() throws {
        struct Fixture: Decodable {
            struct Case: Decodable { let name: String; let text: String; let pieces: [String] }
            let cases: [Case]
        }
        let fx = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: TokTestSupport.fixtureURL("gpt2_split.json")))
        #expect(fx.cases.count >= 200)
        // compare scalars, not Strings (String == is canonical equivalence)
        let bad = fx.cases.filter { c in
            gpt2.pieces(c.text).map { Array($0.unicodeScalars) } != c.pieces.map { Array($0.unicodeScalars) }
        }.map(\.name)
        #expect(bad.isEmpty, "mismatching cases: \(bad)")
    }

    @Test func splitCoversInput() {
        let text = "Mixed 文本 with ✨ emoji, numbers 42 and\ttabs\n\n  end"
        for pre in [qwen, gpt2] as [PreTokenizer] {
            #expect(pre.pieces(text).joined() == text)
        }
    }

    @Test func registry() throws {
        #expect(try PreTokenizerRegistry.make("qwen2") is Qwen2PreTokenizer)
        #expect(try PreTokenizerRegistry.make("olmo") is GPT2PreTokenizer)
        #expect(try PreTokenizerRegistry.make(nil) is GPT2PreTokenizer)
        #expect(throws: ColibriError.self) { try PreTokenizerRegistry.make("no-such-pre") }
    }
}

// MARK: - BPE on tiny vocabularies

@Suite struct TokenizerBPETests {
    @Test func mergesByRank() throws {
        let tok = try TokTestSupport.tinyTokenizer(merges: [("l", "o"), ("lo", "w"), ("e", "r"), ("low", "er")])
        let low = try #require(tok.id(of: "low"))
        let lower = try #require(tok.id(of: "lower"))
        #expect(tok.encode("low") == [low])
        #expect(tok.encode("lower") == [lower])
        #expect(tok.decode(tok.encode("lowest lower")) == "lowest lower")
    }

    @Test func leftmostOnTies() throws {
        let tok = try TokTestSupport.tinyTokenizer(merges: [("a", "a")])
        let aa = try #require(tok.id(of: "aa"))
        let a = try #require(tok.id(of: "a"))
        #expect(tok.encode("aaa") == [aa, a])
        #expect(tok.encode("aaaa") == [aa, aa])
    }

    @Test func addedTokensLongestFirstAndSplitBeforeRegex() throws {
        let tok = try TokTestSupport.tinyTokenizer(
            merges: [(" ", " ")],
            added: [("<|im_end|>", .control), ("<|im", .userDefined), ("<think>", .userDefined)])
        let end = try #require(tok.id(of: "<|im_end|>"))
        let im = try #require(tok.id(of: "<|im"))
        let x = try #require(tok.id(of: "X")), dot = try #require(tok.id(of: "."))
        #expect(tok.encode("X.<|im_end|>") == [x, dot, end])
        #expect(tok.encode("<|im_start") .first == im)
        // the added token ends the ordinary text: two spaces before it stay one piece
        let twoSpaces = try #require(tok.id(of: ByteLevel.encode(Array("  ".utf8))))
        #expect(tok.encode("X  <|im_end|>") == [x, twoSpaces, end])
        // control tokens decode to nothing unless asked; user-defined ones to their text
        #expect(tok.decode([x, end]) == "X")
        #expect(tok.decode([x, end], skipControl: false) == "X<|im_end|>")
        #expect(tok.decode([try #require(tok.id(of: "<think>"))]) == "<think>")
        #expect(tok.tokenType(end) == .control)
        #expect(tok.specialTokens.stop.contains(end))
    }

    @Test func heapBPEMatchesNaiveLoop() throws {
        // a vocabulary with overlapping merges over a small alphabet
        let pairs: [(String, String)] = [
            ("a", "b"), ("b", "a"), ("a", "a"), ("ab", "a"), ("ba", "b"), ("aa", "b"),
            ("ab", "ab"), ("c", "a"), ("ca", "b"), ("b", "c"), ("aba", "b"), ("bc", "a"),
        ]
        let tok = try TokTestSupport.tinyTokenizer(merges: pairs)
        let lines = pairs.map { "\(ByteLevel.encode(Array($0.0.utf8))) \(ByteLevel.encode(Array($0.1.utf8)))" }
        var rng = Xorshift64Star(seed: 12345)
        for _ in 0..<500 {
            let n = Int(rng.next() % 40) + 1
            let bytes = (0..<n).map { _ in Array("abc".utf8)[Int(rng.next() % 3)] }
            var got: [Int32] = []
            tok.bpe(bytes, into: &got)
            #expect(got == TokTestSupport.naiveBPE(tok, bytes, merges: lines), "\(String(decoding: bytes, as: UTF8.self))")
        }
    }

    @Test func longPieceIsFast() throws {
        let tok = try TokTestSupport.tinyTokenizer(merges: [("a", "a"), ("aa", "aa")])
        let start = Date()
        let ids = tok.encode(String(repeating: "a", count: 20_000))
        #expect(ids.count == 5000)
        #expect(Date().timeIntervalSince(start) < 5)
    }
}

// MARK: - Streaming decode

@Suite struct TokenizerStreamingTests {
    @Test func holdsBackPartialUTF8() throws {
        // one token per byte: every multi-byte character arrives split
        let tok = try TokTestSupport.tinyTokenizer(merges: [])
        let text = "aé€😀 Ω\n中"
        let ids = tok.encode(text)
        #expect(ids.count == text.utf8.count)
        var dec = StreamingDecoder(tokenizer: tok)
        var out = ""
        var pieces: [String] = []
        for id in ids {
            let s = dec.push(id)
            if !s.isEmpty { pieces.append(s) }
            out += s
        }
        out += dec.flush()
        #expect(out == text)
        #expect(!pieces.contains { $0.contains("\u{FFFD}") })
        #expect(pieces == ["a", "é", "€", "😀", " ", "Ω", "\n", "中"])
    }

    @Test func invalidBytesAreNotHeldForever() {
        #expect(StreamingDecoder.completePrefix([0x61, 0xE2, 0x82]) == 1)
        #expect(StreamingDecoder.completePrefix([0x61, 0xE2, 0x82, 0xAC]) == 4)
        #expect(StreamingDecoder.completePrefix([0x80, 0x80]) == 2)
        #expect(StreamingDecoder.completePrefix([0xFF]) == 1)
        #expect(StreamingDecoder.completePrefix([]) == 0)
    }
}

// MARK: - Qwen3-0.6B against upstream colibri

@Suite(.enabled(if: TokTestSupport.hasQwen3, "needs ~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"))
struct TokenizerQwen3Tests {
    @Test func loadsVocabulary() throws {
        let tok = try #require(TokTestSupport.qwen3)
        #expect(tok.vocabularySize == 151_936)
        #expect(tok.mergeCount == 151_387)
        #expect(tok.preTokenizer is Qwen2PreTokenizer)
        #expect(tok.id(of: "<|endoftext|>") == 151_643)
        #expect(tok.id(of: "<|im_start|>") == 151_644)
        #expect(tok.id(of: "<|im_end|>") == 151_645)
        #expect(tok.id(of: "<think>") == 151_667)
        var counts: [TokenType: Int] = [:]
        for i in 0..<tok.vocabularySize { counts[tok.tokenType(Int32(i)), default: 0] += 1 }
        #expect(counts[.control] == 20)
        #expect(counts[.userDefined] == 6)
        #expect(counts[.unused] == 267)
        #expect(tok.specialTokens.stop.isSuperset(of: [151_643, 151_645]))
        #expect(tok.specialTokens.addBOS == false)
    }

    @Test func goldenPrompt() throws {
        let tok = try #require(TokTestSupport.qwen3)
        let prompt = ChatMLTemplate(thinking: false).render([ChatMessage(role: .user, content: "What is the capital of France?")])
        #expect(prompt == "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        #expect(tok.encode(prompt) == [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198,
                                       151644, 77091, 198, 151667, 271, 151668, 271])
    }

    @Test func encodeMatchesUpstreamOracle() throws {
        let tok = try #require(TokTestSupport.qwen3)
        let fx = try TokTestSupport.encodeFixture()
        #expect(fx.cases.count >= 30)
        var mismatches: [String] = []
        for c in fx.cases where tok.encode(c.text) != c.ids {
            mismatches.append(c.name)
        }
        #expect(mismatches.isEmpty, "mismatching cases: \(mismatches)")
    }

    @Test func encodeMatchesUpstreamOracleOnRandomStrings() throws {
        let tok = try #require(TokTestSupport.qwen3)
        let fx = try TokTestSupport.encodeFixture("qwen3_encode_fuzz.json")
        #expect(fx.cases.count == 200)
        let mismatches = fx.cases.filter { tok.encode($0.text) != $0.ids }.map(\.name)
        #expect(mismatches.isEmpty, "mismatching cases: \(mismatches)")
    }

    @Test func decodeRoundTrips() throws {
        let tok = try #require(TokTestSupport.qwen3)
        for c in try TokTestSupport.encodeFixture().cases {
            // control tokens are kept, so the text comes back whole (NFC-normalized)
            let back = tok.decode(c.ids, skipControl: false)
            // bytes, not String ==, which is canonical equivalence and would hide NFC
            #expect(Array(back.utf8) == Array(c.text.precomposedStringWithCanonicalMapping.utf8), "\(c.name)")
            // streamed token by token, the same text
            var dec = StreamingDecoder(tokenizer: tok, skipControl: false)
            var streamed = ""
            for id in c.ids { streamed += dec.push(id) }
            streamed += dec.flush()
            #expect(Array(streamed.utf8) == Array(back.utf8), "\(c.name)")
        }
    }

    /// The production path: metadata from the GGUF workstream's reader.
    @Test func loadsThroughGGUFFile() throws {
        let file = try GGUFFile(path: TokTestSupport.qwen3Path)
        let tok = try BPETokenizer(metadata: file.metadata)
        let reference = try #require(TokTestSupport.qwen3)
        let fx = try TokTestSupport.encodeFixture()
        #expect(tok.vocabularySize == reference.vocabularySize)
        #expect(tok.specialTokens == reference.specialTokens)
        for c in fx.cases { #expect(tok.encode(c.text) == c.ids, "\(c.name)") }
    }

    @Test func controlTokensDecodeToNothingForDisplay() throws {
        let tok = try #require(TokTestSupport.qwen3)
        let ids = tok.encode("<|im_start|>assistant\n<think>\nok\n</think>\n\nHi!<|im_end|>")
        #expect(tok.decode(ids) == "assistant\n<think>\nok\n</think>\n\nHi!")
    }
}
