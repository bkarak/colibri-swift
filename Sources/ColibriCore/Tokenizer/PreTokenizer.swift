// Pre-tokenizers: the regex split HF's ByteLevel/Split pre-tokenizers apply before BPE,
// written as scanners over Unicode scalars. Ported from colibri c/qwen36.c pretok_end (in
// its exact mode, g_tok_exact: full Unicode classes, marks are not letters) and c/tok.h
// pretok_chunk_gpt2. docs/port/models.md section 6.

/// The character classes the split regexes use. `\s` is the Unicode White_Space property,
/// `\p{L}` the L* general categories, `\p{N}` the N* ones; everything else (marks,
/// punctuation, symbols, controls) is `other`.
public enum CharClass: UInt8, Sendable {
    case whitespace, letter, number, other

    private static let ascii: [CharClass] = (0..<128).map { v -> CharClass in
        switch v {
        case 0x09...0x0D, 0x20: return .whitespace
        case 0x30...0x39: return .number
        case 0x41...0x5A, 0x61...0x7A: return .letter
        default: return .other
        }
    }

    public init(_ s: Unicode.Scalar) {
        if s.value < 128 { self = CharClass.ascii[Int(s.value)]; return }
        let p = s.properties
        if p.isWhitespace { self = .whitespace; return }
        switch p.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            self = .letter
        case .decimalNumber, .letterNumber, .otherNumber:
            self = .number
        default:
            self = .other
        }
    }
}

/// Splits text into the pieces BPE runs on. One implementation per regex family
/// (`tokenizer.ggml.pre`). Implemented as hand-written scanners over Unicode scalars with
/// `Unicode.Scalar.Properties` classes (letters `L*`, numbers `N*`, White_Space),
/// not with NSRegularExpression, so behaviour is HF's and does not depend on ICU.
public protocol PreTokenizer: Sendable {
    /// The `tokenizer.ggml.pre` values this splitter implements.
    static var names: [String] { get }

    /// Whether the text between added tokens is NFC-normalized before the split (HF's
    /// `normalizer: NFC`, which Qwen2's tokenizer has). Default false.
    var normalizesNFC: Bool { get }

    /// Whether a piece that is itself a vocabulary entry is emitted whole, before any
    /// merge (HF's `ignore_merges`, tiktoken's rule). Default false.
    var ignoreMerges: Bool { get }

    /// The exclusive end offsets of the pieces of `scalars`, in order; the last one is
    /// `scalars.count` (empty input gives no pieces).
    func pieceEnds(_ scalars: [Unicode.Scalar], _ classes: [CharClass]) -> [Int]

    /// The pieces as ranges of `text`, in order, together covering all of it. (No NFC
    /// here: the ranges index `text` itself.)
    func split(_ text: String) -> [Range<String.Index>]
}

extension PreTokenizer {
    public var normalizesNFC: Bool { false }
    public var ignoreMerges: Bool { false }

    public func split(_ text: String) -> [Range<String.Index>] {
        let view = text.unicodeScalars
        let scalars = Array(view)
        var idx = Array(view.indices)
        idx.append(view.endIndex)
        let ends = pieceEnds(scalars, scalars.map(CharClass.init))
        var out: [Range<String.Index>] = []
        out.reserveCapacity(ends.count)
        var start = 0
        for e in ends {
            out.append(idx[start]..<idx[e])
            start = e
        }
        return out
    }

    /// The pieces as strings (tests and debugging).
    public func pieces(_ text: String) -> [String] {
        split(text).map { String(text.unicodeScalars[$0]) }
    }
}

@inline(__always) private func isCRLF(_ s: Unicode.Scalar) -> Bool { s.value == 0x0A || s.value == 0x0D }

@inline(__always) private func lowerASCII(_ s: Unicode.Scalar) -> UInt32 {
    (s.value >= 0x41 && s.value <= 0x5A) ? s.value + 32 : s.value
}

/// Qwen2 (`pre = "qwen2"`):
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// with an NFC normalizer in front, as Qwen2Tokenizer's tokenizer.json has.
public struct Qwen2PreTokenizer: PreTokenizer {
    public static let names = ["qwen2"]
    public init() {}
    public var normalizesNFC: Bool { true }

    public func pieceEnds(_ s: [Unicode.Scalar], _ cls: [CharClass]) -> [Int] {
        var ends: [Int] = []
        var i = 0
        let n = s.count
        while i < n {
            var j = end(s, cls, i, n)
            if j <= i { j = i + 1 }
            ends.append(j)
            i = j
        }
        return ends
    }

    /// The end of the leftmost-first match at `i` (the alternatives in the regex's order).
    private func end(_ s: [Unicode.Scalar], _ cls: [CharClass], _ i: Int, _ n: Int) -> Int {
        let c = s[i]
        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == "'" && i + 1 < n {
            if i + 2 < n {
                let a = lowerASCII(s[i + 1]), b = lowerASCII(s[i + 2])
                if (a == 0x72 && b == 0x65) || (a == 0x76 && b == 0x65) || (a == 0x6C && b == 0x6C) { return i + 3 }
            }
            switch lowerASCII(s[i + 1]) {
            case 0x73, 0x74, 0x6D, 0x64: return i + 2   // s t m d
            default: break
            }
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        do {
            var k = i
            if cls[i] != .letter && cls[i] != .number && !isCRLF(c) && i + 1 < n && cls[i + 1] == .letter { k = i + 1 }
            if cls[k] == .letter {
                while k < n && cls[k] == .letter { k += 1 }
                return k
            }
        }
        // 3. \p{N}
        if cls[i] == .number { return i + 1 }
        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        do {
            var k = i
            if c == " " && i + 1 < n && cls[i + 1] == .other { k = i + 1 }
            if cls[k] == .other {
                while k < n && cls[k] == .other { k += 1 }
                while k < n && isCRLF(s[k]) { k += 1 }
                return k
            }
        }
        // 5-7. whitespace: \s*[\r\n]+ (up to the run's last newline), then \s+(?!\S) (a run
        // before a non-space leaves its last char for the next piece), then \s+.
        var r = i, nlEnd = -1
        while r < n && cls[r] == .whitespace {
            if isCRLF(s[r]) { nlEnd = r + 1 }
            r += 1
        }
        if nlEnd > i { return nlEnd }
        if r < n && r - 1 > i { return r - 1 }
        return max(r, i + 1)
    }
}

/// GPT-2 (`pre = "olmo"`, `"gpt-2"`, `"default"`):
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// (case-sensitive contractions, one optional leading space, no normalizer).
public struct GPT2PreTokenizer: PreTokenizer {
    public static let names = ["olmo", "gpt-2", "gpt2", "default"]
    public init() {}

    public func pieceEnds(_ s: [Unicode.Scalar], _ cls: [CharClass]) -> [Int] {
        var ends: [Int] = []
        var i = 0
        let n = s.count
        while i < n {
            var j = end(s, cls, i, n)
            if j <= i { j = i + 1 }
            ends.append(j)
            i = j
        }
        return ends
    }

    private func end(_ s: [Unicode.Scalar], _ cls: [CharClass], _ i: Int, _ n: Int) -> Int {
        let c = s[i]
        if c == "'" && i + 1 < n {
            let a = s[i + 1].value
            if i + 2 < n {
                let b = s[i + 2].value
                if (a == 0x72 && b == 0x65) || (a == 0x76 && b == 0x65) || (a == 0x6C && b == 0x6C) { return i + 3 }
            }
            if a == 0x73 || a == 0x74 || a == 0x6D || a == 0x64 { return i + 2 }
        }
        var j = i
        if c == " " && i + 1 < n { j = i + 1 }
        switch cls[j] {
        case .letter:
            while j < n && cls[j] == .letter { j += 1 }
            return j
        case .number:
            while j < n && cls[j] == .number { j += 1 }
            return j
        case .other:
            while j < n && cls[j] == .other { j += 1 }
            return j
        case .whitespace:
            break
        }
        var r = i
        while r < n && cls[r] == .whitespace { r += 1 }
        let e = r < n ? r - 1 : r
        return e > i ? e : i + 1
    }
}

/// `tokenizer.ggml.pre` -> splitter. A new pre-tokenizer is one new type plus one line here.
public enum PreTokenizerRegistry {
    public static let all: [PreTokenizer.Type] = [Qwen2PreTokenizer.self, GPT2PreTokenizer.self]

    /// The splitter for `name`. A GGUF without `tokenizer.ggml.pre` is GPT-2's ("default",
    /// as llama.cpp reads it); an unknown name is refused rather than guessed.
    public static func make(_ name: String?) throws -> PreTokenizer {
        let key = name ?? "default"
        if Qwen2PreTokenizer.names.contains(key) { return Qwen2PreTokenizer() }
        if GPT2PreTokenizer.names.contains(key) { return GPT2PreTokenizer() }
        let known = all.flatMap { $0.names }.joined(separator: ", ")
        throw ColibriError.unsupported("tokenizer.ggml.pre '\(key)' has no pre-tokenizer (known: \(known)); add one in Tokenizer/PreTokenizer.swift")
    }
}
