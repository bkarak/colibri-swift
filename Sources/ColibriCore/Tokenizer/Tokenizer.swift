// Byte-level BPE tokenizer read from GGUF metadata. Ported from colibri c/qwen36.c
// (load_tokenizer_gguf, encode_text, bpe_piece, decode_id_to_bytes, utf8_drain) and
// c/tok.h; behaviour follows HF tokenizers where upstream deviates from it (it matches
// upstream's exact mode, g_tok_exact). docs/port/models.md section 6.

import Foundation

/// ggml token types (`tokenizer.ggml.token_type`).
public enum TokenType: Int32, Sendable {
    case normal = 1, unknown = 2, control = 3, userDefined = 4, unused = 5, byte = 6
}

/// The special ids a GGUF names (`tokenizer.ggml.*_token_id`, `eog_token_ids`).
public struct SpecialTokens: Sendable, Equatable {
    public var bos: Int32?
    public var eos: Int32?
    public var eot: Int32?
    public var eom: Int32?
    public var padding: Int32?
    /// `tokenizer.ggml.eog_token_ids`, as stored.
    public var endOfGeneration: [Int32] = []
    /// `tokenizer.ggml.add_bos_token` (informational: `encode` never adds one).
    public var addBOS: Bool = false
    /// Where generation stops: the eog ids, eos, eot, eom, and `<|endoftext|>` /
    /// `<|im_end|>` when the vocabulary has them (docs/port/models.md section 8).
    public var stop: Set<Int32> = []

    public init() {}
}

/// Text <-> token ids. Implementations are immutable after loading and safe to share.
public protocol Tokenizer: AnyObject, Sendable {
    var vocabularySize: Int { get }

    /// bos/eos/eot/eom/padding and the stop set.
    var specialTokens: SpecialTokens { get }

    /// Encodes text. Added tokens (types control and user-defined) are matched literally
    /// first, longest match wins; the spans between them are pre-tokenized and BPE-merged.
    /// No BOS is added; templates write whatever the model needs.
    func encode(_ text: String) -> [Int32]

    /// The raw bytes of one token (control tokens decode to nothing when
    /// `skipControl`). Bytes, not a String: a token can end inside a UTF-8 sequence.
    func decodeBytes(_ id: Int32, skipControl: Bool) -> [UInt8]

    func tokenType(_ id: Int32) -> TokenType
    /// The id of a piece spelled exactly `piece` (e.g. "<|im_end|>"), if any.
    func id(of piece: String) -> Int32?
    /// The stored piece string of `id` (byte-level spelling).
    func piece(_ id: Int32) -> String
}

extension Tokenizer {
    /// Decodes a whole sequence to text (invalid UTF-8 is repaired with U+FFFD).
    public func decode(_ ids: [Int32], skipControl: Bool = true) -> String {
        var bytes: [UInt8] = []
        for id in ids { bytes += decodeBytes(id, skipControl: skipControl) }
        return String(decoding: bytes, as: UTF8.self)
    }
}

/// Byte-level BPE built from `tokenizer.ggml.*` keys (`model == "gpt2"`; anything else is
/// refused, as upstream does). The pre-tokenizer is chosen by `tokenizer.ggml.pre`
/// through `PreTokenizerRegistry`, not by architecture.
public final class BPETokenizer: Tokenizer, @unchecked Sendable {
    public let vocabularySize: Int
    public let preTokenizer: PreTokenizer
    public let specialTokens: SpecialTokens
    /// Number of merges loaded (merges whose parts or result are not in the vocabulary
    /// are skipped).
    public let mergeCount: Int

    private let pieces: [String]
    private let types: [TokenType]
    private let pieceToID: [String: Int32]
    /// byte -> id of its one-byte piece, or -1.
    private let byteIDs: [Int32]
    /// (left id << 32 | right id) -> (rank << 32 | merged id).
    private let merges: [UInt64: UInt64]
    /// Added tokens (control and user-defined) by first byte, longest first.
    private let added: [[(bytes: [UInt8], id: Int32)]]

    public convenience init(metadata: GGUFMetadata) throws {
        guard let tokens = metadata.strings("tokenizer.ggml.tokens"), !tokens.isEmpty else {
            throw ColibriError.unsupported("the GGUF carries no tokenizer.ggml.tokens")
        }
        if let model = metadata.string("tokenizer.ggml.model"), model != "gpt2" {
            throw ColibriError.unsupported("tokenizer model '\(model)' is not byte-level BPE (gpt2)")
        }
        let pre = try PreTokenizerRegistry.make(metadata.string("tokenizer.ggml.pre"))
        let rawTypes = metadata.ints("tokenizer.ggml.token_type")
        let types: [TokenType]
        if let rawTypes, rawTypes.count == tokens.count {
            types = rawTypes.map { TokenType(rawValue: Int32(truncatingIfNeeded: $0)) ?? .normal }
        } else {
            types = Array(repeating: .normal, count: tokens.count)
        }
        func id(_ key: String) -> Int32? {
            guard let v = metadata.int("tokenizer.ggml.\(key)"), v >= 0, v < tokens.count else { return nil }
            return Int32(v)
        }
        var sp = SpecialTokens()
        sp.bos = id("bos_token_id")
        sp.eos = id("eos_token_id")
        sp.eot = id("eot_token_id")
        sp.eom = id("eom_token_id")
        sp.padding = id("padding_token_id")
        sp.endOfGeneration = (metadata.ints("tokenizer.ggml.eog_token_ids") ?? [])
            .filter { $0 >= 0 && $0 < tokens.count }.map { Int32($0) }
        sp.addBOS = metadata.bool("tokenizer.ggml.add_bos_token") ?? false
        try self.init(tokens: tokens, types: types,
                      merges: metadata.strings("tokenizer.ggml.merges") ?? [],
                      preTokenizer: pre, special: sp)
    }

    /// Builds a tokenizer from its parts (tests and tiny synthetic vocabularies).
    /// `special.stop` is completed from the other fields and the vocabulary.
    public init(tokens: [String], types: [TokenType], merges mergeList: [String],
                preTokenizer: PreTokenizer, special: SpecialTokens = SpecialTokens()) throws {
        guard types.count == tokens.count else {
            throw ColibriError.invalidArgument("\(tokens.count) tokens but \(types.count) token types")
        }
        guard tokens.count < Int(Int32.max) else { throw ColibriError.invalidGGUF("vocabulary too large") }
        vocabularySize = tokens.count
        self.preTokenizer = preTokenizer
        pieces = tokens
        self.types = types

        var map: [String: Int32] = [:]
        map.reserveCapacity(tokens.count)
        for (i, t) in tokens.enumerated() where map[t] == nil { map[t] = Int32(i) }
        pieceToID = map

        // Merge results are ordinary pieces; control tokens never take part in BPE.
        func bpeID(_ s: String) -> Int32? {
            guard let id = map[s], types[Int(id)] != .control else { return nil }
            return id
        }
        var byteIDs = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            if let id = bpeID(String(ByteLevel.scalar(for: UInt8(b)))) { byteIDs[b] = id }
        }
        self.byteIDs = byteIDs

        var m: [UInt64: UInt64] = [:]
        m.reserveCapacity(mergeList.count)
        for (rank, line) in mergeList.enumerated() {
            guard let sp = line.firstIndex(of: " ") else { continue }
            let a = String(line[..<sp]), b = String(line[line.index(after: sp)...])
            guard let ia = bpeID(a), let ib = bpeID(b), let im = bpeID(a + b) else { continue }
            let key = UInt64(UInt32(ia)) << 32 | UInt64(UInt32(ib))
            if m[key] == nil { m[key] = UInt64(rank) << 32 | UInt64(UInt32(im)) }
        }
        merges = m
        mergeCount = m.count

        var addedTable = [[(bytes: [UInt8], id: Int32)]](repeating: [], count: 256)
        for (i, t) in tokens.enumerated() where types[i] == .control || types[i] == .userDefined {
            let bytes = Array(t.utf8)
            guard let first = bytes.first else { continue }
            addedTable[Int(first)].append((bytes, Int32(i)))
        }
        for k in 0..<256 { addedTable[k].sort { $0.bytes.count > $1.bytes.count } }
        added = addedTable

        var sp = special
        var stop = Set(sp.endOfGeneration)
        for v in [sp.eos, sp.eot, sp.eom] { if let v { stop.insert(v) } }
        for name in ["<|endoftext|>", "<|im_end|>"] { if let v = map[name] { stop.insert(v) } }
        sp.stop = stop
        specialTokens = sp
    }

    // MARK: Encoding

    /// The added token starting at `i`, longest first.
    @inline(__always)
    private func matchAdded(_ text: UnsafeBufferPointer<UInt8>, _ i: Int) -> (length: Int, id: Int32)? {
        let cands = added[Int(text[i])]
        if cands.isEmpty { return nil }
        let n = text.count
        for c in cands where i + c.bytes.count <= n {
            var ok = true
            for (k, b) in c.bytes.enumerated() where text[i + k] != b { ok = false; break }
            if ok { return (c.bytes.count, c.id) }
        }
        return nil
    }

    public func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        let utf8 = Array(text.utf8)
        utf8.withUnsafeBufferPointer { buf in
            let n = buf.count
            var i = 0
            while i < n {
                if let m = matchAdded(buf, i) {
                    out.append(m.id)
                    i += m.length
                    continue
                }
                // Ordinary text runs to the next added token; the pre-tokenizer sees that
                // boundary as the end of its input, as HF's does.
                var end = i + 1
                while end < n && matchAdded(buf, end) == nil { end += 1 }
                encodeOrdinary(UnsafeBufferPointer(rebasing: buf[i..<end]), into: &out)
                i = end
            }
        }
        return out
    }

    private func encodeOrdinary(_ span: UnsafeBufferPointer<UInt8>, into out: inout [Int32]) {
        var text = String(decoding: span, as: UTF8.self)
        if preTokenizer.normalizesNFC && span.contains(where: { $0 >= 0x80 }) {
            text = text.precomposedStringWithCanonicalMapping
        }
        let scalars = Array(text.unicodeScalars)
        let utf8 = Array(text.utf8)
        var offset = [Int](repeating: 0, count: scalars.count + 1)
        for (k, s) in scalars.enumerated() { offset[k + 1] = offset[k] + UTF8.width(s) }
        let ends = preTokenizer.pieceEnds(scalars, scalars.map(CharClass.init))
        var start = 0
        for e in ends {
            bpe(Array(utf8[offset[start]..<offset[e]]), into: &out)
            start = e
        }
    }

    /// BPE over one piece: repeatedly merge the adjacent pair of lowest rank (leftmost on
    /// ties) until no pair is in the merges table. A min-heap of (rank, position) keeps
    /// long pieces linear-logarithmic; stale entries are skipped on pop.
    func bpe(_ bytes: [UInt8], into out: inout [Int32]) {
        let n = bytes.count
        if n == 0 { return }
        if preTokenizer.ignoreMerges, let id = pieceToID[ByteLevel.encode(bytes)], types[Int(id)] != .control {
            out.append(id)
            return
        }
        var ids = bytes.map { byteIDs[Int($0)] }
        if n == 1 {
            out.append(max(ids[0], 0))
            return
        }
        // a linked list over the symbols; -1 ends it, a merged-away symbol has id -2
        var next = Array(1...n)
        next[n - 1] = -1
        var prev = Array(-1..<(n - 1))
        @inline(__always) func rankOf(_ a: Int32, _ b: Int32) -> UInt64? {
            guard a >= 0, b >= 0 else { return nil }
            return merges[UInt64(UInt32(a)) << 32 | UInt64(UInt32(b))]
        }
        var heap = MinHeap()
        for p in 0..<(n - 1) {
            if let r = rankOf(ids[p], ids[p + 1]) { heap.push((r >> 32) << 32 | UInt64(p)) }
        }
        while let top = heap.pop() {
            let rank = top >> 32
            let p = Int(top & 0xFFFF_FFFF)
            let q = next[p]
            // stale entries: p was merged away (id -2) or the pair at p changed
            guard q >= 0, let r = rankOf(ids[p], ids[q]), r >> 32 == rank else { continue }
            ids[p] = Int32(truncatingIfNeeded: r & 0xFFFF_FFFF)
            // unlink q
            let qn = next[q]
            next[p] = qn
            if qn >= 0 { prev[qn] = p }
            ids[q] = -2
            let pp = prev[p]
            if pp >= 0, let r2 = rankOf(ids[pp], ids[p]) { heap.push((r2 >> 32) << 32 | UInt64(pp)) }
            if qn >= 0, let r3 = rankOf(ids[p], ids[qn]) { heap.push((r3 >> 32) << 32 | UInt64(p)) }
        }
        var p = 0
        while p >= 0 {
            out.append(max(ids[p], 0))  // a byte missing from the vocabulary: id 0, as upstream
            p = next[p]
        }
    }

    // MARK: Decoding and lookup

    public func decodeBytes(_ id: Int32, skipControl: Bool) -> [UInt8] {
        guard id >= 0, Int(id) < vocabularySize else { return [] }
        let piece = pieces[Int(id)]
        switch types[Int(id)] {
        case .control, .unused:
            return skipControl ? [] : Array(piece.utf8)
        case .userDefined, .unknown:
            return Array(piece.utf8)
        case .byte:
            // "<0xXX>" byte-fallback token
            let u = Array(piece.utf8)
            if u.count == 6, u[0] == 0x3C, u[1] == 0x30, u[2] == 0x78, u[5] == 0x3E,
               let v = UInt8(String(decoding: u[3...4], as: UTF8.self), radix: 16) {
                return [v]
            }
            return ByteLevel.decode(piece)
        case .normal:
            return ByteLevel.decode(piece)
        }
    }

    public func tokenType(_ id: Int32) -> TokenType {
        guard id >= 0, Int(id) < vocabularySize else { return .unknown }
        return types[Int(id)]
    }

    public func id(of piece: String) -> Int32? { pieceToID[piece] }

    public func piece(_ id: Int32) -> String {
        guard id >= 0, Int(id) < vocabularySize else { return "" }
        return pieces[Int(id)]
    }
}

/// Binary min-heap of packed (rank << 32 | position) keys.
private struct MinHeap {
    private var a: [UInt64] = []

    mutating func push(_ v: UInt64) {
        a.append(v)
        var i = a.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            if a[parent] <= a[i] { break }
            a.swapAt(parent, i)
            i = parent
        }
    }

    mutating func pop() -> UInt64? {
        guard let top = a.first else { return nil }
        let last = a.removeLast()
        if !a.isEmpty {
            a[0] = last
            var i = 0
            let n = a.count
            while true {
                let l = 2 * i + 1, r = l + 1
                var m = i
                if l < n && a[l] < a[m] { m = l }
                if r < n && a[r] < a[m] { m = r }
                if m == i { break }
                a.swapAt(i, m)
                i = m
            }
        }
        return top
    }
}

/// Turns a token stream into text for printing, holding back an incomplete UTF-8
/// sequence until the token that completes it arrives.
public struct StreamingDecoder {
    public let tokenizer: Tokenizer
    public var skipControl: Bool
    private var pending: [UInt8] = []

    public init(tokenizer: Tokenizer, skipControl: Bool = true) {
        self.tokenizer = tokenizer
        self.skipControl = skipControl
    }

    /// Text that became printable with `id` (possibly empty).
    public mutating func push(_ id: Int32) -> String {
        pending += tokenizer.decodeBytes(id, skipControl: skipControl)
        let cut = StreamingDecoder.completePrefix(pending)
        if cut == 0 { return "" }
        let text = String(decoding: pending[..<cut], as: UTF8.self)
        pending.removeFirst(cut)
        return text
    }

    /// Whatever is still buffered, repaired with U+FFFD.
    public mutating func flush() -> String {
        defer { pending.removeAll() }
        return String(decoding: pending, as: UTF8.self)
    }

    /// The length of the longest prefix of `b` that does not end inside a UTF-8 sequence
    /// that could still be completed. Invalid bytes are not held back (they print as
    /// U+FFFD).
    static func completePrefix(_ b: [UInt8]) -> Int {
        let n = b.count
        var i = n - 1
        var cont = 0
        while i >= 0 && cont < 3 && b[i] & 0xC0 == 0x80 {
            i -= 1
            cont += 1
        }
        if i < 0 { return n }
        let lead = b[i]
        let need: Int
        if lead & 0xE0 == 0xC0 { need = 2 }
        else if lead & 0xF0 == 0xE0 { need = 3 }
        else if lead & 0xF8 == 0xF0 { need = 4 }
        else { return n }
        return cont + 1 < need ? i : n
    }
}
