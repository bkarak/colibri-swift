// Deterministic tiny random GGUFs for the architectures in scope, so the MoE families can
// be tested without a real MoE model on disk. The oracle harness (scripts/oracle/) runs
// upstream colibri on files written by this builder and records its greedy token ids and
// logits as fixtures (Tests/ColibriCoreTests/Fixtures/oracle/).
//
// Default shapes (docs/port/models.md section 10, scaled up to exercise real block sizes):
//
//   arch      L  D    H  KV hd   attention width  FFN
//   qwen3     2  256  4  2  128  512 (!= D)       dense F 512
//   qwen3moe  3  256  8  2  64   512 (!= D)       E 16, top-4, I 256, renormalised
//   olmoe     2  256  4  4  64   256 (= D)        E 8, top-2, I 256, not renormalised
//
// Every matrix's input width is a multiple of 256, so the K-quants (Q4_K, Q6_K) can be
// used for every matrix. Norms and routers stay F32, as llama.cpp writes them.
//
// The vocabulary is either the 256 GPT-2 byte symbols plus a few control tokens and no
// merges (any text encodes byte by byte, so token id == byte value for ASCII), or the
// tokenizer copied from a real GGUF (the Qwen3-0.6B file), which makes the embedding
// and head large.
//
// Random numbers come from SplitMix64 and an Irwin-Hall approximation of a Gaussian
// (sum of four uniforms), which uses only exact IEEE float arithmetic: the same spec gives
// the same bytes on any machine, without depending on libm. `fingerprint` (FNV-1a 64 of
// the serialised file) is recorded with each oracle fixture so a change to the generator
// or to the quantisers is caught before it is mistaken for an engine bug.

import Foundation

/// Where a tiny model's tokenizer comes from.
public enum TinyVocabulary: Sendable, Equatable {
    /// 256 byte symbols (id == byte value) plus the family's control tokens, no merges.
    case bytes
    /// Every `tokenizer.*` key copied from the GGUF at this path.
    case copied(from: String)
}

public struct TinyModelSpec: Sendable {
    public var architecture: String
    /// Type of every matrix unless `overrides` names it. Norms and routers are always F32.
    public var weightType: GGMLType
    /// Tensor name without the `blk.N.` prefix -> type, e.g. `["ffn_down_exps.weight": .q6_K]`
    /// for a Q4_K_M-like mix.
    public var overrides: [String: GGMLType]
    /// No `output.weight` (the head reads `token_embd.weight`).
    public var tied: Bool
    public var seed: UInt64
    public var vocabulary: TinyVocabulary

    public var layers: Int
    public var hidden: Int
    public var heads: Int
    public var kvHeads: Int
    public var headDim: Int
    /// Routed experts (0 for qwen3).
    public var experts: Int
    public var expertsUsed: Int
    public var expertFFN: Int
    /// Dense FFN width (qwen3).
    public var denseFFN: Int
    public var ropeTheta: Float
    public var rmsEps: Float
    public var contextLength: Int

    public init(architecture: String, weightType: GGMLType = .f32, tied: Bool = false, seed: UInt64 = 1,
                vocabulary: TinyVocabulary = .bytes, overrides: [String: GGMLType] = [:]) {
        self.architecture = architecture
        self.weightType = weightType
        self.overrides = overrides
        self.tied = tied
        self.seed = seed
        self.vocabulary = vocabulary
        self.hidden = 256
        self.contextLength = 4096
        switch architecture {
        case "olmoe":
            layers = 2; heads = 4; kvHeads = 4; headDim = 64
            experts = 8; expertsUsed = 2; expertFFN = 256; denseFFN = 256
            ropeTheta = 10000; rmsEps = 1e-5
        case "qwen3moe":
            layers = 3; heads = 8; kvHeads = 2; headDim = 64
            experts = 16; expertsUsed = 4; expertFFN = 256; denseFFN = 0
            ropeTheta = 1_000_000; rmsEps = 1e-6
        default:  // qwen3
            layers = 2; heads = 4; kvHeads = 2; headDim = 128
            experts = 0; expertsUsed = 0; expertFFN = 0; denseFFN = 512
            ropeTheta = 1_000_000; rmsEps = 1e-6
        }
    }

    var isMoE: Bool { architecture != "qwen3" }

    /// The type a matrix named `name` is stored in.
    func matrixType(_ name: String) -> GGMLType {
        var base = Substring(name)
        if base.hasPrefix("blk."), let dot = base.dropFirst(4).firstIndex(of: ".") {
            base = base[base.index(after: dot)...]
        }
        return overrides[String(base)] ?? weightType
    }
}

/// SplitMix64 plus an Irwin-Hall Gaussian; exact float arithmetic only.
struct TinyRNG {
    var state: UInt64
    init(seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 ^ 0xD1B5_4A32_D192_ED03 }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 bits.
    mutating func uniform() -> Float { Float(next() >> 40) * (1.0 / 16_777_216.0) }

    /// Mean 0, standard deviation `std` (approximately normal, bounded by 2*sqrt(3)*std).
    mutating func gaussian(_ std: Float) -> Float {
        let s = uniform() + uniform() + uniform() + uniform() - 2
        return s * (1.7320508 * std)
    }
}

public enum TinyModel {
    /// GPT-2 ByteLevel: the code point each byte maps to, as a one-character string.
    static func byteSymbols() -> [String] {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b); cs.append(256 + n); n += 1
        }
        var out = [String](repeating: "", count: 256)
        for (b, c) in zip(bs, cs) { out[b] = String(Character(Unicode.Scalar(UInt32(c))!)) }
        return out
    }

    /// Control tokens appended after the 256 byte symbols (ids 256...).
    public static func specialTokens(architecture: String) -> [String] {
        architecture == "olmoe"
            ? ["|||IP_ADDRESS|||", "<|endoftext|>", "<|padding|>"]
            : ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    }

    /// Builds the GGUF for `spec` (same bytes for the same spec, on every run).
    public static func build(_ spec: TinyModelSpec) throws -> GGUFWriter {
        let arch = spec.architecture
        guard ["qwen3", "qwen3moe", "olmoe"].contains(arch) else {
            throw ColibriError.invalidArgument("TinyModel: no generator for architecture \(arch)")
        }
        if arch == "olmoe" && spec.kvHeads != spec.heads {
            throw ColibriError.invalidArgument("TinyModel: olmoe needs kvHeads == heads (upstream refuses GQA)")
        }
        if arch == "olmoe" && spec.heads * spec.headDim != spec.hidden {
            throw ColibriError.invalidArgument("TinyModel: olmoe needs heads * headDim == hidden")
        }
        if arch == "olmoe" && spec.tied {
            throw ColibriError.invalidArgument("TinyModel: upstream olmoe needs output.weight (no tied head)")
        }
        var w = GGUFWriter()
        let V = try addMetadata(spec, to: &w)
        let D = spec.hidden, H = spec.heads, KV = spec.kvHeads, hd = spec.headDim

        var rng = TinyRNG(seed: spec.seed)
        func randoms(_ n: Int, _ std: Float) -> [Float] {
            var v = [Float](repeating: 0, count: n)
            for i in 0..<n { v[i] = rng.gaussian(std) }
            return v
        }
        func norm(_ name: String, _ n: Int) throws {
            var v = [Float](repeating: 0, count: n)
            for i in 0..<n { v[i] = 1 + (rng.uniform() - 0.5) * 0.2 }  // 1 +- 0.1
            try w.addTensor(name, dims: [n], type: .f32, values: v)
        }
        /// `[inputs, outputs, n]`, standard deviation `gain / sqrt(inputs)`.
        func matrix(_ name: String, _ dims: [Int], gain: Float, type: GGMLType? = nil) throws {
            let n = dims.reduce(1, *)
            let std = gain / Float(dims[0]).squareRoot()
            try w.addTensor(name, dims: dims, type: type ?? spec.matrixType(name), values: randoms(n, std))
        }

        // Embedding rows are ~unit RMS; the head gives logits with a spread of a few units.
        // A tied head computes logit[t] = e_t . h, and h still contains the current token's
        // embedding, so with unit-RMS rows the model just repeats its input; tied models get
        // embeddings ten times smaller, which the layers' contributions then dominate.
        try matrix("token_embd.weight", [D, V], gain: Float(D).squareRoot() * (spec.tied ? 0.1 : 1))
        try norm("output_norm.weight", D)
        if !spec.tied { try matrix("output.weight", [D, V], gain: 3) }
        for l in 0..<spec.layers {
            let b = "blk.\(l)."
            try norm(b + "attn_norm.weight", D)
            try matrix(b + "attn_q.weight", [D, H * hd], gain: 2)
            try matrix(b + "attn_k.weight", [D, KV * hd], gain: 2)
            try matrix(b + "attn_v.weight", [D, KV * hd], gain: 1)
            try matrix(b + "attn_output.weight", [H * hd, D], gain: 1)
            try norm(b + "attn_q_norm.weight", arch == "olmoe" ? H * hd : hd)
            try norm(b + "attn_k_norm.weight", arch == "olmoe" ? KV * hd : hd)
            try norm(b + "ffn_norm.weight", D)
            if arch == "qwen3" {
                try matrix(b + "ffn_gate.weight", [D, spec.denseFFN], gain: 1.5)
                try matrix(b + "ffn_up.weight", [D, spec.denseFFN], gain: 1.5)
                try matrix(b + "ffn_down.weight", [spec.denseFFN, D], gain: 1)
            } else {
                try matrix(b + "ffn_gate_inp.weight", [D, spec.experts], gain: 2, type: .f32)
                try matrix(b + "ffn_gate_exps.weight", [D, spec.expertFFN, spec.experts], gain: 1.5)
                try matrix(b + "ffn_up_exps.weight", [D, spec.expertFFN, spec.experts], gain: 1.5)
                try matrix(b + "ffn_down_exps.weight", [spec.expertFFN, D, spec.experts], gain: 1)
            }
        }
        return w
    }

    /// The metadata `build` writes (hyperparameters and tokenizer), without the weights.
    public static func metadata(_ spec: TinyModelSpec) throws -> GGUFMetadata {
        var w = GGUFWriter()
        _ = try addMetadata(spec, to: &w)
        return w.metadata
    }

    /// Writes hyperparameters and tokenizer; returns the vocabulary size.
    static func addMetadata(_ spec: TinyModelSpec, to w: inout GGUFWriter) throws -> Int {
        let arch = spec.architecture
        let D = spec.hidden, H = spec.heads, KV = spec.kvHeads, hd = spec.headDim
        let p = arch + "."
        w.set("general.architecture", .string(arch))
        w.set("general.name", .string("tiny-\(arch)"))
        w.set(p + "block_count", .u32(UInt32(spec.layers)))
        w.set(p + "context_length", .u32(UInt32(spec.contextLength)))
        w.set(p + "embedding_length", .u32(UInt32(D)))
        w.set(p + "attention.head_count", .u32(UInt32(H)))
        w.set(p + "attention.head_count_kv", .u32(UInt32(KV)))
        w.set(p + "rope.freq_base", .f32(spec.ropeTheta))
        w.set(p + "attention.layer_norm_rms_epsilon", .f32(spec.rmsEps))
        if arch != "olmoe" {
            w.set(p + "attention.key_length", .u32(UInt32(hd)))
            w.set(p + "attention.value_length", .u32(UInt32(hd)))
        }
        if arch == "qwen3" {
            w.set(p + "feed_forward_length", .u32(UInt32(spec.denseFFN)))
        } else {
            w.set(p + "expert_count", .u32(UInt32(spec.experts)))
            w.set(p + "expert_used_count", .u32(UInt32(spec.expertsUsed)))
            w.set(p + "expert_feed_forward_length", .u32(UInt32(spec.expertFFN)))
            if arch == "olmoe" { w.set(p + "feed_forward_length", .u32(UInt32(spec.expertFFN))) }
        }

        let V: Int
        switch spec.vocabulary {
        case .bytes:
            let specials = specialTokens(architecture: arch)
            let tokens = byteSymbols() + specials
            var types = [Int32](repeating: 1, count: 256) + [Int32](repeating: 3, count: specials.count)
            if arch != "olmoe" { types[types.count - 2] = 4; types[types.count - 1] = 4 }  // <think> </think>
            V = tokens.count
            w.set("tokenizer.ggml.model", .string("gpt2"))
            w.set("tokenizer.ggml.pre", .string(arch == "olmoe" ? "olmo" : "qwen2"))
            w.set("tokenizer.ggml.tokens", .array(.string, tokens.map { .string($0) }))
            w.set("tokenizer.ggml.token_type", .array(.i32, types.map { .i32($0) }))
            w.set("tokenizer.ggml.merges", .array(.string, []))
            w.set("tokenizer.ggml.eos_token_id", .u32(arch == "olmoe" ? 256 : 258))  // |||IP_ADDRESS||| / <|im_end|>
            w.set("tokenizer.ggml.bos_token_id", .u32(256))
            w.set("tokenizer.ggml.add_bos_token", .bool(false))
        case .copied(let path):
            let source = try GGUFFile(path: path)
            w.copyMetadata(from: source.metadata) { $0.hasPrefix("tokenizer.") }
            guard let n = source.metadata.strings("tokenizer.ggml.tokens")?.count, n > 0 else {
                throw ColibriError.invalidArgument("TinyModel: \(path) has no tokenizer.ggml.tokens")
            }
            V = n
        }
        return V
    }

    /// Writes it to `path`.
    public static func write(_ spec: TinyModelSpec, to path: String) throws {
        try build(spec).write(to: path)
    }

    /// FNV-1a 64 of `bytes`, as 16 lowercase hex digits.
    public static func fingerprint(_ bytes: [UInt8]) -> String {
        var h: UInt64 = 0xCBF2_9CE4_8422_2325
        for b in bytes { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01B3 }
        let s = String(h, radix: 16)
        return String(repeating: "0", count: 16 - s.count) + s
    }
}

/// The synthetic models the oracle harness records upstream outputs for. The names are
/// the fixture file names (`Fixtures/oracle/tiny/<name>.json`).
public struct TinyOracleCase: Sendable {
    public var name: String
    public var spec: TinyModelSpec
    /// The case copies the tokenizer of the real Qwen3-0.6B GGUF (skipped when it is absent).
    public var needsRealTokenizer: Bool { spec.vocabulary != .bytes }
}

public enum TinyOracleCases {
    /// Default location of the one real model on this machine.
    public static var qwen3TokenizerSource: String {
        ProcessInfo.processInfo.environment["COLIBRI_QWEN3_GGUF"]
            ?? (NSHomeDirectory() + "/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf")
    }

    /// A Q4_K_M-like mix: Q4_K matrices, Q6_K for attn_v, the down projections and the head.
    static let q4KM: [String: GGMLType] = [
        "attn_v.weight": .q6_K, "ffn_down.weight": .q6_K, "ffn_down_exps.weight": .q6_K, "output.weight": .q6_K,
    ]

    public static var all: [TinyOracleCase] {
        var c: [TinyOracleCase] = []
        func add(_ name: String, _ spec: TinyModelSpec) { c.append(TinyOracleCase(name: name, spec: spec)) }
        for arch in ["qwen3moe", "olmoe", "qwen3"] {
            add("\(arch)_f32", TinyModelSpec(architecture: arch, weightType: .f32, seed: 1))
            add("\(arch)_q8_0", TinyModelSpec(architecture: arch, weightType: .q8_0, seed: 2))
            add("\(arch)_q4_K", TinyModelSpec(architecture: arch, weightType: .q4_K, seed: 3))
            add("\(arch)_q4_K_M", TinyModelSpec(architecture: arch, weightType: .q4_K, seed: 4, overrides: q4KM))
        }
        add("qwen3_q8_0_tied", TinyModelSpec(architecture: "qwen3", weightType: .q8_0, tied: true, seed: 5))
        add("qwen3moe_q8_0_tied", TinyModelSpec(architecture: "qwen3moe", weightType: .q8_0, tied: true, seed: 6))
        let real = qwen3TokenizerSource
        add("qwen3moe_q8_0_qwenvocab",
            TinyModelSpec(architecture: "qwen3moe", weightType: .q8_0, seed: 7, vocabulary: .copied(from: real)))
        add("olmoe_q4_K_qwenvocab",
            TinyModelSpec(architecture: "olmoe", weightType: .q4_K, seed: 8, vocabulary: .copied(from: real)))
        return c
    }
}
