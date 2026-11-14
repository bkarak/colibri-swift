import Foundation
import Testing
@testable import ColibriCore

// Tests for the model families (workstream D): a host-side reference forward pass written
// straight from docs/port/models.md section 4 (no Backend) against the families on the
// CPU backend, prefill chunking, resident vs streamed experts, chat KV reuse, config
// parsing, and the real Qwen3-0.6B against upstream colibri's golden output.
//
// The tiny models are built here (not with Synthetic/TinyModel) so these tests pin the
// exact shapes they need: D 64, 2 layers, 4 heads, head dim 32 (qwen: H*hd != D) or 16
// (olmoe: whole-vector q/k norm over D), 8 experts top-2, a byte vocabulary.

// MARK: - Tiny GGUF builder

struct ModelTestRNG {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-a, a).
    mutating func uniform(_ a: Float) -> Float { (Float(next() >> 40) / Float(1 << 24) * 2 - 1) * a }
}

struct TinySpec {
    var arch: String
    var type: GGMLType = .f32
    var tied = false
    var seed: UInt64 = 7
    var normTopK: Bool? = nil  // olmoe.expert_weights_norm when set
    var gatingFunc: Int? = nil

    var layers = 2
    var hidden = 64
    var heads = 4
    var kvHeads: Int { arch == "olmoe" ? 4 : 2 }
    var headDim: Int { arch == "olmoe" ? hidden / heads : 32 }
    var experts = 8
    var used = 2
    var expertFFN = 32
    var ffn = 96
    var isMoE: Bool { arch != "qwen3" }
    var specials: [String] {
        arch == "olmoe" ? ["|||IP_ADDRESS|||", "<|endoftext|>", "<|padding|>"]
            : ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    }
    var vocab: Int { 256 + specials.count }
}

enum TinyGGUF {
    static func write(_ s: TinySpec) throws -> String {
        var w = GGUFWriter()
        var rng = ModelTestRNG(state: s.seed)
        let a = s.arch
        func u32(_ v: Int) -> GGUFValue { .u32(UInt32(v)) }
        w.set("general.architecture", .string(a))
        w.set("\(a).block_count", u32(s.layers))
        w.set("\(a).context_length", u32(512))
        w.set("\(a).embedding_length", u32(s.hidden))
        w.set("\(a).attention.head_count", u32(s.heads))
        w.set("\(a).attention.head_count_kv", u32(s.kvHeads))
        w.set("\(a).rope.freq_base", .f32(a == "olmoe" ? 10000 : 1_000_000))
        if a != "olmoe" {
            w.set("\(a).attention.key_length", u32(s.headDim))
            w.set("\(a).attention.value_length", u32(s.headDim))
            w.set("\(a).attention.layer_norm_rms_epsilon", .f32(1e-6))
        } else {
            w.set("\(a).attention.layer_norm_rms_epsilon", .f32(1e-5))
        }
        if s.isMoE {
            w.set("\(a).expert_count", u32(s.experts))
            w.set("\(a).expert_used_count", u32(s.used))
            w.set("\(a).expert_feed_forward_length", u32(s.expertFFN))
        } else {
            w.set("\(a).feed_forward_length", u32(s.ffn))
        }
        if let n = s.normTopK { w.set("\(a).expert_weights_norm", .bool(n)) }
        if let g = s.gatingFunc { w.set("\(a).expert_gating_func", u32(g)) }

        var tokens = (0..<256).map { ByteLevel.encode([UInt8($0)]) }
        var types = [GGUFValue](repeating: .i32(1), count: 256)
        for sp in s.specials {
            tokens.append(sp)
            types.append(.i32(sp.hasPrefix("<think") || sp.hasPrefix("</think") ? 4 : 3))
        }
        w.set("tokenizer.ggml.model", .string("gpt2"))
        w.set("tokenizer.ggml.pre", .string(a == "olmoe" ? "olmo" : "qwen2"))
        w.set("tokenizer.ggml.tokens", .array(.string, tokens.map { .string($0) }))
        w.set("tokenizer.ggml.token_type", .array(.i32, types))
        w.set("tokenizer.ggml.merges", .array(.string, []))
        w.set("tokenizer.ggml.eos_token_id", u32(256 + (a == "olmoe" ? 0 : 2)))

        let D = s.hidden, H = s.heads, KV = s.kvHeads, hd = s.headDim, V = s.vocab
        func mat(_ name: String, _ cols: Int, _ rows: Int, _ extra: Int = 1, type: GGMLType? = nil, scale: Float? = nil) throws {
            let n = cols * rows * extra
            let amp = scale ?? (1.7 / Float(cols).squareRoot())
            let values = (0..<n).map { _ in rng.uniform(amp) }
            let dims = extra == 1 ? [cols, rows] : [cols, rows, extra]
            try w.addTensor(name, dims: dims, type: type ?? s.type, values: values)
        }
        func norm(_ name: String, _ n: Int) throws {
            try w.addTensor(name, dims: [n], type: .f32, values: (0..<n).map { _ in 1 + rng.uniform(0.2) })
        }
        try mat("token_embd.weight", D, V, scale: 1)
        try norm("output_norm.weight", D)
        if !s.tied { try mat("output.weight", D, V) }
        for l in 0..<s.layers {
            let p = "blk.\(l)."
            try norm(p + "attn_norm.weight", D)
            try mat(p + "attn_q.weight", D, H * hd)
            try mat(p + "attn_k.weight", D, KV * hd)
            try mat(p + "attn_v.weight", D, KV * hd)
            try mat(p + "attn_output.weight", H * hd, D)
            if a == "olmoe" {
                try norm(p + "attn_q_norm.weight", H * hd)
                try norm(p + "attn_k_norm.weight", KV * hd)
            } else {
                try norm(p + "attn_q_norm.weight", hd)
                try norm(p + "attn_k_norm.weight", hd)
            }
            try norm(p + "ffn_norm.weight", D)
            if s.isMoE {
                try mat(p + "ffn_gate_inp.weight", D, s.experts, type: .f32, scale: 1)
                try mat(p + "ffn_gate_exps.weight", D, s.expertFFN, s.experts)
                try mat(p + "ffn_up_exps.weight", D, s.expertFFN, s.experts)
                try mat(p + "ffn_down_exps.weight", s.expertFFN, D, s.experts)
            } else {
                try mat(p + "ffn_gate.weight", D, s.ffn)
                try mat(p + "ffn_up.weight", D, s.ffn)
                try mat(p + "ffn_down.weight", s.ffn, D)
            }
        }
        let path = NSTemporaryDirectory() + "colibri-model-\(a)-\(UUID().uuidString).gguf"
        try w.write(to: path)
        return path
    }
}

/// Builds a tiny model file, runs `body`, deletes the file (and any routing history).
@discardableResult
func withTinyModel<T>(_ spec: TinySpec, _ body: (String) throws -> T) throws -> T {
    let path = try TinyGGUF.write(spec)
    defer {
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.removeItem(atPath: path + ".coli_usage")
    }
    return try body(path)
}

func loadTiny(_ path: String, maxBatch: Int = 64, options: ExpertStoreOptions = ExpertStoreOptions(saveHistory: false),
              backend: Backend = CPUBackend(threads: 2)) throws -> LoadedModel {
    try ModelRegistry.load(path: path, context: LoadContext(backend: backend, expertOptions: options,
                                                            maxContext: 128, kvType: .f32, maxBatch: maxBatch))
}

/// Logits of the prompt's last row, then of each greedily chosen token, `steps` in all.
func greedyLogits(_ m: LoadedModel, prompt: [Int32], steps: Int) throws -> [[Float]] {
    let g = try Generator(loaded: m, sampler: .greedy)
    var all: [[Float]] = []
    g.onLogits = { _, l in all.append(Array(l)) }
    _ = try g.generate(prompt: prompt, maxTokens: steps, ignoreStop: true) { _ in true }
    return all
}

// MARK: - Reference forward pass (docs/port/models.md section 4, plain Swift)

final class ReferenceModel {
    let file: GGUFFile
    let c: ModelConfig
    private var cache: [String: [Float]] = [:]
    private var keys: [[[Float]]]    // [layer][pos] -> KV*hd
    private var values: [[[Float]]]

    init(path: String) throws {
        file = try GGUFFile(path: path)
        let fam = ModelRegistry.family(for: file.metadata.architecture!)!
        c = try fam.config(from: file)
        keys = Array(repeating: [], count: c.layers)
        values = Array(repeating: [], count: c.layers)
    }

    func w(_ name: String) -> [Float] {
        if let v = cache[name] { return v }
        let v = try! file.floats(name)
        cache[name] = v
        return v
    }

    func matvec(_ m: ArraySlice<Float>, cols: Int, rows: Int, _ x: [Float]) -> [Float] {
        let b = m.startIndex
        return (0..<rows).map { o in
            var s: Float = 0
            for i in 0..<cols { s += m[b + o * cols + i] * x[i] }
            return s
        }
    }
    func matvec(_ name: String, cols: Int, rows: Int, _ x: [Float]) -> [Float] { matvec(w(name)[...], cols: cols, rows: rows, x) }

    func rmsNorm(_ x: ArraySlice<Float>, _ weight: ArraySlice<Float>) -> [Float] {
        var ms = 0.0
        for v in x { ms += Double(v) * Double(v) }
        let r = 1 / sqrtf(Float(ms / Double(x.count)) + c.rmsEps)
        return zip(x, weight).map { $0 * r * $1 }
    }

    func rope(_ v: inout [Float], at base: Int, pos: Int) {
        let h2 = c.ropeDims / 2
        for j in 0..<h2 {
            let ang = Float(pos) * powf(c.ropeTheta, -2 * Float(j) / Float(c.ropeDims))
            let co = cosf(ang), si = sinf(ang)
            let a = v[base + j], b = v[base + j + h2]
            v[base + j] = a * co - b * si
            v[base + j + h2] = b * co + a * si
        }
    }

    func softmax(_ x: [Float]) -> [Float] {
        let m = x.max()!
        let e = x.map { expf($0 - m) }
        let s = e.reduce(0, +)
        return e.map { $0 / s }
    }

    func silu(_ z: Float) -> Float { z / (1 + expf(-z)) }

    /// Runs one token at position `pos` (positions must be fed in order) and returns logits.
    func step(_ token: Int32, pos: Int) -> [Float] {
        let D = c.hidden, H = c.heads, KV = c.kvHeads, hd = c.headDim
        let emb = w("token_embd.weight")
        var x = Array(emb[Int(token) * D ..< (Int(token) + 1) * D])
        for l in 0..<c.layers {
            let p = "blk.\(l)."
            let a = rmsNorm(x[...], w(p + "attn_norm.weight")[...])
            var q = matvec(p + "attn_q.weight", cols: D, rows: H * hd, a)
            var k = matvec(p + "attn_k.weight", cols: D, rows: KV * hd, a)
            let v = matvec(p + "attn_v.weight", cols: D, rows: KV * hd, a)
            let qn = w(p + "attn_q_norm.weight"), kn = w(p + "attn_k_norm.weight")
            if c.qkNorm == .perHead {
                for h in 0..<H { q.replaceSubrange(h * hd ..< (h + 1) * hd, with: rmsNorm(q[h * hd ..< (h + 1) * hd], qn[...])) }
                for g in 0..<KV { k.replaceSubrange(g * hd ..< (g + 1) * hd, with: rmsNorm(k[g * hd ..< (g + 1) * hd], kn[...])) }
            } else {
                q = rmsNorm(q[...], qn[...])
                k = rmsNorm(k[...], kn[...])
            }
            for h in 0..<H { rope(&q, at: h * hd, pos: pos) }
            for g in 0..<KV { rope(&k, at: g * hd, pos: pos) }
            keys[l].append(k)
            values[l].append(v)
            var ctx = [Float](repeating: 0, count: H * hd)
            for h in 0..<H {
                let g = h / (H / KV)
                var sc: [Float] = []
                for u in 0...pos {
                    var d: Float = 0
                    for i in 0..<hd { d += q[h * hd + i] * keys[l][u][g * hd + i] }
                    sc.append(d / sqrtf(Float(hd)))
                }
                let pr = softmax(sc)
                for u in 0...pos { for i in 0..<hd { ctx[h * hd + i] += pr[u] * values[l][u][g * hd + i] } }
            }
            let o = matvec(p + "attn_output.weight", cols: H * hd, rows: D, ctx)
            for i in 0..<D { x[i] += o[i] }

            let f = rmsNorm(x[...], w(p + "ffn_norm.weight")[...])
            var y = [Float](repeating: 0, count: D)
            if c.isMoE {
                let E = c.experts, K = c.expertsUsed, I = c.expertFFNDim
                let pr = softmax(matvec(p + "ffn_gate_inp.weight", cols: D, rows: E, f))
                var taken = Set<Int>(), ids: [Int] = [], ws: [Float] = []
                for _ in 0..<K {
                    var best = -1
                    var bv: Float = -1e30
                    for e in 0..<E where !taken.contains(e) && pr[e] > bv { bv = pr[e]; best = e }
                    taken.insert(best); ids.append(best); ws.append(bv)
                }
                if c.normTopK { let s = ws.reduce(0, +); ws = ws.map { $0 / s } }
                let gw = w(p + "ffn_gate_exps.weight"), uw = w(p + "ffn_up_exps.weight"), dw = w(p + "ffn_down_exps.weight")
                for (e, wk) in zip(ids, ws) {
                    let gg = matvec(gw[e * I * D ..< (e + 1) * I * D], cols: D, rows: I, f)
                    let uu = matvec(uw[e * I * D ..< (e + 1) * I * D], cols: D, rows: I, f)
                    let hmid = zip(gg, uu).map { silu($0) * $1 }
                    let dd = matvec(dw[e * D * I ..< (e + 1) * D * I], cols: I, rows: D, hmid)
                    for i in 0..<D { y[i] += wk * dd[i] }
                }
            } else {
                let F = c.ffnDim
                let gg = matvec(p + "ffn_gate.weight", cols: D, rows: F, f)
                let uu = matvec(p + "ffn_up.weight", cols: D, rows: F, f)
                y = matvec(p + "ffn_down.weight", cols: F, rows: D, zip(gg, uu).map { silu($0) * $1 })
            }
            for i in 0..<D { x[i] += y[i] }
        }
        let xn = rmsNorm(x[...], w("output_norm.weight")[...])
        return matvec(c.tiedEmbeddings ? "token_embd.weight" : "output.weight", cols: D, rows: c.vocab, xn)
    }
}

func maxRelDiff(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var m: Float = 0
    for (x, y) in zip(a, b) { m = max(m, abs(x - y) / max(1, abs(y))) }
    return m
}

func argmax(_ a: [Float]) -> Int32 {
    var best = 0
    for i in a.indices where a[i] > a[best] { best = i }
    return Int32(best)
}

// MARK: - Tests

@Suite struct ModelTests {
    static let prompt: [Int32] = [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100]  // "Hello, world"

    static let referenceCases: [TinySpec] = [
        TinySpec(arch: "qwen3"),
        TinySpec(arch: "qwen3", tied: true),
        TinySpec(arch: "qwen3moe"),
        TinySpec(arch: "olmoe"),
        TinySpec(arch: "olmoe", normTopK: true),
    ]

    @Test(arguments: referenceCases)
    func matchesReferenceForward(_ spec: TinySpec) throws {
        try withTinyModel(spec) { path in
            let m = try loadTiny(path)
            let steps = 6
            let got = try greedyLogits(m, prompt: Self.prompt, steps: steps)
            let ref = try ReferenceModel(path: path)
            var expected: [[Float]] = []
            var last: [Float] = []
            for (i, t) in Self.prompt.enumerated() { last = ref.step(t, pos: i) }
            expected.append(last)
            for s in 1..<steps {
                last = ref.step(argmax(last), pos: Self.prompt.count + s - 1)
                expected.append(last)
            }
            #expect(got.count == steps)
            for s in 0..<steps {
                let d = maxRelDiff(got[s], expected[s])
                #expect(d < 1e-4, "\(spec.arch) step \(s): max relative logit difference \(d)")
                #expect(argmax(got[s]) == argmax(expected[s]))
            }
        }
    }

    @Test func configReadsFamilyDifferences() throws {
        try withTinyModel(TinySpec(arch: "qwen3moe")) { path in
            let c = try Qwen3MoE.config(from: GGUFFile(path: path))
            #expect(c.qkNorm == .perHead && c.normTopK && c.headDim == 32 && c.heads * c.headDim != c.hidden)
            #expect(c.experts == 8 && c.expertsUsed == 2 && c.expertFFNDim == 32 && !c.tiedEmbeddings)
            #expect(c.rmsEps == Float(1e-6) && c.ropeTheta == 1_000_000)
        }
        try withTinyModel(TinySpec(arch: "olmoe")) { path in
            let c = try OLMoE.config(from: GGUFFile(path: path))
            #expect(c.qkNorm == .wholeVector && !c.normTopK && c.headDim == 16)
            #expect(c.rmsEps == Float(1e-5) && c.ropeTheta == 10000 && c.vocab == 259)
        }
        try withTinyModel(TinySpec(arch: "qwen3", tied: true)) { path in
            let c = try Qwen3.config(from: GGUFFile(path: path))
            #expect(c.tiedEmbeddings && c.ffnDim == 96 && !c.isMoE && c.kvHeads == 2)
        }
    }

    @Test func refusesSigmoidRouting() throws {
        try withTinyModel(TinySpec(arch: "qwen3moe", gatingFunc: 2)) { path in
            #expect(throws: ColibriError.self) { try loadTiny(path) }
        }
    }

    @Test func refusesTypesTheBackendLacks() throws {
        final class NoQ8: Backend {
            let cpu = CPUBackend(threads: 1)
            var capabilities: BackendCapabilities {
                var c = cpu.capabilities
                c.weightTypes.remove(.q8_0)
                return c
            }
            func register(_ region: MemoryRegion) throws {}
            func makeBuffer(_ dtype: ScalarType, count: Int, label: String) throws -> DeviceBuffer { try cpu.makeBuffer(dtype, count: count, label: label) }
            func makeKVCache(_ layout: KVCacheLayout) throws -> KVCache { try cpu.makeKVCache(layout) }
            func beginBatch(label: String) {}
            func commitBatch() {}
            func synchronize() throws {}
            func makeSignal() -> HostSignal { cpu.makeSignal() }
            func encodeWait(for signal: HostSignal, value: UInt64) {}
            func embed(table: WeightView, tokens: [Int32], into out: BufferSlice) {}
            func matmul(_ w: WeightView, _ x: BufferSlice, into y: BufferSlice, batch: Int) {}
            func matmulExperts(_ w: StackedWeights, indices: BufferSlice, rows: Int, perRow: Int, _ x: BufferSlice, sharedInput: Bool, into y: BufferSlice) {}
            func rmsNorm(_ x: BufferSlice, weight: WeightView, into out: BufferSlice, rows: Int, dim: Int, eps: Float) {}
            func ropeNeox(_ x: BufferSlice, _ params: RopeParams) {}
            func storeKV(k: BufferSlice, v: BufferSlice, cache: KVCache, layer: Int, positionStart: Int, rows: Int) {}
            func attention(q: BufferSlice, cache: KVCache, layer: Int, _ params: AttentionParams, into out: BufferSlice) {}
            func softmax(_ x: BufferSlice, rows: Int, cols: Int) {}
            func topK(_ scores: BufferSlice, rows: Int, cols: Int, k: Int, renormalize: Bool, ids: BufferSlice, weights: BufferSlice) {}
            func siluMul(gate: BufferSlice, up: BufferSlice, into out: BufferSlice, count: Int) {}
            func add(_ a: BufferSlice, _ b: BufferSlice, into out: BufferSlice, count: Int) {}
            func weightedSum(_ x: BufferSlice, weights: BufferSlice, rows: Int, terms: Int, dim: Int, into out: BufferSlice, accumulate: Bool) {}
            func copy(_ src: BufferSlice, into dst: BufferSlice, count: Int) {}
        }
        try withTinyModel(TinySpec(arch: "qwen3", type: .q8_0)) { path in
            #expect(throws: ColibriError.self) { try loadTiny(path, backend: NoQ8()) }
        }
    }

    @Test(arguments: ["qwen3", "qwen3moe", "olmoe"])
    func prefillChunkingDoesNotChangeLogits(_ arch: String) throws {
        try withTinyModel(TinySpec(arch: arch, type: .q8_0)) { path in
            let whole = try greedyLogits(loadTiny(path, maxBatch: 64), prompt: Self.prompt, steps: 3)
            for batch in [1, 5] {
                let chunked = try greedyLogits(loadTiny(path, maxBatch: batch), prompt: Self.prompt, steps: 3)
                for s in 0..<3 {
                    #expect(maxRelDiff(chunked[s], whole[s]) < 1e-5, "\(arch) batch \(batch) step \(s)")
                }
            }
        }
    }

    /// Placement never changes the math: streamed experts through a minimal slot cache
    /// (evicting constantly, prefill split into groups) give byte-identical logits.
    @Test(arguments: ["qwen3moe", "olmoe"])
    func streamingIsByteIdenticalToResident(_ arch: String) throws {
        try withTinyModel(TinySpec(arch: arch, type: .q8_0)) { path in
            let resident = try loadTiny(path, maxBatch: 8)
            #expect((resident.model as? Qwen3MoE)?.expertStore?.isFullyResident ?? (resident.model as! OLMoE).expertStore!.isFullyResident)
            let expected = try greedyLogits(resident, prompt: Self.prompt, steps: 10)
            for slots in [2, 3, 8] {
                let opts = ExpertStoreOptions(mode: .stream, slotsPerLayer: slots, ioThreads: 2, saveHistory: false, uncached: false)
                let streamed = try loadTiny(path, maxBatch: 8, options: opts)
                let store = (streamed.model as? Qwen3MoE)?.expertStore ?? (streamed.model as! OLMoE).expertStore!
                #expect(!store.isFullyResident)
                let got = try greedyLogits(streamed, prompt: Self.prompt, steps: 10)
                #expect(got == expected, "\(arch) with \(slots) slots per layer")
                if slots < 8 { #expect(store.stats.evictions > 0) }
            }
        }
    }

    @Test func routingStaysOnTheDeviceAndIsRecorded() throws {
        try withTinyModel(TinySpec(arch: "qwen3moe")) { path in
            let m = try loadTiny(path)
            let model = m.model as! Qwen3MoE
            let g = try Generator(loaded: m, sampler: .greedy)
            _ = try g.generate(prompt: Self.prompt, maxTokens: 1) { _ in true }
            let routing = model.stack.lastRouting(rows: Self.prompt.count, state: g.state)
            #expect(routing.count == 2)
            for ids in routing {
                #expect(ids.count == Self.prompt.count * 2)
                for r in 0..<Self.prompt.count { #expect(ids[2 * r] != ids[2 * r + 1]) }
                #expect(ids.allSatisfy { $0 >= 0 && $0 < 8 })
            }
            let store = model.expertStore as! ResidentExpertStore
            #expect(store.history!.total == UInt64(2 * 2 * Self.prompt.count))
        }
    }

    /// A chat turn re-renders the whole conversation; the generator keeps the cached
    /// prefix and must give the same logits as a fresh run, also after a divergence.
    @Test func chatReusesTheCachedPrefix() throws {
        try withTinyModel(TinySpec(arch: "qwen3", type: .q8_0)) { path in
            let m = try loadTiny(path, maxBatch: 4)
            let a: [Int32] = Array(Self.prompt.prefix(7))
            let full: [Int32] = Self.prompt + [33, 10]
            let fork: [Int32] = Array(Self.prompt.prefix(5)) + [90, 91, 92]

            func fresh(_ seq: [Int32]) throws -> [Float] { try greedyLogits(m, prompt: seq, steps: 1)[0] }

            let g = try Generator(loaded: m, sampler: .greedy)
            var first: [Float] = []
            g.onLogits = { s, l in if s == 0 { first = Array(l) } }
            var st = try g.generate(sequence: a, maxTokens: 3) { _ in true }
            #expect(st.reusedTokens == 0 && st.promptTokens == 7)
            #expect(g.context.count == 9)  // prompt + 2 fed-back tokens; the third is not cached

            st = try g.generate(sequence: full, maxTokens: 2) { _ in true }
            #expect(st.reusedTokens == 7 && st.promptTokens == full.count - 7)
            #expect(maxRelDiff(first, try fresh(full)) < 1e-5)

            st = try g.generate(sequence: fork, maxTokens: 2) { _ in true }
            #expect(st.reusedTokens == 5 && st.promptTokens == 3)
            #expect(maxRelDiff(first, try fresh(fork)) < 1e-5)

            // The whole sequence already cached: the last token is recomputed.
            let cached = Array(g.context.prefix(6))
            st = try g.generate(sequence: cached, maxTokens: 1) { _ in true }
            #expect(st.reusedTokens == 5 && st.promptTokens == 1)
            #expect(maxRelDiff(first, try fresh(cached)) < 1e-5)
        }
    }

    @Test func stopsOnStopTokensAndLimits() throws {
        try withTinyModel(TinySpec(arch: "qwen3")) { path in
            let m = try loadTiny(path)
            #expect(m.stopTokens.contains(258))  // <|im_end|>
            #expect(m.stopTokens.contains(256))  // <|endoftext|>
            let g = try Generator(loaded: m, sampler: .greedy)
            var ids: [Int32] = []
            let st = try g.generate(prompt: Self.prompt, maxTokens: 5) { ids.append($0); return true }
            #expect(ids.count == st.generatedTokens && ids.count <= 5)
            if ids.count < 5 { #expect(m.stopTokens.contains(ids.last!)) }
            // Forcing a stop id: callback returning false ends at once.
            g.reset()
            var n = 0
            _ = try g.generate(prompt: Self.prompt, maxTokens: 50) { _ in n += 1; return n < 3 }
            #expect(n == 3)
        }
        try withTinyModel(TinySpec(arch: "olmoe")) { path in
            let m = try loadTiny(path)
            #expect(m.stopTokens.isSuperset(of: [256, 257, 258]))  // IP_ADDRESS + control tokens
        }
    }

    @Test func refusesContextOverflow() throws {
        try withTinyModel(TinySpec(arch: "qwen3")) { path in
            let m = try loadTiny(path)
            let g = try Generator(loaded: m, sampler: .greedy)
            #expect(throws: ColibriError.self) {
                _ = try g.generate(prompt: [Int32](repeating: 65, count: 129), maxTokens: 1) { _ in true }
            }
            // Generation stops at the cache's end instead of overflowing.
            g.reset()
            let st = try g.generate(prompt: [Int32](repeating: 65, count: 120), maxTokens: 50, ignoreStop: true) { _ in true }
            #expect(g.state.position == 128 && st.generatedTokens == 9)
        }
    }

    @Test func registryRefusesUnknownArchitectures() throws {
        var w = GGUFWriter()
        w.set("general.architecture", .string("llama"))
        try w.addTensor("token_embd.weight", dims: [32, 4], type: .f32, values: [Float](repeating: 0, count: 128))
        let path = NSTemporaryDirectory() + "colibri-model-llama-\(UUID().uuidString).gguf"
        try w.write(to: path)
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: ColibriError.self) { try loadTiny(path) }
    }

    @Test func topKHelperOrdersAndBreaksTies() {
        let l: [Float] = [1, 5, 3, 5, -2, 4]
        let t = l.withUnsafeBufferPointer { Generator.top($0, 3) }
        #expect(t.map(\.id) == [1, 3, 5])
    }
}

// MARK: - The real model against upstream colibri

let modelTestsQwen3Path = NSString(string: "~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf").expandingTildeInPath
let modelTestsHaveQwen3 = FileManager.default.fileExists(atPath: modelTestsQwen3Path)

@Suite struct ModelRealQwen3Tests {
    /// docs/port/models.md sections 6 and 10: the golden ChatML prompt, upstream's greedy
    /// continuation (`qwen36` oracle mode) and its GQ_EXACT=1 prefill top-3 logits.
    @Test(.enabled(if: modelTestsHaveQwen3, "Qwen3-0.6B-Q8_0.gguf is not on disk"))
    func goldenPromptMatchesUpstream() throws {
        let m = try ModelRegistry.load(path: modelTestsQwen3Path,
                                       context: LoadContext(backend: CPUBackend(), maxContext: 64, kvType: .f32, maxBatch: 32))
        let text = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let prompt = m.tokenizer.encode(text)
        #expect(prompt == [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271])

        let g = try Generator(loaded: m, sampler: .greedy)
        var top: [(id: Int32, logit: Float)] = []
        g.traceTopK = 3
        g.onTrace = { t in if t.step == 0 { top = t.top } }
        var ids: [Int32] = []
        // Debug builds are slow; four tokens cover prefill and decode.
        _ = try g.generate(prompt: prompt, maxTokens: 4, ignoreStop: true) { ids.append($0); return true }
        #expect(ids == [785, 6722, 315, 9625])  // "The capital of France"
        #expect(top.map(\.id) == [785, 49000, 63593])
        for (got, want) in zip(top.map(\.logit), [30.2135, 23.4182, 19.7609] as [Float]) {
            #expect(abs(got - want) < 2e-3, "prefill logit \(got) vs upstream GQ_EXACT \(want)")
        }
    }
}
