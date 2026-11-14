import ColibriCore
import Foundation
import Metal
import Testing
import ColibriMetal

// Whole transformer layers (attention + dense FFN, attention + routed MoE) encoded into
// ONE batch per step on Metal, compared with the CPU backend running the same encoding.
// This is the execution model the forward passes rely on: ops see each other's writes
// in program order with no host round trip inside a batch.

/// Tensors packed into one region at 32-byte aligned offsets, like a GGUF data section.
struct PackedWeights {
    private(set) var bytes: [UInt8] = []
    private var pending: [(name: String, offset: Int, type: GGMLType, cols: Int, rows: Int, count: Int, stride: Int)] = []

    mutating func add(_ name: String, _ values: [Float], type: GGMLType, cols: Int, rows: Int, count: Int = 1) throws {
        let offset = alignUp(bytes.count, 32)
        bytes += [UInt8](repeating: 0, count: offset - bytes.count)
        bytes += try Quantize.encode(values, as: type)
        pending.append((name, offset, type, cols, rows, count, type.rowBytes(cols) * rows))
    }

    func build() -> (MemoryRegion, [String: WeightView], [String: StackedWeights]) {
        let region = makeRegion(bytes, label: "layer.weights")
        var views: [String: WeightView] = [:], stacks: [String: StackedWeights] = [:]
        for t in pending {
            if t.count == 1 {
                views[t.name] = WeightView(region: region, offset: t.offset, type: t.type, cols: t.cols, rows: t.rows)
            } else {
                stacks[t.name] = StackedWeights(region: region, baseOffset: t.offset, stride: t.stride, count: t.count,
                                                type: t.type, cols: t.cols, rows: t.rows)
            }
        }
        return (region, views, stacks)
    }
}

@Suite(.enabled(if: metalAvailable)) struct MetalLayerTests {
    static let D = 256, H = 4, KV = 2, hd = 64, F = 512, E = 8, K = 2, vocab = 50

    static func weights(dense: GGMLType, experts: GGMLType, seed: UInt64) throws -> (MemoryRegion, [String: WeightView], [String: StackedWeights]) {
        var rng = TestRNG(seed)
        var p = PackedWeights()
        let D = Self.D, H = Self.H, KV = Self.KV, hd = Self.hd, F = Self.F, E = Self.E
        func m(_ n: Int, _ fanIn: Int) -> [Float] { rng.floats(n, scale: 1.7 / Float(fanIn).squareRoot()) }
        try p.add("tok", rng.floats(vocab * D), type: dense, cols: D, rows: vocab)
        try p.add("attn_norm", rng.floats(D).map { 1 + 0.2 * $0 }, type: .f32, cols: D, rows: 1)
        try p.add("q", m(H * hd * D, D), type: dense, cols: D, rows: H * hd)
        try p.add("k", m(KV * hd * D, D), type: dense, cols: D, rows: KV * hd)
        try p.add("v", m(KV * hd * D, D), type: dense, cols: D, rows: KV * hd)
        try p.add("q_norm", rng.floats(hd).map { 1 + 0.2 * $0 }, type: .f32, cols: hd, rows: 1)
        try p.add("k_norm", rng.floats(hd).map { 1 + 0.2 * $0 }, type: .f32, cols: hd, rows: 1)
        try p.add("o", m(D * H * hd, H * hd), type: dense, cols: H * hd, rows: D)
        try p.add("ffn_norm", rng.floats(D).map { 1 + 0.2 * $0 }, type: .f32, cols: D, rows: 1)
        try p.add("gate", m(F * D, D), type: dense, cols: D, rows: F)
        try p.add("up", m(F * D, D), type: dense, cols: D, rows: F)
        try p.add("down", m(D * F, F), type: dense, cols: F, rows: D)
        try p.add("router", m(E * D, D).map { $0 * 4 }, type: .f32, cols: D, rows: E)
        try p.add("gate_exps", m(E * F * D, D), type: experts, cols: D, rows: F, count: E)
        try p.add("up_exps", m(E * F * D, D), type: experts, cols: D, rows: F, count: E)
        try p.add("down_exps", m(E * D * F, F), type: experts, cols: F, rows: D, count: E)
        return p.build()
    }

    /// Scratch buffers for `rows` tokens on one backend.
    final class Scratch {
        let x, normed, q, k, v, attn, proj, gate, up, logits, eids, ew, egate, eup, edown: DeviceBuffer
        init(_ b: Backend, rows: Int) throws {
            let D = MetalLayerTests.D, H = MetalLayerTests.H, KV = MetalLayerTests.KV, hd = MetalLayerTests.hd
            let F = MetalLayerTests.F, E = MetalLayerTests.E, K = MetalLayerTests.K
            x = try b.zeros(rows * D); normed = try b.zeros(rows * D)
            q = try b.zeros(rows * H * hd); k = try b.zeros(rows * KV * hd); v = try b.zeros(rows * KV * hd)
            attn = try b.zeros(rows * H * hd); proj = try b.zeros(rows * D)
            gate = try b.zeros(rows * F); up = try b.zeros(rows * F)
            logits = try b.zeros(rows * E); eids = try b.zeros(rows * K, .i32); ew = try b.zeros(rows * K)
            egate = try b.zeros(rows * K * F); eup = try b.zeros(rows * K * F); edown = try b.zeros(rows * K * D)
        }
    }

    /// One layer for `tokens` at `pos`, encoded as a single batch. Returns the residual.
    static func encodeStep(_ b: Backend, _ w: [String: WeightView], _ st: [String: StackedWeights], cache: KVCache,
                           tokens: [Int32], pos: Int, moe: Bool) throws -> Scratch {
        let S = tokens.count
        let s = try Scratch(b, rows: S)
        b.beginBatch(label: "layer pos \(pos)")
        b.embed(table: w["tok"]!, tokens: tokens, into: s.x.slice())
        // attention
        b.rmsNorm(s.x.slice(), weight: w["attn_norm"]!, into: s.normed.slice(), rows: S, dim: D, eps: 1e-6)
        b.matmul(w["q"]!, s.normed.slice(), into: s.q.slice(), batch: S)
        b.matmul(w["k"]!, s.normed.slice(), into: s.k.slice(), batch: S)
        b.matmul(w["v"]!, s.normed.slice(), into: s.v.slice(), batch: S)
        b.rmsNorm(s.q.slice(), weight: w["q_norm"]!, into: s.q.slice(), rows: S * H, dim: hd, eps: 1e-6)
        b.rmsNorm(s.k.slice(), weight: w["k_norm"]!, into: s.k.slice(), rows: S * KV, dim: hd, eps: 1e-6)
        b.ropeNeox(s.q.slice(), RopeParams(rows: S, heads: H, headDim: hd, ropeDims: hd, theta: 1e6, positionStart: pos))
        b.ropeNeox(s.k.slice(), RopeParams(rows: S, heads: KV, headDim: hd, ropeDims: hd, theta: 1e6, positionStart: pos))
        b.storeKV(k: s.k.slice(), v: s.v.slice(), cache: cache, layer: 0, positionStart: pos, rows: S)
        b.attention(q: s.q.slice(), cache: cache, layer: 0,
                    AttentionParams(rows: S, heads: H, kvHeads: KV, headDim: hd, valueDim: hd,
                                    scale: 1 / Float(hd).squareRoot(), positionStart: pos), into: s.attn.slice())
        b.matmul(w["o"]!, s.attn.slice(), into: s.proj.slice(), batch: S)
        b.add(s.x.slice(), s.proj.slice(), into: s.x.slice(), count: S * D)
        // FFN
        b.rmsNorm(s.x.slice(), weight: w["ffn_norm"]!, into: s.normed.slice(), rows: S, dim: D, eps: 1e-6)
        if moe {
            b.matmul(w["router"]!, s.normed.slice(), into: s.logits.slice(), batch: S)
            b.softmax(s.logits.slice(), rows: S, cols: E)
            b.topK(s.logits.slice(), rows: S, cols: E, k: K, renormalize: true, ids: s.eids.slice(), weights: s.ew.slice())
            b.matmulExperts(st["gate_exps"]!, indices: s.eids.slice(), rows: S, perRow: K, s.normed.slice(), sharedInput: true, into: s.egate.slice())
            b.matmulExperts(st["up_exps"]!, indices: s.eids.slice(), rows: S, perRow: K, s.normed.slice(), sharedInput: true, into: s.eup.slice())
            b.siluMul(gate: s.egate.slice(), up: s.eup.slice(), into: s.egate.slice(), count: S * K * F)
            b.matmulExperts(st["down_exps"]!, indices: s.eids.slice(), rows: S, perRow: K, s.egate.slice(), sharedInput: false, into: s.edown.slice())
            b.weightedSum(s.edown.slice(), weights: s.ew.slice(), rows: S, terms: K, dim: D, into: s.x.slice(), accumulate: true)
        } else {
            b.matmulPair(w["gate"]!, w["up"]!, s.normed.slice(), into: s.gate.slice(), s.up.slice(), batch: S)
            b.siluMul(gate: s.gate.slice(), up: s.up.slice(), into: s.gate.slice(), count: S * F)
            b.matmul(w["down"]!, s.gate.slice(), into: s.proj.slice(), batch: S)
            b.add(s.x.slice(), s.proj.slice(), into: s.x.slice(), count: S * D)
        }
        b.commitBatch()
        return s
    }

    @Test(arguments: [(GGMLType.q8_0, GGMLType.q8_0, false), (.q4_K, .q6_K, false), (.q8_0, .q8_0, true), (.q5_K, .q4_K, true)])
    func layerInOneBatchMatchesCPU(_ dense: GGMLType, _ experts: GGMLType, _ moe: Bool) throws {
        let twin = try Twin()
        let (_, w, st) = try Self.weights(dense: dense, experts: experts, seed: 42)
        let layout = KVCacheLayout(layers: 1, kvHeads: Self.KV, keyDim: Self.hd, valueDim: Self.hd, maxTokens: 64, dtype: .f16)
        let prompt: [Int32] = [3, 17, 4, 49, 0, 22, 8]
        let before = twin.metal.batchesCommitted
        let r = try twin.both { b -> () -> ([Float], [Int32], [Float], [Int32]) in
            let cache = try b.makeKVCache(layout)
            let pre = try Self.encodeStep(b, w, st, cache: cache, tokens: prompt, pos: 0, moe: moe)
            try b.synchronize()
            let dec = try Self.encodeStep(b, w, st, cache: cache, tokens: [11], pos: prompt.count, moe: moe)
            return { (pre.x.f, pre.eids.i, dec.x.f, dec.eids.i) }
        }
        #expect(twin.metal.batchesCommitted - before == 2, "one command buffer per step")
        let what = "\(moe ? "moe" : "dense") \(dense)/\(experts)"
        if moe {
            #expect(r.cpu.1 == r.metal.1, "\(what): prefill routing differs")
            #expect(r.cpu.3 == r.metal.3, "\(what): decode routing differs")
        }
        let scale = r.cpu.0.map { _ in Float(1) }
        expectClose(r.cpu.0, r.metal.0, rel: 0, abs: 2e-4, scale: scale, "\(what) prefill residual")
        expectClose(r.cpu.2, r.metal.2, rel: 0, abs: 2e-4, scale: [Float](repeating: 1, count: r.cpu.2.count), "\(what) decode residual")
    }
}
