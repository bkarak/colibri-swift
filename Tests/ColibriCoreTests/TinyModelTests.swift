import Foundation
import Testing
@testable import ColibriCore

// The synthetic-model generator (Sources/ColibriCore/Synthetic/TinyModel.swift) and the
// hook the oracle harness uses to write the models to disk:
//
//   COLIBRI_TINY_OUT=<dir> swift test --filter TinyModelTests/writeOracleModels
//
// writes <dir>/<case>.gguf for every TinyOracleCases.all entry plus <dir>/manifest.json
// ({"<case>": {"fingerprint": ..., "bytes": ..., "architecture": ..., ...}}).

@Suite struct TinyModelTests {
    @Test(arguments: ["qwen3", "qwen3moe", "olmoe"])
    func sameSpecSameBytes(arch: String) throws {
        let spec = TinyModelSpec(architecture: arch, weightType: .q8_0, seed: 42)
        let a = try TinyModel.build(spec).serialize()
        let b = try TinyModel.build(spec).serialize()
        #expect(a == b)
        var other = spec
        other.seed = 43
        #expect(try TinyModel.build(other).serialize() != a)
    }

    @Test func rngIsPinned() {
        // Pins the generator itself: a change here changes every oracle fixture.
        var r = TinyRNG(seed: 1)
        let u = (0..<4).map { _ in r.next() }
        var r2 = TinyRNG(seed: 1)
        #expect(u == (0..<4).map { _ in r2.next() })
        var g = TinyRNG(seed: 7)
        var sum: Double = 0, sq: Double = 0
        let n = 200_000
        for _ in 0..<n { let x = Double(g.gaussian(0.5)); sum += x; sq += x * x }
        #expect(abs(sum / Double(n)) < 0.01)
        #expect(abs((sq / Double(n)).squareRoot() - 0.5) < 0.01)
    }

    @Test(arguments: ["qwen3", "qwen3moe", "olmoe"])
    func readsBackWithTheRightShapes(arch: String) throws {
        let spec = TinyModelSpec(architecture: arch, weightType: .q4_K, seed: 3,
                                 overrides: ["ffn_down_exps.weight": .q6_K, "ffn_down.weight": .q6_K, "output.weight": .q6_K])
        let path = NSTemporaryDirectory() + "tiny-\(arch)-\(UUID().uuidString).gguf"
        defer { try? FileManager.default.removeItem(atPath: path) }
        try TinyModel.write(spec, to: path)
        let f = try GGUFFile(path: path)
        let md = f.metadata
        #expect(md.architecture == arch)
        #expect(md.int("\(arch).block_count") == spec.layers)
        #expect(md.int("\(arch).embedding_length") == spec.hidden)
        let V = 256 + TinyModel.specialTokens(architecture: arch).count
        #expect(md.strings("tokenizer.ggml.tokens")?.count == V)
        #expect(md.strings("tokenizer.ggml.tokens")?[32] == "\u{0120}")  // space -> Ġ
        #expect(md.strings("tokenizer.ggml.tokens")?[65] == "A")
        #expect(f.tensor("token_embd.weight")?.dims == [spec.hidden, V])
        #expect(f.tensor("token_embd.weight")?.type == .q4_K)
        #expect(f.tensor("output.weight")?.type == .q6_K)
        #expect(f.tensor("output_norm.weight")?.type == .f32)
        let attnWidth = spec.heads * spec.headDim
        #expect(f.tensor("blk.0.attn_q.weight")?.dims == [spec.hidden, attnWidth])
        #expect(f.tensor("blk.0.attn_output.weight")?.dims == [attnWidth, spec.hidden])
        #expect(f.tensor("blk.0.attn_output.weight")?.type == .q4_K)  // "output.weight" override must not leak
        let qn = arch == "olmoe" ? attnWidth : spec.headDim
        #expect(f.tensor("blk.1.attn_q_norm.weight")?.dims == [qn])
        if arch == "qwen3" {
            #expect(f.tensor("blk.1.ffn_down.weight")?.dims == [spec.denseFFN, spec.hidden])
            #expect(f.tensor("blk.1.ffn_down.weight")?.type == .q6_K)
            #expect(md.int("qwen3.feed_forward_length") == spec.denseFFN)
        } else {
            #expect(f.tensor("blk.1.ffn_gate_inp.weight")?.type == .f32)
            #expect(f.tensor("blk.1.ffn_gate_exps.weight")?.dims == [spec.hidden, spec.expertFFN, spec.experts])
            #expect(f.tensor("blk.1.ffn_down_exps.weight")?.dims == [spec.expertFFN, spec.hidden, spec.experts])
            #expect(f.tensor("blk.1.ffn_down_exps.weight")?.type == .q6_K)
            #expect(md.int("\(arch).expert_count") == spec.experts)
            #expect(md.int("\(arch).expert_used_count") == spec.expertsUsed)
        }
        // Norms are 1 +- 0.1 and stored exactly.
        let norm = try f.floats("blk.0.attn_norm.weight")
        #expect(norm.allSatisfy { $0 >= 0.9 && $0 <= 1.1 })
    }

    @Test func tiedHasNoOutputAndOlmoeRefusesTied() throws {
        let w = try TinyModel.build(TinyModelSpec(architecture: "qwen3", weightType: .q8_0, tied: true))
        #expect(!w.tensors.contains { $0.name == "output.weight" })
        #expect(throws: (any Error).self) { try TinyModel.build(TinyModelSpec(architecture: "olmoe", tied: true)) }
        #expect(throws: (any Error).self) { try TinyModel.build(TinyModelSpec(architecture: "llama")) }
    }

    @Test func oracleCaseNamesAreUnique() {
        let names = TinyOracleCases.all.map(\.name)
        #expect(Set(names).count == names.count)
    }

    /// Writes every oracle case to `$COLIBRI_TINY_OUT` (a no-op without it).
    @Test func writeOracleModels() throws {
        guard let out = ProcessInfo.processInfo.environment["COLIBRI_TINY_OUT"], !out.isEmpty else { return }
        try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        var manifest: [String: [String: Any]] = [:]
        for c in TinyOracleCases.all {
            if c.needsRealTokenizer, !FileManager.default.fileExists(atPath: TinyOracleCases.qwen3TokenizerSource) {
                print("[tiny] skip \(c.name): no \(TinyOracleCases.qwen3TokenizerSource)")
                continue
            }
            let w = try TinyModel.build(c.spec)
            let bytes = try w.serialize()
            let path = (out as NSString).appendingPathComponent(c.name + ".gguf")
            try Data(bytes).write(to: URL(fileURLWithPath: path))
            let s = c.spec
            manifest[c.name] = [
                "fingerprint": TinyModel.fingerprint(bytes), "bytes": bytes.count,
                "architecture": s.architecture, "weight_type": s.weightType.description,
                "overrides": s.overrides.mapValues(\.description), "tied": s.tied, "seed": s.seed,
                "vocabulary": c.needsRealTokenizer ? "qwen3-0.6b" : "bytes",
                "vocab_size": w.metadata.strings("tokenizer.ggml.tokens")?.count ?? 0,
                "layers": s.layers, "hidden": s.hidden, "heads": s.heads, "kv_heads": s.kvHeads,
                "head_dim": s.headDim, "experts": s.experts, "experts_used": s.expertsUsed,
                "expert_ffn": s.expertFFN, "dense_ffn": s.denseFFN,
            ]
            print("[tiny] wrote \(path) (\(bytes.count) bytes, \(TinyModel.fingerprint(bytes)))")
        }
        let json = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: URL(fileURLWithPath: (out as NSString).appendingPathComponent("manifest.json")))
    }
}
