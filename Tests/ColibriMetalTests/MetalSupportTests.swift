import ColibriCore
import Foundation
import Metal
import Testing
import ColibriMetal

// Load-time limits of the Metal backend (they must throw, since release builds remove
// preconditions), the expert read-failure path on Metal, and the decoders against
// upstream's own golden blocks (not only bytes from the Swift encoder).

@Suite(.enabled(if: metalAvailable)) struct MetalSupportTests {
    static func tinyQwen3(headDim: Int) throws -> String {
        var spec = TinyModelSpec(architecture: "qwen3", weightType: .q8_0, seed: 5)
        spec.headDim = headDim
        let path = NSTemporaryDirectory() + "colibri-metal-hd\(headDim)-\(UUID().uuidString).gguf"
        try TinyModel.write(spec, to: path)
        return path
    }

    @Test func wideHeadsAreRefusedAtLoad() throws {
        let path = try Self.tinyQwen3(headDim: 512)
        defer { try? FileManager.default.removeItem(atPath: path) }
        do {
            _ = try ModelRegistry.load(path: path, context: LoadContext(backend: try MetalBackend(), maxContext: 32))
            Issue.record("Metal loaded a head dim of 512")
        } catch let ColibriError.unsupported(msg) {
            #expect(msg.contains("head dim 512"), "\(msg)")
        }
    }

    @Test func supportedShapesLoadAndRun() throws {
        let path = try Self.tinyQwen3(headDim: 128)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let m = try ModelRegistry.load(path: path, context: LoadContext(backend: try MetalBackend(), maxContext: 32))
        let logits = try m.model.forward([10, 20, 30, 40], state: try m.model.makeState())
        #expect(logits.floats.allSatisfy { $0.isFinite })
    }

    @Test func checkSupportLimits() throws {
        let metal = try MetalBackend()
        func base() -> ModelRequirements {
            var r = ModelRequirements()
            r.matrices = [.init(name: "blk.0.attn_q.weight", type: .q8_0, cols: 256, rows: 256, offset: 32)]
            r.headDim = 128; r.valueDim = 128; r.heads = 4; r.kvHeads = 2; r.ropeDims = 128
            return r
        }
        try metal.checkSupport(base())

        var r = base(); r.matrices[0].cols = 48; r.matrices[0].type = .f16
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
        r = base(); r.matrices[0].offset = 33
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
        r = base(); r.matrices[0].type = .f32; r.matrices[0].offset = 34
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
        r = base(); r.headDim = 130
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
        r = base(); r.valueDim = 512
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
        r = base(); r.routerExperts = 4096; r.expertsUsed = 8
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
        r = base(); r.routerExperts = 2048; r.expertsUsed = 8
        try metal.checkSupport(r)
        r = base(); r.matrices[0].type = .iq2_xxs
        #expect(throws: ColibriError.self) { try metal.checkSupport(r) }
    }

    @Test func failedExpertReadFailsTheForward() throws {
        try metalExpertReadFailureCheck()
    }

    /// Upstream's golden blocks (Tests/ColibriCoreTests/Fixtures/quant) through the Metal
    /// embed and decode matvec. Embed is bit-exact except that Metal's f32 arithmetic
    /// flushes subnormals to zero (only BF16 can decode to one); the matvec agrees with
    /// upstream's float-path `gq_matmul_f32` up to summation order.
    @Test(arguments: goldenFixtureTypes)
    func upstreamGoldenBlocks(_ type: GGMLType) throws {
        let g = try GoldenBlocks.load(type)
        let metal = try MetalBackend()
        let region = makeRegion(g.bytes, leading: 64)
        let table = WeightView(region: region, offset: 64, type: type, cols: g.cols, rows: g.rows)

        let out = try metal.zeros(g.rows * g.cols)
        metal.embed(table: table, tokens: (0..<g.rows).map(Int32.init), into: out.slice())
        try metal.synchronize()
        let got = out.f
        var mismatches = 0, flushed = 0
        for i in 0..<got.count where got[i].bitPattern != g.deq[i].bitPattern {
            if got[i] == 0 && g.deq[i].isSubnormal { flushed += 1 } else { mismatches += 1 }
        }
        #expect(mismatches == 0, "\(type): \(mismatches) of \(got.count) decoded values differ from upstream")
        if type != .bf16 { #expect(flushed == 0) }

        let xb = try metal.upload(g.x), yb = try metal.zeros(g.rows)
        metal.matmul(table, xb.slice(), into: yb.slice(), batch: 1)
        try metal.synchronize()
        var mag = [Float](repeating: 0, count: g.rows)
        for o in 0..<g.rows { for i in 0..<g.cols { mag[o] += abs(g.deq[o * g.cols + i] * g.x[i]) } }
        expectClose(g.yExact, yb.f, rel: 1e-5, abs: 1e-6, scale: mag, "\(type) matvec vs upstream gq_matmul_f32")
    }
}

/// The 14 types with upstream golden files.
let goldenFixtureTypes: [GGMLType] = [.f16, .bf16, .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl,
                                      .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_xs]

/// A golden file of Tests/ColibriCoreTests/Fixtures/quant (layout: gen_golden.c there).
struct GoldenBlocks {
    var cols: Int, rows: Int
    var bytes: [UInt8]
    var deq: [Float]
    var x: [Float]
    var yExact: [Float]

    static func load(_ type: GGMLType) throws -> GoldenBlocks {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("ColibriCoreTests/Fixtures/quant/\(type.description).golden")
        let d = [UInt8](try Data(contentsOf: url))
        var pos = 0
        func u32() -> Int {
            defer { pos += 4 }
            return Int(UInt32(d[pos]) | UInt32(d[pos + 1]) << 8 | UInt32(d[pos + 2]) << 16 | UInt32(d[pos + 3]) << 24)
        }
        func floats(_ n: Int) -> [Float] { (0..<n).map { _ in Float(bitPattern: UInt32(u32())) } }
        let t = u32(), cols = u32(), rows = u32(), nbytes = u32()
        #expect(t == Int(type.rawValue))
        let bytes = Array(d[pos ..< pos + nbytes]); pos += nbytes
        let deq = floats(rows * cols)
        let x = floats(cols)
        _ = floats(rows)  // integer-path result (not used here)
        let ye = floats(rows)
        return GoldenBlocks(cols: cols, rows: rows, bytes: bytes, deq: deq, x: x, yExact: ye)
    }
}

/// The Metal twin of ColibriCoreTests' `expertReadFailureCheck`: a streamed qwen3moe
/// whose last expert matrix was truncated after load must fail the prefill that routes
/// to it, not return logits computed from a stale slot.
func metalExpertReadFailureCheck() throws {
    let spec = TinyModelSpec(architecture: "qwen3moe", weightType: .q8_0, seed: 2)
    let dir = NSTemporaryDirectory() + "colibri-metal-iofail-\(UUID().uuidString)/"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }
    let path = dir + "broken.gguf"
    try TinyModel.write(spec, to: path)
    let file = try GGUFFile(path: path)
    let last = try #require(file.tensors.max { $0.fileOffset + $0.byteCount < $1.fileOffset + $1.byteCount })
    #expect(last.name.hasSuffix("_exps.weight"))

    let opts = ExpertStoreOptions(mode: .stream, slotsPerLayer: spec.experts, ioThreads: 2, saveHistory: false)
    let m = try ModelRegistry.load(path: path, context: LoadContext(backend: try MetalBackend(), expertOptions: opts,
                                                                    maxContext: 64, maxBatch: 64))
    let size = try #require(try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int)
    #expect(truncate(path, off_t(size - 1000)) == 0)
    let prompt: [Int32] = (0..<48).map { Int32(($0 * 37 + 11) % 200) }
    #expect(throws: (any Error).self) { _ = try m.model.forward(prompt, state: try m.model.makeState()) }
}
