import ColibriCore
import Foundation
import Metal
import Testing
import ColibriMetal

let metalQwen3Path = NSString(string: "~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf").expandingTildeInPath
let metalHasQwen3 = FileManager.default.fileExists(atPath: metalQwen3Path)

/// The real Qwen3-0.6B Q8_0 file, wrapped zero-copy: the whole mapping is registered as one
/// MTLBuffer (`bytesNoCopy`) and Metal products of real tensors match the CPU backend.
@Suite(.enabled(if: metalAvailable && metalHasQwen3, "needs Metal and ~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"))
struct MetalRealModelTests {
    @Test func realTensorsMatchCPU() throws {
        let file = try GGUFFile(path: metalQwen3Path)
        let twin = try Twin()
        try twin.metal.register(file.file.region)
        try twin.metal.register(file.file.region)   // idempotent
        var rng = TestRNG(9)
        for name in ["blk.0.ffn_down.weight", "blk.27.attn_q.weight", "blk.13.ffn_gate.weight"] {
            let w = try file.weight(name)
            #expect(w.type == .q8_0)
            let deq = Dequant.dequantize(w.type, bytes: Array(file.data(file.tensor(name)!)), count: w.rows * w.cols)
            for batch in [1, 4] {
                let x = rng.floats(batch * w.cols)
                let (c, m) = try twin.both { b -> () -> [Float] in
                    let xb = try b.upload(x), yb = try b.zeros(batch * w.rows)
                    b.matmul(w, xb.slice(), into: yb.slice(), batch: batch)
                    return { yb.f }
                }
                let mag = MetalBackendTests.magnitudes(deq, rows: w.rows, cols: w.cols, rowOffset: 0, viewRows: w.rows,
                                                       x, batch: batch, scale: w.scale)
                expectClose(c, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(name) batch \(batch)")
            }
        }
        let table = try file.weight("token_embd.weight")
        let tokens: [Int32] = [0, 9707, 151_643, 151_935, 42]
        let (c, m) = try twin.both { b -> () -> [Float] in
            let out = try b.zeros(tokens.count * table.cols)
            b.embed(table: table, tokens: tokens, into: out.slice())
            return { out.f }
        }
        #expect(c == m, "token_embd rows decode identically")
        // A norm weight read straight from the mapping (F32, zero-copy path).
        let norm = try file.weight("blk.0.attn_norm.weight")
        let x = rng.floats(3 * norm.cols, scale: 2)
        let (nc, nm) = try twin.both { b -> () -> [Float] in
            let xb = try b.upload(x)
            b.rmsNorm(xb.slice(), weight: norm, into: xb.slice(), rows: 3, dim: norm.cols, eps: 1e-6)
            return { xb.f }
        }
        expectClose(nc, nm, rel: 2e-6, abs: 1e-6, "blk.0.attn_norm")
    }
}
