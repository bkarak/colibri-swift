import Foundation
import Testing
@testable import ColibriCore

// Failure modes that must surface as errors, not as wrong output: a backend refusing a
// model's weight types (with the tensors named), and a failed expert read in the
// streaming store (the forward that consumed the stale slot must throw).

@Suite struct FailureModeTests {
    @Test func weightTypeRefusalNamesTheTensors() throws {
        var r = ModelRequirements()
        r.matrices = [
            .init(name: "token_embd.weight", type: .q8_0, cols: 256, rows: 10, offset: 0),
            .init(name: "blk.0.attn_q.weight", type: .q4_K, cols: 256, rows: 256, offset: 4096),
            .init(name: "blk.0.attn_k.weight", type: .q4_K, cols: 256, rows: 64, offset: 8192),
        ]
        r.norms = [.init(name: "output_norm.weight", type: .f32, cols: 256, rows: 1, offset: 0)]
        let caps = BackendCapabilities(name: "toy", weightTypes: [.f32, .q8_0], maxBufferLength: .max,
                                       workingSetLimit: .max, isAsynchronous: false)
        do {
            try r.checkWeightTypes(caps)
            Issue.record("expected a refusal")
        } catch let ColibriError.unsupported(msg) {
            #expect(msg.contains("toy"))
            #expect(msg.contains("Q4_K"))
            #expect(msg.contains("blk.0.attn_q.weight"))
            #expect(msg.contains("blk.0.attn_k.weight"))
            #expect(!msg.contains("token_embd"))
        }
        // Everything supported: no throw. The CPU backend's default check is this one.
        let all = BackendCapabilities(name: "toy", weightTypes: [.f32, .q8_0, .q4_K], maxBufferLength: .max,
                                      workingSetLimit: .max, isAsynchronous: false)
        try r.checkWeightTypes(all)
        try CPUBackend(threads: 1).checkSupport(r)
    }

    /// The CPU backend has no shape limits: a head dim the Metal kernels refuse loads and
    /// runs to finite logits (the `--backend auto` fallback target).
    @Test func cpuRunsWideHeads() throws {
        var spec = TinyModelSpec(architecture: "qwen3", weightType: .q8_0, seed: 5)
        spec.headDim = 512
        let path = NSTemporaryDirectory() + "colibri-hd512-\(UUID().uuidString).gguf"
        try TinyModel.write(spec, to: path)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let m = try ModelRegistry.load(path: path, context: LoadContext(backend: CPUBackend(threads: 2), maxContext: 32))
        let state = try m.model.makeState()
        let logits = try m.model.forward([10, 20, 30, 40], state: state)
        #expect(logits.floats.allSatisfy { $0.isFinite })
    }

    @Test func failedExpertReadFailsTheForward() throws {
        try expertReadFailureCheck(backend: CPUBackend(threads: 2))
    }
}

/// Loads a tiny qwen3moe twice (resident from the intact file, streamed from a copy),
/// truncates the copy so the last expert matrix in the file can no longer be read, then
/// prefills a prompt long enough to route to every expert. Every forward of the streamed
/// model must either match the resident logits or throw; the prefill must throw.
func expertReadFailureCheck(backend: Backend) throws {
    let spec = TinyModelSpec(architecture: "qwen3moe", weightType: .q8_0, seed: 2)
    let dir = NSTemporaryDirectory() + "colibri-iofail-\(UUID().uuidString)/"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: dir) }
    let intact = dir + "intact.gguf", broken = dir + "broken.gguf"
    try TinyModel.write(spec, to: intact)
    try FileManager.default.copyItem(atPath: intact, toPath: broken)

    // The truncated bytes must belong to a routed-expert tensor (only those are pread;
    // the rest of the file is mapped and touching it past EOF would be a SIGBUS).
    let file = try GGUFFile(path: intact)
    let last = try #require(file.tensors.max { $0.fileOffset + $0.byteCount < $1.fileOffset + $1.byteCount })
    #expect(last.name.hasSuffix("_exps.weight"))
    #expect(last.byteCount / spec.experts > 1000)

    let ctx = { (o: ExpertStoreOptions) in LoadContext(backend: backend, expertOptions: o, maxContext: 64, maxBatch: 64) }
    let resident = try ModelRegistry.load(path: intact, context: ctx(ExpertStoreOptions(mode: .resident, saveHistory: false)))
    let streamed = try ModelRegistry.load(path: broken,
                                          context: ctx(ExpertStoreOptions(mode: .stream, slotsPerLayer: spec.experts,
                                                                          ioThreads: 2, saveHistory: false)))
    let size = try #require(try FileManager.default.attributesOfItem(atPath: broken)[.size] as? Int)
    #expect(truncate(broken, off_t(size - 1000)) == 0)

    let prompt: [Int32] = (0..<48).map { Int32(($0 * 37 + 11) % 200) }
    let rs = try resident.model.makeState(), ss = try streamed.model.makeState()
    let want = Array(try resident.model.forward(prompt, state: rs).floats)
    do {
        let got = Array(try streamed.model.forward(prompt, state: ss).floats)
        let diff = zip(want, got).map { abs($0 - $1) }.max() ?? 0
        Issue.record("prefill over a failed expert read returned normally (max |logit diff| \(diff))")
    } catch {
        #expect("\(error)".contains("errno") || "\(error)".contains("failed"), "unexpected error: \(error)")
    }
}
