import ColibriCore
import ColibriMetal
import Foundation
import Testing

// A split model on Metal: the shards are mapped back to back and wrapped one tensor at a
// time, so the whole file and its shards must give the same logits, resident and streamed.
@Suite(.enabled(if: metalAvailable)) struct MetalSplitTests {
    static func run(path: String, options: ExpertStoreOptions) throws -> ([Float], [Int32]) {
        let context = LoadContext(backend: try MetalBackend(), expertOptions: options, maxContext: 128)
        let loaded = try ModelRegistry.load(path: path, context: context)
        let state = try loaded.model.makeState()
        var logits = Array(try loaded.model.forward([1, 7, 42, 3, 99], state: state).floats)
        let prefill = logits
        var ids: [Int32] = []
        for _ in 0..<12 {
            let t = Int32(logits.indices.max { logits[$0] < logits[$1] || (logits[$0] == logits[$1] && $0 > $1) }!)
            ids.append(t)
            logits = Array(try loaded.model.forward([t], state: state).floats)
        }
        return (prefill, ids)
    }

    @Test(arguments: ["resident", "stream"])
    func splitModelRunsLikeTheWholeFile(mode: String) throws {
        let dir = NSTemporaryDirectory() + "colibri-msplit-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let whole = dir + "/tiny.gguf"
        try TinyModel.write(TinyOracleCases.all.first { $0.name == "qwen3moe_q4_K_M" }!.spec, to: whole)
        let shards = try GGUFSplit.write(GGUFFile(path: whole), parts: 3, prefix: dir + "/tiny")
        var options = ExpertStoreOptions()
        options.mode = mode == "stream" ? .stream : .resident
        if mode == "stream" { options.slotsPerLayer = 4 }
        let a = try Self.run(path: whole, options: options)
        let b = try Self.run(path: shards[0], options: options)
        #expect(a.0 == b.0, "prefill logits")
        #expect(a.1 == b.1, "greedy ids")
    }
}
