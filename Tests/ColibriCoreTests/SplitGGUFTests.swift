@testable import ColibriCore
import Foundation
import Testing

// Split models (llama.cpp's gguf-split layout): a tiny MoE model written whole and as
// shards must parse to the same tensors and run to the same logits, resident and with
// experts streamed from the shards.
@Suite struct SplitGGUFTests {
    /// Writes a tiny qwen3moe (Q4_K_M mix) whole and as `parts` shards in a fresh
    /// directory; returns (directory, whole path, shard paths).
    static func fixture(parts: Int) throws -> (dir: String, whole: String, shards: [String]) {
        let dir = NSTemporaryDirectory() + "colibri-split-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let whole = dir + "/tiny.gguf"
        let spec = TinyOracleCases.all.first { $0.name == "qwen3moe_q4_K_M" }!.spec
        try TinyModel.write(spec, to: whole)
        let shards = try GGUFSplit.write(GGUFFile(path: whole), parts: parts, prefix: dir + "/tiny")
        return (dir, whole, shards)
    }

    static func logits(path: String, options: ExpertStoreOptions, prompt: [Int32] = [1, 7, 42, 3, 99]) throws -> ([Float], [Int32]) {
        let context = LoadContext(backend: CPUBackend(), expertOptions: options, maxContext: 128)
        let loaded = try ModelRegistry.load(path: path, context: context)
        let state = try loaded.model.makeState()
        var logits = Array(try loaded.model.forward(prompt, state: state).floats)
        let prefill = logits
        var ids: [Int32] = []
        for _ in 0..<12 {
            let t = Int32(logits.indices.max { logits[$0] < logits[$1] || (logits[$0] == logits[$1] && $0 > $1) }!)
            ids.append(t)
            logits = Array(try loaded.model.forward([t], state: state).floats)
        }
        return (prefill, ids)
    }

    @Test func shardsParseToTheSameTensors() throws {
        let fx = try Self.fixture(parts: 3)
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        let a = try GGUFFile(path: fx.whole)
        for opened in [fx.shards[0], fx.shards[2]] {  // any shard opens the whole model
            let b = try GGUFFile(path: opened)
            #expect(b.file.segments.map(\.path) == fx.shards)
            #expect(b.file.segments.allSatisfy { $0.start % hostPageSize == 0 })
            #expect(b.tensors.map(\.name) == a.tensors.map(\.name))
            #expect(b.metadata.architecture == a.metadata.architecture)
            #expect(b.metadata.strings("tokenizer.ggml.tokens") == a.metadata.strings("tokenizer.ggml.tokens"))
            for (ta, tb) in zip(a.tensors, b.tensors) {
                #expect(ta.dims == tb.dims && ta.type == tb.type && ta.byteCount == tb.byteCount)
                #expect(Array(a.data(ta)) == Array(b.data(tb)), "\(ta.name) bytes")
            }
        }
    }

    @Test(arguments: ["resident", "stream"])
    func splitModelRunsLikeTheWholeFile(mode: String) throws {
        let fx = try Self.fixture(parts: 3)
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        var options = ExpertStoreOptions()
        if mode == "stream" {
            options.mode = .stream
            options.slotsPerLayer = 4  // of 16: evictions on every token, reads from every shard
        } else {
            options.mode = .resident
        }
        let a = try Self.logits(path: fx.whole, options: options)
        let b = try Self.logits(path: fx.shards[1], options: options)
        #expect(a.0 == b.0, "prefill logits")
        #expect(a.1 == b.1, "greedy ids")
    }

    @Test func brokenSplitsAreRefused() throws {
        let fx = try Self.fixture(parts: 2)
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        // A shard renamed away from the -NNNNN-of-NNNNN pattern.
        let renamed = fx.dir + "/renamed.gguf"
        try FileManager.default.copyItem(atPath: fx.shards[0], toPath: renamed)
        #expect(throws: ColibriError.self) { _ = try GGUFFile(path: renamed) }
        // A missing shard.
        try FileManager.default.removeItem(atPath: fx.shards[1])
        #expect(throws: ColibriError.self) { _ = try GGUFFile(path: fx.shards[0]) }
        // Shards swapped: split.no does not match the position.
        let g = try GGUFFile(path: fx.whole)
        let other = try GGUFSplit.write(g, parts: 2, prefix: fx.dir + "/swap")
        let tmp = fx.dir + "/t.gguf"
        try FileManager.default.moveItem(atPath: other[0], toPath: tmp)
        try FileManager.default.moveItem(atPath: other[1], toPath: other[0])
        try FileManager.default.moveItem(atPath: tmp, toPath: other[1])
        #expect(throws: ColibriError.self) { _ = try GGUFFile(path: other[0]) }
    }

    @Test func splitPathsFromAnyShard() throws {
        #expect(try GGUFFile.splitPaths(path: "/m/Big-Q2_K-00002-of-00003.gguf", count: 3)
                == ["/m/Big-Q2_K-00001-of-00003.gguf", "/m/Big-Q2_K-00002-of-00003.gguf", "/m/Big-Q2_K-00003-of-00003.gguf"])
        #expect(throws: ColibriError.self) { _ = try GGUFFile.splitPaths(path: "/m/Big-00002-of-00004.gguf", count: 3) }
    }
}
