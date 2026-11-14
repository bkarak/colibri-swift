import Foundation
import Testing
@testable import ColibriCore

// Workstream A: GGUF reader, writer and mapping.

let qwen3Path = NSString(string: "~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf").expandingTildeInPath
let haveQwen3 = FileManager.default.fileExists(atPath: qwen3Path)

/// A unique path in the temporary directory, removed by the caller.
func tempGGUFPath(_ tag: String) -> String {
    NSTemporaryDirectory() + "colibri-\(tag)-\(UUID().uuidString).gguf"
}

@Suite struct GGUFTests {
    @Test(.enabled(if: haveQwen3, "Qwen3-0.6B-Q8_0.gguf is not on disk"))
    func parsesRealQwen3Model() throws {
        let start = Date()
        let f = try GGUFFile(path: qwen3Path)
        let seconds = Date().timeIntervalSince(start)
        let md = f.metadata
        let arch = try #require(md.architecture)
        func hp(_ k: String) -> String { md["\(arch).\(k)"].map { "\($0)" } ?? "-" }
        print("""
            GGUF \(qwen3Path)
              version \(f.version), alignment \(f.alignment), dataStart \(f.dataStart), parsed in \(String(format: "%.3f", seconds)) s
              general.architecture \(arch)
              block_count \(hp("block_count")), context_length \(hp("context_length")), embedding_length \(hp("embedding_length"))
              feed_forward_length \(hp("feed_forward_length")), head_count \(hp("attention.head_count")), head_count_kv \(hp("attention.head_count_kv"))
              key_length \(hp("attention.key_length")), value_length \(hp("attention.value_length")), rope.freq_base \(hp("rope.freq_base"))
              rms_eps \(hp("attention.layer_norm_rms_epsilon"))
              tensors \(f.tensors.count): \(Dictionary(grouping: f.tensors, by: \.type).map { "\($0.key) \($0.value.count)" }.sorted().joined(separator: ", "))
              metadata keys \(md.keys.count), tokens \(md.strings("tokenizer.ggml.tokens")?.count ?? 0), merges \(md.strings("tokenizer.ggml.merges")?.count ?? 0)
            """)
        #expect(f.version == 3)
        #expect(arch == "qwen3")
        #expect(md.int("qwen3.block_count") == 28)
        #expect(md.int("qwen3.embedding_length") == 1024)
        #expect(md.int("qwen3.feed_forward_length") == 3072)
        #expect(md.int("qwen3.attention.head_count") == 16)
        #expect(md.int("qwen3.attention.head_count_kv") == 8)
        #expect(md.int("qwen3.attention.key_length") == 128)
        #expect(md.int("qwen3.context_length") == 40960)
        #expect(md.float("qwen3.rope.freq_base") == 1_000_000)
        #expect(md["qwen3.attention.layer_norm_rms_epsilon"] == .f32(1e-6))
        #expect(md.int("general.file_type") == 7)
        #expect(md.string("tokenizer.ggml.model") == "gpt2")
        #expect(md.string("tokenizer.ggml.pre") == "qwen2")
        #expect(md.strings("tokenizer.ggml.tokens")?.count == 151_936)
        #expect(md.strings("tokenizer.ggml.merges")?.count == 151_387)
        #expect(md.int("tokenizer.ggml.eos_token_id") == 151_645)
        #expect(md.bool("tokenizer.ggml.add_bos_token") == false)
        #expect(f.alignment == 32)
        #expect(f.dataStart == alignUp(5_951_108, 32))
        #expect(f.tensors.count == 310)
        #expect(f.tensors.filter { $0.type == .q8_0 }.count == 197)
        #expect(f.tensors.filter { $0.type == .f32 }.count == 113)
        #expect(f.tensors.allSatisfy { $0.relativeOffset % 32 == 0 })
        #expect(f.tensor("output.weight") == nil)  // tied head

        let emb = try #require(f.tensor("token_embd.weight"))
        #expect(emb.dims == [1024, 151_936] && emb.type == .q8_0)
        let q = try f.weight("blk.0.attn_q.weight")
        #expect(q.cols == 1024 && q.rows == 2048 && q.type == .q8_0 && q.scale == 1)
        #expect(q.region === f.file.region && q.offset == f.tensor("blk.0.attn_q.weight")!.fileOffset)
        let qn = try f.weight("blk.0.attn_q_norm.weight")
        #expect(qn.cols == 128 && qn.rows == 1 && qn.type == .f32)
        #expect(f.file.mappedLength % hostPageSize == 0 && f.file.mappedLength >= f.file.fileSize)
        #expect(Int(bitPattern: f.file.base) % hostPageSize == 0)
        #expect(throws: ColibriError.self) { try f.weight("no.such.tensor") }
        #expect(throws: ColibriError.self) { try f.stackedWeights("blk.0.attn_q.weight") }

        // Every Q8_0 tensor decodes to finite values (first row of each).
        for t in f.tensors where t.type == .q8_0 {
            var row = [Float](repeating: .nan, count: t.ne(0))
            row.withUnsafeMutableBufferPointer {
                Dequant.dequantizeRow(t.type, f.file.base + t.fileOffset, into: $0.baseAddress!, count: t.ne(0))
            }
            #expect(row.allSatisfy { $0.isFinite }, "\(t.name)")
        }
        let norm = try f.floats("output_norm.weight")
        #expect(norm.count == 1024 && norm.allSatisfy { $0.isFinite })
    }

    @Test(.enabled(if: haveQwen3, "Qwen3-0.6B-Q8_0.gguf is not on disk"))
    func copiesTokenizerMetadataIntoASyntheticFile() throws {
        let src = try GGUFFile(path: qwen3Path)
        var w = GGUFWriter()
        w.set("general.architecture", .string("qwen3"))
        w.copyMetadata(from: src.metadata) { $0.hasPrefix("tokenizer.") }
        try w.addTensor("token_embd.weight", dims: [64, 4], type: .q8_0, values: { var r = TestRNG(3); return r.floats(256) }())
        let path = tempGGUFPath("tok")
        defer { try? FileManager.default.removeItem(atPath: path) }
        try w.write(to: path)
        let f = try GGUFFile(path: path)
        for k in src.metadata.keys where k.hasPrefix("tokenizer.") {
            #expect(f.metadata[k] == src.metadata[k], "\(k)")
        }
        #expect(f.metadata.keys.first == "general.architecture")
    }

    @Test(arguments: [32, 64, 4096])
    func writerRoundTripsEveryValueTypeAndTensor(_ alignment: Int) throws {
        var w = GGUFWriter(alignment: alignment)
        let values: [(String, GGUFValue)] = [
            ("general.architecture", .string("olmoe")),
            ("t.u8", .u8(200)), ("t.i8", .i8(-5)), ("t.u16", .u16(60000)), ("t.i16", .i16(-30000)),
            ("t.u32", .u32(4_000_000_000)), ("t.i32", .i32(-2_000_000_000)),
            ("t.u64", .u64(1 << 40)), ("t.i64", .i64(-(1 << 40))),
            ("t.f32", .f32(1e-5)), ("t.f64", .f64(.pi)), ("t.bool", .bool(true)),
            ("t.str", .string("héllo, ĠGGUF")), ("t.empty", .string("")),
            ("t.astr", .array(.string, [.string("a b"), .string(""), .string("Ċ")])),
            ("t.ai32", .array(.i32, [.i32(1), .i32(-3), .i32(6)])),
            ("t.af32", .array(.f32, [.f32(0.5), .f32(-2)])),
            ("t.abool", .array(.bool, [.bool(false), .bool(true)])),
            ("t.au8empty", .array(.u8, [])),
        ]
        for (k, v) in values { w.set(k, v) }
        var rng = TestRNG(UInt64(alignment))
        var expected: [String: [UInt8]] = [:]
        for (i, type) in portTypes.enumerated() {
            let cols = 256, rows = 2 + i % 3
            let bytes = try Quantize.encode(rng.floats(cols * rows), as: type)
            w.addTensor("t\(i).\(type)", dims: [cols, rows], type: type, bytes: bytes)
            expected["t\(i).\(type)"] = bytes
        }
        let e3 = try Quantize.encode(rng.floats(64 * 3 * 4), as: .q8_0)
        w.addTensor("blk.0.ffn_gate_exps.weight", dims: [64, 3, 4], type: .q8_0, bytes: e3)
        w.addTensor("blk.0.ffn_gate_exps.scale", dims: [4], type: .f32, bytes: try Quantize.encode([1, 2, 3, 4], as: .f32))
        w.addTensor("blk.0.attn_q.weight", dims: [32, 2], type: .f32, bytes: try Quantize.encode(rng.floats(64), as: .f32))
        w.addTensor("blk.0.attn_q.scale", dims: [1], type: .f32, bytes: try Quantize.encode([0.5], as: .f32))

        let path = tempGGUFPath("rt")
        defer { try? FileManager.default.removeItem(atPath: path) }
        try w.write(to: path)
        let f = try GGUFFile(path: path)
        #expect(f.version == 3)
        #expect(f.alignment == alignment)
        #expect(f.dataStart % alignment == 0)
        for (k, v) in values { #expect(f.metadata[k] == v, "\(k)") }
        #expect(Array(f.metadata.keys.prefix(values.count)) == values.map(\.0))
        if alignment != 32 { #expect(f.metadata["general.alignment"] == .u32(UInt32(alignment))) }
        #expect(f.tensors.count == portTypes.count + 4)
        for t in f.tensors {
            #expect(t.relativeOffset % alignment == 0)
            if let bytes = expected[t.name] {
                #expect(Array(f.data(t)) == bytes, "\(t.name)")
            }
        }
        let st = try f.stackedWeights("blk.0.ffn_gate_exps.weight")
        #expect(st.count == 4 && st.cols == 64 && st.rows == 3 && st.stride == GGMLType.q8_0.rowBytes(64) * 3)
        #expect(st.scales == [1, 2, 3, 4])
        #expect(st.matrix(2).scale == 3 && st.matrix(2).offset == st.baseOffset + 2 * st.stride)
        let qv = try f.weight("blk.0.attn_q.weight")
        #expect(qv.scale == 0.5 && qv.cols == 32 && qv.rows == 2)
    }

    // MARK: Malformed input

    func expectInvalid(_ bytes: [UInt8], _ why: String, sourceLocation: SourceLocation = #_sourceLocation) {
        let path = tempGGUFPath("bad")
        defer { try? FileManager.default.removeItem(atPath: path) }
        FileManager.default.createFile(atPath: path, contents: Data(bytes))
        #expect(throws: ColibriError.self, "\(why)", sourceLocation: sourceLocation) { _ = try GGUFFile(path: path) }
    }

    func minimal() throws -> [UInt8] {
        var w = GGUFWriter()
        w.set("general.architecture", .string("qwen3"))
        w.set("a", .array(.u32, [.u32(1), .u32(2)]))
        w.addTensor("x.weight", dims: [32, 2], type: .q8_0, bytes: try Quantize.encode({ var r = TestRNG(1); return r.floats(64) }(), as: .q8_0))
        return try w.serialize()
    }

    @Test func rejectsMalformedFiles() throws {
        let good = try minimal()
        let path = tempGGUFPath("good")
        defer { try? FileManager.default.removeItem(atPath: path) }
        FileManager.default.createFile(atPath: path, contents: Data(good))
        _ = try GGUFFile(path: path)

        var bad = good; bad[0] = UInt8(ascii: "X")
        expectInvalid(bad, "bad magic")
        bad = good; bad[4] = 1
        expectInvalid(bad, "version 1")
        expectInvalid([], "empty file")
        for cut in [3, 12, 30, 60, good.count - 80, good.count - 30] {
            expectInvalid(Array(good.prefix(cut)), "truncated at \(cut)")
        }
        // n_tensors huge
        bad = good; bad[15] = 0x7F
        expectInvalid(bad, "too many tensors")
        // String length past the end: first key's length field at offset 24.
        bad = good; bad[24 + 6] = 0x7F
        expectInvalid(bad, "huge string")
        // Nested array: the "a" entry's element type -> 9.
        let aKey = Array("a".utf8)
        if let i = (0 ..< good.count - 9).first(where: { good[$0] == 1 && good[$0 + 8] == aKey[0] && good[$0 + 9] == 9 }) {
            bad = good; bad[i + 13] = 9
            expectInvalid(bad, "nested array")
        } else {
            Issue.record("could not locate the array entry")
        }

        // general.alignment 64 patched to 48 (not a power of two).
        var w = GGUFWriter(alignment: 64)
        w.addTensor("x", dims: [32], type: .f32, bytes: [UInt8](repeating: 0, count: 128))
        var a48 = try w.serialize()
        let key = Array("general.alignment".utf8)
        let k = try #require((0 ... a48.count - key.count).first { Array(a48[$0 ..< $0 + key.count]) == key })
        #expect(a48[k + key.count] == 4 && a48[k + key.count + 4] == 64)
        a48[k + key.count + 4] = 48
        expectInvalid(a48, "alignment 48")
    }

    @Test func rejectsTensorsOutsideTheFileAndDuplicates() throws {
        var w = GGUFWriter()
        w.addTensor("a", dims: [32, 4], type: .q8_0, bytes: [UInt8](repeating: 0, count: 34 * 4))
        w.addTensor("b", dims: [32], type: .f32, bytes: [UInt8](repeating: 0, count: 128))
        let good = try w.serialize()
        // Drop the last tensor's data: b now leaves the file.
        expectInvalid(Array(good.prefix(good.count - 64)), "tensor past end of file")
        // Rename "b" to "a": a duplicate name.
        var dup = good
        let idx = try #require((0 ..< good.count - 9).last(where: { good[$0] == 1 && good[$0 + 1 ..< $0 + 8].allSatisfy { $0 == 0 } && good[$0 + 8] == UInt8(ascii: "b") }))
        dup[idx + 8] = UInt8(ascii: "a")
        expectInvalid(dup, "duplicate tensor")
        // An unknown ggml type id (4, retired Q4_2): rewrite b's type field.
        var badType = good
        badType[idx + 9 + 4 + 8] = 4
        expectInvalid(badType, "unknown ggml type")
        // ne[0] not a multiple of the block size: a's ne[0] 32 -> 33.
        let ia = try #require((0 ..< good.count - 9).first(where: { good[$0] == 1 && good[$0 + 1 ..< $0 + 8].allSatisfy { $0 == 0 } && good[$0 + 8] == UInt8(ascii: "a") }))
        var badRow = good
        badRow[ia + 9 + 4] = 33
        expectInvalid(badRow, "row not a multiple of the block")
    }

    @Test func mappedFileRegionAndPrefault() throws {
        let path = tempGGUFPath("map")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let data = (0 ..< 40_000).map { UInt8(truncatingIfNeeded: $0 * 7) }
        FileManager.default.createFile(atPath: path, contents: Data(data))
        var region: MemoryRegion?
        do {
            let m = try MappedFile(path: path)
            #expect(m.fileSize == 40_000)
            #expect(m.mappedLength == 49_152)
            #expect(m.region.length == m.mappedLength && m.region.kind == .fileMapping)
            #expect(Array(m.bytes(39_990, 10)) == Array(data[39_990...]))
            m.prefault(0 ..< 40_000)
            m.prefault(39_999 ..< 1 << 30)
            region = m.region
        }
        // The region keeps the mapping alive after the MappedFile is gone.
        let r = try #require(region)
        #expect(r.base.load(fromByteOffset: 12_345, as: UInt8.self) == data[12_345])
        #expect(r.base.load(fromByteOffset: 45_000, as: UInt8.self) == 0)  // zero-filled tail of the last page
        #expect(throws: ColibriError.self) { _ = try MappedFile(path: path + ".missing") }
    }
}
