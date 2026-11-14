import Foundation
import Testing
@testable import ColibriCore

// A GGUF-shaped file for the expert-store tests, written as raw bytes (no dependency on
// the GGUF writer): random "header" bytes, then per MoE layer three fused expert tensors
// at 32-byte aligned offsets, filled with seeded random bytes. Layer 0 is dense.

struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Holds a page-aligned copy of the file bytes standing in for the file mapping.
final class FakeMapping {
    let raw: UnsafeMutableRawPointer
    let length: Int
    init(bytes: [UInt8]) {
        length = alignUp(max(bytes.count, 1), hostPageSize)
        var p: UnsafeMutableRawPointer?
        precondition(posix_memalign(&p, hostPageSize, length) == 0)
        raw = p!
        memset(raw, 0, length)
        bytes.withUnsafeBytes { raw.copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
    }
    deinit { free(raw) }
}

final class ExpertFixture {
    static let layerCount = 4
    static let experts = 8
    let dir: String
    let path: String
    let bytes: [UInt8]
    let tensors: [GGUFTensorInfo]
    let layout: ExpertLayout
    private var mapping: FakeMapping?

    /// `layers` 1..3 route; layer 3's down projection is F16 (mixed types per layer).
    init(seed: UInt64 = 42, experts: Int = ExpertFixture.experts) throws {
        var rng = SplitMix64(seed: seed)
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("colibri-expert-tests-\(UUID().uuidString)").path
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        path = dir + "/tiny-moe.gguf"

        var data: [UInt8] = (0 ..< 1000).map { _ in UInt8.random(in: 0 ... 255, using: &rng) }
        var infos: [GGUFTensorInfo] = []
        func add(_ name: String, _ dims: [Int], _ type: GGMLType) {
            while data.count % 32 != 0 { data.append(0) }
            let n = type.rowBytes(dims[0]) * dims[1...].reduce(1, *)
            infos.append(GGUFTensorInfo(name: name, dims: dims, type: type, relativeOffset: data.count - 1024,
                                        fileOffset: data.count, byteCount: n))
            for _ in 0 ..< n { data.append(UInt8.random(in: 0 ... 255, using: &rng)) }
        }
        add("blk.0.ffn_gate.weight", [64, 32], .q8_0)  // dense layer: ignored
        for l in 1 ..< Self.layerCount {
            add("blk.\(l).ffn_gate_exps.weight", [64, 32, experts], .q8_0)
            add("blk.\(l).ffn_up_exps.weight", [64, 32, experts], .q8_0)
            add("blk.\(l).ffn_down_exps.weight", [32, 64, experts], l == 3 ? .f16 : .q8_0)
        }
        for _ in 0 ..< 77 { data.append(0xEE) }  // trailing bytes, like other tensors
        bytes = data
        tensors = infos
        try Data(data).write(to: URL(fileURLWithPath: path))
        layout = try ExpertLayout(tensors: infos, layerCount: Self.layerCount)
    }

    deinit { try? FileManager.default.removeItem(atPath: dir) }

    /// A region standing in for the whole-file mapping (region offset = file offset).
    func fileRegion() -> MemoryRegion {
        if mapping == nil { mapping = FakeMapping(bytes: bytes) }
        return MemoryRegion(base: UnsafeRawPointer(mapping!.raw), length: mapping!.length, kind: .fileMapping,
                            label: "fake mapping", keepAlive: mapping)
    }

    /// The file bytes of expert `e` of one matrix kind.
    func fileBytes(_ s: ExpertTensorSlice, expert e: Int) -> ArraySlice<UInt8> {
        bytes[s.range(expert: e)]
    }

    /// Whether matrix `index` of `w` holds exactly the file bytes of expert `e`.
    func matches(_ w: StackedWeights, index: Int, slice s: ExpertTensorSlice, expert e: Int) -> Bool {
        let p = w.region.base + w.baseOffset + index * w.stride
        let got = UnsafeRawBufferPointer(start: p, count: s.bytesPerExpert)
        return got.elementsEqual(fileBytes(s, expert: e))
    }

    /// Gate, up and down of index `index` in `weights` equal expert `e` of `layer`.
    func matches(_ weights: ExpertLayerWeights, layer: Int, index: Int, expert e: Int) -> Bool {
        let l = layout.layers[layer]!
        return matches(weights.gate, index: index, slice: l.gate, expert: e)
            && matches(weights.up, index: index, slice: l.up, expert: e)
            && matches(weights.down, index: index, slice: l.down, expert: e)
    }
}

@Suite struct ExpertLayoutTests {
    @Test func layoutFromTensorTable() throws {
        let fx = try ExpertFixture()
        let layout = fx.layout
        #expect(layout.layers.count == 4)
        #expect(layout.layers[0] == nil)
        #expect(layout.moeLayers == [1, 2, 3])
        #expect(layout.maxExperts == 8)
        let l1 = try #require(layout.layers[1])
        #expect(l1.count == 8)
        #expect(l1.gate.type == .q8_0 && l1.gate.cols == 64 && l1.gate.rows == 32)
        #expect(l1.gate.bytesPerExpert == 68 * 32)
        #expect(l1.down.cols == 32 && l1.down.rows == 64 && l1.down.bytesPerExpert == 34 * 64)
        let l3 = try #require(layout.layers[3])
        #expect(l3.down.type == .f16 && l3.down.bytesPerExpert == 64 * 64)
        #expect(l3.slotBytes == 2 * 68 * 32 + 64 * 64)
        let t = try #require(fx.tensors.first { $0.name == "blk.2.ffn_up_exps.weight" })
        #expect(layout.layers[2]!.up.fileOffset == t.fileOffset)
        #expect(layout.totalBytes == 8 * (l1.slotBytes * 2 + l3.slotBytes))
        #expect(layout.fileEnd <= fx.bytes.count)
    }

    @Test func layoutRejectsBrokenTables() throws {
        let fx = try ExpertFixture()
        // a layer with only some expert tensors
        let partial = fx.tensors.filter { $0.name != "blk.2.ffn_down_exps.weight" }
        #expect(throws: ColibriError.self) { try ExpertLayout(tensors: partial, layerCount: 4) }
        // byte count disagreeing with the shape
        var bad = fx.tensors
        let i = bad.firstIndex { $0.name == "blk.1.ffn_gate_exps.weight" }!
        bad[i].byteCount += 1
        #expect(throws: ColibriError.self) { try ExpertLayout(tensors: bad, layerCount: 4) }
        // expert counts differing between gate and up
        bad = fx.tensors
        let j = bad.firstIndex { $0.name == "blk.1.ffn_up_exps.weight" }!
        bad[j].dims[2] = 4
        bad[j].byteCount = 68 * 32 * 4
        #expect(throws: ColibriError.self) { try ExpertLayout(tensors: bad, layerCount: 4) }
        // a row that is not a whole number of blocks (would trap in rowBytes)
        bad = fx.tensors
        bad[i].dims[0] = 63
        #expect(throws: ColibriError.self) { try ExpertLayout(tensors: bad, layerCount: 4) }
        // a 2-D tensor under an expert name
        bad = fx.tensors
        bad[i].dims = [64, 256]
        #expect(throws: ColibriError.self) { try ExpertLayout(tensors: bad, layerCount: 4) }
    }

    @Test func arenaIsPageAligned() throws {
        let fx = try ExpertFixture()
        let l3 = fx.layout.layers[3]!
        let arena = try ExpertArena(layout: l3, slots: 5)
        #expect(Int(bitPattern: arena.region.base) % hostPageSize == 0)
        #expect(arena.region.length % hostPageSize == 0)
        #expect(arena.stride % hostPageSize == 0)
        #expect(arena.region.length == arena.stride * 5)
        #expect(arena.region.kind == .expertArena)
        #expect(arena.gateOffset == 0)
        #expect(arena.upOffset >= l3.gate.bytesPerExpert && arena.upOffset % ExpertArena.matrixAlignment == 0)
        #expect(arena.downOffset >= arena.upOffset + l3.up.bytesPerExpert && arena.downOffset % ExpertArena.matrixAlignment == 0)
        #expect(arena.downOffset + l3.down.bytesPerExpert <= arena.stride)
        #expect(arena.slotPointer(4) == UnsafeMutableRawPointer(mutating: arena.region.base) + 4 * arena.stride)
        #expect(throws: ColibriError.self) { try ExpertArena(layout: l3, slots: 0) }
    }
}
