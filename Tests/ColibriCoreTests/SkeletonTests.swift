import Testing
@testable import ColibriCore

// Package smoke tests from the skeleton step. Workstreams add their own files
// (see docs/port/architecture.md, "Ownership").

@Suite struct SkeletonTests {
    @Test func blockGeometry() {
        #expect(GGMLType.q8_0.blockSize == 32)
        #expect(GGMLType.q8_0.rowBytes(1024) == 1088)
        #expect(GGMLType.q4_K.rowBytes(2048) == 1152)
        #expect(GGMLType.q6_K.typeSize == 210)
        #expect(GGMLType(rawValue: 4) == nil)  // retired Q4_2
        #expect(GGMLType.q4_K.description == "Q4_K")
    }

    @Test func registryNamesTheThreeFamilies() {
        #expect(ModelRegistry.family(for: "qwen3") === Qwen3.self)
        #expect(ModelRegistry.family(for: "qwen3moe") === Qwen3MoE.self)
        #expect(ModelRegistry.family(for: "olmoe") === OLMoE.self)
        #expect(ModelRegistry.family(for: "llama") == nil)
    }

    @Test func tierHysteresis() {
        #expect(!Tier.shouldPromote(hot: 9, cold: 4))
        #expect(Tier.shouldPromote(hot: 10, cold: 4))
    }

    @Test func cpuBufferIsZeroedAndCopies() throws {
        let cpu = CPUBackend(threads: 1)
        let a = try cpu.makeBuffer(.f32, count: 8, label: "a")
        let b = try cpu.makeBuffer(.f32, count: 8, label: "b")
        #expect(a.slice().floats.allSatisfy { $0 == 0 })
        for i in 0..<8 { a.slice().floats[i] = Float(i) }
        cpu.copy(a.slice(2, 4), into: b.slice(0, 4), count: 4)
        try cpu.synchronize()
        #expect(Array(b.slice(0, 4).floats) == [2, 3, 4, 5])
    }
}
