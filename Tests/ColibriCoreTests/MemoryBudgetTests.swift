import Foundation
import Testing
@testable import ColibriCore

@Suite struct MemoryBudgetTests {
    @Test func availableMemoryIsMeasured() throws {
        let avail = try #require(MemoryBudget.availableBytes())
        #expect(avail > 0 && avail <= MemoryBudget.physicalBytes)
    }

    @Test func residentWhenEverythingFits() throws {
        let fx = try ExpertFixture()
        let total = fx.layout.totalBytes
        #expect(MemoryBudget(budgetBytes: total + 100, reservedBytes: 100).plan(layout: fx.layout, topK: 2) == .resident)
        #expect(MemoryBudget(budgetBytes: total + 99, reservedBytes: 100).plan(layout: fx.layout, topK: 2) != .resident)
        #expect(MemoryBudget(budgetBytes: 0, reservedBytes: 0).plan(layout: ExpertLayout(layers: [nil, nil]), topK: 8) == .resident)
    }

    @Test func streamingSlotsAreEqualPerLayerAndBounded() throws {
        let fx = try ExpertFixture()
        let perSlot = (1 ... 3).reduce(0) { $0 + ExpertArena.slotStride(for: fx.layout.layers[$1]!) }
        // (the tiny fixture's slots are padded to 16 KiB, so 3 slots is below the experts' total)
        #expect(3 * perSlot < fx.layout.totalBytes)
        #expect(MemoryBudget(budgetBytes: 3 * perSlot + 1000, reservedBytes: 1000).plan(layout: fx.layout, topK: 2)
            == .streaming(slotsPerLayer: [0, 3, 3, 3]))
        #expect(MemoryBudget(budgetBytes: perSlot, reservedBytes: 0).plan(layout: fx.layout, topK: 3)
            == .streaming(slotsPerLayer: [0, 3, 3, 3]))  // never under top-k
        #expect(MemoryBudget(budgetBytes: 10, reservedBytes: 1 << 30).plan(layout: fx.layout, topK: 20)
            == .streaming(slotsPerLayer: [0, 8, 8, 8]))  // never over the expert count
    }

    @Test func deriveHonoursAnExplicitCacheAndTheWorkingSet() {
        let explicit = MemoryBudget.derive(workingSetLimit: 1 << 40, cacheBytes: 3 << 30, residentBytes: 5 << 30)
        #expect(explicit.roomBytes == 3 << 30)
        let small = MemoryBudget.derive(workingSetLimit: 1 << 30, cacheBytes: nil, residentBytes: 1 << 20)
        #expect(small.budgetBytes <= 1 << 30)
        #expect(small.reservedBytes == (1 << 20) + MemoryBudget.activationHeadroom)
        let big = MemoryBudget.derive(workingSetLimit: Int.max, cacheBytes: nil, residentBytes: 0)
        #expect(big.budgetBytes <= Int(Double(MemoryBudget.physicalBytes) * 0.85))
        #expect(big.budgetBytes > 0)
    }

    @Test func derivedStreamingGivesTheSlotsHalfTheRoom() throws {
        let fx = try ExpertFixture()
        let perSlot = (1 ... 3).reduce(0) { $0 + ExpertArena.slotStride(for: fx.layout.layers[$1]!) }
        // Resident or not is still decided on the whole room.
        let total = fx.layout.totalBytes
        #expect(MemoryBudget(budgetBytes: total, reservedBytes: 0, streamShare: 0.5).plan(layout: fx.layout, topK: 2) == .resident)
        // Streaming: 3 slots' worth of room (below the experts' total, as above), half of
        // it for slots: 1.5 slots, rounded down.
        #expect(3 * perSlot < total)
        #expect(MemoryBudget(budgetBytes: 3 * perSlot, reservedBytes: 0, streamShare: 0.5).plan(layout: fx.layout, topK: 1)
            == .streaming(slotsPerLayer: [0, 1, 1, 1]))
        #expect(MemoryBudget(budgetBytes: 3 * perSlot, reservedBytes: 0).plan(layout: fx.layout, topK: 1)
            == .streaming(slotsPerLayer: [0, 3, 3, 3]))
        // A derived budget gets the share; an explicit --expert-cache-gb is the slots' size.
        #expect(MemoryBudget.derive(workingSetLimit: Int.max, cacheBytes: nil, residentBytes: 0).streamShare
            == MemoryBudget.derivedStreamShare)
        #expect(MemoryBudget.derive(workingSetLimit: Int.max, cacheBytes: 1 << 30, residentBytes: 0).streamShare == 1)
    }
}
