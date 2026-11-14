import Foundation
import Testing
@testable import ColibriCore

private func streaming(_ fx: ExpertFixture, slots: Int, topK: Int = 2, ioThreads: Int = 4, uncached: Bool = true,
                       history: RoutingHistory? = nil, repinEvery: Int = 0) throws -> StreamingExpertStore {
    try StreamingExpertStore(path: fx.path, layout: fx.layout, slotsPerLayer: [0, slots, slots, slots],
                             backend: CPUBackend(threads: 1), ioThreads: ioThreads, history: history,
                             uncached: uncached, topK: topK, repinEvery: repinEvery)
}

/// Every position of the lease addresses the requested expert's file bytes.
private func leaseServesFileBytes(_ fx: ExpertFixture, _ store: ExpertStore, _ lease: ExpertLease) -> Bool {
    let w = store.weights(layer: lease.layer)
    return zip(lease.experts, lease.slots).allSatisfy { e, s in
        fx.matches(w, layer: lease.layer, index: Int(s), expert: Int(e))
    }
}

@Suite struct ExpertStoreTests {
    @Test func residentStoreAddressesTheMappingInPlace() throws {
        let fx = try ExpertFixture()
        let region = fx.fileRegion()
        let history = RoutingHistory(layers: 4, experts: 8, engineName: "qwen3moe", moeLayers: [1, 2, 3])
        let store = ResidentExpertStore(region: region, layout: fx.layout, history: history)
        #expect(store.isFullyResident)
        #expect(store.signal == nil)
        for layer in 1 ... 3 {
            let w = store.weights(layer: layer)
            #expect(w.gate.region === region && w.gate.count == 8)
            #expect(w.gate.baseOffset == fx.layout.layers[layer]!.gate.fileOffset)
            for e in 0 ..< 8 { #expect(fx.matches(w, layer: layer, index: e, expert: e)) }
        }
        let lease = try store.acquire(layer: 2, experts: [5, 1, 5])
        #expect(lease.slots == [5, 1, 5] && lease.misses.isEmpty && lease.readyValue == 0)
        store.release(lease)
        store.recordUsage(layer: 2, experts: [5])
        #expect(history.count(layer: 2, expert: 5) == 3)
        #expect(store.stats.lookups == 2 && store.stats.hits == 2)
        #expect(throws: ColibriError.self) { try store.acquire(layer: 0, experts: [0]) }
        #expect(throws: ColibriError.self) { try store.acquire(layer: 1, experts: [8]) }
    }

    @Test(arguments: [true, false])
    func streamingServesTheSameBytesAsTheFile(uncached: Bool) throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 3, uncached: uncached)
        var rng = SplitMix64(seed: 7)
        var requested = 0
        for step in 0 ..< 300 {
            let layer = 1 + step % 3
            let k = 1 + Int.random(in: 0 ..< 3, using: &rng)
            let ids = (0 ..< k).map { _ in Int32.random(in: 0 ..< 8, using: &rng) }
            let lease = try store.acquire(layer: layer, experts: ids)
            requested += Set(ids).count
            store.waitForMisses(lease)
            #expect(leaseServesFileBytes(fx, store, lease))
            #expect(lease.misses.allSatisfy { ids.indices.contains($0) })
            store.release(lease)
        }
        store.waitUntilIdle()
        let s = store.stats
        #expect(s.lookups == requested)
        #expect(s.hits + s.misses == s.lookups)
        #expect(s.misses > 0 && s.hits > 0 && s.evictions > 0)
        #expect(s.reads == s.misses)  // no prefetches: one read per miss, none duplicated
        let slotBytes = (1 ... 3).map { fx.layout.layers[$0]!.slotBytes }
        #expect(s.bytesRead >= s.reads * slotBytes.min()! && s.bytesRead <= s.reads * slotBytes.max()!)
        #expect(s.readGBPerSecond > 0 && s.ioWallSeconds > 0 && s.diskSeconds > 0)
        #expect(store.signal!.value == UInt64(s.reads))
        for layer in 1 ... 3 { #expect(store.slotTable(layer: layer).allSatisfy { $0.busy == 0 }) }
    }

    @Test func hitsAreServedFromTheCacheWithoutReading() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 4)
        let a = try store.acquire(layer: 1, experts: [3, 6])
        #expect(a.misses == [0, 1] && a.readyValue > 0)
        store.waitForMisses(a)
        store.release(a)
        let before = store.stats
        let b = try store.acquire(layer: 1, experts: [6, 3, 6])
        #expect(b.misses.isEmpty && b.readyValue == 0)
        #expect(b.slots[0] == b.slots[2] && b.slots[0] == a.slots[1])
        #expect(leaseServesFileBytes(fx, store, b))
        store.release(b)
        let after = store.stats
        #expect(after.hits == before.hits + 2 && after.reads == before.reads)
    }

    @Test func prefetchedExpertsAreHits() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 4)
        store.prefetch(layer: 2, experts: [1, 7, 1])
        store.waitUntilIdle()
        #expect(store.stats.prefetchIssued == 2)
        let lease = try store.acquire(layer: 2, experts: [7, 1])
        #expect(lease.misses.isEmpty)
        #expect(leaseServesFileBytes(fx, store, lease))
        store.release(lease)
        let s = store.stats
        #expect(s.prefetchHits == 2 && s.hits == 2 && s.misses == 0 && s.reads == 2)
        // A prefetch replaces used, unleased slots (7 and 1) but never a leased slot (0
        // and 2) nor a prefetch not yet used (3 and 4): the third prefetch is dropped.
        let hold = try store.acquire(layer: 2, experts: [0, 2])
        store.waitForMisses(hold)
        store.prefetch(layer: 2, experts: [3, 4])
        store.waitUntilIdle()
        store.prefetch(layer: 2, experts: [5])
        store.waitUntilIdle()
        #expect(Set(store.slotTable(layer: 2).map(\.expert)) == [0, 2, 3, 4])
        #expect(store.stats.prefetchIssued == 4)
        store.release(hold)
    }

    @Test func aSecondRequestWaitsForTheReadInFlight() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 3, ioThreads: 2)
        store.readDelayMicros = 100_000
        store.prefetch(layer: 1, experts: [4])
        Thread.sleep(forTimeInterval: 0.02)  // the prefetch read has started (and sleeps)
        let lease = try store.acquire(layer: 1, experts: [4])
        #expect(lease.misses == [0] && lease.readyValue > 0)
        store.waitForMisses(lease)
        #expect(leaseServesFileBytes(fx, store, lease))
        store.release(lease)
        store.waitUntilIdle()
        let s = store.stats
        #expect(s.inFlightWaits == 1 && s.reads == 1 && s.waitSeconds > 0)
    }

    @Test func aQueuedPrefetchIsPromotedByDemand() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 8, topK: 2, ioThreads: 1)
        store.readDelayMicros = 20_000
        store.prefetch(layer: 3, experts: [0, 1, 2, 3, 4, 5])  // one worker: a queue forms
        let lease = try store.acquire(layer: 3, experts: [5])
        #expect(lease.misses == [0])
        let t0 = Date()
        store.waitForMisses(lease)
        // Promoted ahead of the queued prefetches: about two reads (the running one and
        // this one), not six.
        #expect(Date().timeIntervalSince(t0) < 0.1)
        #expect(leaseServesFileBytes(fx, store, lease))
        store.release(lease)
        store.waitUntilIdle()
        #expect(store.stats.reads == 6)  // the promoted prefetch was not read twice
        #expect(store.signal!.value == 6)
    }

    @Test func leasedSlotsAreNeverEvicted() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 3)
        let hold = try store.acquire(layer: 1, experts: [0, 1])
        store.waitForMisses(hold)
        for e: Int32 in 2 ..< 8 {  // cycles through the one free slot
            let l = try store.acquire(layer: 1, experts: [e])
            store.waitForMisses(l)
            #expect(leaseServesFileBytes(fx, store, l))
            store.release(l)
        }
        #expect(leaseServesFileBytes(fx, store, hold))
        #expect(hold.slots.allSatisfy { s in store.slotTable(layer: 1)[Int(s)].busy == 1 })
        store.release(hold)
        #expect(store.stats.evictions == 5)
    }

    @Test func acquiringWhenEverySlotIsLeasedFailsCleanly() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 2)
        store.leaseWaitTimeout = 0.2
        let hold = try store.acquire(layer: 1, experts: [0, 1])
        store.waitForMisses(hold)
        #expect(throws: ColibriError.self) { try store.acquire(layer: 1, experts: [2]) }
        #expect(throws: ColibriError.self) { try store.acquire(layer: 1, experts: [2, 3, 4]) }  // more than the slots
        store.release(hold)
        let l = try store.acquire(layer: 1, experts: [2])
        store.waitForMisses(l)
        #expect(leaseServesFileBytes(fx, store, l))
        store.release(l)
        #expect(store.slotTable(layer: 1).allSatisfy { $0.busy == 0 })
    }

    @Test func aBackendBatchCanWaitOnTheSignal() throws {
        let fx = try ExpertFixture()
        let cpu = CPUBackend(threads: 1)
        let store = try StreamingExpertStore(path: fx.path, layout: fx.layout, slotsPerLayer: [0, 4, 4, 4],
                                             backend: cpu, ioThreads: 2, topK: 2)
        store.readDelayMicros = 10_000
        let lease = try store.acquire(layer: 2, experts: [6, 2])
        cpu.encodeWait(for: store.signal!, value: lease.readyValue)  // CPU: blocks right here
        #expect(store.signal!.value >= lease.readyValue)
        #expect(leaseServesFileBytes(fx, store, lease))
        store.release(lease)
    }

    @Test func concurrentLeasesAndPrefetchesStayConsistent() throws {
        let fx = try ExpertFixture()
        // cap = topK: the configuration that exposed upstream's busy-slot race
        let store = try streaming(fx, slots: 4, topK: 2, ioThreads: 8)
        let failures = Atomicish()
        let group = DispatchGroup()
        for t in 0 ..< 4 {
            group.enter()
            Thread {
                var rng = SplitMix64(seed: UInt64(100 + t))
                for _ in 0 ..< 150 {
                    let layer = 1 + Int.random(in: 0 ..< 3, using: &rng)
                    let ids = (0 ..< 2).map { _ in Int32.random(in: 0 ..< 8, using: &rng) }
                    do {
                        let lease = try store.acquire(layer: layer, experts: ids)
                        store.waitForMisses(lease)
                        if !leaseServesFileBytes(fx, store, lease) { failures.add() }
                        store.release(lease)
                    } catch { failures.add() }
                }
                group.leave()
            }.start()
        }
        group.enter()
        Thread {
            var rng = SplitMix64(seed: 99)
            for _ in 0 ..< 300 {
                let layer = 1 + Int.random(in: 0 ..< 3, using: &rng)
                store.prefetch(layer: layer, experts: (0 ..< 3).map { _ in Int32.random(in: 0 ..< 8, using: &rng) })
            }
            group.leave()
        }.start()
        #expect(group.wait(timeout: .now() + 60) == .success)
        store.waitUntilIdle()
        #expect(failures.value == 0)
        let s = store.stats
        #expect(s.hits + s.misses == s.lookups && s.lookups > 0)
        #expect(store.signal!.value == UInt64(s.reads))
        for layer in 1 ... 3 {
            let table = store.slotTable(layer: layer)
            #expect(table.allSatisfy { $0.busy == 0 && $0.state != .loading })
            let resident = table.filter { $0.state == .ready }.map(\.expert)
            #expect(Set(resident).count == resident.count)  // one slot per expert
        }
    }

    @Test func pinnedExpertsSurviveEvictionPressure() throws {
        let fx = try ExpertFixture()
        let history = RoutingHistory(layers: 4, experts: 8, engineName: "olmoe", moeLayers: [1, 2, 3])
        history.record(layer: 1, experts: [6, 6, 6, 2, 2])
        let store = try streaming(fx, slots: 4, topK: 2, history: history)  // pin cap 2
        store.pinFromHistory()
        store.waitUntilIdle()
        #expect(store.pinnedExperts(layer: 1) == [2, 6])
        #expect(store.pinnedExperts(layer: 2).isEmpty)
        #expect(store.stats.pinned == 2)
        for e: Int32 in [0, 1, 3, 4, 5, 7, 0, 1, 3] {
            let l = try store.acquire(layer: 1, experts: [e])
            store.waitForMisses(l)
            store.release(l)
        }
        let readsBefore = store.stats.reads
        let l = try store.acquire(layer: 1, experts: [6, 2])
        #expect(l.misses.isEmpty)
        #expect(leaseServesFileBytes(fx, store, l))
        store.release(l)
        #expect(store.stats.reads == readsBefore)
    }

    @Test func liveRepinPromotesHotExpertsWithHysteresis() throws {
        let fx = try ExpertFixture()
        let store = try streaming(fx, slots: 4, topK: 2, repinEvery: 2)  // pin cap 2
        store.pin(layer: 1, experts: [0, 1])
        store.waitUntilIdle()
        for _ in 0 ..< 20 { store.recordUsage(layer: 1, experts: [5]) }
        store.recordUsage(layer: 1, experts: [0, 1])
        store.endToken()
        #expect(store.pinnedExperts(layer: 1) == [0, 1])  // not a boundary yet
        store.endToken()
        let pins = store.pinnedExperts(layer: 1)
        #expect(pins.contains(5) && pins.count == 2)
        #expect(store.stats.repinSwaps == 1)
        store.waitUntilIdle()
        #expect(store.slotTable(layer: 1).contains { $0.expert == 5 && $0.state == .ready })
        // barely hotter than a pinned expert: no swap (25% + 4 hysteresis)
        for _ in 0 ..< 3 { store.recordUsage(layer: 1, experts: [7]) }
        store.endToken(); store.endToken()
        #expect(!store.pinnedExperts(layer: 1).contains(7))
    }

    @Test func tierPickersFollowUpstream() {
        // tier_pick_swap: hottest unpinned vs coldest pinned
        #expect(Tier.pickSwap(heat: [10, 1, 30, 2], pinned: [0, 1]) == Tier.Swap(slot: 1, expert: 2, gain: 29))
        #expect(Tier.pickSwap(heat: [10, 8, 9, 2], pinned: [0, 1]) == nil)
        // tier_pick_lfru: recency breaks ties only
        let last: [UInt32] = [100, 100, 100, 100]
        #expect(Tier.pickLFRU(heat: [10, 1, 30, 2], last: last, clock: 100, pinned: [0, 1])?.expert == 2)
        #expect(Tier.pickLFRU(heat: [10, 8, 9, 2], last: last, clock: 100, pinned: [0, 1]) == nil)
        #expect(Tier.lfruScore(heat: 1, last: 5, clock: 5) == 256 + 255)
        #expect(Tier.lfruScore(heat: 1, last: 0, clock: 1000) == 256)
        #expect(Tier.lfruScore(heat: 0, last: UInt32.max, clock: 1) == 253)  // wrapping age 2
    }

    @Test func factoryPicksResidentOrStreaming() throws {
        let fx = try ExpertFixture()
        let cpu = CPUBackend(threads: 1)
        let size = fx.bytes.count
        // everything fits: resident, zero copy
        let a = try ExpertStores.make(path: fx.path, fileSize: size, fileRegion: { fx.fileRegion() }, layout: fx.layout,
                                      backend: cpu, options: ExpertStoreOptions(saveHistory: false), engineName: "olmoe", topK: 2)
        #expect(a is ResidentExpertStore)
        // forced streaming with a slot count
        let b = try ExpertStores.make(path: fx.path, fileSize: size, fileRegion: { fx.fileRegion() }, layout: fx.layout,
                                      backend: cpu, options: ExpertStoreOptions(mode: .stream, slotsPerLayer: 3),
                                      engineName: "olmoe", topK: 2)
        let bs = try #require(b as? StreamingExpertStore)
        #expect(bs.slotsPerLayer == [0, 3, 3, 3])
        // a cache budget smaller than the experts: streaming, at least top-k slots
        let stride = (1 ... 3).reduce(0) { $0 + ExpertArena.slotStride(for: fx.layout.layers[$1]!) }
        let c = try ExpertStores.make(path: fx.path, fileSize: size, fileRegion: { fx.fileRegion() }, layout: fx.layout,
                                      backend: cpu, options: ExpertStoreOptions(cacheBytes: 3 * stride),
                                      engineName: "olmoe", topK: 2)
        #expect((c as? StreamingExpertStore)?.slotsPerLayer == [0, 3, 3, 3])
        let d = try ExpertStores.make(path: fx.path, fileSize: size, fileRegion: { fx.fileRegion() }, layout: fx.layout,
                                      backend: cpu, options: ExpertStoreOptions(cacheBytes: 1), engineName: "olmoe", topK: 2)
        #expect((d as? StreamingExpertStore)?.slotsPerLayer == [0, 2, 2, 2])
        // --experts stream on a model that fits: a cache of every expert
        let e = try ExpertStores.make(path: fx.path, fileSize: size, fileRegion: { fx.fileRegion() }, layout: fx.layout,
                                      backend: cpu, options: ExpertStoreOptions(mode: .stream), engineName: "olmoe", topK: 2)
        #expect((e as? StreamingExpertStore)?.slotsPerLayer == [0, 8, 8, 8])
        // a file shorter than the layout says
        #expect(throws: ColibriError.self) {
            try ExpertStores.make(path: fx.path, fileSize: 100, fileRegion: { fx.fileRegion() }, layout: fx.layout,
                                  backend: cpu, options: ExpertStoreOptions(), engineName: "olmoe")
        }
    }

    @Test func factoryLoadsHistoryPinsAndSavesIt() throws {
        let fx = try ExpertFixture()
        let cpu = CPUBackend(threads: 1)
        let path = ExpertStores.defaultHistoryPath(modelPath: fx.path)
        try "-1 4 8\n-2 1 \(RoutingHistory.engineID("olmoe"))\n1 3 9\n1 4 7\n2 0 5\n".write(toFile: path, atomically: true, encoding: .utf8)
        let store = try ExpertStores.make(path: fx.path, fileSize: fx.bytes.count, fileRegion: { fx.fileRegion() },
                                          layout: fx.layout, backend: cpu,
                                          options: ExpertStoreOptions(mode: .stream, slotsPerLayer: 4), engineName: "olmoe", topK: 2)
        let s = try #require(store as? StreamingExpertStore)
        #expect(s.pinnedExperts(layer: 1) == [3, 4])
        #expect(s.pinnedExperts(layer: 2) == [0])
        s.waitUntilIdle()
        let l = try s.acquire(layer: 1, experts: [3, 4, 3])
        #expect(l.misses.isEmpty)
        s.release(l)
        try s.saveHistory()
        let saved = try String(contentsOfFile: path, encoding: .utf8)
        #expect(saved == "-1 4 8\n-2 1 \(RoutingHistory.engineID("olmoe"))\n1 3 11\n1 4 8\n2 0 5\n")
        // another engine's history is refused (not named explicitly): no pins
        let other = try ExpertStores.make(path: fx.path, fileSize: fx.bytes.count, fileRegion: { fx.fileRegion() },
                                          layout: fx.layout, backend: cpu,
                                          options: ExpertStoreOptions(mode: .stream, slotsPerLayer: 4, saveHistory: false),
                                          engineName: "qwen3moe", topK: 2)
        #expect((other as? StreamingExpertStore)?.pinnedExperts(layer: 1) == [])
    }
}

@Suite struct ExpertGGUFIntegrationTests {
    /// Through workstream A's writer and reader: a real GGUF with fused expert tensors.
    @Test func storesServeTheTensorsOfAParsedGGUF() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("colibri-expert-gguf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("tiny.gguf").path
        var rng = SplitMix64(seed: 3)
        func random(_ n: Int) -> [UInt8] { (0 ..< n).map { _ in UInt8.random(in: 0 ... 255, using: &rng) } }
        var w = GGUFWriter()
        w.addTensor("blk.0.attn_norm.weight", dims: [64], type: .f32, bytes: random(256))
        for l in 1 ... 2 {
            w.addTensor("blk.\(l).ffn_gate_exps.weight", dims: [64, 32, 6], type: .q8_0, bytes: random(68 * 32 * 6))
            w.addTensor("blk.\(l).ffn_up_exps.weight", dims: [64, 32, 6], type: .q8_0, bytes: random(68 * 32 * 6))
            w.addTensor("blk.\(l).ffn_down_exps.weight", dims: [32, 64, 6], type: .f16, bytes: random(64 * 64 * 6))
        }
        try w.write(to: path)
        let file = try GGUFFile(path: path)
        let layout = try ExpertLayout(file: file, layerCount: 3)
        #expect(layout.moeLayers == [1, 2])
        let raw = try [UInt8](Data(contentsOf: URL(fileURLWithPath: path)))
        let cpu = CPUBackend(threads: 1)

        let resident = try ExpertStores.make(file: file, layout: layout, backend: cpu,
                                             options: ExpertStoreOptions(saveHistory: false), engineName: "olmoe", topK: 2)
        #expect(resident is ResidentExpertStore)
        let streamed = try ExpertStores.make(file: file, layout: layout, backend: cpu,
                                             options: ExpertStoreOptions(mode: .stream, slotsPerLayer: 3, saveHistory: false),
                                             engineName: "olmoe", topK: 2)
        #expect(streamed is StreamingExpertStore)
        for l in 1 ... 2 {
            // the resident view is exactly the reader's own stacked view, in place
            let stacked = try file.stackedWeights("blk.\(l).ffn_down_exps.weight")
            let rw = resident.weights(layer: l)
            #expect(rw.down.region === stacked.region && rw.down.baseOffset == stacked.baseOffset)
            #expect(rw.down.stride == stacked.stride && rw.down.count == stacked.count)
            for e: Int32 in [5, 0, 3, 5, 1] {
                let a = try resident.acquire(layer: l, experts: [e])
                let b = try streamed.acquire(layer: l, experts: [e])
                streamed.waitForMisses(b)
                for (kind, s) in [(\ExpertLayerWeights.gate, layout.layers[l]!.gate), (\.up, layout.layers[l]!.up),
                                  (\.down, layout.layers[l]!.down)] {
                    let x = resident.weights(layer: l)[keyPath: kind].matrix(Int(a.slots[0]))
                    let y = streamed.weights(layer: l)[keyPath: kind].matrix(Int(b.slots[0]))
                    let bx = UnsafeRawBufferPointer(start: x.rowPointer(0), count: x.byteCount)
                    let by = UnsafeRawBufferPointer(start: y.rowPointer(0), count: y.byteCount)
                    #expect(bx.elementsEqual(by))
                    #expect(bx.elementsEqual(raw[s.range(expert: Int(e))]))
                }
                resident.release(a)
                streamed.release(b)
            }
        }
    }
}

/// A tiny lock-protected counter for the stress test.
final class Atomicish: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func add() { lock.lock(); n += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}
