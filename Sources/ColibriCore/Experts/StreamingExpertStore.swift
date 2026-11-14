import Foundation

// Expert streaming: a per-layer slot cache filled by `pread`, ported in behaviour from
// colibri c/olmoe.c (LCache, Slot, expert_get/expert_put, olmoe_victim, pin_hot_experts),
// c/qwen36.c (load_expert_merged: one expert = three contiguous reads from the fused
// GGUF tensors) and c/colibri.c (page-aligned slabs registered with Metal, the PIPE I/O
// pool). Design: docs/port/runtime-and-metal.md sections 2 and 6.

/// Owns one arena allocation; the arena's `MemoryRegion` keeps it alive.
final class ArenaMemory {
    let raw: UnsafeMutableRawPointer
    let length: Int
    init(length: Int) throws {
        var p: UnsafeMutableRawPointer?
        guard posix_memalign(&p, hostPageSize, length) == 0, let p else {
            throw ColibriError.backend("cannot allocate an expert arena of \(length) bytes")
        }
        memset(p, 0, length)
        raw = p
        self.length = length
    }
    deinit { free(raw) }
}

/// One MoE layer's slot memory: `slots` slots at a 16 KiB-aligned `stride`, one
/// page-aligned allocation wrapped once as a `MemoryRegion` (Metal: one buffer per layer,
/// kernels index `base + slot * stride`). Slot layout: gate | up | down at the offsets
/// below.
public final class ExpertArena {
    public let layer: Int
    public let slots: Int
    public let stride: Int
    public let gateOffset: Int
    public let upOffset: Int
    public let downOffset: Int
    /// The whole arena: base and length are multiples of `hostPageSize`.
    public let region: MemoryRegion
    private let memory: ArenaMemory

    /// Alignment of the up and down matrices inside a slot.
    public static let matrixAlignment = 256

    /// Offsets of gate, up and down inside a slot, and the slot stride.
    public static func slotGeometry(for l: ExpertLayerLayout) -> (gate: Int, up: Int, down: Int, stride: Int) {
        let up = alignUp(l.gate.bytesPerExpert, matrixAlignment)
        let down = alignUp(up + l.up.bytesPerExpert, matrixAlignment)
        return (0, up, down, alignUp(down + l.down.bytesPerExpert, hostPageSize))
    }

    /// Bytes one slot of this layer takes in an arena.
    public static func slotStride(for l: ExpertLayerLayout) -> Int { slotGeometry(for: l).stride }

    public init(layout: ExpertLayerLayout, slots: Int) throws {
        guard slots >= 1 else { throw ColibriError.invalidArgument("expert arena needs at least one slot") }
        let g = Self.slotGeometry(for: layout)
        let (length, overflow) = g.stride.multipliedReportingOverflow(by: slots)
        guard !overflow else { throw ColibriError.invalidArgument("expert arena size overflows") }
        layer = layout.layer
        self.slots = slots
        stride = g.stride
        gateOffset = g.gate
        upOffset = g.up
        downOffset = g.down
        memory = try ArenaMemory(length: length)
        region = MemoryRegion(base: UnsafeRawPointer(memory.raw), length: length, kind: .expertArena,
                              label: "experts L\(layout.layer)", keepAlive: memory)
    }

    public var byteCount: Int { slots * stride }

    public func slotPointer(_ slot: Int) -> UnsafeMutableRawPointer {
        precondition(slot >= 0 && slot < slots, "slot \(slot) out of range 0..<\(slots)")
        return memory.raw + slot * stride
    }
}

// MARK: - Cache state

enum SlotState: Equatable { case empty, loading, ready }

struct Slot {
    var expert: Int32 = -1
    var state: SlotState = .empty
    /// Leases holding the slot (upstream's `busy`): a busy slot is never evicted.
    var busy = 0
    /// LRU clock of the last use.
    var used: UInt64 = 0
    /// Loaded (or loading) by a prefetch and not yet used by a lease.
    var prefetched = false
    /// Sequence number on the store's signal; 0 = a queued prefetch whose read has not
    /// started (it gets one when it starts, or when a lease promotes it).
    var seq: UInt64 = 0
    /// A worker is reading into the slot.
    var claimed = false
    /// Load generation: identifies the read job that fills the slot.
    var gen: UInt64 = 0
}

/// One layer's slot table, guarded by `cond`.
final class LayerCache: @unchecked Sendable {
    let layer: Int
    let layout: ExpertLayerLayout
    let arena: ExpertArena
    let pinCap: Int
    /// Number of slots (immutable, readable without the lock).
    let capacity: Int
    let cond = NSCondition()
    var slots: [Slot]
    /// expert -> slot, -1 when not resident or loading.
    var slotOf: [Int32]
    /// The frequency-protected segment: experts LRU eviction skips.
    var pinned: [Bool]
    /// Session heat (decayed at each repin) and last access on `clock`, for LFRU.
    var heat: [UInt32]
    var lastAccess: [UInt32]
    var clock: UInt32 = 0
    var tick: UInt64 = 0
    var nextGen: UInt64 = 0
    var leases: [UInt64: [Int]] = [:]

    init(layout: ExpertLayerLayout, arena: ExpertArena, pinCap: Int) {
        layer = layout.layer
        self.layout = layout
        self.arena = arena
        self.pinCap = pinCap
        capacity = arena.slots
        slots = [Slot](repeating: Slot(), count: arena.slots)
        slotOf = [Int32](repeating: -1, count: layout.count)
        pinned = [Bool](repeating: false, count: layout.count)
        heat = [UInt32](repeating: 0, count: layout.count)
        lastAccess = [UInt32](repeating: 0, count: layout.count)
    }

    /// `olmoe_victim` plus its fallbacks. Called with `cond` held.
    /// 1. an empty slot nobody holds;
    /// 2. least recently used ready slot that is not pinned and not busy (and, for a
    ///    prefetch, not an unconsumed prefetch: prefetches never evict each other);
    /// 3. demand only: least recently used ready slot that is not busy, pinned or not.
    /// Loading and busy slots are never chosen.
    func victim(demand: Bool, sparePrefetched: Bool) -> Int? {
        if let i = slots.firstIndex(where: { $0.state == .empty && $0.busy == 0 }) { return i }
        var best = -1
        for i in slots.indices {
            let s = slots[i]
            guard s.state == .ready, s.busy == 0, !pinned[Int(s.expert)] else { continue }
            if sparePrefetched && s.prefetched { continue }
            if best < 0 || s.used < slots[best].used { best = i }
        }
        if best >= 0 || !demand { return best >= 0 ? best : nil }
        for i in slots.indices {
            let s = slots[i]
            guard s.state == .ready, s.busy == 0 else { continue }
            if best < 0 || s.used < slots[best].used { best = i }
        }
        return best >= 0 ? best : nil
    }

    /// Points slot `i` at `expert` in the loading state. Returns whether a ready expert
    /// was evicted. Called with `cond` held.
    func reserve(_ i: Int, expert: Int32, busy: Int, prefetched: Bool, seq: UInt64) -> (evicted: Bool, gen: UInt64) {
        let old = slots[i]
        if old.expert >= 0 { slotOf[Int(old.expert)] = -1 }
        tick += 1
        nextGen += 1
        slots[i] = Slot(expert: expert, state: .loading, busy: busy, used: tick, prefetched: prefetched,
                        seq: seq, claimed: false, gen: nextGen)
        slotOf[Int(expert)] = Int32(i)
        return (old.state == .ready, nextGen)
    }

    var isLoadingFreeSlot: Bool { slots.contains { $0.state == .loading && $0.busy == 0 } }
}

/// State shared by the engine thread and the I/O jobs; jobs capture this (and a
/// `LayerCache`), never the store, so the store can shut its pool down in `deinit`.
final class StreamShared: @unchecked Sendable {
    let lock = NSCondition()
    let signal: HostSignal
    let fd: Int32
    var lastSeq: UInt64 = 0
    /// Every sequence number up to here has completed; the signal's value.
    var watermark: UInt64 = 0
    var completedAhead = Set<UInt64>()
    var counters = ExpertStoreStats()
    var activeReads = 0
    var activeSince: UInt64 = 0
    var firstError: Error?
    var nextToken: UInt64 = 1
    var tokens = 0
    /// Tests: microseconds each read sleeps before reading, to widen race windows.
    var readDelayMicros: UInt32 = 0
    /// How long an acquire waits for a slot when every slot is leased (single-threaded
    /// use would otherwise deadlock on its own leases).
    var leaseWaitTimeout: Double = 10

    init(signal: HostSignal, fd: Int32) {
        self.signal = signal
        self.fd = fd
    }

    func assignSeq() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        lastSeq += 1
        return lastSeq
    }

    func makeToken() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        let t = nextToken
        nextToken += 1
        return t
    }

    func beginRead() -> UInt32 {
        lock.lock(); defer { lock.unlock() }
        if activeReads == 0 { activeSince = DispatchTime.now().uptimeNanoseconds }
        activeReads += 1
        return readDelayMicros
    }

    /// Ends a read: stats, then advances the watermark past `seq` (out-of-order
    /// completions wait for the ones before them) and signals it.
    func finish(seq: UInt64, bytes: Int, seconds: Double, error: Error?) {
        lock.lock()
        activeReads -= 1
        if activeReads == 0 {
            counters.ioWallSeconds += Double(DispatchTime.now().uptimeNanoseconds - activeSince) / 1e9
        }
        counters.diskSeconds += seconds
        if let error {
            if firstError == nil { firstError = error }
        } else {
            counters.bytesRead += bytes
            counters.reads += 1
        }
        completedAhead.insert(seq)
        while completedAhead.remove(watermark + 1) != nil { watermark += 1 }
        signal.signal(watermark)
        lock.broadcast()
        lock.unlock()
    }
}

/// Per-layer slot cache over `pread` (F_NOCACHE), ported in behaviour from colibri
/// c/olmoe.c (LCache, Slot, expert_get/expert_put, olmoe_victim) and c/qwen36.c
/// (load_expert_merged): slot states empty / loading / ready, a busy refcount that
/// blocks eviction, in-flight waits instead of duplicate reads, LRU eviction that skips
/// pinned, loading and busy slots, pins from the routing history, optional live repin
/// (Tier) and router lookahead prefetch. Misses are published on `signal`.
///
/// Cache policy: LRU over the slots, with a frequency-protected segment of at most
/// `slots - topK` pinned experts per layer. Pins come from the persisted routing
/// history at startup (`pinFromHistory`) and, with `repinEvery`, are re-ranked at token
/// boundaries by session heat (LFRU score, colibri c/tier.h hysteresis, at most four
/// swaps per boundary). Eviction takes an empty slot, else the least recently used
/// unpinned slot, else (demand only) the least recently used pinned one; it never takes
/// a loading or leased slot.
///
/// Signal protocol: every read gets a sequence number when it is queued as a demand
/// read or starts as a prefetch. The store signals the highest number below which all
/// reads have completed, so `lease.readyValue` (the largest number among the lease's
/// misses) is reached only once every miss of the lease has landed.
///
/// Thread-safe: any thread may acquire, release, prefetch or read stats; one lock per
/// layer plus one for the shared counters (layer lock first).
public final class StreamingExpertStore: ExpertStore {
    public let layout: ExpertLayout
    public var isFullyResident: Bool { false }
    public var signal: HostSignal? { shared.signal }
    public let history: RoutingHistory?
    /// Where `saveHistory` writes; nil = read-only history.
    public let historyPath: String?
    public let topK: Int
    public let repinEvery: Int
    public let uncached: Bool
    public let path: String

    let caches: [LayerCache?]
    let shared: StreamShared
    private let pool: IOPool
    private let weightsByLayer: [ExpertLayerWeights?]
    private let closeLock = NSLock()
    private var closed = false

    public convenience init(file: GGUFFile, layout: ExpertLayout, slotsPerLayer: [Int], backend: Backend,
                            ioThreads: Int, history: RoutingHistory?) throws {
        try self.init(path: file.file.path, layout: layout, slotsPerLayer: slotsPerLayer, backend: backend,
                      ioThreads: ioThreads, history: history)
    }

    /// - Parameters:
    ///   - path: the GGUF file; opened again for `pread` (with `F_NOCACHE` when `uncached`).
    ///   - slotsPerLayer: slot count for each layer (indexed by layer; dense layers ignored),
    ///     clamped to the layer's expert count and to what one backend buffer can hold.
    ///   - topK: experts per token; pins never take more than `slots - topK` slots.
    public init(path: String, layout: ExpertLayout, slotsPerLayer: [Int], backend: Backend,
                ioThreads: Int = 8, history: RoutingHistory? = nil, historyPath: String? = nil,
                uncached: Bool = true, topK: Int = 8, repinEvery: Int = 0) throws {
        guard slotsPerLayer.count == layout.layers.count else {
            throw ColibriError.invalidArgument("\(slotsPerLayer.count) slot counts for \(layout.layers.count) layers")
        }
        let fd = try IOPool.open(path, uncached: uncached)
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            let e = errno
            Darwin.close(fd)
            throw ColibriError.io(path: path, operation: "fstat", errno: e)
        }
        guard layout.fileEnd <= Int(st.st_size) else {
            Darwin.close(fd)
            throw ColibriError.invalidGGUF("expert tensors end at byte \(layout.fileEnd), \(path) has \(st.st_size)")
        }
        var caches: [LayerCache?] = []
        var weights: [ExpertLayerWeights?] = []
        do {
            for (l, ll) in layout.layers.enumerated() {
                guard let ll else {
                    caches.append(nil)
                    weights.append(nil)
                    continue
                }
                let stride = ExpertArena.slotStride(for: ll)
                let fit = backend.capabilities.maxBufferLength / stride
                let n = min(ll.count, slotsPerLayer[l], fit)
                guard n >= 1 else {
                    throw ColibriError.invalidArgument("layer \(l): \(slotsPerLayer[l]) slots requested, \(fit) fit one backend buffer")
                }
                let arena = try ExpertArena(layout: ll, slots: n)
                try backend.register(arena.region)
                caches.append(LayerCache(layout: ll, arena: arena, pinCap: max(0, n - topK)))
                func stack(_ s: ExpertTensorSlice, _ off: Int) -> StackedWeights {
                    StackedWeights(region: arena.region, baseOffset: off, stride: arena.stride, count: n,
                                   type: s.type, cols: s.cols, rows: s.rows)
                }
                weights.append(ExpertLayerWeights(gate: stack(ll.gate, arena.gateOffset), up: stack(ll.up, arena.upOffset),
                                                  down: stack(ll.down, arena.downOffset)))
            }
        } catch {
            Darwin.close(fd)
            throw error
        }
        self.layout = layout
        self.path = path
        self.history = history
        self.historyPath = historyPath
        self.topK = topK
        self.repinEvery = repinEvery
        self.uncached = uncached
        self.caches = caches
        self.weightsByLayer = weights
        self.shared = StreamShared(signal: backend.makeSignal(), fd: fd)
        self.pool = IOPool(threads: ioThreads)
    }

    deinit { close() }

    /// Waits for queued reads, stops the I/O threads and closes the file. Idempotent;
    /// the store must not be used afterwards.
    public func close() {
        closeLock.lock()
        defer { closeLock.unlock() }
        guard !closed else { return }
        closed = true
        pool.shutdown()
        Darwin.close(shared.fd)
    }

    // MARK: ExpertStore

    public var stats: ExpertStoreStats {
        shared.lock.lock()
        var s = shared.counters
        shared.lock.unlock()
        for lc in caches.compactMap({ $0 }) {
            lc.cond.lock()
            s.pinned += lc.pinned.reduce(0) { $0 + ($1 ? 1 : 0) }
            lc.cond.unlock()
        }
        return s
    }

    public func weights(layer: Int) -> ExpertLayerWeights {
        guard layer >= 0, layer < weightsByLayer.count, let w = weightsByLayer[layer] else {
            preconditionFailure("layer \(layer) has no routed experts")
        }
        return w
    }

    /// Slots per layer (0 for dense layers).
    public var slotsPerLayer: [Int] { caches.map { $0?.capacity ?? 0 } }

    private func cache(_ layer: Int) throws -> LayerCache {
        guard layer >= 0, layer < caches.count, let lc = caches[layer] else {
            throw ColibriError.invalidArgument("layer \(layer) has no routed experts")
        }
        return lc
    }

    private func distinctValid(_ experts: [Int32], count: Int, layer: Int) throws -> [Int32] {
        var seen = Set<Int32>()
        var out: [Int32] = []
        for e in experts {
            guard e >= 0, Int(e) < count else {
                throw ColibriError.invalidArgument("expert \(e) out of range 0..<\(count) in layer \(layer)")
            }
            if seen.insert(e).inserted { out.append(e) }
        }
        return out
    }

    private func submit(_ priority: IOPool.Priority, _ lc: LayerCache, slot: Int, gen: UInt64) {
        let shared = self.shared
        pool.submit(priority) { Self.load(lc, shared, slot: slot, gen: gen) }
    }

    /// The read job: three `pread`s (gate, up, down) straight into the slot, then publish.
    private static func load(_ lc: LayerCache, _ shared: StreamShared, slot s: Int, gen: UInt64) {
        lc.cond.lock()
        guard lc.slots[s].gen == gen, lc.slots[s].state == .loading, !lc.slots[s].claimed else {
            lc.cond.unlock()  // the other job of a promoted prefetch got there first
            return
        }
        lc.slots[s].claimed = true
        if lc.slots[s].seq == 0 { lc.slots[s].seq = shared.assignSeq() }
        let seq = lc.slots[s].seq
        let e = Int(lc.slots[s].expert)
        lc.cond.unlock()

        let delay = shared.beginRead()
        let t0 = DispatchTime.now().uptimeNanoseconds
        var failure: Error?
        do {
            if delay > 0 { usleep(delay) }
            let dst = lc.arena.slotPointer(s)
            let l = lc.layout
            try IOPool.readFully(fd: shared.fd, into: dst + lc.arena.gateOffset, count: l.gate.bytesPerExpert,
                                 offset: l.gate.fileOffset + e * l.gate.bytesPerExpert)
            try IOPool.readFully(fd: shared.fd, into: dst + lc.arena.upOffset, count: l.up.bytesPerExpert,
                                 offset: l.up.fileOffset + e * l.up.bytesPerExpert)
            try IOPool.readFully(fd: shared.fd, into: dst + lc.arena.downOffset, count: l.down.bytesPerExpert,
                                 offset: l.down.fileOffset + e * l.down.bytesPerExpert)
        } catch {
            failure = error
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9

        lc.cond.lock()
        lc.slots[s].claimed = false
        if failure == nil {
            lc.slots[s].state = .ready
        } else {
            lc.slots[s].state = .empty
            lc.slots[s].prefetched = false
            if lc.slotOf[e] == Int32(s) { lc.slotOf[e] = -1 }
            lc.slots[s].expert = -1
        }
        lc.cond.broadcast()
        lc.cond.unlock()
        shared.finish(seq: seq, bytes: lc.layout.slotBytes, seconds: seconds, error: failure)
    }

    private func checkError() throws {
        shared.lock.lock()
        defer { shared.lock.unlock() }
        if let e = shared.firstError {
            shared.firstError = nil
            throw e
        }
    }

    public func acquire(layer: Int, experts: [Int32]) throws -> ExpertLease {
        let lc = try cache(layer)
        try checkError()
        let distinct = try distinctValid(experts, count: lc.layout.count, layer: layer)
        guard distinct.count <= lc.capacity else {
            throw ColibriError.invalidArgument(
                "layer \(layer): \(distinct.count) distinct experts in one lease, the cache has \(lc.capacity) slots; acquire fewer rows at a time")
        }
        history?.record(layer: layer, experts: experts)

        lc.cond.lock()
        defer { lc.cond.unlock() }
        lc.clock &+= 1
        for e in experts {
            let (h, o) = lc.heat[Int(e)].addingReportingOverflow(1)
            lc.heat[Int(e)] = o ? .max : h
            lc.lastAccess[Int(e)] = lc.clock
        }

        var slotFor: [Int32: Int32] = [:]
        var missed = Set<Int32>()
        var taken: [Int] = []
        var readyValue: UInt64 = 0
        var hits = 0, misses = 0, inFlight = 0, prefetchHits = 0, evictions = 0
        let deadline = Date(timeIntervalSinceNow: shared.leaseWaitTimeout)
        var i = 0
        while i < distinct.count {
            let e = distinct[i]
            let cur = Int(lc.slotOf[Int(e)])
            if cur >= 0 {
                lc.tick += 1
                lc.slots[cur].used = lc.tick
                lc.slots[cur].busy += 1
                if lc.slots[cur].state == .ready {
                    hits += 1
                    if lc.slots[cur].prefetched { prefetchHits += 1 }
                } else {
                    // In flight (a prefetch, a pin load, or another thread's miss): wait for
                    // that read instead of starting a second one.
                    misses += 1
                    inFlight += 1
                    missed.insert(e)
                    if lc.slots[cur].seq == 0 {
                        // A prefetch still queued behind other work: promote it to a demand read.
                        lc.slots[cur].seq = shared.assignSeq()
                        submit(.demand, lc, slot: cur, gen: lc.slots[cur].gen)
                    }
                    readyValue = max(readyValue, lc.slots[cur].seq)
                }
                lc.slots[cur].prefetched = false
                taken.append(cur)
                slotFor[e] = Int32(cur)
                i += 1
                continue
            }
            guard let v = lc.victim(demand: true, sparePrefetched: false) else {
                // No evictable slot. Drop this acquire's partial reservations before
                // waiting (no hold-and-wait: concurrent acquirers cannot deadlock each
                // other), wait for a read to land or a lease to end, then start over.
                for s in taken { lc.slots[s].busy -= 1 }
                lc.cond.broadcast()
                taken.removeAll()
                slotFor.removeAll()
                missed.removeAll()
                readyValue = 0
                hits = 0; misses = 0; inFlight = 0; prefetchHits = 0
                i = 0
                if lc.isLoadingFreeSlot {
                    lc.cond.wait()  // an unleased read will land and become evictable
                } else if !lc.cond.wait(until: deadline), lc.victim(demand: true, sparePrefetched: false) == nil {
                    throw ColibriError.invalidArgument(
                        "layer \(layer): every slot is leased; release earlier leases before acquiring more experts")
                }
                continue
            }
            let seq = shared.assignSeq()
            let r = lc.reserve(v, expert: e, busy: 1, prefetched: false, seq: seq)
            if r.evicted { evictions += 1 }
            submit(.demand, lc, slot: v, gen: r.gen)
            misses += 1
            missed.insert(e)
            readyValue = max(readyValue, seq)
            taken.append(v)
            slotFor[e] = Int32(v)
            i += 1
        }

        let token = shared.makeToken()
        lc.leases[token] = taken
        var missPositions: [Int] = []
        let slots = experts.enumerated().map { (pos, e) -> Int32 in
            if missed.contains(e) { missPositions.append(pos) }
            return slotFor[e]!
        }
        shared.lock.lock()
        shared.counters.lookups += distinct.count
        shared.counters.hits += hits
        shared.counters.misses += misses
        shared.counters.inFlightWaits += inFlight
        shared.counters.prefetchHits += prefetchHits
        shared.counters.evictions += evictions
        shared.lock.unlock()
        return ExpertLease(layer: layer, experts: experts, slots: slots, misses: missPositions,
                           readyValue: missPositions.isEmpty ? 0 : readyValue, token: token)
    }

    public func waitForMisses(_ lease: ExpertLease) {
        guard lease.readyValue > 0 else { return }
        shared.lock.lock()
        if shared.watermark < lease.readyValue {
            let t0 = DispatchTime.now().uptimeNanoseconds
            while shared.watermark < lease.readyValue { shared.lock.wait() }
            shared.counters.waitSeconds += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        }
        shared.lock.unlock()
    }

    /// Every slot of the lease must be ready and still hold the expert it was leased
    /// for (a leased slot is never evicted, so only a failed read leaves it otherwise).
    /// Throws the read's error (taking it, so the next `acquire` does not report it
    /// again), or a generic one if another failure already took that.
    public func checkLease(_ lease: ExpertLease) throws {
        guard !lease.misses.isEmpty else { return }  // hits were ready and stay so while leased
        let lc = try cache(lease.layer)
        lc.cond.lock()
        var bad: Int32?
        for p in lease.misses {
            let s = Int(lease.slots[p])
            if lc.slots[s].state != .ready || lc.slots[s].expert != lease.experts[p] { bad = lease.experts[p]; break }
        }
        lc.cond.unlock()
        guard let bad else { return }
        try checkError()
        throw ColibriError.backend("layer \(lease.layer): reading expert \(bad) failed")
    }

    public func release(_ lease: ExpertLease) {
        guard let lc = try? cache(lease.layer) else { preconditionFailure("lease for a dense layer") }
        lc.cond.lock()
        guard let slots = lc.leases.removeValue(forKey: lease.token) else {
            lc.cond.unlock()
            preconditionFailure("expert lease \(lease.token) released twice or not from this store")
        }
        for s in slots { lc.slots[s].busy -= 1 }
        lc.cond.broadcast()
        lc.cond.unlock()
    }

    /// Advisory read-ahead: queues reads (behind demand reads) for experts neither
    /// resident nor in flight, into empty slots or least recently used unpinned ones.
    /// Never evicts a leased, loading, pinned or not-yet-used prefetched slot; drops
    /// what does not fit.
    public func prefetch(layer: Int, experts: [Int32]) {
        guard let lc = try? cache(layer),
              let distinct = try? distinctValid(experts, count: lc.layout.count, layer: layer) else { return }
        var issued = 0, evictions = 0
        lc.cond.lock()
        for e in distinct where lc.slotOf[Int(e)] < 0 {
            guard let v = lc.victim(demand: false, sparePrefetched: true) else { break }
            let r = lc.reserve(v, expert: e, busy: 0, prefetched: true, seq: 0)
            if r.evicted { evictions += 1 }
            submit(.prefetch, lc, slot: v, gen: r.gen)
            issued += 1
        }
        lc.cond.unlock()
        shared.lock.lock()
        shared.counters.prefetchIssued += issued
        shared.counters.evictions += evictions
        shared.lock.unlock()
    }

    public func recordUsage(layer: Int, experts: [Int32]) {
        history?.record(layer: layer, experts: experts)
        guard let lc = try? cache(layer) else { return }
        lc.cond.lock()
        lc.clock &+= 1
        for e in experts where e >= 0 && Int(e) < lc.layout.count {
            let (h, o) = lc.heat[Int(e)].addingReportingOverflow(1)
            lc.heat[Int(e)] = o ? .max : h
            lc.lastAccess[Int(e)] = lc.clock
        }
        lc.cond.unlock()
    }

    public func endToken() {
        shared.lock.lock()
        shared.tokens += 1
        let due = repinEvery > 0 && shared.tokens % repinEvery == 0
        shared.lock.unlock()
        guard due else { return }
        var swaps = 0
        for lc in caches.compactMap({ $0 }) { swaps += repin(lc) }
        shared.lock.lock()
        shared.counters.repinSwaps += swaps
        shared.lock.unlock()
    }

    public func saveHistory() throws {
        guard let history, let historyPath else { return }
        try history.save(path: historyPath)
    }

    // MARK: Pins

    /// Makes `experts` (most important first) the layer's pinned set, truncated to
    /// `slots - topK`, and queues reads for those not resident. Pinned experts are
    /// skipped by LRU eviction.
    public func pin(layer: Int, experts: [Int32]) {
        guard let lc = try? cache(layer),
              let distinct = try? distinctValid(experts, count: lc.layout.count, layer: layer) else { return }
        let chosen = Array(distinct.prefix(lc.pinCap))
        lc.cond.lock()
        for e in lc.pinned.indices { lc.pinned[e] = false }
        for e in chosen { lc.pinned[Int(e)] = true }
        loadPinned(lc, chosen)
        lc.cond.unlock()
    }

    /// Pins each layer's most used experts from the routing history (upstream PIN=auto).
    public func pinFromHistory() {
        guard let history else { return }
        for lc in caches.compactMap({ $0 }) where lc.pinCap > 0 {
            let top = history.top(layer: lc.layer, lc.pinCap)
            if !top.isEmpty { pin(layer: lc.layer, experts: top) }
        }
    }

    /// The pinned experts of `layer`, ascending.
    public func pinnedExperts(layer: Int) -> [Int32] {
        guard let lc = try? cache(layer) else { return [] }
        lc.cond.lock(); defer { lc.cond.unlock() }
        return lc.pinned.indices.filter { lc.pinned[$0] }.map { Int32($0) }
    }

    /// Queues reads for pinned experts not resident. Called with `lc.cond` held.
    private func loadPinned(_ lc: LayerCache, _ experts: [Int32]) {
        for e in experts where lc.slotOf[Int(e)] < 0 {
            guard let v = lc.victim(demand: false, sparePrefetched: false) else { break }
            let r = lc.reserve(v, expert: e, busy: 0, prefetched: false, seq: 0)
            submit(.prefetch, lc, slot: v, gen: r.gen)
        }
    }

    /// Live repin at a token boundary (colibri c/tier.h): fill free pin seats with the
    /// hottest unpinned experts, then swap via `Tier.pickLFRU`, at most
    /// `Tier.maxSwapsPerRepin` changes in all; then decay the session heat. Returns the
    /// number of changes.
    private func repin(_ lc: LayerCache) -> Int {
        lc.cond.lock()
        defer { lc.cond.unlock() }
        guard lc.pinCap > 0 else { return 0 }
        var pinnedList = lc.pinned.indices.filter { lc.pinned[$0] }
        var added: [Int32] = []
        var changes = 0
        while pinnedList.count < lc.pinCap && changes < Tier.maxSwapsPerRepin {
            var best = -1
            var bestScore: UInt64 = 0
            for e in lc.heat.indices where !lc.pinned[e] && lc.heat[e] > 0 {
                let s = Tier.lfruScore(heat: lc.heat[e], last: lc.lastAccess[e], clock: lc.clock)
                if best < 0 || s > bestScore { best = e; bestScore = s }
            }
            guard best >= 0 else { break }
            lc.pinned[best] = true
            pinnedList.append(best)
            added.append(Int32(best))
            changes += 1
        }
        while changes < Tier.maxSwapsPerRepin,
              let sw = Tier.pickLFRU(heat: lc.heat, last: lc.lastAccess, clock: lc.clock, pinned: pinnedList) {
            lc.pinned[pinnedList[sw.slot]] = false
            lc.pinned[sw.expert] = true
            pinnedList[sw.slot] = sw.expert
            added.append(Int32(sw.expert))
            changes += 1
        }
        for e in lc.heat.indices { lc.heat[e] = Tier.decay(lc.heat[e]) }
        loadPinned(lc, added)
        return changes
    }

    // MARK: Test and diagnostics hooks

    /// Blocks until no read is queued or running.
    public func waitUntilIdle() { pool.waitUntilIdle() }

    /// `(expert, state, busy)` of every slot of `layer`.
    func slotTable(layer: Int) -> [(expert: Int32, state: SlotState, busy: Int)] {
        guard let lc = try? cache(layer) else { return [] }
        lc.cond.lock(); defer { lc.cond.unlock() }
        return lc.slots.map { ($0.expert, $0.state, $0.busy) }
    }

    var readDelayMicros: UInt32 {
        get { shared.lock.lock(); defer { shared.lock.unlock() }; return shared.readDelayMicros }
        set { shared.lock.lock(); shared.readDelayMicros = newValue; shared.lock.unlock() }
    }

    var leaseWaitTimeout: Double {
        get { shared.lock.lock(); defer { shared.lock.unlock() }; return shared.leaseWaitTimeout }
        set { shared.lock.lock(); shared.leaseWaitTimeout = newValue; shared.lock.unlock() }
    }
}
