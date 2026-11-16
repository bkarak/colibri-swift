import Foundation

// colibri's core idea: dense weights stay resident, routed experts are streamed from
// disk through a RAM cache. The lease contract is upstream's c/expert_store.h
// (lookup/release/prefetch): release exactly once, a leased slot is never evicted,
// prefetch holds no lease. Placement never changes the math: every store returns the
// same bytes, so tokens are identical whichever store runs.
// Design: docs/port/runtime-and-metal.md sections 2 and 6.

/// One routed expert of one layer.
public struct ExpertKey: Hashable, Sendable, CustomStringConvertible {
    public var layer: Int32
    public var expert: Int32
    public init(layer: Int, expert: Int) {
        self.layer = Int32(layer)
        self.expert = Int32(expert)
    }
    public var description: String { "L\(layer)E\(expert)" }
}

/// The gate, up and down matrices of every addressable expert (or slot) of one layer.
/// The index space is whatever the store's leases hand out: expert ids for a resident
/// store, slot numbers for a streaming one. Pass these to `Backend.matmulExperts` with
/// the lease's `slots` (or, resident, the router's ids) as indices.
public struct ExpertLayerWeights: @unchecked Sendable {
    public var gate: StackedWeights
    public var up: StackedWeights
    public var down: StackedWeights
    public init(gate: StackedWeights, up: StackedWeights, down: StackedWeights) {
        self.gate = gate
        self.up = up
        self.down = down
    }
}

/// Experts pinned for the duration of one MoE block.
public struct ExpertLease: Sendable {
    public let layer: Int
    /// Requested expert ids, in request order (duplicates allowed: prefill rows share experts).
    public let experts: [Int32]
    /// For each requested expert, its index into `ExpertStore.weights(layer:)`.
    public let slots: [Int32]
    /// Positions in `experts` whose bytes were not resident when acquired.
    public let misses: [Int]
    /// When `misses` is non-empty: the value of the store's `HostSignal` that marks the
    /// last miss as loaded. A backend batch may `encodeWait(for:value:)` on it instead of
    /// the host calling `waitForMisses`.
    public let readyValue: UInt64
    /// Store-private lease identity.
    public let token: UInt64

    public init(layer: Int, experts: [Int32], slots: [Int32], misses: [Int], readyValue: UInt64, token: UInt64) {
        self.layer = layer
        self.experts = experts
        self.slots = slots
        self.misses = misses
        self.readyValue = readyValue
        self.token = token
    }
}

/// Counters for the end-of-run footer (`--stats`) and the streaming tests.
public struct ExpertStoreStats: Sendable, Equatable {
    public var lookups = 0
    public var hits = 0
    public var misses = 0
    /// Misses that found the expert already in flight (waited instead of re-reading).
    public var inFlightWaits = 0
    public var prefetchIssued = 0
    /// Prefetched experts that a later lookup hit.
    public var prefetchHits = 0
    public var evictions = 0
    public var bytesRead = 0
    /// Total time I/O threads spent in `pread`.
    public var diskSeconds = 0.0
    /// Time the engine thread spent blocked on misses (smaller than `diskSeconds` when
    /// overlap works).
    public var waitSeconds = 0.0
    /// Wall-clock time during which at least one read was in progress. With several I/O
    /// threads this is smaller than `diskSeconds`; `bytesRead / ioWallSeconds` is the
    /// aggregate read rate.
    public var ioWallSeconds = 0.0
    /// Expert reads that completed (demand, prefetch and pin loads).
    public var reads = 0
    /// Experts pinned (protected from LRU eviction) right now, summed over layers.
    public var pinned = 0
    /// Pin changes made by live repin (`endToken`).
    public var repinSwaps = 0
    public init() {}

    /// Aggregate read throughput in GB/s (10^9 bytes): `bytesRead / ioWallSeconds`.
    public var readGBPerSecond: Double { ioWallSeconds > 0 ? Double(bytesRead) / ioWallSeconds / 1e9 : 0 }

    /// Hits over lookups (1 when there were no lookups).
    public var hitRate: Double { lookups > 0 ? Double(hits) / Double(lookups) : 1 }
}

/// Where routed experts live and how the engine gets at them.
public protocol ExpertStore: AnyObject {
    var layout: ExpertLayout { get }

    /// True when every expert of every layer is always addressable (no cache): the model
    /// may then route on the device and index the fused tensors with the router's ids,
    /// never calling `acquire`.
    var isFullyResident: Bool { get }

    /// The stacked gate/up/down views of a MoE layer.
    func weights(layer: Int) -> ExpertLayerWeights

    /// Leases `experts` of `layer`. Hits return at once; misses are queued for reading
    /// (demand reads ahead of prefetches) and reported in the lease. Never evicts a
    /// leased or loading slot; if no slot is evictable it waits for one.
    func acquire(layer: Int, experts: [Int32]) throws -> ExpertLease

    /// Blocks until every miss of `lease` is loaded.
    func waitForMisses(_ lease: ExpertLease)

    /// Throws if a read that `lease` depends on failed: its slot then holds stale or
    /// zero bytes, so whatever the lease's batch computed is wrong. Call after the
    /// lease's batch has completed (and its misses have landed), before `release`.
    /// A failed read still advances the signal, so a waiting batch never hangs; this is
    /// where the failure surfaces instead.
    func checkLease(_ lease: ExpertLease) throws

    /// Ends the lease. Exactly once per lease.
    func release(_ lease: ExpertLease)

    /// Advisory read-ahead (router lookahead / PILOT). Never evicts a leased slot and
    /// takes no lease.
    func prefetch(layer: Int, experts: [Int32])

    /// Counts a routing decision that did not go through `acquire` (resident mode with
    /// device-side routing reads the ids back after the token) for the routing history.
    func recordUsage(layer: Int, experts: [Int32])

    var stats: ExpertStoreStats { get }

    /// The signal misses are published on (nil for stores that never miss).
    var signal: HostSignal? { get }

    /// Marks a token boundary: the safe point for live repin (`--repin N`). Call once per
    /// generated token (and once after a prefill), with no lease outstanding.
    func endToken()

    /// Writes the routing history (`.coli_usage`) atomically, if the store keeps one and
    /// saving is enabled. Call after each CLI turn and at exit.
    func saveHistory() throws
}

extension ExpertStore {
    public func checkLease(_ lease: ExpertLease) throws {}
    public func endToken() {}
    public func saveHistory() throws {}
}

/// How the CLI asks for a store (`--experts`, `--expert-cache-gb`).
public struct ExpertStoreOptions: Sendable {
    public enum Mode: String, Sendable { case auto, resident, stream }
    public var mode: Mode
    /// Byte budget for expert slots (`--expert-cache-gb`); nil = derive from MemoryBudget.
    public var cacheBytes: Int?
    /// Force a slot count per layer (tests: exercise eviction on tiny models).
    public var slotsPerLayer: Int?
    /// Concurrent `pread` workers (measured: 8 gives 2.5x one on this SSD).
    public var ioThreads: Int
    /// Router lookahead prefetch.
    public var pilot: Bool
    /// `.coli_usage` path; nil = next to the model. Pins from it at startup when streaming.
    public var historyPath: String?
    public var saveHistory: Bool
    /// Read experts with `F_NOCACHE` (upstream `DIRECT=1`): the reads go to the SSD and
    /// leave no duplicate in the unified buffer cache, so streaming is measured honestly.
    /// Off (the default, as in the CLI without `--direct`) = buffered `pread` (a warm page
    /// cache then serves the reads at RAM speed).
    public var uncached: Bool
    /// Live repin every N tokens (`--repin N`, colibri c/tier.h); 0 = off.
    public var repinEvery: Int
    /// Memory the model keeps resident besides the file's dense weights (KV cache,
    /// activation scratch), subtracted from the budget before experts are placed.
    public var extraResidentBytes: Int

    public init(mode: Mode = .auto, cacheBytes: Int? = nil, slotsPerLayer: Int? = nil, ioThreads: Int = 8,
                pilot: Bool = false, historyPath: String? = nil, saveHistory: Bool = true,
                uncached: Bool = false, repinEvery: Int = 0, extraResidentBytes: Int = 0) {
        self.mode = mode
        self.cacheBytes = cacheBytes
        self.slotsPerLayer = slotsPerLayer
        self.ioThreads = ioThreads
        self.pilot = pilot
        self.historyPath = historyPath
        self.saveHistory = saveHistory
        self.uncached = uncached
        self.repinEvery = repinEvery
        self.extraResidentBytes = extraResidentBytes
    }
}

public enum ExpertStores {
    /// Picks and builds a store: resident when the expert bytes fit the budget (or
    /// `.resident` was asked for), streaming otherwise. Registers the store's regions
    /// with `backend`. Loads the routing history (`options.historyPath`, else
    /// `<model>.coli_usage`) and, when streaming, pins each layer's most used experts
    /// from it (at most `slots - topK` per layer). `topK` is the family's experts per
    /// token: the floor for the slot count.
    public static func make(file: GGUFFile, layout: ExpertLayout, backend: Backend,
                            options: ExpertStoreOptions, engineName: String, topK: Int = 8) throws -> ExpertStore {
        try make(path: file.file.path, fileSize: file.file.fileSize, fileRegion: { file.file.region },
                 layout: layout, backend: backend, options: options, engineName: engineName, topK: topK,
                 segments: file.file.segments)
    }

    /// The history file used when `ExpertStoreOptions.historyPath` is nil.
    public static func defaultHistoryPath(modelPath: String) -> String { modelPath + ".coli_usage" }

    /// `make(file:)` without a parsed `GGUFFile`: the model's path and size, and the
    /// whole-file mapping (only asked for when the store is resident). `segments`: the
    /// model's files (a split model's shards); nil = the single file at `path`.
    public static func make(path: String, fileSize: Int, fileRegion: () throws -> MemoryRegion,
                            layout: ExpertLayout, backend: Backend, options: ExpertStoreOptions,
                            engineName: String, topK: Int = 8, segments: [FileSegment]? = nil) throws -> ExpertStore {
        guard layout.fileEnd <= fileSize else {
            throw ColibriError.invalidGGUF("expert tensors end at byte \(layout.fileEnd), file has \(fileSize)")
        }
        let explicitHistory = options.historyPath != nil
        let historyPath = options.historyPath ?? defaultHistoryPath(modelPath: path)
        let history = RoutingHistory(layers: layout.layers.count, experts: layout.maxExperts,
                                     engineName: engineName, moeLayers: Set(layout.moeLayers))
        if FileManager.default.fileExists(atPath: historyPath) {
            do {
                try history.load(path: historyPath, explicit: explicitHistory)
            } catch {
                FileHandle.standardError.write(Data("[USAGE] \(error)\n".utf8))
            }
        }

        let slots = try decideSlots(fileSize: fileSize, layout: layout, backend: backend, options: options, topK: topK)
        guard let slots else {
            let region = try fileRegion()
            try backend.register(region)
            return ResidentExpertStore(region: region, layout: layout, history: history,
                                       historyPath: options.saveHistory ? historyPath : nil)
        }
        let store = try StreamingExpertStore(
            segments: segments ?? [FileSegment(path: path, start: 0, size: fileSize)], layout: layout, slotsPerLayer: slots, backend: backend, ioThreads: options.ioThreads,
            history: history, historyPath: options.saveHistory ? historyPath : nil,
            uncached: options.uncached, topK: topK, repinEvery: options.repinEvery)
        if !history.isEmpty { store.pinFromHistory() }
        return store
    }

    /// nil = resident; otherwise slot counts per layer (0 for dense layers).
    static func decideSlots(fileSize: Int, layout: ExpertLayout, backend: Backend,
                            options: ExpertStoreOptions, topK: Int) throws -> [Int]? {
        if layout.moeLayers.isEmpty || options.mode == .resident { return nil }
        if let forced = options.slotsPerLayer {
            guard forced >= 1 else { throw ColibriError.invalidArgument("slots per layer must be >= 1, not \(forced)") }
            return layout.layers.map { $0.map { min($0.count, forced) } ?? 0 }
        }
        let budget = MemoryBudget.derive(workingSetLimit: backend.capabilities.workingSetLimit,
                                         cacheBytes: options.cacheBytes,
                                         residentBytes: max(0, fileSize - layout.totalBytes) + options.extraResidentBytes)
        switch budget.plan(layout: layout, topK: topK) {
        case .streaming(let slots): return slots
        case .resident:
            // `--experts stream` on a model that fits: stream through a cache of every expert.
            return options.mode == .stream ? layout.layers.map { $0?.count ?? 0 } : nil
        }
    }
}
