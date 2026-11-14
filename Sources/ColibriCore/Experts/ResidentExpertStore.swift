import Foundation

/// Every expert addressed in place inside the file mapping; no slots, no I/O. Leases
/// only count usage. The default when the experts fit the memory budget, which on a
/// 64 GB machine is true for every model in scope. Zero copy: the views point into the
/// `mmap` of the GGUF, which a Metal backend wraps once with `bytesNoCopy`.
///
/// Thread-safe.
public final class ResidentExpertStore: ExpertStore {
    public let layout: ExpertLayout
    public var isFullyResident: Bool { true }
    public var signal: HostSignal? { nil }
    public let history: RoutingHistory?
    /// Where `saveHistory` writes; nil = read-only history.
    public let historyPath: String?

    private let region: MemoryRegion
    private let weightsByLayer: [ExpertLayerWeights?]
    private let lock = NSLock()
    private var counters = ExpertStoreStats()
    private var nextToken: UInt64 = 1

    public convenience init(file: GGUFFile, layout: ExpertLayout, history: RoutingHistory?) {
        self.init(region: file.file.region, layout: layout, history: history, historyPath: nil)
    }

    /// `region` is the whole-file mapping: region offset = file offset.
    public init(region: MemoryRegion, layout: ExpertLayout, history: RoutingHistory?, historyPath: String? = nil) {
        precondition(layout.fileEnd <= region.length, "expert tensors extend past the file mapping")
        self.layout = layout
        self.region = region
        self.history = history
        self.historyPath = historyPath
        self.weightsByLayer = layout.layers.map { l in
            l.map { l in
                func stack(_ s: ExpertTensorSlice) -> StackedWeights {
                    StackedWeights(region: region, baseOffset: s.fileOffset, stride: s.bytesPerExpert, count: l.count,
                                   type: s.type, cols: s.cols, rows: s.rows)
                }
                return ExpertLayerWeights(gate: stack(l.gate), up: stack(l.up), down: stack(l.down))
            }
        }
    }

    public var stats: ExpertStoreStats {
        lock.lock(); defer { lock.unlock() }
        return counters
    }

    public func weights(layer: Int) -> ExpertLayerWeights {
        guard layer >= 0, layer < weightsByLayer.count, let w = weightsByLayer[layer] else {
            preconditionFailure("layer \(layer) has no routed experts")
        }
        return w
    }

    public func acquire(layer: Int, experts: [Int32]) throws -> ExpertLease {
        guard layer >= 0, layer < layout.layers.count, let l = layout.layers[layer] else {
            throw ColibriError.invalidArgument("layer \(layer) has no routed experts")
        }
        for e in experts where e < 0 || Int(e) >= l.count {
            throw ColibriError.invalidArgument("expert \(e) out of range 0..<\(l.count) in layer \(layer)")
        }
        history?.record(layer: layer, experts: experts)
        let distinct = Set(experts).count
        lock.lock()
        counters.lookups += distinct
        counters.hits += distinct
        let token = nextToken
        nextToken += 1
        lock.unlock()
        return ExpertLease(layer: layer, experts: experts, slots: experts, misses: [], readyValue: 0, token: token)
    }

    public func waitForMisses(_ lease: ExpertLease) {}
    public func release(_ lease: ExpertLease) {}
    public func prefetch(layer: Int, experts: [Int32]) {}

    public func recordUsage(layer: Int, experts: [Int32]) {
        history?.record(layer: layer, experts: experts)
    }

    public func saveHistory() throws {
        guard let history, let historyPath else { return }
        try history.save(path: historyPath)
    }
}
