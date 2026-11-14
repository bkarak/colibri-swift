import Darwin
import Foundation

/// How much memory experts may use, after upstream's c/olmoe.c init formula with the GPU
/// working-set ceiling added (docs/port/runtime-and-metal.md section 6.4):
///
///     budget = min(backend.workingSetLimit, --expert-cache-gb or 0.85 * physical - others)
///     room   = budget - dense resident - KV cache - scratch - 0.5 GB
///     mode   = all experts fit in room ? resident : streaming
///     slots  = equal count per MoE layer, at least top-k, from per-layer slot bytes
public struct MemoryBudget: Sendable {
    public var budgetBytes: Int
    public var reservedBytes: Int

    public init(budgetBytes: Int, reservedBytes: Int) {
        self.budgetBytes = budgetBytes
        self.reservedBytes = reservedBytes
    }

    /// Activations and other headroom upstream keeps out of the expert room (0.5 GB).
    public static let activationHeadroom = 512 << 20

    /// Bytes left for experts.
    public var roomBytes: Int { max(0, budgetBytes - reservedBytes) }

    /// Physical memory.
    public static var physicalBytes: Int { Int(ProcessInfo.processInfo.physicalMemory) }

    /// Free + inactive + purgeable from `host_statistics64`; nil if it reads as
    /// unmeasured (under 2% of physical memory), as upstream treats it.
    public static func availableBytes() -> Int? {
        var info = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        let page = Int(vm_kernel_page_size)
        let pages = Int(info.free_count) + Int(info.inactive_count) + Int(info.purgeable_count)
        let bytes = pages * page
        guard bytes >= physicalBytes / 50 else { return nil }
        return bytes
    }

    /// The budget for a model: an explicit expert-cache size wins (then the room is that
    /// size exactly); otherwise upstream's `resident + 0.88 * available`, capped at 85%
    /// of physical memory, and in every case at the backend's working-set limit.
    ///
    /// - Parameters:
    ///   - workingSetLimit: `BackendCapabilities.workingSetLimit`.
    ///   - cacheBytes: `--expert-cache-gb`, if given.
    ///   - residentBytes: dense weights, KV cache and scratch (everything but experts).
    public static func derive(workingSetLimit: Int, cacheBytes: Int?, residentBytes: Int) -> MemoryBudget {
        if let cacheBytes {
            return MemoryBudget(budgetBytes: max(0, cacheBytes), reservedBytes: 0)
        }
        let reserved = residentBytes + activationHeadroom
        let physicalCap = Int(Double(physicalBytes) * 0.85)
        var host = physicalCap
        if let avail = availableBytes() {
            host = min(physicalCap, residentBytes + Int(Double(avail) * 0.88))
        }
        return MemoryBudget(budgetBytes: min(workingSetLimit, host), reservedBytes: reserved)
    }

    public enum Plan: Sendable, Equatable {
        case resident
        case streaming(slotsPerLayer: [Int])
    }

    /// Decides resident versus streaming and the slot counts for `layout`. Slot counts
    /// are indexed by layer (0 for dense layers), equal across MoE layers, sized from each
    /// layer's arena slot stride, at least `topK` (the hard floor: one token's experts)
    /// and at most the layer's expert count.
    public func plan(layout: ExpertLayout, topK: Int) -> Plan {
        let room = roomBytes
        if layout.totalBytes <= room { return .resident }
        let moe = layout.layers.compactMap { $0 }
        guard !moe.isEmpty else { return .resident }
        let perSlotAcrossLayers = moe.reduce(0) { $0 + ExpertArena.slotStride(for: $1) }
        let n = room / max(1, perSlotAcrossLayers)
        let slots = layout.layers.map { l -> Int in
            guard let l else { return 0 }
            return min(l.count, max(topK, n))
        }
        return .streaming(slotsPerLayer: slots)
    }
}
