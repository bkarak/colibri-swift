// Ported from colibri c/tier.h (Apache-2.0): promotion rule with hysteresis, LFRU score,
// heat decay, and the swap pickers. Pure functions, used by the streaming store's live
// repin (StreamingExpertStore.endToken).

public enum Tier {
    /// Promote an unpinned expert over a pinned one only when clearly hotter:
    /// `hot > cold + cold/4 + 4` (25% plus 4 hysteresis against ping-pong).
    @inlinable public static func shouldPromote(hot: UInt32, cold: UInt32) -> Bool {
        UInt64(hot) > UInt64(cold) + UInt64(cold / 4) + 4
    }

    /// LFRU score `heat << 8 | recency`, recency `max(0, 255 - age)`: recency only breaks ties.
    @inlinable public static func lfruScore(heat: UInt32, age: UInt32) -> UInt64 {
        (UInt64(heat) << 8) | UInt64(age >= 255 ? 0 : 255 - age)
    }

    /// `tier_lfru_score(heat, last, clock)`: the age is `clock - last` in wrapping 32-bit
    /// arithmetic, as upstream.
    @inlinable public static func lfruScore(heat: UInt32, last: UInt32, clock: UInt32) -> UInt64 {
        lfruScore(heat: heat, age: clock &- last)
    }

    /// Session heat decay (`>> 1`). The persistent history is never decayed.
    @inlinable public static func decay(_ heat: UInt32) -> UInt32 { heat >> 1 }

    /// At most this many pin swaps per repin boundary.
    public static let maxSwapsPerRepin = 4

    /// A proposed swap: replace `pinned[slot]` (the coldest pinned expert) with `expert`.
    public struct Swap: Sendable, Equatable {
        public var slot: Int
        public var expert: Int
        public var gain: Int
    }

    /// `tier_pick_swap`: the hottest unpinned expert against the coldest pinned one, by
    /// heat alone, with the `shouldPromote` hysteresis.
    public static func pickSwap(heat: [UInt32], pinned: [Int]) -> Swap? {
        guard !heat.isEmpty, !pinned.isEmpty else { return nil }
        var cold = 0
        for z in 1 ..< pinned.count where heat[pinned[z]] < heat[pinned[cold]] { cold = z }
        let pinnedSet = Set(pinned)
        var hot = -1
        var fh: UInt32 = 0
        for e in heat.indices where !pinnedSet.contains(e) && heat[e] > fh {
            fh = heat[e]
            hot = e
        }
        guard hot >= 0 else { return nil }
        let fc = heat[pinned[cold]]
        guard shouldPromote(hot: fh, cold: fc) else { return nil }
        return Swap(slot: cold, expert: hot, gain: Int(fh) - Int(fc))
    }

    /// `tier_pick_lfru`: as `pickSwap` but ranked by LFRU score, keeping the 25% + 4
    /// frequency hysteresis in score units (`hs > cs + cs/4 + 4 << 8`).
    public static func pickLFRU(heat: [UInt32], last: [UInt32], clock: UInt32, pinned: [Int]) -> Swap? {
        guard !heat.isEmpty, heat.count == last.count, !pinned.isEmpty else { return nil }
        func score(_ e: Int) -> UInt64 { lfruScore(heat: heat[e], last: last[e], clock: clock) }
        var cold = 0
        for z in 1 ..< pinned.count where score(pinned[z]) < score(pinned[cold]) { cold = z }
        let pinnedSet = Set(pinned)
        var hot = -1
        var hs: UInt64 = 0
        for e in heat.indices where !pinnedSet.contains(e) {
            let s = score(e)
            if hot < 0 || s > hs {
                hot = e
                hs = s
            }
        }
        guard hot >= 0 else { return nil }
        let cs = score(pinned[cold])
        guard hs > cs + (cs >> 2) + (4 << 8) else { return nil }
        return Swap(slot: cold, expert: hot, gain: Int((hs - cs) >> 8))
    }
}
