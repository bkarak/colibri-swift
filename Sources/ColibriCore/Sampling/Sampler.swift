// Token sampling. Greedy is upstream's argmax_first (c/argmax_first.h): first index of
// the maximum, strict `>` scan from id 0, NaN skipped. Stochastic sampling follows
// c/qwen36_sample.h (temperature, nucleus sorted by probability with ties by id) plus
// top-k, min-p and a repetition penalty, drawing with the xorshift generator of
// c/sample.h. docs/port/models.md section 9.
//
// The order of the stages, all optional except the draw:
//   1. repetition penalty on the logits of recently seen tokens (also before greedy)
//   2. greedy (temperature <= 0) stops here with argmax
//   3. temperature: logits / T
//   4. top-k: the k largest logits (ties: lower id first)
//   5. softmax over the survivors
//   6. top-p: the shortest prefix, by probability (ties: lower id), whose mass reaches p
//   7. min-p: drop tokens whose probability is below minP * the largest one
//   8. draw from what is left, in that order, with one uniform from the generator (with
//      no truncation at all the walk goes over the vocabulary in id order instead)

import Foundation

public struct SamplerConfig: Sendable, Equatable {
    /// <= 0 means greedy.
    public var temperature: Float
    /// Nucleus mass in (0, 1]; 1 disables.
    public var topP: Float
    /// Keep the k most likely tokens before top-p; 0 disables.
    public var topK: Int
    /// Drop tokens less likely than minP times the most likely one; 0 disables.
    public var minP: Float
    /// Divide positive (multiply negative) logits of recently seen tokens by this
    /// (HF / llama.cpp `repetition_penalty`); 1 disables.
    public var repetitionPenalty: Float
    /// How many of the most recent tokens the penalty looks at; 0 means all of them.
    public var repetitionWindow: Int
    public var seed: UInt64

    public init(temperature: Float = 0, topP: Float = 1, topK: Int = 0, minP: Float = 0,
                repetitionPenalty: Float = 1, repetitionWindow: Int = 64,
                seed: UInt64 = 0x9E37_79B9_7F4A_7C15) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
        self.seed = seed
    }

    public static let greedy = SamplerConfig()
}

/// The generator of c/sample.h (`rndu`): xorshift64 with shifts 13, 7, 17, and a uniform
/// double from the top 53 bits of the state. Upstream calls it xorshift64*, but there is
/// no multiply; this reproduces upstream's sequence. Deterministic for a given seed.
public struct Xorshift64Star: RandomNumberGenerator, Sendable {
    public private(set) var state: UInt64
    public init(seed: UInt64) { state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed }

    public mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    /// A uniform double in [0, 1), as upstream's `rndu`.
    public mutating func nextUniform() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }
}

public struct Sampler {
    public var config: SamplerConfig
    public var rng: Xorshift64Star
    /// Tokens the repetition penalty looks at (most recent last). `sample` appends what it
    /// picks; feed the prompt with `accept` to penalize it too, as HF does.
    public private(set) var history: [Int32] = []

    private var scratch: [Float] = []
    private var probs: [Float] = []
    private var candidates: [Candidate] = []

    private struct Candidate {
        var id: Int32
        var value: Float  // logit, then probability
    }

    public init(_ config: SamplerConfig) {
        self.config = config
        self.rng = Xorshift64Star(seed: config.seed)
    }

    /// Records tokens for the repetition penalty (the prompt, or tokens chosen elsewhere).
    public mutating func accept<S: Sequence>(_ ids: S) where S.Element == Int32 {
        guard config.repetitionPenalty != 1 else { return }
        history.append(contentsOf: ids)
        trimHistory()
    }

    /// Forgets the history (a new conversation).
    public mutating func reset() {
        history.removeAll()
        rng = Xorshift64Star(seed: config.seed)
    }

    private mutating func trimHistory() {
        let w = config.repetitionWindow
        if w > 0 && history.count > 2 * w + 1024 { history.removeFirst(history.count - w) }
    }

    /// Picks the next token from the logits of one position.
    public mutating func sample(_ logits: UnsafeBufferPointer<Float>) -> Int32 {
        let id: Int32
        if config.repetitionPenalty != 1 && !history.isEmpty {
            scratch.removeAll(keepingCapacity: true)
            scratch.append(contentsOf: logits)
            applyPenalty(&scratch)
            id = scratch.withUnsafeBufferPointer { pick($0) }
        } else {
            id = pick(logits)
        }
        if config.repetitionPenalty != 1 {
            history.append(id)
            trimHistory()
        }
        return id
    }

    /// Convenience for an array of logits.
    public mutating func sample(_ logits: [Float]) -> Int32 {
        logits.withUnsafeBufferPointer { sample($0) }
    }

    private func applyPenalty(_ x: inout [Float]) {
        let p = config.repetitionPenalty
        let w = config.repetitionWindow
        let recent = w > 0 ? history.suffix(w) : history[...]
        var seen = Set<Int32>()
        for id in recent where id >= 0 && Int(id) < x.count && seen.insert(id).inserted {
            let v = x[Int(id)]
            x[Int(id)] = v > 0 ? v / p : v * p
        }
    }

    private mutating func pick(_ logits: UnsafeBufferPointer<Float>) -> Int32 {
        let V = logits.count
        if config.temperature <= 0 || V <= 1 { return Sampler.argmax(logits) }
        let invT = 1 / config.temperature

        // the largest finite logit; +inf wins outright, NaN and -inf never take part
        var mx = -Float.infinity
        var any = false
        for i in 0..<V {
            let v = logits[i]
            if v == .infinity { return Int32(i) }
            if v.isFinite && (!any || v > mx) { mx = v; any = true }
        }
        if !any { return Sampler.argmax(logits) }

        candidates.removeAll(keepingCapacity: true)
        var total = 0.0  // the mass the nucleus is a fraction of
        let k = config.topK
        if k > 0 && k < V {
            selectTop(k, logits)  // best first; ties: lower id
            for i in candidates.indices {
                let p = expf((candidates[i].value - mx) * invT)
                candidates[i].value = p
                total += Double(p)
            }
        } else {
            if probs.count != V { probs = [Float](repeating: 0, count: V) }
            probs.withUnsafeMutableBufferPointer { p in
                for i in 0..<V {
                    let v = logits[i]
                    let e: Float = v.isFinite ? expf((v - mx) * invT) : 0
                    p[i] = e
                    total += Double(e)
                }
            }
            let topP = Double(config.topP)
            if topP > 0 && topP < 1 {
                gatherNucleus(neededMass: topP * total)
            } else if config.minP > 0 {
                gather(threshold: config.minP)  // the largest probability is exactly 1 here
            } else {
                // no truncation: draw over the whole vocabulary in id order
                let r = rng.nextUniform() * total
                var acc = 0.0
                var last: Int32 = 0
                for i in 0..<V where probs[i] > 0 {
                    acc += Double(probs[i])
                    last = Int32(i)
                    if acc > r { return last }
                }
                return last
            }
        }
        guard total.isFinite, total > 0, !candidates.isEmpty else { return Sampler.argmax(logits) }

        // top-p over the sorted candidates
        let topP = Double(config.topP)
        if topP > 0 && topP < 1 {
            var kept = 0.0, n = 0
            while n < candidates.count && kept < topP * total {
                kept += Double(candidates[n].value)
                n += 1
            }
            candidates.removeSubrange(max(n, 1)...)
        }

        // min-p, relative to the most likely token (candidates[0])
        if config.minP > 0 {
            let floor = candidates[0].value * config.minP
            if let cut = candidates.firstIndex(where: { $0.value < floor }) {
                candidates.removeSubrange(max(cut, 1)...)
            }
        }

        var sum = 0.0
        for c in candidates { sum += Double(c.value) }
        let r = rng.nextUniform() * sum
        var acc = 0.0
        for c in candidates {
            acc += Double(c.value)
            if acc > r { return c.id }
        }
        return candidates[candidates.count - 1].id
    }

    private static func byProbability(_ a: Candidate, _ b: Candidate) -> Bool {
        a.value != b.value ? a.value > b.value : a.id < b.id
    }

    /// Fills `candidates` with the k largest finite logits, best first (ties: lower id).
    private mutating func selectTop(_ k: Int, _ logits: UnsafeBufferPointer<Float>) {
        var top: [Candidate] = []
        top.reserveCapacity(k + 1)
        for i in 0..<logits.count {
            let v = logits[i]
            guard v.isFinite else { continue }
            // ids come in order, so an equal value later is worse and never displaces one
            if top.count == k {
                guard v > top[k - 1].value else { continue }
                top.removeLast()
            }
            let c = Candidate(id: Int32(i), value: v)
            var lo = 0, hi = top.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if Sampler.byProbability(top[mid], c) { lo = mid + 1 } else { hi = mid }
            }
            top.insert(c, at: lo)
        }
        candidates = top
    }

    /// `candidates` = the tokens with probability >= threshold, sorted.
    private mutating func gather(threshold: Float) {
        candidates.removeAll(keepingCapacity: true)
        probs.withUnsafeBufferPointer { p in
            for i in 0..<p.count where p[i] >= threshold && p[i] > 0 {
                candidates.append(Candidate(id: Int32(i), value: p[i]))
            }
        }
        candidates.sort(by: Sampler.byProbability)
    }

    /// `candidates` = enough of the most likely tokens, sorted, to hold the nucleus without
    /// sorting the long tail: the tokens above a probability threshold, lowered until
    /// their mass reaches `neededMass` (c/qwen36_sample.h does the same with exponent
    /// buckets). Every token of the nucleus is above the threshold, so dropping the rest
    /// changes nothing.
    private mutating func gatherNucleus(neededMass: Double) {
        var threshold: Float = 1e-4  // relative to the largest probability, which is 1
        while threshold >= Float.leastNormalMagnitude {
            var mass = 0.0
            probs.withUnsafeBufferPointer { p in
                for i in 0..<p.count where p[i] >= threshold { mass += Double(p[i]) }
            }
            // margin: the walk sums the same values in another order
            if mass >= neededMass * (1 + 1e-6) {
                gather(threshold: threshold)
                return
            }
            threshold *= 1e-3
        }
        gather(threshold: 0)
    }

    /// First index of the maximum (strict `>`), skipping NaN; 0 when every value is NaN
    /// or the buffer is empty.
    public static func argmax(_ logits: UnsafeBufferPointer<Float>) -> Int32 {
        var best = -1
        var bv = -Float.infinity
        for i in 0..<logits.count {
            let v = logits[i]
            if v.isNaN { continue }
            if best < 0 || v > bv {
                best = i
                bv = v
            }
        }
        return Int32(max(best, 0))
    }
}
