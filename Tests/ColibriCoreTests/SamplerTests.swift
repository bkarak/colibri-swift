import Foundation
import Testing
@testable import ColibriCore

@Suite struct SamplerTests {
    private func logits(_ probs: [Double]) -> [Float] { probs.map { Float(log($0)) } }

    private func frequencies(_ config: SamplerConfig, _ x: [Float], draws: Int = 20_000) -> [Double] {
        var s = Sampler(config)
        var counts = [Int](repeating: 0, count: x.count)
        for _ in 0..<draws { counts[Int(s.sample(x))] += 1 }
        return counts.map { Double($0) / Double(draws) }
    }

    @Test func argmaxFirstMaximumSkippingNaN() {
        func am(_ x: [Float]) -> Int32 { x.withUnsafeBufferPointer { Sampler.argmax($0) } }
        #expect(am([1, 3, 3, 2]) == 1)
        #expect(am([.nan, 1, 5, .nan, 5]) == 2)
        #expect(am([.nan, .nan]) == 0)
        #expect(am([]) == 0)
        #expect(am([-.infinity, -.infinity]) == 0)
        #expect(am([0, -0.0]) == 0)
    }

    @Test func greedyIsDefault() {
        var s = Sampler(.greedy)
        #expect(s.sample([0.1, 2, 0.5]) == 1)
        var t = Sampler(SamplerConfig(temperature: 0, topP: 0.5, topK: 1))
        #expect(t.sample([3, 2, 9, 9]) == 2)
    }

    @Test func generatorMatchesUpstream() {
        // c/sample.h rndu from its default seed: x ^= x<<13; x ^= x>>7; x ^= x<<17; (x>>11)/2^53
        var g = Xorshift64Star(seed: 0x9E37_79B9_7F4A_7C15)
        #expect(g.next() == 0xDC1B_77AE_0BF3_4DAD)
        #expect(g.next() == 0x64F0_EEB9_026E_6076)
        #expect(abs(g.nextUniform() - 0.48058787404949177) < 1e-15)
        #expect(Xorshift64Star(seed: 0).state == 0x9E37_79B9_7F4A_7C15)
    }

    @Test func seededRunsRepeat() {
        let x = logits([0.4, 0.3, 0.2, 0.1])
        let c = SamplerConfig(temperature: 1, seed: 42)
        var a = Sampler(c), b = Sampler(c)
        let ra = (0..<100).map { _ in a.sample(x) }
        let rb = (0..<100).map { _ in b.sample(x) }
        #expect(ra == rb)
        var d = Sampler(SamplerConfig(temperature: 1, seed: 43))
        #expect((0..<100).map { _ in d.sample(x) } != ra)
    }

    @Test func temperatureOneFollowsTheDistribution() {
        let f = frequencies(SamplerConfig(temperature: 1, seed: 7), logits([0.5, 0.3, 0.2]))
        #expect(abs(f[0] - 0.5) < 0.02 && abs(f[1] - 0.3) < 0.02 && abs(f[2] - 0.2) < 0.02)
    }

    @Test func lowTemperatureSharpens() {
        let f = frequencies(SamplerConfig(temperature: 0.25, seed: 7), logits([0.5, 0.3, 0.2]))
        // p ∝ p^4: 0.0625, 0.0081, 0.0016 -> 0.866, 0.112, 0.022
        #expect(abs(f[0] - 0.866) < 0.02 && abs(f[1] - 0.112) < 0.02)
    }

    @Test func topK() {
        let f = frequencies(SamplerConfig(temperature: 1, topK: 2, seed: 1), logits([0.1, 0.5, 0.3, 0.1]))
        #expect(f[0] == 0 && f[3] == 0)
        #expect(abs(f[1] - 0.625) < 0.02)
        // ties at the cut: the lower id is kept
        let g = frequencies(SamplerConfig(temperature: 1, topK: 2, seed: 1), logits([0.2, 0.4, 0.2, 0.2]), draws: 2000)
        #expect(g[0] > 0 && g[2] == 0 && g[3] == 0)
    }

    @Test func topP() {
        // 0.5 + 0.3 reaches 0.6: the nucleus is {1, 2}
        let f = frequencies(SamplerConfig(temperature: 1, topP: 0.6, seed: 3), logits([0.1, 0.5, 0.3, 0.1]))
        #expect(f[0] == 0 && f[3] == 0)
        #expect(abs(f[1] - 0.625) < 0.02)
        // a tiny p keeps only the most likely token
        let g = frequencies(SamplerConfig(temperature: 1, topP: 0.01, seed: 3), logits([0.1, 0.5, 0.3, 0.1]), draws: 500)
        #expect(g[1] == 1)
    }

    @Test func topPOnALargeVocabulary() {
        // 150k logits with a long flat tail: the partial sort must keep the same nucleus
        var x = [Float](repeating: -10, count: 150_000)
        x[777] = 5; x[12] = 4.5; x[99_999] = 4.5
        let f = frequencies(SamplerConfig(temperature: 1, topP: 0.9, seed: 5), x, draws: 40)
        let kept = f.enumerated().filter { $0.element > 0 }.map(\.offset)
        #expect(Set(kept).isSubset(of: [12, 777, 99_999]))
        #expect(f[777] > 0.2)
    }

    @Test func minP() {
        // max 0.5: the floor at minP 0.5 is 0.25, which drops 0.2 and 0.1
        let f = frequencies(SamplerConfig(temperature: 1, minP: 0.5, seed: 9), logits([0.5, 0.2, 0.3, 0.1]))
        #expect(f[1] == 0 && f[3] == 0)
        #expect(abs(f[0] - 0.625) < 0.02)
    }

    @Test func repetitionPenalty() {
        var s = Sampler(SamplerConfig(temperature: 0, repetitionPenalty: 2))
        s.accept([0])
        // 3 / 2 = 1.5 < 2: the seen token loses
        #expect(s.sample([3, 2, 1]) == 1)
        // negative logits are multiplied (dividing would make -0.6 the best): [-2, -1.2, -1.1]
        #expect(s.sample([-1, -0.6, -1.1]) == 2)
        #expect(s.history == [0, 1, 2])
        // window: only the last N tokens count ([2.6, 1.5, 1], not [1.3, 1.5, 1])
        var w = Sampler(SamplerConfig(temperature: 0, repetitionPenalty: 2, repetitionWindow: 1))
        w.accept([0, 1])
        #expect(w.sample([2.6, 3, 1]) == 0)
        // penalty 1 keeps no history
        var none = Sampler(.greedy)
        none.accept([1, 2])
        #expect(none.history.isEmpty)
    }

    @Test func nonFiniteLogits() {
        var s = Sampler(SamplerConfig(temperature: 0.8, topP: 0.9, seed: 1))
        #expect(s.sample([.nan, 1, .infinity, 2]) == 2)
        for _ in 0..<50 {
            let id = s.sample([.nan, 0, -.infinity, 0])
            #expect(id == 1 || id == 3)
        }
        #expect(s.sample([.nan, .nan]) == 0)
    }
}
