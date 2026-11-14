import Foundation
import Testing
@testable import ColibriCore

// Workstream A: per-op tests of the CPU backend against naive references written from
// the formulas in Backend.swift and docs/port/models.md.

/// Weight bytes held in a page-aligned allocation, exposed as a `MemoryRegion`.
final class TestWeights {
    let region: MemoryRegion
    private let raw: UnsafeMutableRawPointer
    private var used = 0

    init(capacity: Int) {
        let len = alignUp(max(capacity, 1), hostPageSize)
        var p: UnsafeMutableRawPointer?
        posix_memalign(&p, hostPageSize, len)
        raw = p!
        memset(raw, 0, len)
        region = MemoryRegion(base: UnsafeRawPointer(raw), length: len, kind: .other, label: "test", keepAlive: nil)
    }

    deinit { free(raw) }

    /// Appends `bytes` at a 32-byte aligned offset and returns that offset.
    func append(_ bytes: [UInt8]) -> Int {
        let off = alignUp(used, 32)
        precondition(off + bytes.count <= region.length)
        bytes.withUnsafeBytes { (raw + off).copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
        used = off + bytes.count
        return off
    }

    /// A matrix of random values in `type`; returns the view and its dequantised values.
    func matrix(_ type: GGMLType, cols: Int, rows: Int, rng: inout TestRNG, scale: Float = 0.3) throws -> (WeightView, [Float]) {
        let bytes = try Quantize.encode(rng.floats(cols * rows, scale: scale), as: type)
        let off = append(bytes)
        return (WeightView(region: region, offset: off, type: type, cols: cols, rows: rows),
                Dequant.dequantize(type, bytes: bytes, count: cols * rows))
    }
}

func buffer(_ cpu: CPUBackend, _ values: [Float]) throws -> DeviceBuffer {
    let b = try cpu.makeBuffer(.f32, count: values.count, label: "t")
    for (i, v) in values.enumerated() { b.slice().floats[i] = v }
    return b
}

func ints(_ cpu: CPUBackend, _ values: [Int32]) throws -> DeviceBuffer {
    let b = try cpu.makeBuffer(.i32, count: values.count, label: "ids")
    for (i, v) in values.enumerated() { b.slice().int32s[i] = v }
    return b
}

func expectClose(_ a: [Float], _ b: [Float], rel: Float = 1e-5, abs absTol: Float = 1e-6,
                 _ what: String = "", sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(a.count == b.count, sourceLocation: sourceLocation)
    var worst: Float = 0, at = -1
    for i in 0..<min(a.count, b.count) {
        let e = abs(a[i] - b[i]) - (absTol + rel * max(abs(a[i]), abs(b[i])))
        if e > worst || (a[i].isNaN != b[i].isNaN) { worst = max(e, worst); at = i }
    }
    #expect(at < 0, "\(what): element \(at): \(at >= 0 ? a[at] : 0) vs \(at >= 0 ? b[at] : 0)", sourceLocation: sourceLocation)
}

/// `y[s][o] = scale * sum_i W[o][i] * x[s][i]` in Double.
func referenceMatmul(_ w: [Float], cols: Int, rows: Int, _ x: [Float], batch: Int, scale: Float = 1) -> [Float] {
    var y = [Float](repeating: 0, count: batch * rows)
    for s in 0..<batch {
        for o in 0..<rows {
            var acc = 0.0
            for i in 0..<cols { acc += Double(w[o * cols + i]) * Double(x[s * cols + i]) }
            y[s * rows + o] = Float(acc) * scale
        }
    }
    return y
}

@Suite struct CPUBackendTests {
    static let matmulTypes: [GGMLType] = [.f32, .f16, .bf16, .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl, .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_xs]

    @Test(arguments: matmulTypes)
    func matmulMatchesReference(_ type: GGMLType) throws {
        var rng = TestRNG(UInt64(type.rawValue) * 31 + 1)
        let cols = 512, rows = 77
        let store = TestWeights(capacity: 1 << 20)
        var (w, deq) = try store.matrix(type, cols: cols, rows: rows, rng: &rng)
        w.scale = 0.75
        for threads in [1, 8] {
            let cpu = CPUBackend(threads: threads)
            for batch in [1, 3] {
                let x = rng.floats(batch * cols)
                let xb = try buffer(cpu, x)
                let yb = try cpu.makeBuffer(.f32, count: batch * rows + 5, label: "y")
                cpu.matmul(w, xb.slice(), into: yb.slice(0, batch * rows), batch: batch)
                try cpu.synchronize()
                expectClose(Array(yb.slice(0, batch * rows).floats), referenceMatmul(deq, cols: cols, rows: rows, x, batch: batch, scale: 0.75),
                            rel: 1e-4, abs: 1e-4, "\(type) threads \(threads) batch \(batch)")
                #expect(yb.slice(batch * rows, 5).floats.allSatisfy { $0 == 0 }, "wrote past the output")
            }
        }
    }

    @Test(arguments: [GGMLType.q8_0, .q4_K, .q6_K, .q4_0, .iq4_xs])
    func integerActivationPathIsCloseToFloatPath(_ type: GGMLType) throws {
        var rng = TestRNG(UInt64(type.rawValue) + 5)
        let cols = 1024, rows = 64, batch = 2
        let store = TestWeights(capacity: 1 << 20)
        let (w, deq) = try store.matrix(type, cols: cols, rows: rows, rng: &rng)
        let x = rng.floats(batch * cols)
        let cpu = CPUBackend(threads: 4, integerActivations: true)
        let yb = try cpu.makeBuffer(.f32, count: batch * rows, label: "y")
        cpu.matmul(w, try buffer(cpu, x).slice(), into: yb.slice(), batch: batch)
        let ref = referenceMatmul(deq, cols: cols, rows: rows, x, batch: batch)
        // Relative to the row's magnitude sum |w x|: int8 activations cost ~0.4% per term.
        for s in 0..<batch {
            for o in 0..<rows {
                var mag: Float = 0
                for i in 0..<cols { mag += abs(deq[o * cols + i] * x[s * cols + i]) }
                #expect(abs(yb.slice().floats[s * rows + o] - ref[s * rows + o]) <= 0.01 * mag)
            }
        }
    }

    @Test func matmulPairDefaultsToTwoMatmuls() throws {
        var rng = TestRNG(9)
        let store = TestWeights(capacity: 1 << 16)
        let (w0, d0) = try store.matrix(.q8_0, cols: 64, rows: 10, rng: &rng)
        let (w1, d1) = try store.matrix(.q4_K, cols: 256, rows: 6, rng: &rng)
        _ = w1; _ = d1
        let (w2, d2) = try store.matrix(.q8_0, cols: 64, rows: 7, rng: &rng)
        let cpu = CPUBackend()
        let x = rng.floats(64)
        let xb = try buffer(cpu, x)
        let y0 = try cpu.makeBuffer(.f32, count: 10, label: "y0"), y1 = try cpu.makeBuffer(.f32, count: 7, label: "y1")
        cpu.matmulPair(w0, w2, xb.slice(), into: y0.slice(), y1.slice(), batch: 1)
        expectClose(Array(y0.slice().floats), referenceMatmul(d0, cols: 64, rows: 10, x, batch: 1), rel: 1e-5)
        expectClose(Array(y1.slice().floats), referenceMatmul(d2, cols: 64, rows: 7, x, batch: 1), rel: 1e-5)
    }

    @Test(arguments: [false, true])
    func matmulExpertsMatchesPerExpertProducts(_ integer: Bool) throws {
        var rng = TestRNG(integer ? 77 : 78)
        let E = 6, cols = 256, rows = 40, S = 3, K = 2
        let store = TestWeights(capacity: 1 << 20)
        let values = rng.floats(E * cols * rows, scale: 0.3)
        let bytes = try Quantize.encode(values, as: .q4_K)
        let off = store.append(bytes)
        let stride = GGMLType.q4_K.rowBytes(cols) * rows
        let scales: [Float] = (0..<E).map { 1 + Float($0) / 4 }
        let stack = StackedWeights(region: store.region, baseOffset: off, stride: stride, count: E,
                                   type: .q4_K, cols: cols, rows: rows, scales: scales)
        let deq = Dequant.dequantize(.q4_K, bytes: bytes, count: E * cols * rows)
        let ids: [Int32] = [5, 0, 2, 2, 1, 4]
        let cpu = CPUBackend(threads: 8, integerActivations: integer)
        let idb = try ints(cpu, ids)
        let tol: Float = integer ? 2e-2 : 1e-4
        for shared in [true, false] {
            let xRows = shared ? S : S * K
            let x = rng.floats(xRows * cols)
            let y = try cpu.makeBuffer(.f32, count: S * K * rows, label: "y")
            cpu.matmulExperts(stack, indices: idb.slice(), rows: S, perRow: K, try buffer(cpu, x).slice(),
                              sharedInput: shared, into: y.slice())
            try cpu.synchronize()
            for p in 0..<(S * K) {
                let e = Int(ids[p])
                let xr = shared ? p / K : p
                let ref = referenceMatmul(Array(deq[e * cols * rows ..< (e + 1) * cols * rows]), cols: cols, rows: rows,
                                          Array(x[xr * cols ..< (xr + 1) * cols]), batch: 1, scale: scales[e])
                expectClose(Array(y.slice(p * rows, rows).floats), ref, rel: tol, abs: integer ? 0.1 : 1e-4, "pair \(p) shared \(shared)")
            }
        }
    }

    @Test func embedDequantisesRows() throws {
        var rng = TestRNG(4)
        let store = TestWeights(capacity: 1 << 16)
        let (table, deq) = try store.matrix(.q8_0, cols: 64, rows: 20, rng: &rng)
        let cpu = CPUBackend()
        let out = try cpu.makeBuffer(.f32, count: 3 * 64, label: "e")
        cpu.embed(table: table, tokens: [7, 0, 19], into: out.slice())
        for (s, t) in [7, 0, 19].enumerated() {
            #expect(Array(out.slice(s * 64, 64).floats) == Array(deq[t * 64 ..< (t + 1) * 64]))
        }
    }

    @Test func rmsNormMatchesUpstreamFormula() throws {
        var rng = TestRNG(12)
        let store = TestWeights(capacity: 1 << 16)
        let dim = 128, rows = 6
        let (w, wv) = try store.matrix(.f32, cols: dim, rows: 1, rng: &rng, scale: 2)
        let (w16, w16v) = try store.matrix(.f16, cols: dim, rows: 1, rng: &rng, scale: 2)
        let x = rng.floats(rows * dim, scale: 5)
        let cpu = CPUBackend(threads: 4)
        func ref(_ wv: [Float]) -> [Float] {
            var out = [Float](repeating: 0, count: rows * dim)
            for r in 0..<rows {
                var ms = 0.0
                for i in 0..<dim { ms += Double(x[r * dim + i]) * Double(x[r * dim + i]) }
                let rinv = 1 / sqrtf(Float(ms / Double(dim)) + 1e-6)
                for i in 0..<dim { out[r * dim + i] = x[r * dim + i] * rinv * wv[i] }
            }
            return out
        }
        let xb = try buffer(cpu, x)
        let out = try cpu.makeBuffer(.f32, count: rows * dim, label: "o")
        cpu.rmsNorm(xb.slice(), weight: w, into: out.slice(), rows: rows, dim: dim, eps: 1e-6)
        #expect(Array(out.slice().floats) == ref(wv))  // same operations, bit-identical
        cpu.rmsNorm(xb.slice(), weight: w16, into: xb.slice(), rows: rows, dim: dim, eps: 1e-6)  // in place, f16 weight
        #expect(Array(xb.slice().floats) == ref(w16v))
    }

    @Test func ropeNeoxMatchesUpstreamRotation() throws {
        var rng = TestRNG(21)
        let heads = 3, hd = 16, rows = 4, start = 1000
        let x = rng.floats(rows * heads * hd)
        let cpu = CPUBackend()
        let b = try buffer(cpu, x)
        for (ropeDims, theta) in [(16, Float(1_000_000)), (8, Float(10_000))] {
            for i in 0..<x.count { b.slice().floats[i] = x[i] }
            cpu.ropeNeox(b.slice(), RopeParams(rows: rows, heads: heads, headDim: hd, ropeDims: ropeDims, theta: theta, positionStart: start))
            var ref = x
            for s in 0..<rows {
                for h in 0..<heads {
                    let base = (s * heads + h) * hd, half = ropeDims / 2
                    for j in 0..<half {
                        let inv = powf(theta, -2.0 * Float(j) / Float(ropeDims))
                        let ang = Float(start + s) * inv
                        let a = x[base + j], c = x[base + j + half]
                        ref[base + j] = a * cosf(ang) - c * sinf(ang)
                        ref[base + j + half] = c * cosf(ang) + a * sinf(ang)
                    }
                }
            }
            #expect(Array(b.slice().floats) == ref, "ropeDims \(ropeDims)")
        }
        // Position 0 is the identity.
        for i in 0..<x.count { b.slice().floats[i] = x[i] }
        cpu.ropeNeox(b.slice(0, heads * hd), RopeParams(rows: 1, heads: heads, headDim: hd, ropeDims: hd, theta: 1e4, positionStart: 0))
        #expect(Array(b.slice(0, heads * hd).floats) == Array(x[0 ..< heads * hd]))
    }

    @Test(arguments: [ScalarType.f32, .f16])
    func storeKVAndAttentionMatchReference(_ dtype: ScalarType) throws {
        var rng = TestRNG(dtype == .f32 ? 31 : 32)
        let layers = 2, H = 4, KV = 2, hd = 16, vd = 8, maxT = 12
        let cpu = CPUBackend(threads: 4)
        let cache = try cpu.makeKVCache(KVCacheLayout(layers: layers, kvHeads: KV, keyDim: hd, valueDim: vd, maxTokens: maxT, dtype: dtype))
        func round(_ v: [Float]) -> [Float] { dtype == .f16 ? v.map { Float(Float16($0)) } : v }
        // Prefill 5 rows into layer 1, then decode 1 row at position 5.
        var allK = [Float](), allV = [Float]()
        var outs = [[Float]](), qs = [[Float]]()
        for (start, rows) in [(0, 5), (5, 1)] {
            let k = rng.floats(rows * KV * hd), v = rng.floats(rows * KV * vd), q = rng.floats(rows * H * hd)
            allK += round(k); allV += round(v); qs.append(q)
            cpu.storeKV(k: try buffer(cpu, k).slice(), v: try buffer(cpu, v).slice(), cache: cache, layer: 1, positionStart: start, rows: rows)
            let out = try cpu.makeBuffer(.f32, count: rows * H * vd, label: "o")
            let scale = 1 / sqrtf(Float(hd))
            cpu.attention(q: try buffer(cpu, q).slice(), cache: cache, layer: 1,
                          AttentionParams(rows: rows, heads: H, kvHeads: KV, headDim: hd, valueDim: vd, scale: scale, positionStart: start),
                          into: out.slice())
            try cpu.synchronize()
            outs.append(Array(out.slice().floats))
        }
        // Layer 0 untouched.
        let layerK = KV * maxT * hd
        if dtype == .f32 {
            #expect(cache.keys.slice(0, layerK).floats.allSatisfy { $0 == 0 })
        }
        // Reference.
        for (call, (start, rows)) in [(0, 5), (5, 1)].enumerated() {
            var ref = [Float](repeating: 0, count: rows * H * vd)
            for s in 0..<rows {
                let pos = start + s
                for h in 0..<H {
                    let g = h / (H / KV)
                    var sc = [Double]()
                    for t in 0...pos {
                        var d = 0.0
                        for i in 0..<hd { d += Double(qs[call][(s * H + h) * hd + i]) * Double(allK[(t * KV + g) * hd + i]) }
                        sc.append(d / Double(hd).squareRoot())
                    }
                    let m = sc.max()!
                    let e = sc.map { Foundation.exp($0 - m) }
                    let z = e.reduce(0, +)
                    for i in 0..<vd {
                        var acc = 0.0
                        for t in 0...pos { acc += e[t] / z * Double(allV[(t * KV + g) * vd + i]) }
                        ref[(s * H + h) * vd + i] = Float(acc)
                    }
                }
            }
            expectClose(outs[call], ref, rel: 1e-4, abs: 1e-5, "\(dtype) call \(call)")
        }
        // Cache layout: key of position 3, kv head 1, layer 1.
        let at = ((1 * KV + 1) * maxT + 3) * hd
        let stored: Float = dtype == .f32 ? cache.keys.slice().floats[at]
            : Float(cache.keys.contents.assumingMemoryBound(to: Float16.self)[at])
        #expect(stored == allK[(3 * KV + 1) * hd])
    }

    @Test func softmaxTopKAndRenormalisation() throws {
        let cpu = CPUBackend()
        let logits: [Float] = [1, 3, 3, -2, 0.5, 3, 2, 2,   // ties at 3 (cols 1, 2, 5) and 2 (6, 7)
                               0, 0, 0, 0, 0, 0, 0, 0]     // all equal
        let b = try buffer(cpu, logits)
        cpu.softmax(b.slice(), rows: 2, cols: 8)
        for r in 0..<2 {
            let row = logits[r * 8 ..< r * 8 + 8]
            let m = row.max()!
            let e = row.map { expf($0 - m) }
            let s = e.reduce(0, +)
            #expect(Array(b.slice(r * 8, 8).floats) == e.map { $0 / s })
        }
        let ids = try cpu.makeBuffer(.i32, count: 8, label: "ids"), w = try cpu.makeBuffer(.f32, count: 8, label: "w")
        cpu.topK(b.slice(), rows: 2, cols: 8, k: 4, renormalize: false, ids: ids.slice(), weights: w.slice())
        #expect(Array(ids.slice().int32s) == [1, 2, 5, 6, 0, 1, 2, 3])
        let p = b.slice().floats
        #expect(Array(w.slice(0, 4).floats) == [p[1], p[2], p[5], p[6]])
        cpu.topK(b.slice(), rows: 2, cols: 8, k: 3, renormalize: true, ids: ids.slice(0, 6), weights: w.slice(0, 6))
        let sum = p[1] + p[2] + p[5]
        #expect(Array(w.slice(0, 3).floats) == [p[1] / sum, p[2] / sum, p[5] / sum])
        #expect(abs(w.slice(3, 3).floats.reduce(0, +) - 1) < 1e-6)
        #expect(Array(ids.slice(0, 6).int32s) == [1, 2, 5, 0, 1, 2])
    }

    @Test func elementwiseOps() throws {
        var rng = TestRNG(55)
        let n = 40_000  // above the parallel threshold of siluMul
        let g = rng.floats(n, scale: 6), u = rng.floats(n), a = rng.floats(n)
        let cpu = CPUBackend()
        let gb = try buffer(cpu, g), ub = try buffer(cpu, u), ab = try buffer(cpu, a)
        let out = try cpu.makeBuffer(.f32, count: n, label: "o")
        cpu.siluMul(gate: gb.slice(), up: ub.slice(), into: out.slice(), count: n)
        #expect(Array(out.slice().floats) == (0..<n).map { (g[$0] / (1 + expf(-g[$0]))) * u[$0] })
        cpu.add(ab.slice(), ub.slice(), into: ab.slice(), count: n)  // aliasing
        #expect(Array(ab.slice().floats) == (0..<n).map { a[$0] + u[$0] })
    }

    @Test(arguments: [false, true])
    func weightedSumCombinesInRankOrder(_ accumulate: Bool) throws {
        var rng = TestRNG(accumulate ? 61 : 62)
        let rows = 3, terms = 4, dim = 33
        let x = rng.floats(rows * terms * dim), w = rng.floats(rows * terms), base = rng.floats(rows * dim)
        let cpu = CPUBackend()
        let out = try buffer(cpu, base)
        cpu.weightedSum(try buffer(cpu, x).slice(), weights: try buffer(cpu, w).slice(), rows: rows, terms: terms, dim: dim,
                        into: out.slice(), accumulate: accumulate)
        var ref = [Float](repeating: 0, count: rows * dim)
        for r in 0..<rows {
            for i in 0..<dim {
                var acc: Float = 0
                for k in 0..<terms { acc += w[r * terms + k] * x[(r * terms + k) * dim + i] }
                ref[r * dim + i] = accumulate ? base[r * dim + i] + acc : acc
            }
        }
        #expect(Array(out.slice().floats) == ref)
    }

    @Test func encodeWaitBlocksUntilSignalled() throws {
        let cpu = CPUBackend()
        let signal = cpu.makeSignal()
        let start = Date()
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.05) { signal.signal(2) }
        cpu.encodeWait(for: signal, value: 2)
        #expect(Date().timeIntervalSince(start) >= 0.04)
        #expect(signal.value == 2)
        signal.signal(1)  // never goes backwards
        #expect(signal.value == 2)
        cpu.encodeWait(for: signal, value: 1)  // already satisfied: returns at once
    }

    @Test func parallelChunksCoverTheRangeOnce() {
        for (count, threads, minChunk) in [(0, 4, 16), (1, 4, 16), (1000, 1, 16), (1000, 16, 16), (17, 16, 1), (151_936, 16, 2374)] {
            let hits = UnsafeMutablePointer<Int32>.allocate(capacity: max(count, 1))
            hits.initialize(repeating: 0, count: max(count, 1))
            defer { hits.deallocate() }
            let h = UnsafeSendable(hits)
            Parallel.forChunks(count, threads: threads, minChunk: minChunk) { r in
                #expect(r.count >= min(minChunk, count) || r.upperBound == count)
                for i in r { h.value[i] += 1 }
            }
            #expect((0..<count).allSatisfy { hits[$0] == 1 }, "count \(count) threads \(threads)")
        }
    }

    @Test(.enabled(if: haveQwen3, "Qwen3-0.6B-Q8_0.gguf is not on disk"))
    func realModelEmbeddingAndTiedHead() throws {
        let f = try GGUFFile(path: qwen3Path)
        let cpu = CPUBackend()
        try cpu.register(f.file.region)
        let emb = try f.weight("token_embd.weight")
        let x = try cpu.makeBuffer(.f32, count: 1024, label: "x")
        cpu.embed(table: emb, tokens: [9707], into: x.slice())  // "Hello"
        // Logits of the tied head for the token's own embedding: its own id should score high.
        let logits = try cpu.makeBuffer(.f32, count: emb.rows, label: "logits")
        cpu.matmul(emb, x.slice(), into: logits.slice(), batch: 1)
        try cpu.synchronize()
        let l = logits.slice().floats
        var ref = 0.0
        let row = Array(x.slice().floats)
        for i in 0..<1024 { ref += Double(row[i]) * Double(row[i]) }
        #expect(abs(Double(l[9707]) - ref) <= 1e-4 * ref)
        #expect(l.allSatisfy { $0.isFinite })
    }
}
