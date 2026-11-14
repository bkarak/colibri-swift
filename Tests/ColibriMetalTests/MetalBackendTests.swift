import ColibriCore
import Foundation
import Metal
import Testing
import ColibriMetal

// Every Metal op against the CPU backend on random data. Weights are random floats
// quantised with `Quantize.encode`, so both backends decode the same block bytes; the
// difference left is f32 summation order (and FMA contraction), hence the tolerances.

@Suite(.enabled(if: metalAvailable)) struct MetalBackendTests {
    let twin: Twin
    init() throws { twin = try Twin() }

    /// Per-output magnitude sum_i |w[o][i] * x[s][i]| * |scale|, the scale of the rounding error.
    static func magnitudes(_ w: [Float], rows: Int, cols: Int, rowOffset: Int, viewRows: Int,
                           _ x: [Float], batch: Int, scale: Float) -> [Float] {
        var m = [Float](repeating: 0, count: batch * viewRows)
        for s in 0..<batch {
            for o in 0..<viewRows {
                var acc: Float = 0
                let wr = (rowOffset + o) * cols
                for i in 0..<cols { acc += Swift.abs(w[wr + i] * x[s * cols + i]) }
                m[s * viewRows + o] = acc * Swift.abs(scale)
            }
        }
        return m
    }

    @Test(arguments: metalWeightTypes)
    func matmulMatchesCPU(_ type: GGMLType) throws {
        let rows = 45, cols = 512
        var rng = TestRNG(UInt64(type.rawValue) + 100)
        let bytes = try Quantize.encode(rng.floats(rows * cols), as: type)
        let w = dequantRows(type, bytes, rows: rows, cols: cols)
        let leading = 96
        let region = makeRegion(bytes, leading: leading)
        let rb = type.rowBytes(cols)
        // The whole matrix, and a view starting at row 5 (an offset that is not 256-aligned).
        let views: [(WeightView, Int)] = [
            (WeightView(region: region, offset: leading, type: type, cols: cols, rows: rows), 0),
            (WeightView(region: region, offset: leading + 5 * rb, type: type, cols: cols, rows: rows - 5, scale: 0.75), 5),
        ]
        for batch in [1, 3, 8, 13, 21, 33] {  // 21 and 33 take the tiled kernel (one partial tile)
            let x = rng.floats(batch * cols)
            for (view, r0) in views {
                let (c, m) = try twin.both { b -> () -> [Float] in
                    let xb = try b.upload(x), yb = try b.zeros(batch * view.rows)
                    b.matmul(view, xb.slice(), into: yb.slice(), batch: batch)
                    return { yb.f }
                }
                let mag = Self.magnitudes(w, rows: rows, cols: cols, rowOffset: r0, viewRows: view.rows, x, batch: batch, scale: view.scale)
                expectClose(c, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(type) matmul batch \(batch) row0 \(r0)")
            }
        }
    }

    /// The tiled prefill kernel over several 64-row x 32-token tiles with partial edges
    /// in both directions, against the CPU and against the matvec prefill kernel.
    @Test(arguments: metalWeightTypes)
    func tiledMatmulMatchesCPUAndMatvec(_ type: GGMLType) throws {
        let rows = 200, cols = 768, batch = 70
        var rng = TestRNG(UInt64(type.rawValue) + 700)
        let bytes = try Quantize.encode(rng.floats(rows * cols), as: type)
        let w = dequantRows(type, bytes, rows: rows, cols: cols)
        let view = WeightView(region: makeRegion(bytes, leading: 64), offset: 64, type: type, cols: cols, rows: rows, scale: 0.5)
        let x = rng.floats(batch * cols)
        let (c, m) = try twin.both { b -> () -> [Float] in
            let xb = try b.upload(x), yb = try b.zeros(batch * rows)
            b.matmul(view, xb.slice(), into: yb.slice(), batch: batch)
            return { yb.f }
        }
        let mag = Self.magnitudes(w, rows: rows, cols: cols, rowOffset: 0, viewRows: rows, x, batch: batch, scale: 0.5)
        expectClose(c, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(type) tiled matmul")
        // The matvec prefill kernel on the same data.
        twin.metal.tiledMinBatch = .max
        defer { twin.metal.tiledMinBatch = 20 }
        let mv = try twin.both { b -> () -> [Float] in
            let xb = try b.upload(x), yb = try b.zeros(batch * rows)
            b.matmul(view, xb.slice(), into: yb.slice(), batch: batch)
            return { yb.f }
        }.1
        expectClose(mv, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(type) tiled vs matvec")
    }

    /// Tall matrices (output heads) take the two-rows-per-simdgroup decode kernel.
    @Test(arguments: metalWeightTypes)
    func tallMatvecMatchesCPU(_ type: GGMLType) throws {
        let rows = 16_389, cols = 256
        var rng = TestRNG(UInt64(type.rawValue) + 400)
        let bytes = try Quantize.encode(rng.floats(rows * cols), as: type)
        let w = dequantRows(type, bytes, rows: rows, cols: cols)
        let view = WeightView(region: makeRegion(bytes), offset: 0, type: type, cols: cols, rows: rows, scale: 1.25)
        let x = rng.floats(cols)
        let (c, m) = try twin.both { b -> () -> [Float] in
            let xb = try b.upload(x), yb = try b.zeros(rows)
            b.matmul(view, xb.slice(), into: yb.slice(), batch: 1)
            return { yb.f }
        }
        let mag = Self.magnitudes(w, rows: rows, cols: cols, rowOffset: 0, viewRows: rows, x, batch: 1, scale: 1.25)
        expectClose(c, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(type) tall matvec")
    }

    @Test(arguments: metalWeightTypes)
    func matmulExpertsMatchesCPU(_ type: GGMLType) throws {
        let count = 6, rows = 40, cols = 256
        var rng = TestRNG(UInt64(type.rawValue) + 200)
        let rb = type.rowBytes(cols)
        // Stride larger than one matrix (as in an arena slot), base not 256-aligned.
        let stride = alignUp(rb * rows + 100, 2)
        let leading = 64
        var bytes = [UInt8](repeating: 0, count: stride * count)
        for e in 0..<count {
            let m = try Quantize.encode(rng.floats(rows * cols), as: type)
            bytes.replaceSubrange(e * stride ..< e * stride + m.count, with: m)
        }
        let region = makeRegion(bytes, leading: leading)
        for (tokens, perRow, shared, scales) in [(3, 2, true, nil as [Float]?), (2, 3, false, [0.5, 1, 2, -1, 1.5, 0.25] as [Float]?)] {
            let stack = StackedWeights(region: region, baseOffset: leading, stride: stride, count: count,
                                       type: type, cols: cols, rows: rows, scales: scales)
            let pairs = tokens * perRow
            let ids: [Int32] = (0..<pairs).map { _ in Int32(rng.int(count)) }
            let x = rng.floats((shared ? tokens : pairs) * cols)
            let (c, m) = try twin.both { b -> () -> [Float] in
                let xb = try b.upload(x), ib = try b.uploadI32(ids), yb = try b.zeros(pairs * rows)
                b.matmulExperts(stack, indices: ib.slice(), rows: tokens, perRow: perRow, xb.slice(), sharedInput: shared, into: yb.slice())
                return { yb.f }
            }
            var mag = [Float](repeating: 0, count: pairs * rows)
            for p in 0..<pairs {
                let e = Int(ids[p])
                let wbytes = Array(bytes[e * stride ..< e * stride + rb * rows])
                let w = dequantRows(type, wbytes, rows: rows, cols: cols)
                let xr = shared ? p / perRow : p
                let mm = Self.magnitudes(w, rows: rows, cols: cols, rowOffset: 0, viewRows: rows,
                                         Array(x[xr * cols ..< (xr + 1) * cols]), batch: 1, scale: scales?[e] ?? 1)
                for o in 0..<rows { mag[p * rows + o] = mm[o] }
            }
            expectClose(c, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(type) matmulExperts shared \(shared)")
        }
    }

    /// The routed tiled path (GPU grouping + tiled kernel): enough pairs to take it, a
    /// matrix of 100 rows (a partial 64-row tile), and one hot matrix with more pairs than
    /// the parallel token tiles cover (the kernel loops), shared and per-pair inputs.
    @Test(arguments: metalWeightTypes)
    func tiledExpertsMatchCPU(_ type: GGMLType) throws {
        let count = 6, rows = 100, cols = 512
        var rng = TestRNG(UInt64(type.rawValue) + 800)
        let rb = type.rowBytes(cols)
        let stride = alignUp(rb * rows + 100, 2)
        let leading = 64
        var bytes = [UInt8](repeating: 0, count: stride * count)
        for e in 0..<count {
            let m = try Quantize.encode(rng.floats(rows * cols), as: type)
            bytes.replaceSubrange(e * stride ..< e * stride + m.count, with: m)
        }
        let region = makeRegion(bytes, leading: leading)
        for (tokens, perRow, shared, scales) in [(100, 4, true, nil as [Float]?), (400, 1, false, [0.5, 1, 2, -1, 1.5, 0.25] as [Float]?)] {
            let stack = StackedWeights(region: region, baseOffset: leading, stride: stride, count: count,
                                       type: type, cols: cols, rows: rows, scales: scales)
            let pairs = tokens * perRow
            #expect(pairs >= twin.metal.tiledExpertMinPairs)
            // 80% of the pairs go to matrix 2.
            let ids: [Int32] = (0..<pairs).map { _ in rng.int(10) < 8 ? 2 : Int32(rng.int(count)) }
            let x = rng.floats((shared ? tokens : pairs) * cols)
            let (c, m) = try twin.both { b -> () -> [Float] in
                let xb = try b.upload(x), ib = try b.uploadI32(ids), yb = try b.zeros(pairs * rows)
                b.matmulExperts(stack, indices: ib.slice(), rows: tokens, perRow: perRow, xb.slice(), sharedInput: shared, into: yb.slice())
                return { yb.f }
            }
            var mag = [Float](repeating: 0, count: pairs * rows)
            var deq: [Int: [Float]] = [:]
            for p in 0..<pairs {
                let e = Int(ids[p])
                let w = deq[e] ?? dequantRows(type, Array(bytes[e * stride ..< e * stride + rb * rows]), rows: rows, cols: cols)
                deq[e] = w
                let xr = shared ? p / perRow : p
                let mm = Self.magnitudes(w, rows: rows, cols: cols, rowOffset: 0, viewRows: rows,
                                         Array(x[xr * cols ..< (xr + 1) * cols]), batch: 1, scale: scales?[e] ?? 1)
                for o in 0..<rows { mag[p * rows + o] = mm[o] }
            }
            expectClose(c, m, rel: 3e-5, abs: 1e-6, scale: mag, "\(type) tiled matmulExperts shared \(shared)")
        }
    }

    /// Out-of-range ids on the routed tiled path give NaN rows; the others are computed.
    @Test func tiledExpertsMarkBadIndices() throws {
        let count = 4, rows = 64, cols = 256, pairs = 400
        var rng = TestRNG(4242)
        let bytes = try Quantize.encode(rng.floats(count * rows * cols), as: .q8_0)
        let stack = StackedWeights(region: makeRegion(bytes), baseOffset: 0, stride: GGMLType.q8_0.rowBytes(cols) * rows,
                                   count: count, type: .q8_0, cols: cols, rows: rows)
        var ids: [Int32] = (0..<pairs).map { _ in Int32(rng.int(count)) }
        ids[7] = 4; ids[150] = -1
        let x = rng.floats(pairs * cols)
        let b = twin.metal
        let xb = try b.upload(x), ib = try b.uploadI32(ids), yb = try b.zeros(pairs * rows)
        b.matmulExperts(stack, indices: ib.slice(), rows: pairs, perRow: 1, xb.slice(), sharedInput: false, into: yb.slice())
        try b.synchronize()
        let y = yb.f
        for p in [7, 150] {
            let allNaN = y[p * rows ..< (p + 1) * rows].allSatisfy { $0.isNaN }
            #expect(allNaN, "pair \(p)")
        }
        let finite = y[0 ..< rows].allSatisfy { $0.isFinite } && y[8 * rows ..< 9 * rows].allSatisfy { $0.isFinite }
        #expect(finite)
    }

    @Test(arguments: metalWeightTypes)
    func embedMatchesCPU(_ type: GGMLType) throws {
        let rows = 20, cols = 256
        var rng = TestRNG(UInt64(type.rawValue) + 300)
        let bytes = try Quantize.encode(rng.floats(rows * cols), as: type)
        let region = makeRegion(bytes, leading: 32)
        let tokens: [Int32] = [3, 0, 19, 3, 7]
        for scale: Float in [1, 1.5] {
            let table = WeightView(region: region, offset: 32, type: type, cols: cols, rows: rows, scale: scale)
            let (c, m) = try twin.both { b -> () -> [Float] in
                let out = try b.zeros(tokens.count * cols)
                b.embed(table: table, tokens: tokens, into: out.slice())
                return { out.f }
            }
            // Same decoded values; only an FMA in the decoder can move the last bit.
            expectClose(c, m, rel: 1e-6, abs: 1e-7, "\(type) embed scale \(scale)")
        }
    }

    @Test func rmsNormMatchesCPU() throws {
        var rng = TestRNG(1)
        for (rows, dim, wtype) in [(5, 2048, GGMLType.f32), (48, 128, .f32), (3, 96, .f32), (4, 1024, .f16), (2, 256, .bf16)] {
            let wvals = rng.floats(dim, scale: 2)
            let region = makeRegion(try Quantize.encode(wvals, as: wtype), leading: 64)
            let weight = WeightView(region: region, offset: 64, type: wtype, cols: dim, rows: 1)
            let x = rng.floats(rows * dim, scale: 3)
            for inPlace in [false, true] {
                let (c, m) = try twin.both { b -> () -> [Float] in
                    let xb = try b.upload(x)
                    let ob = inPlace ? xb : try b.zeros(rows * dim)
                    b.rmsNorm(xb.slice(), weight: weight, into: ob.slice(), rows: rows, dim: dim, eps: 1e-6)
                    return { ob.f }
                }
                expectClose(c, m, rel: 2e-6, abs: 1e-6, "rmsNorm rows \(rows) dim \(dim) \(wtype) inPlace \(inPlace)")
            }
        }
    }

    @Test func ropeMatchesCPU() throws {
        var rng = TestRNG(2)
        for (rows, heads, hd, rd, pos) in [(4, 3, 64, 64, 0), (5, 2, 128, 128, 1000), (2, 4, 64, 32, 37), (1, 16, 128, 128, 40000)] {
            let x = rng.floats(rows * heads * hd)
            let params = RopeParams(rows: rows, heads: heads, headDim: hd, ropeDims: rd, theta: 1_000_000, positionStart: pos)
            let (c, m) = try twin.both { b -> () -> [Float] in
                let xb = try b.upload(x)
                b.ropeNeox(xb.slice(), params)
                return { xb.f }
            }
            // The angle pos * inv is one f32 rounding away from the CPU's (powf vs Metal pow can
            // differ by an ulp), and an ulp of the angle is ~6e-8 * pos rad: allow a few.
            let absTol: Float = 1e-5 + Float(pos) * 3e-7
            expectClose(c, m, rel: 1e-5, abs: absTol, "rope rows \(rows) hd \(hd) rd \(rd) pos \(pos)")
        }
    }

    @Test(arguments: [ScalarType.f32, .f16])
    func storeKVAndAttentionMatchCPU(_ dtype: ScalarType) throws {
        var rng = TestRNG(dtype == .f32 ? 3 : 4)
        for (heads, kvHeads, hd) in [(6, 2, 64), (4, 4, 128), (8, 1, 32)] {
            let layout = KVCacheLayout(layers: 2, kvHeads: kvHeads, keyDim: hd, valueDim: hd, maxTokens: 300, dtype: dtype)
            // Prefill 37 rows at 0, then 213 rows (crosses several 32-position chunks per
            // simdgroup), then a 1-row decode step at 250, all into layer 1.
            let steps = [(0, 37), (37, 213), (250, 1)]
            let kvData = steps.map { (_, rows) in (rng.floats(rows * kvHeads * hd), rng.floats(rows * kvHeads * hd)) }
            let qData = steps.map { (_, rows) in rng.floats(rows * heads * hd) }
            let (c, m) = try twin.both { b -> () -> ([Float], [Float], [Float]) in
                let cache = try b.makeKVCache(layout)
                var outs: [DeviceBuffer] = []
                for (n, (pos, rows)) in steps.enumerated() {
                    let kb = try b.upload(kvData[n].0), vb = try b.upload(kvData[n].1), qb = try b.upload(qData[n])
                    let ob = try b.zeros(rows * heads * hd)
                    b.storeKV(k: kb.slice(), v: vb.slice(), cache: cache, layer: 1, positionStart: pos, rows: rows)
                    b.attention(q: qb.slice(), cache: cache, layer: 1,
                                AttentionParams(rows: rows, heads: heads, kvHeads: kvHeads, headDim: hd, valueDim: hd,
                                                scale: 1 / Float(hd).squareRoot(), positionStart: pos),
                                into: ob.slice())
                    outs.append(ob)
                }
                return {
                    func raw(_ d: DeviceBuffer) -> [Float] {
                        if d.dtype == .f32 { return d.f }
                        let p = d.contents.assumingMemoryBound(to: Float16.self)
                        return (0..<d.count).map { Float(p[$0]) }
                    }
                    return (outs.flatMap { $0.f }, raw(cache.keys), raw(cache.values))
                }
            }
            #expect(c.1 == m.1, "storeKV keys differ (\(dtype), heads \(heads)/\(kvHeads), hd \(hd))")
            #expect(c.2 == m.2, "storeKV values differ (\(dtype), heads \(heads)/\(kvHeads), hd \(hd))")
            expectClose(c.0, m.0, rel: 1e-4, abs: 2e-5, "attention \(dtype) heads \(heads)/\(kvHeads) hd \(hd)")
        }
    }

    /// Split-KV decode attention (rows <= 4 at long context): many splits, a last split
    /// shorter than the rest, and splits past the end that stay empty (n = 8301 gives 64
    /// splits of 160 positions, the last 12 empty), then a 3-row batch.
    @Test(arguments: [ScalarType.f32, .f16])
    func splitDecodeAttentionMatchesCPU(_ dtype: ScalarType) throws {
        var rng = TestRNG(dtype == .f32 ? 31 : 32)
        let (heads, kvHeads, hd, filled) = (4, 2, 64, 8300)
        let layout = KVCacheLayout(layers: 1, kvHeads: kvHeads, keyDim: hd, valueDim: hd, maxTokens: 8400, dtype: dtype)
        let k = rng.floats(filled * kvHeads * hd), v = rng.floats(filled * kvHeads * hd)
        let steps = [(8300, 1), (5000, 3), (700, 1)]
        let kv1 = steps.map { (_, rows) in (rng.floats(rows * kvHeads * hd), rng.floats(rows * kvHeads * hd)) }
        let qs = steps.map { (_, rows) in rng.floats(rows * heads * hd, scale: 2) }
        let (c, m) = try twin.both { b -> () -> [Float] in
            let cache = try b.makeKVCache(layout)
            let kb = try b.upload(k), vb = try b.upload(v)
            b.storeKV(k: kb.slice(), v: vb.slice(), cache: cache, layer: 0, positionStart: 0, rows: filled)
            var outs: [DeviceBuffer] = []
            for (n, (pos, rows)) in steps.enumerated() {
                let kn = try b.upload(kv1[n].0), vn = try b.upload(kv1[n].1), qb = try b.upload(qs[n])
                let ob = try b.zeros(rows * heads * hd)
                b.storeKV(k: kn.slice(), v: vn.slice(), cache: cache, layer: 0, positionStart: pos, rows: rows)
                b.attention(q: qb.slice(), cache: cache, layer: 0,
                            AttentionParams(rows: rows, heads: heads, kvHeads: kvHeads, headDim: hd, valueDim: hd,
                                            scale: 1 / Float(hd).squareRoot(), positionStart: pos),
                            into: ob.slice())
                outs.append(ob)
            }
            return { outs.flatMap { $0.f } }
        }
        expectClose(c, m, rel: 1e-4, abs: 2e-5, "split attention \(dtype)")
    }

    @Test func softmaxMatchesCPU() throws {
        var rng = TestRNG(5)
        for (rows, cols) in [(3, 100), (2, 1000), (4, 128), (1, 7)] {
            let x = rng.floats(rows * cols, scale: 8)
            let (c, m) = try twin.both { b -> () -> [Float] in
                let xb = try b.upload(x)
                b.softmax(xb.slice(), rows: rows, cols: cols)
                return { xb.f }
            }
            expectClose(c, m, rel: 1e-5, abs: 1e-9, "softmax \(rows)x\(cols)")
        }
    }

    @Test func topKMatchesCPUExactly() throws {
        var rng = TestRNG(6)
        let cols = 128, k = 8
        var scores = rng.floats(cols)                                  // plain row
        var ties = (0..<cols).map { Float($0 % 5) }                    // many ties
        ties[100] = 4
        let nans = [Float](repeating: .nan, count: cols)               // nothing beats -1e30
        var mixed = rng.floats(cols); for i in stride(from: 0, to: cols, by: 3) { mixed[i] = .nan }
        var neg = [Float](repeating: -.infinity, count: cols); neg[77] = -3; neg[5] = -3
        scores += ties + nans + mixed + neg
        let rows = 5
        for renorm in [false, true] {
            for (cc, kk) in [(cols, k), (cols, 1), (cols, cols)] {
                let (c, m) = try twin.both { b -> () -> ([Int32], [Float]) in
                    let sb = try b.upload(scores), ib = try b.zeros(rows * kk, .i32), wb = try b.zeros(rows * kk)
                    b.topK(sb.slice(), rows: rows, cols: cc, k: kk, renormalize: renorm, ids: ib.slice(), weights: wb.slice())
                    return { (ib.i, wb.f) }
                }
                #expect(c.0 == m.0, "topK ids k \(kk) renorm \(renorm)")
                let wEqual = zip(c.1, m.1).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) }
                #expect(wEqual, "topK weights k \(kk) renorm \(renorm)")
            }
        }
    }

    @Test func elementwiseOpsMatchCPU() throws {
        var rng = TestRNG(7)
        let n = 5000
        let g = rng.floats(n, scale: 6), u = rng.floats(n), a = rng.floats(n)
        let (c, m) = try twin.both { b -> () -> ([Float], [Float], [Float]) in
            let gb = try b.upload(g), ub = try b.upload(u), ab = try b.upload(a), o = try b.zeros(n)
            b.siluMul(gate: gb.slice(), up: ub.slice(), into: o.slice(), count: n)
            b.add(ab.slice(), ub.slice(), into: ab.slice(), count: n)             // aliasing out == a
            b.siluMul(gate: gb.slice(1, 100), up: ub.slice(), into: gb.slice(1, 100), count: 99)
            return { (o.f, ab.f, gb.f) }
        }
        expectClose(c.0, m.0, rel: 1e-6, abs: 1e-7, "siluMul")
        #expect(c.1 == m.1, "add")
        expectClose(c.2, m.2, rel: 1e-6, abs: 1e-7, "siluMul on sub-slices")
    }

    @Test func weightedSumMatchesCPU() throws {
        var rng = TestRNG(8)
        let rows = 3, terms = 8, dim = 300
        let x = rng.floats(rows * terms * dim), w = rng.floats(rows * terms), base = rng.floats(rows * dim)
        for accumulate in [false, true] {
            let (c, m) = try twin.both { b -> () -> [Float] in
                let xb = try b.upload(x), wb = try b.upload(w), ob = try b.upload(base)
                b.weightedSum(xb.slice(), weights: wb.slice(), rows: rows, terms: terms, dim: dim, into: ob.slice(), accumulate: accumulate)
                return { ob.f }
            }
            expectClose(c, m, rel: 1e-6, abs: 1e-6, "weightedSum accumulate \(accumulate)")
        }
    }

    @Test func copyHandlesEveryDtype() throws {
        let metal = twin.metal
        let f = try metal.upload((0..<50).map(Float.init)), g = try metal.zeros(50)
        let i = try metal.uploadI32((0..<50).map { Int32($0) * 3 }), j = try metal.zeros(50, .i32)
        let h = try metal.makeBuffer(.f16, count: 50, label: "h"), h2 = try metal.makeBuffer(.f16, count: 50, label: "h2")
        let hp = h.contents.assumingMemoryBound(to: Float16.self)
        for n in 0..<50 { hp[n] = Float16(Float(n) * 0.5) }
        metal.beginBatch(label: "copies")
        metal.copy(f.slice(3, 40), into: g.slice(7, 40), count: 40)
        metal.copy(i.slice(), into: j.slice(), count: 50)
        metal.copy(h.slice(1, 30), into: h2.slice(3, 30), count: 30)
        try metal.synchronize()
        let gf = g.f
        #expect(Array(gf[7..<47]) == (3..<43).map(Float.init))
        #expect(gf[0..<7].allSatisfy { $0 == 0 } && gf[47...].allSatisfy { $0 == 0 })
        #expect(j.i == (0..<50).map { Int32($0) * 3 })
        let h2p = h2.contents.assumingMemoryBound(to: Float16.self)
        #expect((0..<30).allSatisfy { h2p[3 + $0] == hp[1 + $0] })
    }

    /// The GPU waits on the event: work encoded after `encodeWait` does not run before the
    /// host signals, and the batch completes once it does.
    @Test func encodeWaitBlocksUntilSignalled() throws {
        let metal = twin.metal
        let signal = metal.makeSignal()
        let a = try metal.upload([1, 2, 3, 4]), b = try metal.zeros(4)
        metal.beginBatch(label: "gated")
        metal.encodeWait(for: signal, value: 1)
        metal.add(a.slice(), a.slice(), into: b.slice(), count: 4)
        metal.commitBatch()
        Thread.sleep(forTimeInterval: 0.05)
        #expect(b.f == [0, 0, 0, 0], "ran before the signal")
        // A dedicated thread, as the expert store's I/O threads are (GCD's global pool can be
        // saturated by the parallel CPU tests).
        let box = UnsafeSendable(signal)
        let t0 = Date()
        Thread.detachNewThread { box.value.signal(1) }
        try metal.synchronize()
        #expect(Date().timeIntervalSince(t0) < 2, "the GPU took long to see the signal")
        #expect(b.f == [2, 4, 6, 8])
        #expect(signal.value == 1)
    }

    @Test func capabilitiesAdvertiseEveryTestedType() throws {
        let caps = twin.metal.capabilities
        #expect(caps.name == "metal")
        #expect(caps.isAsynchronous)
        #expect(caps.weightTypes == GGMLType.firstPortWeightTypes)
        #expect(caps.maxBufferLength > 0 && caps.workingSetLimit > 0)
    }
}
