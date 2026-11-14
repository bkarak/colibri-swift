import Foundation

/// The reference backend: eager execution on the CPU, parallelised with
/// `DispatchQueue.concurrentPerform` over chunks of output rows (never nested, never
/// spinning; see docs/port/runtime-and-metal.md section 6.6). Every other backend is
/// tested against this one.
///
/// Numerics follow upstream colibri (olmoe.c / qwen36.c, docs/port/models.md sections 4
/// and 5): RMSNorm sums squares in Double, RoPE angles are Float32 `powf`/`cosf`/`sinf`,
/// softmax is max-subtracted with a Float32 sum and a division, top-k is a strict `>`
/// scan. Matrix products default to the float path (dequantised weights, f32
/// activations: upstream's GQ_EXACT=1); `integerActivations` switches block-quantised
/// weights to upstream's default int8-activation path.
public final class CPUBackend: Backend {
    public let capabilities: BackendCapabilities
    /// Worker count for `concurrentPerform` splits (`--threads`; default: active cores).
    public let threads: Int
    /// Use the int8-activation integer path (upstream's default) for block-quantised
    /// weights instead of the float path.
    public let integerActivations: Bool

    private let q8 = Q8Activations()

    public init(threads: Int? = nil, integerActivations: Bool = false) {
        let n = threads ?? ProcessInfo.processInfo.activeProcessorCount
        self.threads = max(1, n)
        self.integerActivations = integerActivations
        self.capabilities = BackendCapabilities(
            name: "cpu",
            weightTypes: GGMLType.firstPortWeightTypes,
            maxBufferLength: Int.max,
            workingSetLimit: Int(ProcessInfo.processInfo.physicalMemory),
            isAsynchronous: false)
    }

    // MARK: Memory

    public func register(_ region: MemoryRegion) throws {}

    public func makeBuffer(_ dtype: ScalarType, count: Int, label: String) throws -> DeviceBuffer {
        let bytes = alignUp(max(count * dtype.size, 1), hostPageSize)
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, hostPageSize, bytes) == 0, let p = raw else {
            throw ColibriError.backend("cpu: cannot allocate \(bytes) bytes for \(label)")
        }
        memset(p, 0, bytes)
        return DeviceBuffer(dtype: dtype, count: count, label: label, contents: p, handle: nil,
                            release: { free(p) })
    }

    public func makeKVCache(_ layout: KVCacheLayout) throws -> KVCache {
        precondition(layout.dtype == .f32 || layout.dtype == .f16, "cpu: KV cache must be f32 or f16")
        return KVCache(layout: layout,
                       keys: try makeBuffer(layout.dtype, count: layout.keyCount, label: "kv.keys"),
                       values: try makeBuffer(layout.dtype, count: layout.valueCount, label: "kv.values"))
    }

    // MARK: Batching (eager: nothing to record)

    public func beginBatch(label: String) {}
    public func commitBatch() {}
    public func synchronize() throws {}
    public func makeSignal() -> HostSignal { BlockingHostSignal() }

    public func encodeWait(for signal: HostSignal, value: UInt64) {
        guard let s = signal as? BlockingHostSignal else {
            preconditionFailure("cpu: foreign HostSignal")
        }
        s.wait(atLeast: value)
    }

    // MARK: Helpers

    @inline(__always) private func f32(_ s: BufferSlice) -> UnsafeMutablePointer<Float> {
        precondition(s.buffer.dtype == .f32, "cpu: \(s.buffer.label) is not f32")
        return (s.buffer.contents + s.byteOffset).assumingMemoryBound(to: Float.self)
    }

    @inline(__always) private func i32(_ s: BufferSlice) -> UnsafeMutablePointer<Int32> {
        precondition(s.buffer.dtype == .i32, "cpu: \(s.buffer.label) is not i32")
        return (s.buffer.contents + s.byteOffset).assumingMemoryBound(to: Int32.self)
    }

    /// Rows per work item for a matrix of `rows` outputs: about four items per worker,
    /// at least 16 rows.
    private func rowChunk(_ rows: Int) -> Int {
        max(16, (rows + threads * 4 - 1) / (threads * 4))
    }

    private func useInteger(_ type: GGMLType, cols: Int) -> Bool {
        integerActivations && IntegerForm.supports(type) && cols % 32 == 0
    }

    /// `y[s * yStride + o] = scale * dot(row o, x[s])` for `o` in `range`, `s < batch`.
    @inline(__always)
    private func rowsProduct(_ type: GGMLType, base: UnsafeRawPointer, rowBytes: Int, cols: Int, range: Range<Int>,
                             scale: Float, x: UnsafePointer<Float>, xRow0: Int, batch: Int,
                             y: UnsafeMutablePointer<Float>, yStride: Int, integer: Bool) {
        if integer && batch > 1 && type != .q8_0 {
            // Unpack each row once for the whole batch (Q8_0's codes are already int8).
            let nu = cols / 16
            withUnsafeTemporaryAllocation(of: Int8.self, capacity: cols) { w8b in
                withUnsafeTemporaryAllocation(of: Float.self, capacity: 2 * nu) { scb in
                    let w8 = w8b.baseAddress!, sw = scb.baseAddress!, mw = sw + nu
                    for o in range {
                        Dot.unpackRowQ8(type, row: base + o * rowBytes, cols: cols, w8: w8, sw: sw, mw: mw)
                        for s in 0..<batch {
                            y[s * yStride + o] = scale * Dot.dotQ8Unpacked(type, w8: w8, sw: sw, mw: mw, q8, row: xRow0 + s)
                        }
                    }
                }
            }
        } else if integer {
            for o in range {
                let row = base + o * rowBytes
                for s in 0..<batch {
                    y[s * yStride + o] = scale * Dot.dotQ8(type, row: row, q8, row: xRow0 + s)
                }
            }
        } else if batch == 1 {
            let xs = x + xRow0 * cols
            for o in range { y[o] = scale * Dot.dot(type, row: base + o * rowBytes, xs, count: cols) }
        } else {
            let xs = x + xRow0 * cols
            for o in range {
                Dot.dotBatch(type, row: base + o * rowBytes, xs, count: cols, batch: batch,
                             into: y + o, yStride: yStride)
                if scale != 1 { for s in 0..<batch { y[s * yStride + o] *= scale } }
            }
        }
    }

    // MARK: Ops

    public func embed(table: WeightView, tokens: [Int32], into out: BufferSlice) {
        precondition(out.count >= tokens.count * table.cols, "embed: output too small")
        let o = f32(out)
        for (s, t) in tokens.enumerated() {
            precondition(t >= 0 && Int(t) < table.rows, "embed: token \(t) outside table of \(table.rows)")
            let dst = o + s * table.cols
            Dequant.dequantizeRow(table.type, table.rowPointer(Int(t)), into: dst, count: table.cols)
            if table.scale != 1 { for i in 0..<table.cols { dst[i] *= table.scale } }
        }
    }

    public func matmul(_ w: WeightView, _ x: BufferSlice, into y: BufferSlice, batch: Int) {
        precondition(x.count >= batch * w.cols && y.count >= batch * w.rows, "matmul: slice too small")
        precondition(w.offset >= 0 && w.offset + w.byteCount <= w.region.length, "matmul: weight outside its region")
        let xp = UnsafePointer(f32(x)), yp = f32(y)
        let integer = useInteger(w.type, cols: w.cols)
        if integer { q8.quantize(xp, rows: batch, cols: w.cols) }
        let base = w.rowPointer(0), rb = w.rowBytes, type = w.type, cols = w.cols, scale = w.scale, rows = w.rows
        let ctx = UnsafeSendable((base, xp, yp))
        Parallel.forChunks(rows, threads: threads, minChunk: rowChunk(rows)) { range in
            let (base, xp, yp) = ctx.value
            rowsProduct(type, base: base, rowBytes: rb, cols: cols, range: range, scale: scale,
                        x: xp, xRow0: 0, batch: batch, y: yp, yStride: rows, integer: integer)
        }
    }

    public func matmulExperts(_ w: StackedWeights, indices: BufferSlice, rows: Int, perRow: Int,
                              _ x: BufferSlice, sharedInput: Bool, into y: BufferSlice) {
        let pairs = rows * perRow
        let xRows = sharedInput ? rows : pairs
        precondition(indices.count >= pairs, "matmulExperts: indices too small")
        precondition(x.count >= xRows * w.cols && y.count >= pairs * w.rows, "matmulExperts: slice too small")
        let ids = i32(indices)
        for p in 0..<pairs {
            precondition(ids[p] >= 0 && Int(ids[p]) < w.count, "matmulExperts: index \(ids[p]) outside 0..<\(w.count)")
        }
        precondition(w.baseOffset >= 0 && w.baseOffset + (w.count - 1) * w.stride + w.type.rowBytes(w.cols) * w.rows <= w.region.length,
                     "matmulExperts: stack outside its region")
        let xp = UnsafePointer(f32(x)), yp = f32(y)
        let integer = useInteger(w.type, cols: w.cols)
        if integer { q8.quantize(xp, rows: xRows, cols: w.cols) }
        let chunk = max(16, (w.rows * pairs + threads * 4 - 1) / (threads * 4))
        let cpm = (w.rows + chunk - 1) / chunk
        let rb = w.type.rowBytes(w.cols), type = w.type, cols = w.cols, outRows = w.rows
        let region = w.region.base + w.baseOffset, stride = w.stride, scales = w.scales
        let ctx = UnsafeSendable((region, xp, yp, UnsafePointer(ids)))
        Parallel.forEach(pairs * cpm, threads: threads) { item in
            let (region, xp, yp, ids) = ctx.value
            let p = item / cpm, c = item % cpm
            let e = Int(ids[p])
            let range = c * chunk ..< min(outRows, (c + 1) * chunk)
            let xr = sharedInput ? p / perRow : p
            rowsProduct(type, base: region + e * stride, rowBytes: rb, cols: cols, range: range,
                        scale: scales?[e] ?? 1, x: xp, xRow0: xr, batch: 1,
                        y: yp + p * outRows, yStride: outRows, integer: integer)
        }
    }

    public func rmsNorm(_ x: BufferSlice, weight: WeightView, into out: BufferSlice, rows: Int, dim: Int, eps: Float) {
        precondition(weight.cols * weight.rows == dim, "rmsNorm: weight has \(weight.cols * weight.rows) values, dim \(dim)")
        precondition(x.count >= rows * dim && out.count >= rows * dim, "rmsNorm: slice too small")
        let minChunk = max(1, 32_768 / max(dim, 1))
        let xp = UnsafePointer(f32(x)), op = f32(out)
        func run(_ wp: UnsafePointer<Float>) {
            let ctx = UnsafeSendable((xp, op, wp))
            Parallel.forChunks(rows, threads: threads, minChunk: minChunk) { range in
                let (xp, op, wp) = ctx.value
                for r in range {
                    let xr = xp + r * dim, orow = op + r * dim
                    var ms: Double = 0
                    for i in 0..<dim { ms += Double(xr[i]) * Double(xr[i]) }
                    let rinv: Float = 1 / sqrtf(Float(ms / Double(dim)) + eps)
                    for i in 0..<dim { orow[i] = xr[i] * rinv * wp[i] }
                }
            }
        }
        if weight.type == .f32 {
            run(weight.rowPointer(0).assumingMemoryBound(to: Float.self))
        } else {
            var w = [Float](repeating: 0, count: dim)
            w.withUnsafeMutableBufferPointer {
                Dequant.dequantizeRow(weight.type, weight.rowPointer(0), into: $0.baseAddress!, count: dim)
                run(UnsafePointer($0.baseAddress!))
            }
        }
    }

    public func ropeNeox(_ x: BufferSlice, _ p: RopeParams) {
        let width = p.heads * p.headDim
        precondition(x.count >= p.rows * width, "ropeNeox: slice too small")
        precondition(p.ropeDims <= p.headDim && p.ropeDims % 2 == 0, "ropeNeox: bad ropeDims \(p.ropeDims)")
        let half = p.ropeDims / 2
        guard half > 0 else { return }
        var inv = [Float](repeating: 0, count: half)
        for j in 0..<half { inv[j] = powf(p.theta, -2.0 * Float(j) / Float(p.ropeDims)) }
        let xp = f32(x)
        inv.withUnsafeBufferPointer { invp in
            let ctx = UnsafeSendable((xp, invp.baseAddress!))
            Parallel.forChunks(p.rows, threads: threads, minChunk: 8) { range in
                let (xp, inv) = ctx.value
                withUnsafeTemporaryAllocation(of: Float.self, capacity: 2 * half) { cs in
                    let c = cs.baseAddress!, s = c + half
                    for r in range {
                        let pos = Float(p.positionStart + r)
                        for j in 0..<half {
                            let ang = pos * inv[j]
                            c[j] = cosf(ang)
                            s[j] = sinf(ang)
                        }
                        for h in 0..<p.heads {
                            let v = xp + r * width + h * p.headDim
                            for j in 0..<half {
                                let a = v[j], b = v[j + half]
                                v[j] = a * c[j] - b * s[j]
                                v[j + half] = b * c[j] + a * s[j]
                            }
                        }
                    }
                }
            }
        }
    }

    public func storeKV(k: BufferSlice, v: BufferSlice, cache: KVCache, layer: Int, positionStart: Int, rows: Int) {
        let L = cache.layout
        precondition(layer >= 0 && layer < L.layers, "storeKV: layer \(layer)")
        precondition(positionStart >= 0 && positionStart + rows <= L.maxTokens, "storeKV: positions past maxTokens \(L.maxTokens)")
        precondition(k.count >= rows * L.kvHeads * L.keyDim && v.count >= rows * L.kvHeads * L.valueDim, "storeKV: slice too small")
        let kp = f32(k), vp = f32(v)
        func store(_ src: UnsafeMutablePointer<Float>, _ dst: DeviceBuffer, _ dim: Int) {
            for s in 0..<rows {
                for g in 0..<L.kvHeads {
                    let from = src + (s * L.kvHeads + g) * dim
                    let to = ((layer * L.kvHeads + g) * L.maxTokens + positionStart + s) * dim
                    if L.dtype == .f32 {
                        (dst.contents.assumingMemoryBound(to: Float.self) + to).update(from: from, count: dim)
                    } else {
                        let d16 = dst.contents.assumingMemoryBound(to: Float16.self) + to
                        for i in 0..<dim { d16[i] = Float16(from[i]) }
                    }
                }
            }
        }
        store(kp, cache.keys, L.keyDim)
        store(vp, cache.values, L.valueDim)
    }

    public func attention(q: BufferSlice, cache: KVCache, layer: Int, _ p: AttentionParams, into out: BufferSlice) {
        let L = cache.layout
        precondition(p.kvHeads == L.kvHeads && p.headDim == L.keyDim && p.valueDim == L.valueDim, "attention: params disagree with the cache layout")
        precondition(p.heads % p.kvHeads == 0, "attention: heads \(p.heads) not a multiple of kvHeads \(p.kvHeads)")
        precondition(p.positionStart + p.rows <= L.maxTokens, "attention: positions past maxTokens")
        precondition(q.count >= p.rows * p.heads * p.headDim && out.count >= p.rows * p.heads * p.valueDim, "attention: slice too small")
        let group = p.heads / p.kvHeads
        let half = L.dtype == .f16
        let kBase = cache.keys.contents, vBase = cache.values.contents
        let ctx = UnsafeSendable((UnsafePointer(f32(q)), f32(out), kBase, vBase))
        let hd = p.headDim, vd = p.valueDim
        Parallel.forEach(p.rows * p.heads, threads: p.rows * p.heads * (p.positionStart + p.rows) >= 4096 ? threads : 1) { item in
            let (qp, op, kBase, vBase) = ctx.value
            let s = item / p.heads, h = item % p.heads
            let g = h / group
            let pos = p.positionStart + s
            let qv = qp + (s * p.heads + h) * hd
            let ov = op + (s * p.heads + h) * vd
            let headK = ((layer * L.kvHeads + g) * L.maxTokens) * hd
            let headV = ((layer * L.kvHeads + g) * L.maxTokens) * vd
            withUnsafeTemporaryAllocation(of: Float.self, capacity: pos + 1 + (half ? max(hd, vd) : 0)) { buf in
                let sc = buf.baseAddress!
                let conv = sc + pos + 1
                for t in 0...pos {
                    let dot: Float
                    if half {
                        let k16 = kBase.assumingMemoryBound(to: Float16.self) + headK + t * hd
                        for i in 0..<hd { conv[i] = Float(k16[i]) }
                        dot = Dot.dotF32(qv, conv, hd)
                    } else {
                        dot = Dot.dotF32(qv, kBase.assumingMemoryBound(to: Float.self) + headK + t * hd, hd)
                    }
                    sc[t] = dot * p.scale
                }
                CPUBackend.softmaxRow(sc, pos + 1)
                for i in 0..<vd { ov[i] = 0 }
                for t in 0...pos {
                    let a = sc[t]
                    if half {
                        let v16 = vBase.assumingMemoryBound(to: Float16.self) + headV + t * vd
                        for i in 0..<vd { ov[i] += a * Float(v16[i]) }
                    } else {
                        let vr = vBase.assumingMemoryBound(to: Float.self) + headV + t * vd
                        for i in 0..<vd { ov[i] += a * vr[i] }
                    }
                }
            }
        }
    }

    /// Upstream's `softmax_row`: max from -1e30, `expf(x - max)`, Float32 sum, divide.
    @inline(__always)
    static func softmaxRow(_ x: UnsafeMutablePointer<Float>, _ n: Int) {
        var m: Float = -1e30
        for i in 0..<n where x[i] > m { m = x[i] }
        var s: Float = 0
        for i in 0..<n {
            x[i] = expf(x[i] - m)
            s += x[i]
        }
        for i in 0..<n { x[i] /= s }
    }

    public func softmax(_ x: BufferSlice, rows: Int, cols: Int) {
        precondition(x.count >= rows * cols, "softmax: slice too small")
        let xp = f32(x)
        for r in 0..<rows { CPUBackend.softmaxRow(xp + r * cols, cols) }
    }

    public func topK(_ scores: BufferSlice, rows: Int, cols: Int, k: Int, renormalize: Bool,
                     ids: BufferSlice, weights: BufferSlice) {
        precondition(k >= 1 && k <= cols, "topK: k \(k) outside 1...\(cols)")
        precondition(scores.count >= rows * cols && ids.count >= rows * k && weights.count >= rows * k, "topK: slice too small")
        let sp = f32(scores), ip = i32(ids), wp = f32(weights)
        var taken = [Bool](repeating: false, count: cols)
        for r in 0..<rows {
            let row = sp + r * cols
            for i in 0..<cols { taken[i] = false }
            for kk in 0..<k {
                var best = -1
                var bv: Float = -1e30
                for e in 0..<cols where !taken[e] && row[e] > bv {
                    bv = row[e]
                    best = e
                }
                if best < 0 { best = taken.firstIndex(of: false)! } // all NaN / below -1e30
                taken[best] = true
                ip[r * k + kk] = Int32(best)
                wp[r * k + kk] = row[best]
            }
            if renormalize {
                var sum: Float = 0
                for kk in 0..<k { sum += wp[r * k + kk] }
                for kk in 0..<k { wp[r * k + kk] /= sum }
            }
        }
    }

    public func siluMul(gate: BufferSlice, up: BufferSlice, into out: BufferSlice, count: Int) {
        precondition(gate.count >= count && up.count >= count && out.count >= count, "siluMul: slice too small")
        let ctx = UnsafeSendable((f32(gate), f32(up), f32(out)))
        Parallel.forChunks(count, threads: threads, minChunk: 16_384) { range in
            let (g, u, o) = ctx.value
            for i in range {
                let gv = g[i]
                o[i] = (gv / (1 + expf(-gv))) * u[i]
            }
        }
    }

    public func add(_ a: BufferSlice, _ b: BufferSlice, into out: BufferSlice, count: Int) {
        precondition(a.count >= count && b.count >= count && out.count >= count, "add: slice too small")
        let ap = f32(a), bp = f32(b), op = f32(out)
        for i in 0..<count { op[i] = ap[i] + bp[i] }
    }

    public func weightedSum(_ x: BufferSlice, weights: BufferSlice, rows: Int, terms: Int, dim: Int,
                            into out: BufferSlice, accumulate: Bool) {
        precondition(x.count >= rows * terms * dim && weights.count >= rows * terms && out.count >= rows * dim,
                     "weightedSum: slice too small")
        let xp = f32(x), wp = f32(weights), op = f32(out)
        withUnsafeTemporaryAllocation(of: Float.self, capacity: dim) { accBuf in
            let acc = accBuf.baseAddress!
            for r in 0..<rows {
                for i in 0..<dim { acc[i] = 0 }
                for k in 0..<terms {
                    let w = wp[r * terms + k]
                    let xr = xp + (r * terms + k) * dim
                    for i in 0..<dim { acc[i] += w * xr[i] }
                }
                let o = op + r * dim
                if accumulate {
                    for i in 0..<dim { o[i] = o[i] + acc[i] }
                } else {
                    for i in 0..<dim { o[i] = acc[i] }
                }
            }
        }
    }

    public func copy(_ src: BufferSlice, into dst: BufferSlice, count: Int) {
        precondition(src.buffer.dtype == dst.buffer.dtype && count <= src.count && count <= dst.count)
        memmove(dst.buffer.contents + dst.byteOffset, src.buffer.contents + src.byteOffset, count * src.buffer.dtype.size)
    }
}

/// `HostSignal` for the CPU backend: a counter behind a condition variable.
public final class BlockingHostSignal: HostSignal, @unchecked Sendable {
    private let cond = NSCondition()
    private var current: UInt64 = 0

    public init() {}

    public var value: UInt64 {
        cond.lock(); defer { cond.unlock() }
        return current
    }

    public func signal(_ value: UInt64) {
        cond.lock()
        if value > current { current = value }
        cond.broadcast()
        cond.unlock()
    }

    public func wait(atLeast value: UInt64) {
        cond.lock()
        while current < value { cond.wait() }
        cond.unlock()
    }
}
