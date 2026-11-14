// Dot products of one quantised weight row with a Float32 activation vector: the inner
// loop of the CPU backend's matmul. This file is the "float path" (dequantise in
// registers or a block-sized scratch, f32 multiply-add), which matches upstream run with
// GQ_EXACT=1 (docs/port/gguf-and-quants.md section 4.1) up to summation order. The
// int8-activation path (upstream's default, gq_dot_i8) is in IntegerDot.swift.
//
// The dequantised weights are bit-exact (BlockDecoder); the dot itself uses fused
// multiply-add and SIMD lanes, so it agrees with a scalar reference to rounding only.
// The summation order is fixed, so results are reproducible run to run.

public enum Dot {
    /// `sum_i dequant(row)[i] * x[i]` over `count` elements (a multiple of the block size).
    public static func dot(_ type: GGMLType, row: UnsafeRawPointer, _ x: UnsafePointer<Float>, count: Int) -> Float {
        let nb = count / type.blockSize
        switch type {
        case .f32: return dotF32(row.assumingMemoryBound(to: Float.self), x, count)
        case .f16: return dotHalf(row, x, count, bf16: false)
        case .bf16: return dotHalf(row, x, count, bf16: true)
        case .q8_0: return dotQ8_0(row, x, nb)
        case .q4_0: return dotVector(row, x, nb, 18, 32, VectorDecoder.q4_0)
        case .q4_1: return dotVector(row, x, nb, 20, 32, VectorDecoder.q4_1)
        case .q5_0: return dotVector(row, x, nb, 22, 32, VectorDecoder.q5_0)
        case .q5_1: return dotVector(row, x, nb, 24, 32, VectorDecoder.q5_1)
        case .iq4_nl: return dotBlocks(row, x, nb, 18, 32, BlockDecoder.iq4_nl)
        case .q2_K: return dotVector(row, x, nb, 84, 256, VectorDecoder.q2_K)
        case .q3_K: return dotVector(row, x, nb, 110, 256, VectorDecoder.q3_K)
        case .q4_K: return dotVector(row, x, nb, 144, 256, VectorDecoder.q4_K)
        case .q5_K: return dotVector(row, x, nb, 176, 256, VectorDecoder.q5_K)
        case .q6_K: return dotVector(row, x, nb, 210, 256, VectorDecoder.q6_K)
        case .iq4_xs: return dotBlocks(row, x, nb, 136, 256, BlockDecoder.iq4_xs)
        default: preconditionFailure("Dot: no kernel for \(type)")
        }
    }

    /// Dots one weight row with `batch` activation rows (`x` is `[batch][count]`), writing
    /// `y[s * yStride]`. Prefill uses this so each weight block is decoded once per batch.
    public static func dotBatch(_ type: GGMLType, row: UnsafeRawPointer, _ x: UnsafePointer<Float>, count: Int,
                                batch: Int, into y: UnsafeMutablePointer<Float>, yStride: Int) {
        if batch == 1 {
            y[0] = dot(type, row: row, x, count: count)
            return
        }
        if type == .f32 {
            let w = row.assumingMemoryBound(to: Float.self)
            for s in 0..<batch { y[s * yStride] = dotF32(w, x + s * count, count) }
            return
        }
        withUnsafeTemporaryAllocation(of: Float.self, capacity: count) { tmp in
            let w = tmp.baseAddress!
            Dequant.dequantizeRow(type, row, into: w, count: count)
            for s in 0..<batch { y[s * yStride] = dotF32(UnsafePointer(w), x + s * count, count) }
        }
    }

    // MARK: Kernels

    /// Plain f32 dot, four 4-lane accumulators per SIMD16 pair, fixed order.
    @inline(__always)
    public static func dotF32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
        var s0 = SIMD16<Float>(), s1 = SIMD16<Float>()
        let ar = UnsafeRawPointer(a), br = UnsafeRawPointer(b)
        var i = 0
        while i + 32 <= n {
            s0 = s0.addingProduct(ar.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self),
                                  br.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self))
            s1 = s1.addingProduct(ar.loadUnaligned(fromByteOffset: i * 4 + 64, as: SIMD16<Float>.self),
                                  br.loadUnaligned(fromByteOffset: i * 4 + 64, as: SIMD16<Float>.self))
            i += 32
        }
        while i + 16 <= n {
            s0 = s0.addingProduct(ar.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self),
                                  br.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self))
            i += 16
        }
        var r = (s0 + s1).sum()
        while i < n {
            r += a[i] * b[i]
            i += 1
        }
        return r
    }

    /// Q8_0, fused: int8 codes converted in registers, f32 FMA, the block scale applied
    /// once per block.
    @inline(__always)
    static func dotQ8_0(_ row: UnsafeRawPointer, _ x: UnsafePointer<Float>, _ nb: Int) -> Float {
        var acc0 = SIMD8<Float>(), acc1 = SIMD8<Float>()
        for b in 0..<nb {
            let blk = row + b * 34
            let d = rdF16(blk, 0)
            let w = (blk + 2).assumingMemoryBound(to: Int8.self)
            let xb = UnsafeRawPointer(x + b * 32)
            var w0 = SIMD8<Float>(), w1 = SIMD8<Float>(), w2 = SIMD8<Float>(), w3 = SIMD8<Float>()
            for j in 0..<8 {
                w0[j] = Float(w[j]); w1[j] = Float(w[8 + j]); w2[j] = Float(w[16 + j]); w3[j] = Float(w[24 + j])
            }
            var s0 = w0 * xb.loadUnaligned(as: SIMD8<Float>.self)
            var s1 = w1 * xb.loadUnaligned(fromByteOffset: 32, as: SIMD8<Float>.self)
            s0 = s0.addingProduct(w2, xb.loadUnaligned(fromByteOffset: 64, as: SIMD8<Float>.self))
            s1 = s1.addingProduct(w3, xb.loadUnaligned(fromByteOffset: 96, as: SIMD8<Float>.self))
            acc0 = acc0.addingProduct(s0, d)
            acc1 = acc1.addingProduct(s1, d)
        }
        return (acc0 + acc1).sum()
    }

    /// Types with a vector decoder: dequantise 16 values in registers, multiply-add.
    @inline(__always)
    static func dotVector(_ row: UnsafeRawPointer, _ x: UnsafePointer<Float>, _ nb: Int, _ bs: Int, _ bl: Int,
                          _ decode: (UnsafeRawPointer, (Int, SIMD16<Float>) -> Void) -> Void) -> Float {
        var a0 = SIMD16<Float>(), a1 = SIMD16<Float>()
        let xr = UnsafeRawPointer(x)
        for b in 0..<nb {
            let xb = xr + b * bl * 4
            decode(row + b * bs) { c, v in
                let xv = xb.loadUnaligned(fromByteOffset: c * 64, as: SIMD16<Float>.self)
                if c & 1 == 0 { a0 = a0.addingProduct(v, xv) } else { a1 = a1.addingProduct(v, xv) }
            }
        }
        return (a0 + a1).sum()
    }

    /// Any block type: decode one block into a stack scratch (bit-exact), then FMA it
    /// against the matching slice of `x`.
    @inline(__always)
    static func dotBlocks(_ row: UnsafeRawPointer, _ x: UnsafePointer<Float>, _ nb: Int, _ bs: Int, _ bl: Int,
                          _ decode: (UnsafeRawPointer, UnsafeMutablePointer<Float>) -> Void) -> Float {
        withUnsafeTemporaryAllocation(of: SIMD16<Float>.self, capacity: bl / 16) { tmpBuf in
            let tmp = UnsafeMutableRawPointer(tmpBuf.baseAddress!)
            let tf = tmp.assumingMemoryBound(to: Float.self)
            var s0 = SIMD16<Float>(), s1 = SIMD16<Float>()
            for b in 0..<nb {
                decode(row + b * bs, tf)
                let xb = UnsafeRawPointer(x + b * bl)
                var i = 0
                while i < bl {
                    s0 = s0.addingProduct(tmp.load(fromByteOffset: i * 4, as: SIMD16<Float>.self),
                                          xb.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self))
                    s1 = s1.addingProduct(tmp.load(fromByteOffset: i * 4 + 64, as: SIMD16<Float>.self),
                                          xb.loadUnaligned(fromByteOffset: i * 4 + 64, as: SIMD16<Float>.self))
                    i += 32
                }
            }
            return (s0 + s1).sum()
        }
    }

    /// F16 / BF16 rows: convert 256 at a time into a scratch, then dot.
    @inline(__always)
    static func dotHalf(_ row: UnsafeRawPointer, _ x: UnsafePointer<Float>, _ n: Int, bf16: Bool) -> Float {
        withUnsafeTemporaryAllocation(of: Float.self, capacity: 256) { tmpBuf in
            let t = tmpBuf.baseAddress!
            var r: Float = 0
            var i = 0
            while i < n {
                let m = min(256, n - i)
                if bf16 {
                    for j in 0..<m { t[j] = HalfFloat.bf16(rd16(row, 2 * (i + j))) }
                } else {
                    for j in 0..<m { t[j] = HalfFloat.f16(rd16(row, 2 * (i + j))) }
                }
                r += dotF32(UnsafePointer(t), x + i, m)
                i += m
            }
            return r
        }
    }
}
