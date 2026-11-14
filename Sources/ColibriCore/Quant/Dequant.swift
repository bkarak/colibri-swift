// Block decoders. Ported from colibri c/ggml_quants.h (gq_blk_*, gq_dequant_row),
// formulas in docs/port/gguf-and-quants.md sections 3.3-3.5.
//
// Exactness contract: the arithmetic is Float32, in the order the doc writes it, with no
// fused multiply-add, so the output is bit-identical to upstream's (and gguf-py's)
// dequantisation. Do not use `addingProduct` / `fma` here.

/// Scalar conversions shared by every decoder.
public enum HalfFloat {
    /// IEEE binary16 bits to Float32 (exact).
    @inlinable public static func f16(_ bits: UInt16) -> Float { Float(Float16(bitPattern: bits)) }
    /// bfloat16 bits to Float32 (exact).
    @inlinable public static func bf16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
}

/// `GGML_KVALUES_IQ4NL` (ggml_grids.h): the IQ4_NL / IQ4_XS codebook.
public let iq4nlValues: [Int8] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

@usableFromInline let iq4nlTuple: (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8,
                                   Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8) =
    (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

@inline(__always) func iq4nl(_ i: Int) -> Int8 {
    withUnsafeBytes(of: iq4nlTuple) { Int8(bitPattern: $0[i]) }
}

@inline(__always) func rd8(_ p: UnsafeRawPointer, _ o: Int) -> UInt8 { p.load(fromByteOffset: o, as: UInt8.self) }
@inline(__always) func rd16(_ p: UnsafeRawPointer, _ o: Int) -> UInt16 { p.loadUnaligned(fromByteOffset: o, as: UInt16.self) }
@inline(__always) func rd32(_ p: UnsafeRawPointer, _ o: Int) -> UInt32 { p.loadUnaligned(fromByteOffset: o, as: UInt32.self) }
@inline(__always) func rdF16(_ p: UnsafeRawPointer, _ o: Int) -> Float { HalfFloat.f16(rd16(p, o)) }

/// `gq_scale_min_k4`: the 6-bit scale and min of sub-block `g` of a Q4_K/Q5_K block.
@inline(__always) func scaleMinK4(_ s: UnsafeRawPointer, _ g: Int) -> (Int, Int) {
    if g < 4 {
        return (Int(rd8(s, g) & 0x3F), Int(rd8(s, g + 4) & 0x3F))
    }
    let sc = Int(rd8(s, g + 4) & 0x0F) | Int((rd8(s, g - 4) >> 2) & 0x30)
    let mn = Int(rd8(s, g + 4) >> 4) | Int((rd8(s, g) >> 2) & 0x30)
    return (sc, mn)
}

// MARK: Vector decoders
//
// The common types decoded 16 values at a time in NEON registers. The arithmetic per
// element is exactly the scalar decoder's (same operations, same order, no FMA), so the
// values are bit-identical; the golden tests check both. Swift's SIMD conversion
// initialisers are generic and slow, but an element-wise loop into a SIMD value is
// recognised by LLVM and becomes a vector widen + `scvtf`.

@inline(__always) func cvtU(_ v: SIMD16<UInt8>) -> SIMD16<Float> {
    var f = SIMD16<Float>()
    for j in 0..<16 { f[j] = Float(v[j]) }
    return f
}

@inline(__always) func cvtS(_ v: SIMD16<Int8>) -> SIMD16<Float> {
    var f = SIMD16<Float>()
    for j in 0..<16 { f[j] = Float(v[j]) }
    return f
}

@inline(__always) func ld16(_ p: UnsafeRawPointer, _ o: Int) -> SIMD16<UInt8> {
    p.loadUnaligned(fromByteOffset: o, as: SIMD16<UInt8>.self)
}

/// Decoders that hand each run of 16 values to `emit(chunk, values)`, where element
/// `16 * chunk` of the block is the first of the run. Used by `Dequant` (emit = store)
/// and `Dot` (emit = multiply-add with the activations).
enum VectorDecoder {
    /// Types with a vector decoder.
    static func supports(_ t: GGMLType) -> Bool {
        switch t {
        case .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .q2_K, .q3_K, .q4_K, .q5_K, .q6_K: return true
        default: return false
        }
    }

    /// Lane j = bit `j + shift` of `qh` (the fifth bit of Q5_0 / Q5_1 element j + shift).
    @inline(__always) static func qhBits(_ qh: UInt32, _ shift: Int) -> SIMD16<UInt8> {
        let b0 = UInt8(truncatingIfNeeded: qh >> UInt32(shift)), b1 = UInt8(truncatingIfNeeded: qh >> UInt32(shift + 8))
        let rep = SIMD16<UInt8>(b0, b0, b0, b0, b0, b0, b0, b0, b1, b1, b1, b1, b1, b1, b1, b1)
        return (rep &>> SIMD16<UInt8>(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7)) & 1
    }

    @inline(__always) static func q5_0(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0)
        let qh = rd32(b, 2)
        let q = ld16(b, 6)
        let lo = (q & 0x0F) | (qhBits(qh, 0) &<< 4), hi = (q &>> 4) | (qhBits(qh, 16) &<< 4)
        let sixteen = SIMD16<Int8>(repeating: 16)
        emit(0, d * cvtS(unsafeBitCast(lo, to: SIMD16<Int8>.self) &- sixteen))
        emit(1, d * cvtS(unsafeBitCast(hi, to: SIMD16<Int8>.self) &- sixteen))
    }

    @inline(__always) static func q5_1(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0), m = rdF16(b, 2)
        let qh = rd32(b, 4)
        let q = ld16(b, 8)
        emit(0, d * cvtU((q & 0x0F) | (qhBits(qh, 0) &<< 4)) + m)
        emit(1, d * cvtU((q &>> 4) | (qhBits(qh, 16) &<< 4)) + m)
    }

    @inline(__always) static func q2_K(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 80), dmin = rdF16(b, 82)
        for g in 0..<16 {
            let sc = rd8(b, g)
            let dl = d * Float(sc & 0x0F), ml = dmin * Float(sc >> 4)
            let i0 = g * 16
            let c = i0 >> 7, s = (i0 >> 5) & 3, j0 = i0 & 31
            let q = (ld16(b, 16 + c * 32 + j0) &>> SIMD16<UInt8>(repeating: UInt8(2 * s))) & 3
            emit(g, dl * cvtU(q) - ml)
        }
    }

    @inline(__always) static func q3_K(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 108)
        let s = b + 96
        for k in 0..<16 {
            let l = k < 8 ? Int(rd8(s, k) & 0x0F) : Int(rd8(s, k - 8) >> 4)
            let h = Int((rd8(s, 8 + (k & 3)) >> UInt8(2 * (k >> 2))) & 3)
            let dl = d * Float((l | (h << 4)) - 32)
            let i0 = k * 16
            let c = i0 >> 7, sh = (i0 >> 5) & 3, j0 = i0 & 31
            let ql = (ld16(b, 32 + c * 32 + j0) &>> SIMD16<UInt8>(repeating: UInt8(2 * sh))) & 3
            let hb = (ld16(b, j0) &>> SIMD16<UInt8>(repeating: UInt8(i0 >> 5))) & 1
            // ql - (hb == 0 ? 4 : 0) == ql + 4 * hb - 4, exact in Int8.
            let q = unsafeBitCast(ql | (hb &<< 2), to: SIMD16<Int8>.self) &- SIMD16<Int8>(repeating: 4)
            emit(k, dl * cvtS(q))
        }
    }

    @inline(__always) static func q8_0(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0)
        emit(0, cvtS(unsafeBitCast(ld16(b, 2), to: SIMD16<Int8>.self)) * d)
        emit(1, cvtS(unsafeBitCast(ld16(b, 18), to: SIMD16<Int8>.self)) * d)
    }

    @inline(__always) static func q4_0(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0)
        let q = ld16(b, 2)
        let eight = SIMD16<Int8>(repeating: 8)
        emit(0, d * cvtS(unsafeBitCast(q & 0x0F, to: SIMD16<Int8>.self) &- eight))
        emit(1, d * cvtS(unsafeBitCast(q &>> 4, to: SIMD16<Int8>.self) &- eight))
    }

    @inline(__always) static func q4_1(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0), m = rdF16(b, 2)
        let q = ld16(b, 4)
        emit(0, d * cvtU(q & 0x0F) + m)
        emit(1, d * cvtU(q &>> 4) + m)
    }

    @inline(__always) static func q4_K(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0), dmin = rdF16(b, 2)
        for p in 0..<4 {
            let (sc0, mn0) = scaleMinK4(b + 4, 2 * p)
            let (sc1, mn1) = scaleMinK4(b + 4, 2 * p + 1)
            let d0 = d * Float(sc0), m0 = dmin * Float(mn0)
            let d1 = d * Float(sc1), m1 = dmin * Float(mn1)
            for h in 0..<2 {
                let q = ld16(b, 16 + p * 32 + h * 16)
                emit(4 * p + h, d0 * cvtU(q & 0x0F) - m0)
                emit(4 * p + 2 + h, d1 * cvtU(q &>> 4) - m1)
            }
        }
    }

    @inline(__always) static func q5_K(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 0), dmin = rdF16(b, 2)
        let qh0 = ld16(b, 16), qh1 = ld16(b, 32)
        for p in 0..<4 {
            let g0 = 2 * p, g1 = 2 * p + 1
            let (sc0, mn0) = scaleMinK4(b + 4, g0)
            let (sc1, mn1) = scaleMinK4(b + 4, g1)
            let d0 = d * Float(sc0), m0 = dmin * Float(mn0)
            let d1 = d * Float(sc1), m1 = dmin * Float(mn1)
            let s0 = SIMD16<UInt8>(repeating: UInt8(g0)), s1 = SIMD16<UInt8>(repeating: UInt8(g1))
            for h in 0..<2 {
                let q = ld16(b, 48 + p * 32 + h * 16)
                let qh = h == 0 ? qh0 : qh1
                let v0 = (q & 0x0F) | (((qh &>> s0) & 1) &<< 4)
                let v1 = (q &>> 4) | (((qh &>> s1) & 1) &<< 4)
                emit(4 * p + h, d0 * cvtU(v0) - m0)
                emit(4 * p + 2 + h, d1 * cvtU(v1) - m1)
            }
        }
    }

    @inline(__always) static func q6_K(_ b: UnsafeRawPointer, _ emit: (Int, SIMD16<Float>) -> Void) {
        let d = rdF16(b, 208)
        let off = SIMD16<Int8>(repeating: 32)
        for g in 0..<16 {
            let dl = d * Float(Int8(bitPattern: rd8(b, 192 + g)))
            let i0 = g * 16
            let c = i0 >> 7, r0 = i0 & 127
            let lo = (ld16(b, c * 64 + (r0 & 63)) &>> SIMD16<UInt8>(repeating: UInt8(4 * (r0 >> 6)))) & 0x0F
            let hi = (ld16(b, 128 + c * 32 + (r0 & 31)) &>> SIMD16<UInt8>(repeating: UInt8(2 * (r0 >> 5)))) & 3
            let q = unsafeBitCast(lo | (hi &<< 4), to: SIMD16<Int8>.self) &- off
            emit(g, dl * cvtS(q))
        }
    }
}

/// One block of each type, decoded to Float32. `b` points at the block, `y` receives
/// `type.blockSize` values.
enum BlockDecoder {
    @inline(__always) static func q8_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0)
        for j in 0..<32 { y[j] = Float(Int8(bitPattern: rd8(b, 2 + j))) * d }
    }

    @inline(__always) static func q4_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0)
        for j in 0..<16 {
            let q = rd8(b, 2 + j)
            y[j] = d * Float(Int(q & 0x0F) - 8)
            y[j + 16] = d * Float(Int(q >> 4) - 8)
        }
    }

    @inline(__always) static func q4_1(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0), m = rdF16(b, 2)
        for j in 0..<16 {
            let q = rd8(b, 4 + j)
            y[j] = d * Float(q & 0x0F) + m
            y[j + 16] = d * Float(q >> 4) + m
        }
    }

    @inline(__always) static func q5_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0)
        let qh = rd32(b, 2)
        for j in 0..<16 {
            let q = rd8(b, 6 + j)
            let lo = Int(q & 0x0F) | Int(((qh >> UInt32(j)) & 1) << 4)
            let hi = Int(q >> 4) | Int(((qh >> UInt32(j + 16)) & 1) << 4)
            y[j] = d * Float(lo - 16)
            y[j + 16] = d * Float(hi - 16)
        }
    }

    @inline(__always) static func q5_1(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0), m = rdF16(b, 2)
        let qh = rd32(b, 4)
        for j in 0..<16 {
            let q = rd8(b, 8 + j)
            let lo = Int(q & 0x0F) | Int(((qh >> UInt32(j)) & 1) << 4)
            let hi = Int(q >> 4) | Int(((qh >> UInt32(j + 16)) & 1) << 4)
            y[j] = d * Float(lo) + m
            y[j + 16] = d * Float(hi) + m
        }
    }

    @inline(__always) static func iq4_nl(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0)
        for j in 0..<16 {
            let q = rd8(b, 2 + j)
            y[j] = d * Float(iq4nl(Int(q & 0x0F)))
            y[j + 16] = d * Float(iq4nl(Int(q >> 4)))
        }
    }

    @inline(__always) static func q2_K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 80), dmin = rdF16(b, 82)
        for g in 0..<16 {
            let sc = rd8(b, g)
            let dl = d * Float(sc & 0x0F)
            let ml = dmin * Float(sc >> 4)
            // Elements g*16 ..< g*16+16 share c, s and a 16-byte run of qs.
            let i0 = g * 16
            let c = i0 >> 7, s = (i0 >> 5) & 3, j0 = i0 & 31
            for t in 0..<16 {
                let q = (rd8(b, 16 + c * 32 + j0 + t) >> UInt8(2 * s)) & 3
                y[i0 + t] = dl * Float(q) - ml
            }
        }
    }

    @inline(__always) static func q3_K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 108)
        let s = b + 96
        for k in 0..<16 {
            let l = k < 8 ? Int(rd8(s, k) & 0x0F) : Int(rd8(s, k - 8) >> 4)
            let h = Int((rd8(s, 8 + (k & 3)) >> UInt8(2 * (k >> 2))) & 3)
            let dl = d * Float((l | (h << 4)) - 32)
            let i0 = k * 16
            let c = i0 >> 7, sh = (i0 >> 5) & 3, j0 = i0 & 31
            for t in 0..<16 {
                let i = i0 + t
                let ql = Int((rd8(b, 32 + c * 32 + j0 + t) >> UInt8(2 * sh)) & 3)
                let hb = Int((rd8(b, (j0 + t)) >> UInt8(i >> 5)) & 1)
                let q = ql - (hb == 0 ? 4 : 0)
                y[i] = dl * Float(q)
            }
        }
    }

    @inline(__always) static func q4_K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0), dmin = rdF16(b, 2)
        for g in 0..<8 {
            let (sc, mn) = scaleMinK4(b + 4, g)
            let d1 = d * Float(sc), m1 = dmin * Float(mn)
            let q = 16 + (g >> 1) * 32
            let shift = UInt8((g & 1) * 4)
            for j in 0..<32 {
                y[g * 32 + j] = d1 * Float((rd8(b, q + j) >> shift) & 0x0F) - m1
            }
        }
    }

    @inline(__always) static func q5_K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0), dmin = rdF16(b, 2)
        for g in 0..<8 {
            let (sc, mn) = scaleMinK4(b + 4, g)
            let d1 = d * Float(sc), m1 = dmin * Float(mn)
            let q = 48 + (g >> 1) * 32
            let shift = UInt8((g & 1) * 4)
            for j in 0..<32 {
                let v = ((rd8(b, q + j) >> shift) & 0x0F) | (((rd8(b, 16 + j) >> UInt8(g)) & 1) << 4)
                y[g * 32 + j] = d1 * Float(v) - m1
            }
        }
    }

    @inline(__always) static func q6_K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 208)
        for g in 0..<16 {
            let dl = d * Float(Int8(bitPattern: rd8(b, 192 + g)))
            let i0 = g * 16
            let c = i0 >> 7, r0 = i0 & 127
            let nib = UInt8(4 * (r0 >> 6)), sh = UInt8(2 * (r0 >> 5))
            for t in 0..<16 {
                let r = r0 + t
                let lo = Int((rd8(b, c * 64 + (r & 63)) >> nib) & 0x0F)
                let hi = Int((rd8(b, 128 + c * 32 + (r & 31)) >> sh) & 3)
                y[i0 + t] = dl * Float((lo | (hi << 4)) - 32)
            }
        }
    }

    @inline(__always) static func iq4_xs(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = rdF16(b, 0)
        let sh = rd16(b, 2)
        for ib in 0..<8 {
            let ls = Int((rd8(b, 4 + (ib >> 1)) >> UInt8(4 * (ib & 1))) & 0x0F) | Int(((sh >> UInt16(2 * ib)) & 3) << 4)
            let dl = d * Float(ls - 32)
            let q = 8 + 16 * ib
            for j in 0..<16 {
                let v = rd8(b, q + j)
                y[ib * 32 + j] = dl * Float(iq4nl(Int(v & 0x0F)))
                y[ib * 32 + j + 16] = dl * Float(iq4nl(Int(v >> 4)))
            }
        }
    }
}

/// Row dequantisation: the CPU reference every other kernel (CPU dot, Metal) is tested
/// against.
public enum Dequant {
    /// True when `dequantizeRow` supports `type`.
    public static func supports(_ type: GGMLType) -> Bool {
        GGMLType.firstPortWeightTypes.contains(type)
    }

    /// Decodes `count` elements (a multiple of `type.blockSize`) starting at `src` into
    /// `dst`. `src` has no alignment guarantee (Q8_0 blocks are 34 bytes).
    public static func dequantizeRow(_ type: GGMLType, _ src: UnsafeRawPointer,
                                     into dst: UnsafeMutablePointer<Float>, count: Int) {
        precondition(count % type.blockSize == 0, "\(type): \(count) is not a multiple of the block size")
        let nb = count / type.blockSize
        let bs = type.typeSize, bl = type.blockSize
        switch type {
        case .f32:
            UnsafeMutableRawPointer(dst).copyMemory(from: src, byteCount: count * 4)
        case .f16:
            for i in 0..<count { dst[i] = HalfFloat.f16(rd16(src, 2 * i)) }
        case .bf16:
            for i in 0..<count { dst[i] = HalfFloat.bf16(rd16(src, 2 * i)) }
        case .q8_0: storeRow(src, dst, nb, bs, bl, VectorDecoder.q8_0)
        case .q4_0: storeRow(src, dst, nb, bs, bl, VectorDecoder.q4_0)
        case .q4_1: storeRow(src, dst, nb, bs, bl, VectorDecoder.q4_1)
        case .q5_0: storeRow(src, dst, nb, bs, bl, VectorDecoder.q5_0)
        case .q5_1: storeRow(src, dst, nb, bs, bl, VectorDecoder.q5_1)
        case .iq4_nl: for i in 0..<nb { BlockDecoder.iq4_nl(src + i * bs, dst + i * bl) }
        case .q2_K: storeRow(src, dst, nb, bs, bl, VectorDecoder.q2_K)
        case .q3_K: storeRow(src, dst, nb, bs, bl, VectorDecoder.q3_K)
        case .q4_K: storeRow(src, dst, nb, bs, bl, VectorDecoder.q4_K)
        case .q5_K: storeRow(src, dst, nb, bs, bl, VectorDecoder.q5_K)
        case .q6_K: storeRow(src, dst, nb, bs, bl, VectorDecoder.q6_K)
        case .iq4_xs: for i in 0..<nb { BlockDecoder.iq4_xs(src + i * bs, dst + i * bl) }
        default:
            preconditionFailure("Dequant: no decoder for \(type)")
        }
    }

    /// The scalar reference decoders (`BlockDecoder`), for every type. Tests compare the
    /// vector decoders used by `dequantizeRow` against these and against upstream.
    static func dequantizeRowScalar(_ type: GGMLType, _ src: UnsafeRawPointer,
                                    into dst: UnsafeMutablePointer<Float>, count: Int) {
        let nb = count / type.blockSize, bs = type.typeSize, bl = type.blockSize
        let f: (UnsafeRawPointer, UnsafeMutablePointer<Float>) -> Void
        switch type {
        case .q8_0: f = BlockDecoder.q8_0
        case .q4_0: f = BlockDecoder.q4_0
        case .q4_1: f = BlockDecoder.q4_1
        case .q5_0: f = BlockDecoder.q5_0
        case .q5_1: f = BlockDecoder.q5_1
        case .q2_K: f = BlockDecoder.q2_K
        case .q3_K: f = BlockDecoder.q3_K
        case .q4_K: f = BlockDecoder.q4_K
        case .q5_K: f = BlockDecoder.q5_K
        case .q6_K: f = BlockDecoder.q6_K
        default:
            dequantizeRow(type, src, into: dst, count: count)
            return
        }
        for i in 0..<nb { f(src + i * bs, dst + i * bl) }
    }

    @inline(__always)
    static func storeRow(_ src: UnsafeRawPointer, _ dst: UnsafeMutablePointer<Float>, _ nb: Int, _ bs: Int, _ bl: Int,
                         _ decode: (UnsafeRawPointer, (Int, SIMD16<Float>) -> Void) -> Void) {
        let y = UnsafeMutableRawPointer(dst)
        for i in 0..<nb {
            let yb = y + i * bl * 4
            decode(src + i * bs) { c, v in yb.storeBytes(of: v, toByteOffset: c * 64, as: SIMD16<Float>.self) }
        }
    }

    /// Convenience for tests: dequantises a whole byte array.
    public static func dequantize(_ type: GGMLType, bytes: [UInt8], count: Int) -> [Float] {
        precondition(bytes.count >= type.rowBytes(count), "dequantize: \(bytes.count) bytes for \(count) \(type) values")
        var out = [Float](repeating: 0, count: count)
        bytes.withUnsafeBytes { raw in
            out.withUnsafeMutableBufferPointer { dequantizeRow(type, raw.baseAddress!, into: $0.baseAddress!, count: count) }
        }
        return out
    }
}
