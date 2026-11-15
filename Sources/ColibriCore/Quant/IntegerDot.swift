import CColibriKernels
import Foundation
// The integer path: activations quantised to int8 per 32 values, weights unpacked to
// int8 codes plus a scale and a min per 16, exact integer inner sums. Ported from
// colibri c/ggml_quants.h (gq_quant_act_ref, gq_ublk, gq_dot_i8, gq_dot_q8_0);
// docs/port/gguf-and-quants.md section 4.2. This is upstream's default CPU path; the
// float path (Dot.swift, = upstream GQ_EXACT=1) is the reference. The only
// approximation against it is the activation's rounding to int8.
//
// Q8_0's 32-wide inner loop is a plain scalar loop, which LLVM turns into NEON `sdot`
// (checked in the assembly). The 16-wide loops of the other types are not vectorised
// (LLVM unrolls them to scalar multiply-adds first), so they call `colibri_dot_i8_16x2`
// (CColibriKernels, one `sdot` per 16 values).

/// Activation rows quantised the way `gq_quant_act_ref` does it: per block of 32,
/// `dx = amax / 127`, `xq = clamp(lrint(v * 127 / amax), -127, 127)`, and the code sums
/// of each half (`xs`, the min term of the K types). Reused across calls.
public final class Q8Activations {
    public private(set) var rows = 0
    public private(set) var cols = 0
    public private(set) var xq: UnsafeMutablePointer<Int8>
    public private(set) var dx: UnsafeMutablePointer<Float>
    public private(set) var xs: UnsafeMutablePointer<Int32>
    private var capacity = 0

    public init() {
        xq = .allocate(capacity: 1)
        dx = .allocate(capacity: 1)
        xs = .allocate(capacity: 1)
    }

    deinit {
        xq.deallocate()
        dx.deallocate()
        xs.deallocate()
    }

    /// Quantises `rows` rows of `cols` (a multiple of 32) values.
    public func quantize(_ x: UnsafePointer<Float>, rows: Int, cols: Int) {
        precondition(cols % 32 == 0, "Q8Activations: \(cols) is not a multiple of 32")
        let n = rows * cols
        if n > capacity {
            xq.deallocate(); dx.deallocate(); xs.deallocate()
            xq = .allocate(capacity: n)
            dx = .allocate(capacity: n / 32)
            xs = .allocate(capacity: n / 16)
            capacity = n
        }
        self.rows = rows
        self.cols = cols
        for b in 0..<(n / 32) {
            Q8Activations.quantizeBlock(x + b * 32, xq + b * 32, dx + b, xs + 2 * b)
        }
    }

    @inline(__always)
    static func quantizeBlock(_ v: UnsafePointer<Float>, _ q: UnsafeMutablePointer<Int8>,
                              _ d: UnsafeMutablePointer<Float>, _ s16: UnsafeMutablePointer<Int32>) {
        var amax: Float = 0
        for i in 0..<32 { amax = max(amax, abs(v[i])) }
        d.pointee = amax / 127
        let id: Float = amax > 0 ? 127 / amax : 0
        var s0: Int32 = 0, s1: Int32 = 0
        for i in 0..<32 {
            let r = (v[i] * id).rounded(.toNearestOrEven)
            let c = Int32(min(max(r, -127), 127))
            q[i] = Int8(truncatingIfNeeded: c)
            if i < 16 { s0 &+= c } else { s1 &+= c }
        }
        s16[0] = s0
        s16[1] = s1
    }

    /// Row `r`'s codes, scales and sums.
    @inline(__always) func row(_ r: Int) -> (UnsafePointer<Int8>, UnsafePointer<Float>, UnsafePointer<Int32>) {
        (UnsafePointer(xq + r * cols), UnsafePointer(dx + r * cols / 32), UnsafePointer(xs + r * cols / 16))
    }
}

/// `gq_ublk`: every block type as int8 codes `w8` and per-16 scale `sw` and min `mw`, such
/// that `sw[g] * Float(w8[i]) - mw[g]` is bit for bit what `Dequant` gives.
enum IntegerForm {
    static func hasMin(_ t: GGMLType) -> Bool {
        t == .q4_1 || t == .q5_1 || t == .q2_K || t == .q4_K || t == .q5_K
    }

    /// Types the integer path covers (every block type the port decodes).
    static func supports(_ t: GGMLType) -> Bool {
        switch t {
        case .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl, .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_xs: return true
        default: return false
        }
    }

    @inline(__always)
    static func unpack(_ t: GGMLType, _ b: UnsafeRawPointer, _ w: UnsafeMutablePointer<Int8>,
                       _ sw: UnsafeMutablePointer<Float>, _ mw: UnsafeMutablePointer<Float>) {
        // Vector unpack for the common types (the same integers as the scalar cases below,
        // which the tests compare against upstream's dequantisation).
        let wr = UnsafeMutableRawPointer(w)
        @inline(__always) func put(_ i: Int, _ v: SIMD16<UInt8>) { wr.storeBytes(of: v, toByteOffset: 16 * i, as: SIMD16<UInt8>.self) }
        switch t {
        case .q4_0:
            let d = rdF16(b, 0), q = ld16(b, 2)
            put(0, (q & 0x0F) &- 8); put(1, (q &>> 4) &- 8)
            sw[0] = d; sw[1] = d; mw[0] = 0; mw[1] = 0
            return
        case .q4_1:
            let d = rdF16(b, 0), m = rdF16(b, 2), q = ld16(b, 4)
            put(0, q & 0x0F); put(1, q &>> 4)
            sw[0] = d; sw[1] = d; mw[0] = -m; mw[1] = -m
            return
        case .q4_K, .q5_K:
            let d = rdF16(b, 0), dmin = rdF16(b, 2)
            let five = t == .q5_K
            let qsOff = five ? 48 : 16
            let qh0 = ld16(b, 16), qh1 = ld16(b, 32)
            for g in 0..<8 {
                let (s, m) = scaleMinK4(b + 4, g)
                let ds = d * Float(s), dm = dmin * Float(m)
                sw[2 * g] = ds; sw[2 * g + 1] = ds; mw[2 * g] = dm; mw[2 * g + 1] = dm
            }
            for p in 0..<4 {
                for h in 0..<2 {
                    let q = ld16(b, qsOff + p * 32 + h * 16)
                    var v0 = q & 0x0F, v1 = q &>> 4
                    if five {
                        let qh = h == 0 ? qh0 : qh1
                        v0 |= ((qh &>> SIMD16<UInt8>(repeating: UInt8(2 * p))) & 1) &<< 4
                        v1 |= ((qh &>> SIMD16<UInt8>(repeating: UInt8(2 * p + 1))) & 1) &<< 4
                    }
                    put(4 * p + h, v0)
                    put(4 * p + 2 + h, v1)
                }
            }
            return
        case .q6_K:
            let d = rdF16(b, 208)
            for g in 0..<16 {
                let i0 = g * 16
                let c = i0 >> 7, r0 = i0 & 127
                let lo = (ld16(b, c * 64 + (r0 & 63)) &>> SIMD16<UInt8>(repeating: UInt8(4 * (r0 >> 6)))) & 0x0F
                let hi = (ld16(b, 128 + c * 32 + (r0 & 31)) &>> SIMD16<UInt8>(repeating: UInt8(2 * (r0 >> 5)))) & 3
                put(g, (lo | (hi &<< 4)) &- 32)
                sw[g] = d * Float(Int8(bitPattern: rd8(b, 192 + g))); mw[g] = 0
            }
            return
        default:
            break
        }
        switch t {
        case .q8_0:
            let d = rdF16(b, 0)
            for j in 0..<32 { w[j] = Int8(bitPattern: rd8(b, 2 + j)) }
            sw[0] = d; sw[1] = d; mw[0] = 0; mw[1] = 0
        case .q5_0:
            let d = rdF16(b, 0); let qh = rd32(b, 2)
            for j in 0..<16 {
                let q = rd8(b, 6 + j)
                w[j] = Int8((Int(q & 15) | Int(((qh >> UInt32(j)) & 1) << 4)) - 16)
                w[j + 16] = Int8((Int(q >> 4) | Int(((qh >> UInt32(j + 16)) & 1) << 4)) - 16)
            }
            sw[0] = d; sw[1] = d; mw[0] = 0; mw[1] = 0
        case .q5_1:
            let d = rdF16(b, 0), m = rdF16(b, 2); let qh = rd32(b, 4)
            for j in 0..<16 {
                let q = rd8(b, 8 + j)
                w[j] = Int8(Int(q & 15) | Int(((qh >> UInt32(j)) & 1) << 4))
                w[j + 16] = Int8(Int(q >> 4) | Int(((qh >> UInt32(j + 16)) & 1) << 4))
            }
            sw[0] = d; sw[1] = d; mw[0] = -m; mw[1] = -m
        case .iq4_nl:
            let d = rdF16(b, 0)
            for j in 0..<16 {
                let q = rd8(b, 2 + j)
                w[j] = iq4nl(Int(q & 15)); w[j + 16] = iq4nl(Int(q >> 4))
            }
            sw[0] = d; sw[1] = d; mw[0] = 0; mw[1] = 0
        case .q2_K:
            let d = rdF16(b, 80), dmin = rdF16(b, 82)
            for i in 0..<256 {
                let c = i >> 7, sh = (i >> 5) & 3, j = i & 31
                w[i] = Int8((rd8(b, 16 + c * 32 + j) >> UInt8(2 * sh)) & 3)
            }
            for g in 0..<16 {
                let sc = rd8(b, g)
                sw[g] = d * Float(sc & 0xF); mw[g] = dmin * Float(sc >> 4)
            }
        case .q3_K:
            let d = rdF16(b, 108)
            let s = b + 96
            for k in 0..<16 {
                let l = k < 8 ? Int(rd8(s, k) & 0x0F) : Int(rd8(s, k - 8) >> 4)
                let h = Int((rd8(s, 8 + (k & 3)) >> UInt8(2 * (k >> 2))) & 3)
                sw[k] = d * Float((l | (h << 4)) - 32); mw[k] = 0
            }
            for i in 0..<256 {
                let c = i >> 7, sh = (i >> 5) & 3, j = i & 31
                let ql = Int((rd8(b, 32 + c * 32 + j) >> UInt8(2 * sh)) & 3)
                let hb = Int((rd8(b, i & 31) >> UInt8(i >> 5)) & 1)
                w[i] = Int8(ql - ((hb ^ 1) << 2))
            }
        case .iq4_xs:
            let d = rdF16(b, 0)
            let shi = rd16(b, 2)
            for ib in 0..<8 {
                let ls = Int((rd8(b, 4 + (ib >> 1)) >> UInt8(4 * (ib & 1))) & 0x0F) | Int(((shi >> UInt16(2 * ib)) & 3) << 4)
                let dl = d * Float(ls - 32)
                sw[2 * ib] = dl; sw[2 * ib + 1] = dl; mw[2 * ib] = 0; mw[2 * ib + 1] = 0
                for j in 0..<16 {
                    let v = rd8(b, 8 + 16 * ib + j)
                    w[ib * 32 + j] = iq4nl(Int(v & 0x0F))
                    w[ib * 32 + j + 16] = iq4nl(Int(v >> 4))
                }
            }
        default:
            preconditionFailure("IntegerForm: no unpack for \(t)")
        }
    }
}

extension Dot {
    /// True when `dotQ8` handles `type` (every block type of the first port).
    public static func supportsInteger(_ type: GGMLType) -> Bool { IntegerForm.supports(type) }

    /// Types with the C rows kernel (`colibri_rows_i8`, CColibriKernels/rows_i8.c): the
    /// same bits as `dotQ8`, four rows per pass. `COLIBRI_NO_ROW_KERNEL=1` turns it off.
    static let rowKernelTypes: Set<GGMLType> = {
        if ProcessInfo.processInfo.environment["COLIBRI_NO_ROW_KERNEL"] == "1" { return [] }
        return Set(GGMLType.allCases.filter { colibri_rows_i8_supported(Int32($0.rawValue)) != 0 })
    }()

    @inline(__always) public static func hasRowKernel(_ type: GGMLType) -> Bool { rowKernelTypes.contains(type) }

    /// Integer-path dot of one weight row with row `r` of `act` (`gq_dot_q8_0` for Q8_0,
    /// `gq_ublk` + `gq_dot_i8` for every other block type). `count == act.cols`.
    public static func dotQ8(_ type: GGMLType, row: UnsafeRawPointer, _ act: Q8Activations, row r: Int) -> Float {
        let (xq, dx, xs) = act.row(r)
        let n = act.cols
        if type == .q8_0 { return dotQ8_0Int(row, xq, dx, n / 32) }
        let bl = type.blockSize, bs = type.typeSize
        return withUnsafeTemporaryAllocation(of: Int8.self, capacity: 256) { w8b in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 32) { scb in
                let w8 = w8b.baseAddress!, sw = scb.baseAddress!, mw = sw + 16
                var acc: Float = 0, accB: Float = 0, m: Float = 0
                let hasMin = IntegerForm.hasMin(type)
                for blk in 0..<(n / bl) {
                    IntegerForm.unpack(type, row + blk * bs, w8, sw, mw)
                    accumulateBlock(w8, sw, mw, base: blk * bl, bl: bl, hasMin: hasMin, xq, dx, xs, &acc, &accB, &m)
                }
                return (acc + accB) - m
            }
        }
    }

    /// Unpacks a whole row of a non-Q8_0 block type for `dotQ8Unpacked`, so a batch of
    /// activation rows pays for the unpack once: `w8` gets `cols` codes, `sw` and `mw`
    /// `cols / 16` scales and mins (block `b`'s at `b * blockSize / 16`).
    public static func unpackRowQ8(_ type: GGMLType, row: UnsafeRawPointer, cols: Int, w8: UnsafeMutablePointer<Int8>,
                                   sw: UnsafeMutablePointer<Float>, mw: UnsafeMutablePointer<Float>) {
        let bl = type.blockSize, bs = type.typeSize, nu = bl / 16
        for blk in 0..<(cols / bl) {
            IntegerForm.unpack(type, row + blk * bs, w8 + blk * bl, sw + blk * nu, mw + blk * nu)
        }
    }

    /// `dotQ8` of a row unpacked by `unpackRowQ8`: the same sums in the same order, so the
    /// same bits.
    public static func dotQ8Unpacked(_ type: GGMLType, w8: UnsafePointer<Int8>, sw: UnsafePointer<Float>,
                                     mw: UnsafePointer<Float>, _ act: Q8Activations, row r: Int) -> Float {
        let (xq, dx, xs) = act.row(r)
        let bl = type.blockSize, nu = bl / 16
        var acc: Float = 0, accB: Float = 0, m: Float = 0
        let hasMin = IntegerForm.hasMin(type)
        for blk in 0..<(act.cols / bl) {
            accumulateBlock(w8 + blk * bl, sw + blk * nu, mw + blk * nu, base: blk * bl, bl: bl, hasMin: hasMin,
                            xq, dx, xs, &acc, &accB, &m)
        }
        return (acc + accB) - m
    }

    /// One block of `gq_dot_i8`: integer sums per 16 codes, scaled per 32 activations,
    /// alternating accumulators, and the min term with upstream's fmaf nesting.
    @inline(__always)
    static func accumulateBlock(_ w8: UnsafePointer<Int8>, _ sw: UnsafePointer<Float>, _ mw: UnsafePointer<Float>,
                                base: Int, bl: Int, hasMin: Bool, _ xq: UnsafePointer<Int8>, _ dx: UnsafePointer<Float>,
                                _ xs: UnsafePointer<Int32>, _ acc: inout Float, _ accB: inout Float, _ m: inout Float) {
        var g = 0
        while g < bl / 16 {
            let b32 = (base >> 5) + (g >> 1)
            let xb = xq + base + g * 16
            let wb = w8 + g * 16
            var s0: Int32 = 0, s1: Int32 = 0
            colibri_dot_i8_16x2(wb, xb, &s0, &s1)  // NEON sdot; exact, so order-free
            let v = dx[b32] * (sw[g] * Float(s0) + sw[g + 1] * Float(s1))
            if g & 2 == 0 { acc += v } else { accB += v }
            if hasMin {
                let g16 = (base >> 4) + g
                // upstream's explicit fmaf: fmaf(dx, fmaf(mw0, xs0, mw1 * xs1), m)
                let inner = (mw[g + 1] * Float(xs[g16 + 1])).addingProduct(mw[g], Float(xs[g16]))
                m = m.addingProduct(dx[b32], inner)
            }
            g += 2
        }
    }

    /// `gq_dot_q8_0`: per block `Float(sum qs * xq) * (d * dx)`.
    @inline(__always)
    static func dotQ8_0Int(_ row: UnsafeRawPointer, _ xq: UnsafePointer<Int8>, _ dx: UnsafePointer<Float>, _ nb: Int) -> Float {
        var acc0: Float = 0, acc1: Float = 0
        var b = 0
        while b + 1 < nb {
            let blk0 = row + b * 34, blk1 = blk0 + 34
            let w0 = (blk0 + 2).assumingMemoryBound(to: Int8.self), w1 = (blk1 + 2).assumingMemoryBound(to: Int8.self)
            let x0 = xq + b * 32, x1 = x0 + 32
            var s0: Int32 = 0, s1: Int32 = 0
            for i in 0..<32 { s0 &+= Int32(w0[i]) &* Int32(x0[i]) }
            for i in 0..<32 { s1 &+= Int32(w1[i]) &* Int32(x1[i]) }
            acc0 += Float(s0) * (rdF16(blk0, 0) * dx[b])
            acc1 += Float(s1) * (rdF16(blk1, 0) * dx[b + 1])
            b += 2
        }
        if b < nb {
            let blk = row + b * 34
            let w = (blk + 2).assumingMemoryBound(to: Int8.self), x = xq + b * 32
            var s: Int32 = 0
            for i in 0..<32 { s &+= Int32(w[i]) &* Int32(x[i]) }
            acc0 += Float(s) * (rdF16(blk, 0) * dx[b])
        }
        return acc0 + acc1
    }
}
