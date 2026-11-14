// Encoders that turn Float32 values into ggml block bytes, for synthetic weights and
// round-trip tests. They are simple round-to-nearest quantisers (ggml's own use a
// weighted scale search for the K and IQ types), so files written here are valid and
// decode within the type's step size, but are not byte-identical to llama.cpp's output.
// Block layouts: docs/port/gguf-and-quants.md sections 3.3-3.5 (the inverse of
// BlockDecoder in Dequant.swift).

/// Encoders that turn Float32 values into block bytes, for synthetic weights.
/// Quantisation need not match ggml's quantiser bit for bit: the tests compare
/// dequantised values, and the oracle reads the same file.
public enum Quantize {
    /// Every type `Dequant` decodes. `values.count` must be a multiple of the block size;
    /// rows are not distinguished (blocks never straddle rows when `ne[0]` is a multiple of
    /// the block size, which `GGUFWriter` checks).
    public static func encode(_ values: [Float], as type: GGMLType) throws -> [UInt8] {
        guard Dequant.supports(type) else {
            throw ColibriError.unsupported("Quantize: no encoder for \(type)")
        }
        guard values.count % type.blockSize == 0 else {
            throw ColibriError.invalidArgument("Quantize: \(values.count) values is not a multiple of \(type)'s block \(type.blockSize)")
        }
        let nb = values.count / type.blockSize
        var out = [UInt8](repeating: 0, count: nb * type.typeSize)
        values.withUnsafeBufferPointer { vb in
            out.withUnsafeMutableBytes { ob in
                let x = vb.baseAddress!, o = ob.baseAddress!
                let bl = type.blockSize, bs = type.typeSize
                switch type {
                case .f32:
                    for i in 0..<values.count { o.storeBytes(of: x[i].bitPattern.littleEndian, toByteOffset: 4 * i, as: UInt32.self) }
                case .f16:
                    for i in 0..<values.count { o.storeBytes(of: Float16(x[i]).bitPattern.littleEndian, toByteOffset: 2 * i, as: UInt16.self) }
                case .bf16:
                    for i in 0..<values.count { o.storeBytes(of: bf16Bits(x[i]).littleEndian, toByteOffset: 2 * i, as: UInt16.self) }
                default:
                    for b in 0..<nb {
                        let v = UnsafeBufferPointer(start: x + b * bl, count: bl)
                        let p = o + b * bs
                        switch type {
                        case .q8_0: q8_0(v, p)
                        case .q4_0: q4_0(v, p)
                        case .q4_1: q4_1(v, p)
                        case .q5_0: q5_0(v, p)
                        case .q5_1: q5_1(v, p)
                        case .iq4_nl: iq4_nl(v, p)
                        case .q2_K: q2_K(v, p)
                        case .q3_K: q3_K(v, p)
                        case .q4_K: q4_K(v, p, fiveBit: false)
                        case .q5_K: q4_K(v, p, fiveBit: true)
                        case .q6_K: q6_K(v, p)
                        case .iq4_xs: iq4_xs(v, p)
                        default: preconditionFailure("unreachable")
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: Helpers

    /// Round-to-nearest-even bfloat16 (NaN kept quiet).
    static func bf16Bits(_ x: Float) -> UInt16 {
        let b = x.bitPattern
        if x.isNaN { return UInt16(b >> 16) | 0x40 }
        return UInt16((b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16)
    }

    /// Stores `x` as f16 at `o` and returns the value a decoder will read back.
    @discardableResult
    static func putF16(_ x: Float, _ p: UnsafeMutableRawPointer, _ off: Int) -> Float {
        let h = Float16(x)
        p.storeBytes(of: h.bitPattern.littleEndian, toByteOffset: off, as: UInt16.self)
        return Float(h)
    }

    /// The element of largest magnitude, with its sign.
    static func maxAbsSigned(_ v: UnsafeBufferPointer<Float>) -> Float {
        var m: Float = 0
        for x in v where abs(x) > abs(m) { m = x }
        return m
    }

    static func roundClamp(_ x: Float, _ lo: Int, _ hi: Int) -> Int {
        guard x.isFinite else { return lo }
        return min(max(Int(x.rounded(.toNearestOrAwayFromZero)), lo), hi)
    }

    static func nearestIQ4(_ x: Float, _ d: Float) -> Int {
        var best = 0
        var bestErr = Float.infinity
        for k in 0..<16 {
            let e = abs(x - d * Float(iq4nlValues[k]))
            if e < bestErr { bestErr = e; best = k }
        }
        return best
    }

    // MARK: 32-element blocks

    static func q8_0(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        let amax = v.reduce(0) { max($0, abs($1)) }
        let d = putF16(amax / 127, p, 0)
        let id: Float = d != 0 ? 1 / d : 0
        for j in 0..<32 { p.storeBytes(of: Int8(roundClamp(v[j] * id, -127, 127)), toByteOffset: 2 + j, as: Int8.self) }
    }

    static func q4_0(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        let d = putF16(maxAbsSigned(v) / -8, p, 0)
        let id: Float = d != 0 ? 1 / d : 0
        for j in 0..<16 {
            let lo = roundClamp(v[j] * id + 8, 0, 15), hi = roundClamp(v[j + 16] * id + 8, 0, 15)
            p.storeBytes(of: UInt8(lo | (hi << 4)), toByteOffset: 2 + j, as: UInt8.self)
        }
    }

    static func q4_1(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        let mn = v.min()!, mx = v.max()!
        let d = putF16((mx - mn) / 15, p, 0)
        let m = putF16(mn, p, 2)
        let id: Float = d != 0 ? 1 / d : 0
        for j in 0..<16 {
            let lo = roundClamp((v[j] - m) * id, 0, 15), hi = roundClamp((v[j + 16] - m) * id, 0, 15)
            p.storeBytes(of: UInt8(lo | (hi << 4)), toByteOffset: 4 + j, as: UInt8.self)
        }
    }

    static func q5_0(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        let d = putF16(maxAbsSigned(v) / -16, p, 0)
        let id: Float = d != 0 ? 1 / d : 0
        var qh: UInt32 = 0
        for j in 0..<16 {
            let lo = roundClamp(v[j] * id + 16, 0, 31), hi = roundClamp(v[j + 16] * id + 16, 0, 31)
            p.storeBytes(of: UInt8((lo & 15) | ((hi & 15) << 4)), toByteOffset: 6 + j, as: UInt8.self)
            qh |= UInt32(lo >> 4) << UInt32(j)
            qh |= UInt32(hi >> 4) << UInt32(j + 16)
        }
        p.storeBytes(of: qh.littleEndian, toByteOffset: 2, as: UInt32.self)
    }

    static func q5_1(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        let mn = v.min()!, mx = v.max()!
        let d = putF16((mx - mn) / 31, p, 0)
        let m = putF16(mn, p, 2)
        let id: Float = d != 0 ? 1 / d : 0
        var qh: UInt32 = 0
        for j in 0..<16 {
            let lo = roundClamp((v[j] - m) * id, 0, 31), hi = roundClamp((v[j + 16] - m) * id, 0, 31)
            p.storeBytes(of: UInt8((lo & 15) | ((hi & 15) << 4)), toByteOffset: 8 + j, as: UInt8.self)
            qh |= UInt32(lo >> 4) << UInt32(j)
            qh |= UInt32(hi >> 4) << UInt32(j + 16)
        }
        p.storeBytes(of: qh.littleEndian, toByteOffset: 4, as: UInt32.self)
    }

    static func iq4_nl(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        let d = putF16(maxAbsSigned(v) / -127, p, 0)
        for j in 0..<16 {
            let lo = nearestIQ4(v[j], d), hi = nearestIQ4(v[j + 16], d)
            p.storeBytes(of: UInt8(lo | (hi << 4)), toByteOffset: 2 + j, as: UInt8.self)
        }
    }

    // MARK: K-quants

    static func q2_K(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        var scale = [Float](repeating: 0, count: 16), minv = [Float](repeating: 0, count: 16)
        for g in 0..<16 {
            let s = UnsafeBufferPointer(rebasing: v[g * 16 ..< g * 16 + 16])
            let mn = min(0, s.min()!), mx = max(mn, s.max()!)
            scale[g] = (mx - mn) / 3
            minv[g] = -mn
        }
        let d = putF16(scale.max()! / 15, p, 80)
        let dmin = putF16(minv.max()! / 15, p, 82)
        for g in 0..<16 {
            let sc = d != 0 ? roundClamp(scale[g] / d, 0, 15) : 0
            let mc = dmin != 0 ? roundClamp(minv[g] / dmin, 0, 15) : 0
            p.storeBytes(of: UInt8(sc | (mc << 4)), toByteOffset: g, as: UInt8.self)
            let dl = d * Float(sc), ml = dmin * Float(mc)
            for t in 0..<16 {
                let i = g * 16 + t
                let q = dl != 0 ? roundClamp((v[i] + ml) / dl, 0, 3) : 0
                let c = i >> 7, sh = (i >> 5) & 3, j = i & 31
                let off = 16 + c * 32 + j
                p.storeBytes(of: p.load(fromByteOffset: off, as: UInt8.self) | UInt8(q << (2 * sh)), toByteOffset: off, as: UInt8.self)
            }
        }
    }

    static func q3_K(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        var s = [Float](repeating: 0, count: 16)
        for k in 0..<16 { s[k] = maxAbsSigned(UnsafeBufferPointer(rebasing: v[k * 16 ..< k * 16 + 16])) / -4 }
        var smax: Float = 0
        for x in s where abs(x) > abs(smax) { smax = x }
        let d = putF16(smax / -32, p, 108)
        for k in 0..<16 {
            let l = d != 0 ? roundClamp(s[k] / d, -32, 31) : 0
            let sc6 = l + 32
            if k < 8 {
                p.storeBytes(of: p.load(fromByteOffset: 96 + k, as: UInt8.self) | UInt8(sc6 & 0xF), toByteOffset: 96 + k, as: UInt8.self)
            } else {
                p.storeBytes(of: p.load(fromByteOffset: 96 + k - 8, as: UInt8.self) | UInt8((sc6 & 0xF) << 4), toByteOffset: 96 + k - 8, as: UInt8.self)
            }
            let ho = 96 + 8 + (k & 3)
            p.storeBytes(of: p.load(fromByteOffset: ho, as: UInt8.self) | UInt8((sc6 >> 4) << (2 * (k >> 2))), toByteOffset: ho, as: UInt8.self)
            let dl = d * Float(l)
            for t in 0..<16 {
                let i = k * 16 + t
                let q = dl != 0 ? roundClamp(v[i] / dl, -4, 3) : 0
                let L = q + 4
                let c = i >> 7, sh = (i >> 5) & 3, j = i & 31
                let qo = 32 + c * 32 + j
                p.storeBytes(of: p.load(fromByteOffset: qo, as: UInt8.self) | UInt8((L & 3) << (2 * sh)), toByteOffset: qo, as: UInt8.self)
                let hm = i & 31
                p.storeBytes(of: p.load(fromByteOffset: hm, as: UInt8.self) | UInt8((L >> 2) << (i >> 5)), toByteOffset: hm, as: UInt8.self)
            }
        }
    }

    /// Q4_K, or Q5_K when `fiveBit`.
    static func q4_K(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer, fiveBit: Bool) {
        let levels: Float = fiveBit ? 31 : 15
        var scale = [Float](repeating: 0, count: 8), minv = [Float](repeating: 0, count: 8)
        for g in 0..<8 {
            let s = UnsafeBufferPointer(rebasing: v[g * 32 ..< g * 32 + 32])
            let mn = min(0, s.min()!), mx = max(mn, s.max()!)
            scale[g] = (mx - mn) / levels
            minv[g] = -mn
        }
        let d = putF16(scale.max()! / 63, p, 0)
        let dmin = putF16(minv.max()! / 63, p, 2)
        var sc = [Int](repeating: 0, count: 8), mc = [Int](repeating: 0, count: 8)
        for g in 0..<8 {
            sc[g] = d != 0 ? roundClamp(scale[g] / d, 0, 63) : 0
            mc[g] = dmin != 0 ? roundClamp(minv[g] / dmin, 0, 63) : 0
        }
        // Inverse of scaleMinK4.
        var s12 = [UInt8](repeating: 0, count: 12)
        for g in 0..<4 {
            s12[g] = UInt8(sc[g] & 0x3F) | UInt8((sc[g + 4] >> 4) << 6)
            s12[g + 4] = UInt8(mc[g] & 0x3F) | UInt8((mc[g + 4] >> 4) << 6)
            s12[g + 8] = UInt8(sc[g + 4] & 0x0F) | UInt8((mc[g + 4] & 0x0F) << 4)
        }
        for i in 0..<12 { p.storeBytes(of: s12[i], toByteOffset: 4 + i, as: UInt8.self) }
        let qsOff = fiveBit ? 48 : 16
        let maxQ = fiveBit ? 31 : 15
        for g in 0..<8 {
            let d1 = d * Float(sc[g]), m1 = dmin * Float(mc[g])
            let qo = qsOff + (g >> 1) * 32
            let shift = (g & 1) * 4
            for j in 0..<32 {
                let q = d1 != 0 ? roundClamp((v[g * 32 + j] + m1) / d1, 0, maxQ) : 0
                p.storeBytes(of: p.load(fromByteOffset: qo + j, as: UInt8.self) | UInt8((q & 0xF) << shift), toByteOffset: qo + j, as: UInt8.self)
                if fiveBit {
                    p.storeBytes(of: p.load(fromByteOffset: 16 + j, as: UInt8.self) | UInt8((q >> 4) << g), toByteOffset: 16 + j, as: UInt8.self)
                }
            }
        }
    }

    static func q6_K(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        var s = [Float](repeating: 0, count: 16)
        for g in 0..<16 { s[g] = maxAbsSigned(UnsafeBufferPointer(rebasing: v[g * 16 ..< g * 16 + 16])) / -32 }
        let d = putF16(s.reduce(0) { max($0, abs($1)) } / 127, p, 208)
        for g in 0..<16 {
            let sc = d != 0 ? roundClamp(s[g] / d, -128, 127) : 0
            p.storeBytes(of: Int8(sc), toByteOffset: 192 + g, as: Int8.self)
            let dl = d * Float(sc)
            for t in 0..<16 {
                let i = g * 16 + t
                let q = dl != 0 ? roundClamp(v[i] / dl, -32, 31) : 0
                let L = q + 32
                let c = i >> 7, r = i & 127
                let lo = c * 64 + (r & 63), hi = 128 + c * 32 + (r & 31)
                p.storeBytes(of: p.load(fromByteOffset: lo, as: UInt8.self) | UInt8((L & 0xF) << (4 * (r >> 6))), toByteOffset: lo, as: UInt8.self)
                p.storeBytes(of: p.load(fromByteOffset: hi, as: UInt8.self) | UInt8((L >> 4) << (2 * (r >> 5))), toByteOffset: hi, as: UInt8.self)
            }
        }
    }

    static func iq4_xs(_ v: UnsafeBufferPointer<Float>, _ p: UnsafeMutableRawPointer) {
        var s = [Float](repeating: 0, count: 8)
        for ib in 0..<8 { s[ib] = maxAbsSigned(UnsafeBufferPointer(rebasing: v[ib * 32 ..< ib * 32 + 32])) / -127 }
        var smax: Float = 0
        for x in s where abs(x) > abs(smax) { smax = x }
        let d = putF16(smax / -32, p, 0)
        var sh: UInt16 = 0
        for ib in 0..<8 {
            let l = d != 0 ? roundClamp(s[ib] / d, -32, 31) : 0
            let ls = l + 32
            sh |= UInt16((ls >> 4) & 3) << UInt16(2 * ib)
            let lo = 4 + (ib >> 1)
            p.storeBytes(of: p.load(fromByteOffset: lo, as: UInt8.self) | UInt8((ls & 0xF) << (4 * (ib & 1))), toByteOffset: lo, as: UInt8.self)
            let dl = d * Float(l)
            for j in 0..<16 {
                let a = nearestIQ4(v[ib * 32 + j], dl), b = nearestIQ4(v[ib * 32 + j + 16], dl)
                p.storeBytes(of: UInt8(a | (b << 4)), toByteOffset: 8 + 16 * ib + j, as: UInt8.self)
            }
        }
        p.storeBytes(of: sh.littleEndian, toByteOffset: 2, as: UInt16.self)
    }
}
