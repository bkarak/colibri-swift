import CColibriKernels
import Foundation
import Testing
@testable import ColibriCore

// Workstream A: block decoders, encoders and dot products.
// Golden data: Fixtures/quant/*.golden, produced by upstream's own c/ggml_quants.h.

/// The 14 block/float types of the first port that have golden files.
let goldenTypes: [GGMLType] = [.f16, .bf16, .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl,
                               .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_xs]

/// Every type `Quantize` / `Dequant` / `Dot` support.
let portTypes: [GGMLType] = [.f32] + goldenTypes

struct Golden {
    var type: GGMLType
    var cols: Int
    var rows: Int
    var bytes: [UInt8]
    var deq: [Float]
    var x: [Float]
    var yInt: [Float]
    var yExact: [Float]

    static func load(_ type: GGMLType) throws -> Golden {
        let url = try #require(Bundle.module.url(forResource: type.description, withExtension: "golden",
                                                 subdirectory: "Fixtures/quant"))
        let d = [UInt8](try Data(contentsOf: url))
        var pos = 0
        func u32() -> Int {
            defer { pos += 4 }
            return Int(UInt32(d[pos]) | UInt32(d[pos + 1]) << 8 | UInt32(d[pos + 2]) << 16 | UInt32(d[pos + 3]) << 24)
        }
        func floats(_ n: Int) -> [Float] {
            (0..<n).map { _ in Float(bitPattern: UInt32(u32())) }
        }
        let t = u32(), cols = u32(), rows = u32(), nbytes = u32()
        #expect(t == Int(type.rawValue))
        let bytes = Array(d[pos ..< pos + nbytes]); pos += nbytes
        let deq = floats(rows * cols)
        let x = floats(cols)
        let yi = floats(rows), ye = floats(rows)
        #expect(pos == d.count)
        return Golden(type: type, cols: cols, rows: rows, bytes: bytes, deq: deq, x: x, yInt: yi, yExact: ye)
    }
}

/// Deterministic test data.
struct TestRNG {
    var s: UInt64
    init(_ seed: UInt64) { s = seed &* 0x9E37_79B9_7F4A_7C15 | 1 }
    mutating func next() -> UInt64 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17
        return s
    }
    /// Uniform in [-1, 1).
    mutating func float() -> Float { Float(Double(next() >> 11) / 9_007_199_254_740_992.0 * 2 - 1) }
    mutating func floats(_ n: Int, scale: Float = 1) -> [Float] { (0..<n).map { _ in float() * scale } }
}

func referenceDot(_ a: [Float], _ b: [Float], _ off: Int = 0) -> (value: Double, magnitude: Double) {
    var s = 0.0, m = 0.0
    for i in 0..<b.count {
        let p = Double(a[off + i]) * Double(b[i])
        s += p; m += abs(p)
    }
    return (s, m)
}

@Suite struct QuantTests {
    @Test(arguments: goldenTypes)
    func dequantIsBitExactAgainstUpstream(_ type: GGMLType) throws {
        let g = try Golden.load(type)
        let out = Dequant.dequantize(type, bytes: g.bytes, count: g.rows * g.cols)
        var mismatches = 0
        for i in 0..<out.count where out[i].bitPattern != g.deq[i].bitPattern { mismatches += 1 }
        #expect(mismatches == 0, "\(type): \(mismatches) of \(out.count) values differ from upstream")
        // The scalar reference decoders too (the vector ones above are what dequantizeRow uses).
        var scalar = [Float](repeating: .nan, count: out.count)
        g.bytes.withUnsafeBytes { raw in
            scalar.withUnsafeMutableBufferPointer {
                Dequant.dequantizeRowScalar(type, raw.baseAddress!, into: $0.baseAddress!, count: out.count)
            }
        }
        #expect(zip(scalar, g.deq).allSatisfy { $0.bitPattern == $1.bitPattern }, "\(type): scalar decoder differs")
    }

    @Test(arguments: goldenTypes)
    func integerUnpackReproducesDequant(_ type: GGMLType) throws {
        guard IntegerForm.supports(type) else { return }
        let g = try Golden.load(type)
        let bl = type.blockSize, bs = type.typeSize
        var w8 = [Int8](repeating: 0, count: 256), sw = [Float](repeating: 0, count: 16), mw = sw
        var mismatches = 0
        g.bytes.withUnsafeBytes { raw in
            for b in 0..<(g.bytes.count / bs) {
                IntegerForm.unpack(type, raw.baseAddress! + b * bs, &w8, &sw, &mw)
                for i in 0..<bl {
                    let v = sw[i / 16] * Float(w8[i]) - mw[i / 16]
                    if v.bitPattern != g.deq[b * bl + i].bitPattern && !(v == 0 && g.deq[b * bl + i] == 0) { mismatches += 1 }
                }
            }
        }
        #expect(mismatches == 0, "\(type): integer form differs from dequant in \(mismatches) values")
    }

    @Test(arguments: goldenTypes)
    func floatDotMatchesUpstreamExactPath(_ type: GGMLType) throws {
        let g = try Golden.load(type)
        let rb = type.rowBytes(g.cols)
        g.bytes.withUnsafeBytes { raw in
            g.x.withUnsafeBufferPointer { x in
                for o in 0..<g.rows {
                    let y = Dot.dot(type, row: raw.baseAddress! + o * rb, x.baseAddress!, count: g.cols)
                    let ref = referenceDot(g.deq, g.x, o * g.cols)
                    #expect(abs(Double(y) - Double(g.yExact[o])) <= 1e-5 * ref.magnitude + 1e-30,
                            "\(type) row \(o): \(y) vs upstream \(g.yExact[o])")
                    #expect(abs(Double(y) - ref.value) <= 1e-5 * ref.magnitude + 1e-30)
                }
            }
        }
    }

    @Test(arguments: goldenTypes)
    func integerDotMatchesUpstreamIntegerPath(_ type: GGMLType) throws {
        guard IntegerForm.supports(type) else { return }
        let g = try Golden.load(type)
        let act = Q8Activations()
        g.x.withUnsafeBufferPointer { act.quantize($0.baseAddress!, rows: 1, cols: g.cols) }
        let rb = type.rowBytes(g.cols)
        g.bytes.withUnsafeBytes { raw in
            for o in 0..<g.rows {
                let y = Dot.dotQ8(type, row: raw.baseAddress! + o * rb, act, row: 0)
                let ref = referenceDot(g.deq, g.x, o * g.cols)
                // Same integers and scales as upstream; only the float accumulation order differs.
                #expect(abs(Double(y) - Double(g.yInt[o])) <= 2e-5 * ref.magnitude + 1e-30,
                        "\(type) row \(o): \(y) vs upstream \(g.yInt[o])")
            }
        }
    }

    @Test func activationQuantisationMatchesUpstreamFormula() {
        var rng = TestRNG(7)
        var x = rng.floats(64, scale: 3)
        x[5] = 0; x[40...].withUnsafeMutableBufferPointer { for i in $0.indices { $0[i] = 0 } }  // an all-zero block
        let act = Q8Activations()
        x.withUnsafeBufferPointer { act.quantize($0.baseAddress!, rows: 1, cols: 64) }
        for b in 0..<2 {
            let blk = x[b * 32 ..< b * 32 + 32]
            let amax = blk.reduce(0) { max($0, abs($1)) }
            #expect(act.dx[b] == amax / 127)
            let id: Float = amax > 0 ? 127 / amax : 0
            var s0: Int32 = 0, s1: Int32 = 0
            for i in 0..<32 {
                let q = Int32(max(-127, min(127, (x[b * 32 + i] * id).rounded(.toNearestOrEven))))
                #expect(Int32(act.xq[b * 32 + i]) == q)
                if i < 16 { s0 += q } else { s1 += q }
            }
            #expect(act.xs[2 * b] == s0 && act.xs[2 * b + 1] == s1)
        }
    }

    /// Worst error of quantise -> dequantise, as a fraction of the block's largest |value|.
    static let roundTripBound: [GGMLType: Float] = [
        .f32: 0, .f16: 1e-3, .bf16: 8e-3,
        .q8_0: 0.006, .q4_0: 0.13, .q4_1: 0.075, .q5_0: 0.065, .q5_1: 0.04, .iq4_nl: 0.12,
        .q2_K: 0.45, .q3_K: 0.3, .q4_K: 0.15, .q5_K: 0.08, .q6_K: 0.05, .iq4_xs: 0.14,
    ]

    @Test(arguments: portTypes)
    func quantizeDequantRoundTrip(_ type: GGMLType) throws {
        var rng = TestRNG(UInt64(type.rawValue) + 11)
        let n = 4096
        // Mixed magnitudes, plus an all-zero block and a constant block.
        var v = (0..<n).map { i -> Float in
            let s: Float = [0.01, 0.3, 2, 17][(i / 256) % 4]
            return rng.float() * s
        }
        for i in 512..<768 { v[i] = 0 }
        for i in 768..<1024 { v[i] = 0.75 }
        let bytes = try Quantize.encode(v, as: type)
        #expect(bytes.count == type.rowBytes(n))
        let back = Dequant.dequantize(type, bytes: bytes, count: n)
        let bound = try #require(Self.roundTripBound[type])
        let bl = max(type.blockSize, 32)
        var worst: Float = 0
        for b in 0..<(n / bl) {
            let blk = b * bl ..< (b + 1) * bl
            let amax = v[blk].reduce(0) { max($0, abs($1)) }
            for i in blk {
                let e = abs(back[i] - v[i])
                if amax > 0 { worst = max(worst, e / amax) } else { #expect(e == 0, "\(type): zero block decodes to \(back[i])") }
            }
        }
        print("round trip \(type): worst error \(worst) of block amax")
        #expect(worst <= bound, "\(type): worst relative error \(worst) exceeds \(bound)")
    }

    @Test(arguments: portTypes)
    func fusedDotEqualsDequantThenDot(_ type: GGMLType) throws {
        var rng = TestRNG(UInt64(type.rawValue) + 101)
        for cols in [256, 1024, 3072] {
            let rows = 3
            let w = rng.floats(rows * cols, scale: 0.5)
            let x = rng.floats(cols, scale: 2)
            let bytes = try Quantize.encode(w, as: type)
            let deq = Dequant.dequantize(type, bytes: bytes, count: rows * cols)
            let rb = type.rowBytes(cols)
            let xs = (0..<3).flatMap { _ in rng.floats(cols) }
            bytes.withUnsafeBytes { raw in
                x.withUnsafeBufferPointer { xp in
                    for o in 0..<rows {
                        let y = Dot.dot(type, row: raw.baseAddress! + o * rb, xp.baseAddress!, count: cols)
                        let ref = referenceDot(deq, x, o * cols)
                        #expect(abs(Double(y) - ref.value) <= 1e-5 * ref.magnitude, "\(type) cols \(cols) row \(o)")
                    }
                }
                // dotBatch: the same row against three activation rows.
                var y = [Float](repeating: 0, count: 6)
                xs.withUnsafeBufferPointer { xp in
                    y.withUnsafeMutableBufferPointer { yp in
                        Dot.dotBatch(type, row: raw.baseAddress! + rb, xp.baseAddress!, count: cols, batch: 3,
                                     into: yp.baseAddress!, yStride: 2)
                    }
                }
                for s in 0..<3 {
                    let ref = referenceDot(deq, Array(xs[s * cols ..< (s + 1) * cols]), cols)
                    #expect(abs(Double(y[2 * s]) - ref.value) <= 1e-5 * ref.magnitude, "\(type) dotBatch s \(s)")
                }
            }
        }
    }

    @Test(arguments: portTypes)
    func integerPathIsCloseToFloatPath(_ type: GGMLType) throws {
        guard IntegerForm.supports(type) else { return }
        var rng = TestRNG(UInt64(type.rawValue) + 303)
        let cols = 2048
        let w = rng.floats(cols, scale: 0.2)
        let x = rng.floats(cols)
        let bytes = try Quantize.encode(w, as: type)
        let deq = Dequant.dequantize(type, bytes: bytes, count: cols)
        let act = Q8Activations()
        x.withUnsafeBufferPointer { act.quantize($0.baseAddress!, rows: 1, cols: cols) }
        let y = bytes.withUnsafeBytes { Dot.dotQ8(type, row: $0.baseAddress!, act, row: 0) }
        let ref = referenceDot(deq, x)
        // int8 activations: about 0.4% of each |x| per element, random signs.
        #expect(abs(Double(y) - ref.value) <= 0.01 * ref.magnitude, "\(type): \(y) vs \(ref.value)")
    }

    /// The batch path (row unpacked once, `unpackRowQ8` + `dotQ8Unpacked`) gives the same
    /// bits as `dotQ8` for every block type and several activation rows.
    @Test(arguments: portTypes)
    func unpackedRowDotIsBitIdentical(_ type: GGMLType) throws {
        guard IntegerForm.supports(type), type != .q8_0 else { return }
        var rng = TestRNG(UInt64(type.rawValue) + 909)
        let cols = 2048, rows = 5
        let bytes = try Quantize.encode(rng.floats(cols, scale: 0.2), as: type)
        let x = rng.floats(rows * cols)
        let act = Q8Activations()
        x.withUnsafeBufferPointer { act.quantize($0.baseAddress!, rows: rows, cols: cols) }
        var w8 = [Int8](repeating: 0, count: cols)
        var sc = [Float](repeating: .nan, count: 2 * cols / 16 + 1)
        sc[2 * cols / 16] = 12345  // guard value just past the mins
        bytes.withUnsafeBytes { raw in
            w8.withUnsafeMutableBufferPointer { w in
                sc.withUnsafeMutableBufferPointer { s in
                    let sw = s.baseAddress!, mw = sw + cols / 16
                    Dot.unpackRowQ8(type, row: raw.baseAddress!, cols: cols, w8: w.baseAddress!, sw: sw, mw: mw)
                    for r in 0..<rows {
                        let a = Dot.dotQ8(type, row: raw.baseAddress!, act, row: r)
                        let b = Dot.dotQ8Unpacked(type, w8: w.baseAddress!, sw: sw, mw: mw, act, row: r)
                        #expect(a.bitPattern == b.bitPattern, "\(type) row \(r): \(a) vs \(b)")
                    }
                }
            }
        }
        #expect(sc[2 * cols / 16] == 12345, "\(type): unpack wrote past cols / 16 mins")
    }

    /// The C rows kernel (`colibri_rows_i8`) gives `scale * dotQ8` bit for bit, for rows
    /// from the encoder and for random bytes (every code and scale bit pattern) with sane
    /// half-float scales, over 7 rows (a 4-row pass and a partial one) and two activations.
    @Test func rowKernelCoversTheCommonTypes() {
        #expect(Dot.rowKernelTypes == [.q4_0, .q4_1, .q8_0, .q2_K, .q3_K, .q4_K, .q5_K, .q6_K])
    }

    @Test(arguments: portTypes)
    func rowKernelIsBitIdentical(_ type: GGMLType) throws {
        guard Dot.hasRowKernel(type) else { return }
        var rng = TestRNG(UInt64(type.rawValue) + 4242)
        let cols = 1024, rows = 7, nact = 2
        let rowBytes = type.rowBytes(cols)
        var encoded: [UInt8] = []
        for _ in 0..<rows { encoded += try Quantize.encode(rng.floats(cols, scale: 0.2), as: type) }
        // Random bytes, half-float scale fields set to small positive values.
        let halfFields: [Int] = [.q4_0: [0], .q4_1: [0, 2], .q8_0: [0], .q2_K: [80, 82], .q3_K: [108],
                                 .q4_K: [0, 2], .q5_K: [0, 2], .q6_K: [208]][type] ?? []
        var random = (0..<(rows * rowBytes)).map { _ in UInt8(truncatingIfNeeded: rng.next()) }
        for blk in 0..<(rows * rowBytes / type.typeSize) {
            for f in halfFields {
                let h = Float16(Float(rng.next() % 1000 + 1) / 4096).bitPattern
                random[blk * type.typeSize + f] = UInt8(h & 0xFF)
                random[blk * type.typeSize + f + 1] = UInt8(h >> 8)
            }
        }
        let x = rng.floats(nact * cols)
        let act = Q8Activations()
        x.withUnsafeBufferPointer { act.quantize($0.baseAddress!, rows: nact, cols: cols) }
        let scale: Float = 0.75
        for (label, data) in [("encoded", encoded), ("random", random)] {
            data.withUnsafeBytes { raw in
                let base = raw.baseAddress!
                for r in 0..<nact {
                    var y = [Float](repeating: .nan, count: rows + 1)
                    y[rows] = 777
                    let (xq, dx, xs) = act.row(r)
                    y.withUnsafeMutableBufferPointer { yp in
                        colibri_rows_i8(Int32(type.rawValue), base.assumingMemoryBound(to: UInt8.self), rowBytes,
                                        Int32(rows), xq, dx, xs, Int32(cols), scale, yp.baseAddress!)
                    }
                    for o in 0..<rows {
                        let want = scale * Dot.dotQ8(type, row: base + o * rowBytes, act, row: r)
                        #expect(want.bitPattern == y[o].bitPattern, "\(type) \(label) act \(r) row \(o): \(want) vs \(y[o])")
                    }
                    #expect(y[rows] == 777, "\(type): kernel wrote past nrows")
                }
            }
        }
    }

    @Test func encoderRefusesUndecodableTypes() {
        #expect(throws: ColibriError.self) { try Quantize.encode([Float](repeating: 0, count: 256), as: .iq2_xxs) }
        #expect(throws: ColibriError.self) { try Quantize.encode([Float](repeating: 0, count: 33), as: .q8_0) }
    }
}
