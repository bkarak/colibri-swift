import ColibriCore
import Foundation
import Metal
import Testing
import ColibriMetal

/// True when this machine has a Metal device (every Metal test is skipped otherwise).
let metalAvailable = MTLCreateSystemDefaultDevice() != nil

/// Deterministic SplitMix64 for test data.
struct TestRNG {
    var state: UInt64
    init(_ seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1 }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    mutating func float() -> Float { Float(Double(next() >> 11) / 9_007_199_254_740_992.0 * 2 - 1) }
    mutating func floats(_ n: Int, scale: Float = 1) -> [Float] { (0..<n).map { _ in float() * scale } }
    mutating func int(_ n: Int) -> Int { Int(next() % UInt64(n)) }
}

/// Page-aligned host memory that a `MemoryRegion` keeps alive.
final class AlignedMemory {
    let base: UnsafeMutableRawPointer
    let length: Int
    init(_ length: Int) {
        var p: UnsafeMutableRawPointer?
        precondition(posix_memalign(&p, hostPageSize, length) == 0)
        base = p!
        self.length = length
        memset(base, 0, length)
    }
    deinit { free(base) }
}

/// A region holding `bytes` at byte offset `leading` (the rest zero).
func makeRegion(_ bytes: [UInt8], leading: Int = 0, label: String = "test.weights") -> MemoryRegion {
    let length = alignUp(max(leading + bytes.count, 1), hostPageSize)
    let mem = AlignedMemory(length)
    bytes.withUnsafeBytes { (mem.base + leading).copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
    return MemoryRegion(base: UnsafeRawPointer(mem.base), length: length, kind: .other, label: label, keepAlive: mem)
}

/// Dequantised rows of a matrix (the CPU reference decoder).
func dequantRows(_ type: GGMLType, _ bytes: [UInt8], rows: Int, cols: Int) -> [Float] {
    Dequant.dequantize(type, bytes: bytes, count: rows * cols)
}

extension Backend {
    func upload(_ v: [Float], label: String = "t") throws -> DeviceBuffer {
        let b = try makeBuffer(.f32, count: max(v.count, 1), label: label)
        for (i, x) in v.enumerated() { b.slice().floats[i] = x }
        return b
    }
    func uploadI32(_ v: [Int32], label: String = "ids") throws -> DeviceBuffer {
        let b = try makeBuffer(.i32, count: max(v.count, 1), label: label)
        for (i, x) in v.enumerated() { b.slice().int32s[i] = x }
        return b
    }
    func zeros(_ n: Int, _ dtype: ScalarType = .f32, label: String = "z") throws -> DeviceBuffer {
        try makeBuffer(dtype, count: max(n, 1), label: label)
    }
}

extension DeviceBuffer {
    var f: [Float] { Array(slice().floats) }
    var i: [Int32] { Array(slice().int32s) }
}

/// A CPU and a Metal backend side by side. `both` runs the same encoding closure on each,
/// synchronises, and returns what the closure's reader produced on each backend.
final class Twin {
    let cpu = CPUBackend()
    let metal: MetalBackend

    init() throws { metal = try MetalBackend() }

    func both<R>(_ body: (Backend) throws -> (() -> R)) throws -> (cpu: R, metal: R) {
        let rc = try body(cpu)
        try cpu.synchronize()
        let a = rc()
        let rm = try body(metal)
        try metal.synchronize()
        let b = rm()
        return (a, b)
    }
}

/// Largest |a - b| / (rel * scale + abs) over the arrays; <= 1 means within tolerance.
/// `scale[i]` defaults to |a[i]|.
@discardableResult
func expectClose(_ a: [Float], _ b: [Float], rel: Float, abs: Float, scale: [Float]? = nil,
                 _ what: String, sourceLocation: SourceLocation = #_sourceLocation) -> Float {
    #expect(a.count == b.count, "\(what): count \(a.count) vs \(b.count)", sourceLocation: sourceLocation)
    var worst: Float = 0, at = -1
    for i in 0..<min(a.count, b.count) {
        let s = scale?[i] ?? Swift.abs(a[i])
        let err = Swift.abs(a[i] - b[i]) / (rel * s + abs)
        if !(err <= worst) { worst = err; at = i }   // NaN counts as worst
        if err.isNaN { worst = .infinity; at = i; break }
    }
    #expect(worst <= 1, "\(what): worst at \(at): cpu \(at >= 0 ? a[at] : 0) metal \(at >= 0 ? b[at] : 0) (ratio \(worst))",
            sourceLocation: sourceLocation)
    return worst
}

/// Every weight type the Metal backend decodes, in id order.
let metalWeightTypes: [GGMLType] = MetalBackend.supportedWeightTypes.sorted { $0.rawValue < $1.rawValue }
