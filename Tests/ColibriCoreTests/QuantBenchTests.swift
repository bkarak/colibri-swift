import Foundation
import Testing
@testable import ColibriCore

// Workstream A: CPU matvec throughput. Off by default; run in release:
//
//   COLIBRI_BENCH=1 swift test -c release -Xswiftc -enable-testing \
//       --scratch-path .build-A-rel --filter QuantBenchTests
//
// Reports weight bytes streamed per second (rows * rowBytes / time).

let benchEnabled = ProcessInfo.processInfo.environment["COLIBRI_BENCH"] == "1"

@Suite(.serialized) struct QuantBenchTests {
    func bench(_ type: GGMLType, rows: Int, cols: Int, integer: Bool, threads: Int? = nil) throws -> (gbps: Double, ms: Double) {
        var rng = TestRNG(42)
        let store = TestWeights(capacity: type.rowBytes(cols) * rows + 64)
        // Quantise in slices to keep the temporary Float array small.
        var off = -1
        let slice = 4096
        var r = 0
        while r < rows {
            let n = min(slice, rows - r)
            let o = store.append(try Quantize.encode(rng.floats(n * cols, scale: 0.1), as: type))
            if off < 0 { off = o }
            r += n
        }
        let w = WeightView(region: store.region, offset: off, type: type, cols: cols, rows: rows)
        let cpu = CPUBackend(threads: threads, integerActivations: integer)
        let x = try buffer(cpu, rng.floats(cols))
        let y = try cpu.makeBuffer(.f32, count: rows, label: "y")
        for _ in 0..<3 { cpu.matmul(w, x.slice(), into: y.slice(), batch: 1) }
        var times = [Double]()
        let target = 0.5  // seconds
        let t0 = Date()
        while Date().timeIntervalSince(t0) < target || times.count < 10 {
            let s = DispatchTime.now().uptimeNanoseconds
            cpu.matmul(w, x.slice(), into: y.slice(), batch: 1)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e9)
        }
        times.sort()
        let median = times[times.count / 2]
        return (Double(w.byteCount) / median / 1e9, median * 1e3)
    }

    @Test(.enabled(if: benchEnabled, "set COLIBRI_BENCH=1"))
    func q8_0Matvec() throws {
        let shapes = [(4096, 4096), (151_936, 1024)]
        print("CPU matvec, \(ProcessInfo.processInfo.activeProcessorCount) threads unless noted, median time")
        for (rows, cols) in shapes {
            for integer in [false, true] {
                let (g, ms) = try bench(.q8_0, rows: rows, cols: cols, integer: integer)
                let bytes = GGMLType.q8_0.rowBytes(cols) * rows
                print(String(format: "  Q8_0 %6d x %5d (%6.1f MB) %@: %7.3f ms  %6.1f GB/s",
                             rows, cols, Double(bytes) / 1e6, integer ? "int8 path " : "float path", ms, g))
            }
        }
        let (g1, ms1) = try bench(.q8_0, rows: 4096, cols: 4096, integer: false, threads: 1)
        print(String(format: "  Q8_0   4096 x  4096 float path, 1 thread: %7.3f ms  %6.1f GB/s", ms1, g1))
        let (g2, ms2) = try bench(.q8_0, rows: 4096, cols: 4096, integer: true, threads: 1)
        print(String(format: "  Q8_0   4096 x  4096 int8 path,  1 thread: %7.3f ms  %6.1f GB/s", ms2, g2))
    }

    @Test(.enabled(if: benchEnabled, "set COLIBRI_BENCH=1"))
    func otherTypesMatvec() throws {
        print("4096 x 4096 matvec per type: weight GB/s, 1 thread / all threads")
        for type in [GGMLType.f32, .f16, .bf16, .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .iq4_nl, .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_xs] {
            for integer in [false, true] where !(integer && !IntegerForm.supports(type)) {
                let (g1, _) = try bench(type, rows: 4096, cols: 4096, integer: integer, threads: 1)
                let (g, _) = try bench(type, rows: 4096, cols: 4096, integer: integer)
                print(String(format: "  %-6@ %@: %6.1f / %6.1f GB/s", type.description, integer ? "int8 path " : "float path", g1, g))
            }
        }
    }
}
