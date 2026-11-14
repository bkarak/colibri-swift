import ColibriCore
import Foundation
import Metal
import Testing
import ColibriMetal

// Workstream E: Metal versus CPU matvec throughput. Off by default; run in release:
//
//   COLIBRI_BENCH=1 swift test -c release --scratch-path .build-E-rel --filter MetalBenchTests
//
// Reports weight bytes streamed per second (rows * rowBytes / time), the same measure as
// QuantBenchTests (workstream A). Two Metal numbers per shape:
// * "GPU": `gpuEndTime - gpuStartTime` of one command buffer holding `reps` matvecs, divided
//   by `reps` (the kernel's own throughput; what a token batched into one CB sees).
// * "1 CB": wall time of one matvec in its own command buffer, commit + wait (includes the
//   ~0.1 ms submit round trip).
// "cold" cycles through enough distinct copies of the matrix (>= 1 GiB) that the system
// level cache cannot hold them; "warm" repeats one matrix.

let metalBenchEnabled = ProcessInfo.processInfo.environment["COLIBRI_BENCH"] == "1"

@Suite(.serialized, .enabled(if: metalAvailable && metalBenchEnabled, "set COLIBRI_BENCH=1 (and use -c release)"))
struct MetalBenchTests {
    /// `copies` back-to-back copies of one random `rows x cols` matrix of `type`.
    static func weights(_ type: GGMLType, rows: Int, cols: Int, copies: Int) throws -> (MemoryRegion, [WeightView]) {
        var rng = TestRNG(42)
        let rb = type.rowBytes(cols)
        let matrixBytes = rb * rows
        let stride = alignUp(matrixBytes, 256)
        let length = alignUp(stride * copies, hostPageSize)
        let mem = AlignedMemory(length)
        var r = 0
        while r < rows {   // quantise in slices to keep the temporary arrays small
            let n = min(4096, rows - r)
            let q = try Quantize.encode(rng.floats(n * cols, scale: 0.1), as: type)
            q.withUnsafeBytes { (mem.base + r * rb).copyMemory(from: $0.baseAddress!, byteCount: q.count) }
            r += n
        }
        for c in 1..<max(copies, 1) { (mem.base + c * stride).copyMemory(from: mem.base, byteCount: matrixBytes) }
        let region = MemoryRegion(base: UnsafeRawPointer(mem.base), length: length, kind: .other, label: "bench", keepAlive: mem)
        let views = (0..<copies).map { WeightView(region: region, offset: $0 * stride, type: type, cols: cols, rows: rows) }
        return (region, views)
    }

    struct Result { var gpuGBps: Double; var gpuMs: Double; var oneCBms: Double; var cpuGBps: Double; var cpuMs: Double }

    static func run(_ type: GGMLType, rows: Int, cols: Int, cold: Bool) throws -> Result {
        let bytes = type.rowBytes(cols) * rows
        let copies = cold ? max(2, (1 << 30) / bytes + 1) : 1
        let (region, views) = try weights(type, rows: rows, cols: cols, copies: copies)
        let metal = try MetalBackend()
        try metal.register(region)
        var rng = TestRNG(7)
        let xv = rng.floats(cols)
        let x = try metal.upload(xv), y = try metal.zeros(rows)

        // Warm-up (pipeline creation, page mapping of the wrapped memory).
        for v in views { metal.matmul(v, x.slice(), into: y.slice(), batch: 1) }
        try metal.synchronize()

        // GPU throughput: many matvecs in one command buffer, several trials, best trial.
        let reps = max(copies, max(8, (2 << 30) / bytes))
        var best = Double.infinity
        for _ in 0..<5 {
            let g0 = metal.gpuSeconds
            metal.beginBatch(label: "bench")
            for i in 0..<reps { metal.matmul(views[i % copies], x.slice(), into: y.slice(), batch: 1) }
            try metal.synchronize()
            best = min(best, (metal.gpuSeconds - g0) / Double(reps))
        }

        // One matvec per command buffer, wall time, median.
        var walls = [Double]()
        for i in 0..<41 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            metal.beginBatch(label: "one")
            metal.matmul(views[i % copies], x.slice(), into: y.slice(), batch: 1)
            try metal.synchronize()
            walls.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        }
        walls.sort()

        // CPU backend on the same bytes (float path, all cores), median.
        let cpu = CPUBackend()
        let cx = try cpu.upload(xv), cy = try cpu.zeros(rows)
        for v in views.prefix(3) { cpu.matmul(v, cx.slice(), into: cy.slice(), batch: 1) }
        var cpuTimes = [Double]()
        let start = Date()
        var i = 0
        while Date().timeIntervalSince(start) < 1.5 || cpuTimes.count < 10 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            cpu.matmul(views[i % copies], cx.slice(), into: cy.slice(), batch: 1)
            cpuTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            i += 1
        }
        cpuTimes.sort()
        let cpuMedian = cpuTimes[cpuTimes.count / 2]

        // The two backends agree on this matrix (a cheap sanity check on the bench itself).
        metal.matmul(views[0], x.slice(), into: y.slice(), batch: 1)
        try metal.synchronize()
        cpu.matmul(views[0], cx.slice(), into: cy.slice(), batch: 1)
        let maxErr = zip(y.f, cy.f).map { abs($0 - $1) }.max() ?? 0
        #expect(maxErr < 1e-3, "bench matrices disagree: \(maxErr)")

        return Result(gpuGBps: Double(bytes) / best / 1e9, gpuMs: best * 1e3, oneCBms: walls[walls.count / 2] * 1e3,
                      cpuGBps: Double(bytes) / cpuMedian / 1e9, cpuMs: cpuMedian * 1e3)
    }

    @Test func matvecThroughput() throws {
        let device = MTLCreateSystemDefaultDevice()!
        print("Metal matvec on \(device.name), CPU backend with \(ProcessInfo.processInfo.activeProcessorCount) threads (float path)")
        print("  type  rows x cols      MB    cache |  GPU ms   GPU GB/s  1-CB ms |  CPU ms   CPU GB/s  speedup")
        for type in [GGMLType.q8_0, .q4_K] {
            for (rows, cols) in [(4096, 4096), (3072, 1024), (1024, 3072), (151_936, 1024)] {
                for cold in [false, true] {
                    let r = try Self.run(type, rows: rows, cols: cols, cold: cold)
                    let mb = Double(type.rowBytes(cols) * rows) / 1e6
                    print(String(format: "  %@ %6d x %5d %7.1f  %@ | %7.3f  %8.1f  %7.3f | %7.3f  %8.1f  %6.1fx",
                                 type.description.padding(toLength: 5, withPad: " ", startingAt: 0), rows, cols, mb,
                                 cold ? "cold" : "warm", r.gpuMs, r.gpuGBps, r.oneCBms, r.cpuMs, r.cpuGBps, r.cpuMs / r.gpuMs))
                }
            }
        }
    }
}

extension MetalBenchTests {
    /// Prefill (batch > 1) throughput of the tiled matvec kernels: GFLOP/s = 2 * rows *
    /// cols * batch / GPU time, against the CPU backend's batched path.
    @Test func prefillThroughput() throws {
        print("Metal prefill matmul (8-row tiles), 4096 x 4096, GPU time vs CPU backend wall time")
        print("  type  batch |  GPU ms  GFLOP/s |  CPU ms  GFLOP/s")
        let metal = try MetalBackend(), cpu = CPUBackend()
        for type in [GGMLType.q8_0, .q4_K] {
            let (region, views) = try Self.weights(type, rows: 4096, cols: 4096, copies: 1)
            try metal.register(region)
            let w = views[0]
            var rng = TestRNG(3)
            for batch in [1, 8, 64, 256] {
                let xv = rng.floats(batch * 4096)
                let x = try metal.upload(xv), y = try metal.zeros(batch * 4096)
                metal.matmul(w, x.slice(), into: y.slice(), batch: batch)
                try metal.synchronize()
                var best = Double.infinity
                for _ in 0..<5 {
                    let g0 = metal.gpuSeconds
                    metal.beginBatch(label: "prefill")
                    for _ in 0..<8 { metal.matmul(w, x.slice(), into: y.slice(), batch: batch) }
                    try metal.synchronize()
                    best = min(best, (metal.gpuSeconds - g0) / 8)
                }
                let cx = try cpu.upload(xv), cy = try cpu.zeros(batch * 4096)
                var cbest = Double.infinity
                for _ in 0..<3 {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    cpu.matmul(w, cx.slice(), into: cy.slice(), batch: batch)
                    cbest = min(cbest, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
                }
                let flop = 2.0 * 4096 * 4096 * Double(batch)
                print(String(format: "  %@ %5d | %7.3f  %7.0f | %7.2f  %7.0f", type.description.padding(toLength: 5, withPad: " ", startingAt: 0),
                             batch, best * 1e3, flop / best / 1e9, cbest * 1e3, flop / cbest / 1e9))
            }
        }
    }
}

/// Kernel-shape sweep for the decode matvec (rows per simdgroup x lanes per unit).
/// `COLIBRI_TUNE=1 swift test -c release --filter MetalTuneTests`.
@Suite(.serialized, .enabled(if: metalAvailable && ProcessInfo.processInfo.environment["COLIBRI_TUNE"] == "1"))
struct MetalTuneTests {
    @Test func sweepDecodeShapes() throws {
        let metal = try MetalBackend()
        for type in [GGMLType.q8_0, .q4_K, .q6_K] {
            for (rows, cols) in [(4096, 4096), (1024, 3072), (3072, 1024), (768, 2048), (2048, 768), (151_936, 1024)] {
                let bytes = type.rowBytes(cols) * rows
                let copies = max(2, (1 << 30) / bytes + 1)
                let (region, views) = try MetalBenchTests.weights(type, rows: rows, cols: cols, copies: copies)
                try metal.register(region)
                var rng = TestRNG(7)
                let x = try metal.upload(rng.floats(cols)), y = try metal.zeros(rows)
                var line = String(format: "%@ %6d x %5d:", type.description, rows, cols)
                for (r, l) in [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4)] {
                    metal.tuneDecode = (r, l)
                    let reps = max(copies, (2 << 30) / bytes)
                    var best = Double.infinity
                    for _ in 0..<4 {
                        let g0 = metal.gpuSeconds
                        metal.beginBatch(label: "tune")
                        for i in 0..<reps { metal.matmul(views[i % copies], x.slice(), into: y.slice(), batch: 1) }
                        try metal.synchronize()
                        best = min(best, (metal.gpuSeconds - g0) / Double(reps))
                    }
                    line += String(format: "  r%dl%d %5.0f", r, l, Double(bytes) / best / 1e9)
                }
                metal.tuneDecode = nil
                print(line)
            }
        }
    }
}
