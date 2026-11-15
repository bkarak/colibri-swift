import CColibriKernels
import Darwin
import Foundation

/// Row-chunked parallel loops for the CPU kernels, on the spinning worker pool of
/// CColibriKernels/pool.c (one flat loop per op; chunks are claimed dynamically, so no
/// worker waits on a fixed share). `concurrentPerform` parked its workers between the
/// few hundred short loops of a decode step and paid to wake them every time.
public enum Parallel {
    /// Performance cores (`hw.perflevel0.physicalcpu`), else active processors. The
    /// pool's workers spin between loops; on the efficiency cores they slow every loop
    /// (Qwen3-4B decode: 16 threads half the speed of 12 on a 12P + 4E M4 Max).
    public static let performanceCores: Int = {
        var n: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.physicalcpu", &n, &size, nil, 0) == 0, n > 0 { return Int(n) }
        return ProcessInfo.processInfo.activeProcessorCount
    }()

    /// Calls `body(range)` for disjoint chunks covering `0 ..< count`, in parallel over at
    /// most `threads` workers. Chunks are at least `minChunk` long. Must not be nested
    /// (a nested call runs serially).
    /// `perThread` chunks per worker at most; `align` rounds the chunk length up.
    public static func forChunks(_ count: Int, threads: Int, minChunk: Int = 16, perThread: Int = 4, align: Int = 1,
                                 _ body: (Range<Int>) -> Void) {
        guard count > 0 else { return }
        let chunks = min(max(1, threads * perThread), (count + max(1, minChunk) - 1) / max(1, minChunk))
        if threads <= 1 || chunks <= 1 {
            body(0 ..< count)
            return
        }
        let a = max(1, align)
        let per = ((count + chunks - 1) / chunks + a - 1) / a * a
        let n = (count + per - 1) / per
        run(n, threads: threads) { i in body(i * per ..< min(count, (i + 1) * per)) }
    }

    /// Calls `body(i)` for every `i < count`, in parallel when `threads > 1` and
    /// `count > 1`. For iteration spaces that are already chunked (expert x row-chunk).
    public static func forEach(_ count: Int, threads: Int, _ body: (Int) -> Void) {
        guard count > 0 else { return }
        if threads <= 1 || count == 1 {
            for i in 0..<count { body(i) }
            return
        }
        run(count, threads: threads, body)
    }

    private final class Job {
        let body: (Int) -> Void
        init(_ body: @escaping (Int) -> Void) { self.body = body }
    }

    private static func run(_ n: Int, threads: Int, _ body: (Int) -> Void) {
        withoutActuallyEscaping(body) { escaping in
            let job = Job(escaping)
            withExtendedLifetime(job) {
                colibri_parallel_for(Int32(n), Int32(threads), { ctx, i in
                    Unmanaged<Job>.fromOpaque(ctx!).takeUnretainedValue().body(Int(i))
                }, Unmanaged.passUnretained(job).toOpaque())
            }
        }
    }
}
