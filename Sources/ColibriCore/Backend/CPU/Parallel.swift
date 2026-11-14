import Foundation

/// Row-chunked parallel loops for the CPU kernels. One flat `concurrentPerform` per op;
/// chunks are small enough that the performance cores take work from the efficiency
/// cores, so no barrier waits for the slowest core.
public enum Parallel {
    /// Calls `body(range)` for disjoint chunks covering `0 ..< count`, in parallel over at
    /// most `threads` workers. Chunks are at least `minChunk` long. Must not be nested.
    public static func forChunks(_ count: Int, threads: Int, minChunk: Int = 16,
                                 _ body: (Range<Int>) -> Void) {
        guard count > 0 else { return }
        let chunks = min(max(1, threads * 4), (count + max(1, minChunk) - 1) / max(1, minChunk))
        if threads <= 1 || chunks <= 1 {
            body(0 ..< count)
            return
        }
        let per = (count + chunks - 1) / chunks
        let n = (count + per - 1) / per
        DispatchQueue.concurrentPerform(iterations: n) { i in
            body(i * per ..< min(count, (i + 1) * per))
        }
    }

    /// Calls `body(i)` for every `i < count`, in parallel when `threads > 1` and
    /// `count > 1`. For iteration spaces that are already chunked (expert x row-chunk).
    public static func forEach(_ count: Int, threads: Int, _ body: (Int) -> Void) {
        guard count > 0 else { return }
        if threads <= 1 || count == 1 {
            for i in 0..<count { body(i) }
            return
        }
        DispatchQueue.concurrentPerform(iterations: count, execute: body)
    }
}
