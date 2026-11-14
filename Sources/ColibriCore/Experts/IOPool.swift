import Foundation

/// Dedicated `Thread`s that run blocking `pread`s (not Swift concurrency, whose
/// cooperative pool a blocking read starves; not `concurrentPerform`, which belongs to
/// compute). Demand jobs run before prefetches; each class is FIFO. Upstream's PIPE
/// pool in c/colibri.c is the model (8 workers, measured 2.5x one on this SSD).
public final class IOPool: @unchecked Sendable {
    public enum Priority: Sendable { case demand, prefetch }
    public typealias Job = @Sendable () -> Void

    private let cond = NSCondition()
    private var demand: [Job] = []
    private var demandHead = 0
    private var prefetch: [Job] = []
    private var prefetchHead = 0
    private var running = 0
    private var alive = 0
    private var stopping = false
    public let threadCount: Int

    public init(threads: Int) {
        threadCount = max(1, threads)
        alive = threadCount
        for i in 0 ..< threadCount {
            let t = Thread { [self] in self.workerLoop() }
            t.name = "colibri-io-\(i)"
            t.qualityOfService = .userInitiated
            t.start()
        }
    }

    public func submit(_ priority: Priority, _ job: @escaping Job) {
        cond.lock()
        precondition(!stopping, "IOPool.submit after shutdown")
        switch priority {
        case .demand: demand.append(job)
        case .prefetch: prefetch.append(job)
        }
        cond.signal()
        cond.unlock()
    }

    /// Jobs queued and not yet started.
    public var queuedCount: Int {
        cond.lock(); defer { cond.unlock() }
        return (demand.count - demandHead) + (prefetch.count - prefetchHead)
    }

    /// Blocks until no job is queued or running.
    public func waitUntilIdle() {
        cond.lock()
        while running > 0 || demandHead < demand.count || prefetchHead < prefetch.count { cond.wait() }
        cond.unlock()
    }

    /// Stops the workers after the queued jobs finish. Idempotent. Must not be called
    /// from a job.
    public func shutdown() {
        cond.lock()
        stopping = true
        cond.broadcast()
        while alive > 0 { cond.wait() }
        cond.unlock()
    }

    private func next() -> Job? {
        if demandHead < demand.count {
            let j = demand[demandHead]
            demandHead += 1
            if demandHead == demand.count { demand.removeAll(keepingCapacity: true); demandHead = 0 }
            return j
        }
        if prefetchHead < prefetch.count {
            let j = prefetch[prefetchHead]
            prefetchHead += 1
            if prefetchHead == prefetch.count { prefetch.removeAll(keepingCapacity: true); prefetchHead = 0 }
            return j
        }
        return nil
    }

    private func workerLoop() {
        cond.lock()
        while true {
            if let job = next() {
                running += 1
                cond.unlock()
                job()
                cond.lock()
                running -= 1
                cond.broadcast()
                continue
            }
            if stopping { break }
            cond.wait()
        }
        alive -= 1
        cond.broadcast()
        cond.unlock()
    }

    /// Opens `path` for uncached reads (`F_NOCACHE`, readahead off with `F_RDAHEAD 0`),
    /// upstream's `DIRECT=1` (`compat_open_direct`). macOS imposes no alignment on
    /// `F_NOCACHE` reads.
    public static func openUncached(_ path: String) throws -> Int32 {
        try open(path, uncached: true)
    }

    /// Opens `path` read-only; `uncached` sets `F_NOCACHE` and `F_RDAHEAD 0`.
    public static func open(_ path: String, uncached: Bool) throws -> Int32 {
        let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw ColibriError.io(path: path, operation: "open", errno: errno) }
        if uncached {
            if fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0 {
                let e = errno
                close(fd)
                throw ColibriError.io(path: path, operation: "fcntl(F_NOCACHE)", errno: e)
            }
        }
        return fd
    }

    /// `pread` until `count` bytes are read or an error occurs. Retries `EINTR` and short
    /// reads; reading past the end of the file is an error.
    public static func readFully(fd: Int32, into dst: UnsafeMutableRawPointer, count: Int, offset: Int) throws {
        var done = 0
        while done < count {
            let chunk = min(count - done, 1 << 30)  // pread's count is capped at INT_MAX
            let n = pread(fd, dst + done, chunk, off_t(offset + done))
            if n < 0 {
                if errno == EINTR { continue }
                throw ColibriError.io(path: "fd \(fd)", operation: "pread \(count) bytes at \(offset)", errno: errno)
            }
            if n == 0 {
                throw ColibriError.io(path: "fd \(fd)", operation: "pread \(count) bytes at \(offset) (end of file after \(done))", errno: EIO)
            }
            done += n
        }
    }
}
