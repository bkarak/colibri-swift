import Foundation

/// One file of a mapping: its bytes `[0, size)` sit at `[start, start + size)` of the
/// mapping. A split model (`-00001-of-0000N.gguf`) has one segment per shard.
public struct FileSegment: Sendable, Equatable {
    public let path: String
    /// Where the file's byte 0 sits in the mapping (a multiple of `hostPageSize`).
    public let start: Int
    /// File size in bytes.
    public let size: Int

    public init(path: String, start: Int, size: Int) {
        self.path = path
        self.start = start
        self.size = size
    }

    public var end: Int { start + size }
}

/// Owns the mapping and the descriptors. Separate from `MappedFile` so the cached
/// `MemoryRegion` can keep the bytes alive without a reference cycle
/// (`MappedFile -> region -> storage`, `MappedFile -> storage`).
final class MappingStorage: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int
    let fds: [Int32]

    init(base: UnsafeMutableRawPointer, length: Int, fds: [Int32]) {
        self.base = base
        self.length = length
        self.fds = fds
    }

    deinit {
        munmap(base, length)
        for fd in fds { close(fd) }
    }
}

/// One or more files mapped read-only (`mmap(PROT_READ, MAP_SHARED)`) back to back in one
/// contiguous range of address space, each starting on a `hostPageSize` boundary and
/// padded to one, so Metal can wrap any page-aligned window with `bytesNoCopy` and a
/// split model reads like a single file. The range is reserved first (`PROT_NONE`) and
/// each file is mapped into it with `MAP_FIXED`.
/// `MAP_SHARED`, not `MAP_PRIVATE`: the mapping is never written, and making a private
/// mapping resident for the GPU cost about 1.5 s per 17 GB on the first command buffer
/// (Qwen3-30B-A3B, file cache warm) against 0.17 s for a shared one.
/// Dense weights are read through this mapping; routed experts in streaming mode are
/// read with `pread` instead (see Experts/), so their pages never count against RSS.
public final class MappedFile: @unchecked Sendable {
    /// The first file's path (the model's name for history files and messages).
    public let path: String
    /// The files, in mapping order.
    public let segments: [FileSegment]
    /// Bytes from the start of the mapping to the end of the last file (the file size for
    /// a single file; shards add their page padding).
    public let fileSize: Int
    public let base: UnsafeRawPointer
    /// Mapping length, a multiple of `hostPageSize`.
    public let mappedLength: Int

    private let storage: MappingStorage
    /// The whole mapping as a weight region (created once).
    public let region: MemoryRegion

    public convenience init(path: String) throws {
        try self.init(paths: [path])
    }

    public init(paths: [String]) throws {
        precondition(!paths.isEmpty, "MappedFile: no paths")
        var fds: [Int32] = []
        var segments: [FileSegment] = []
        var total = 0
        func closeAll() { for fd in fds { close(fd) } }
        for path in paths {
            let fd = open(path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else {
                let e = errno
                closeAll()
                throw ColibriError.io(path: path, operation: "open", errno: e)
            }
            fds.append(fd)
            var st = stat()
            guard fstat(fd, &st) == 0 else {
                let e = errno
                closeAll()
                throw ColibriError.io(path: path, operation: "fstat", errno: e)
            }
            let size = Int(st.st_size)
            guard size > 0 else {
                closeAll()
                throw ColibriError.invalidGGUF("\(path) is empty")
            }
            segments.append(FileSegment(path: path, start: total, size: size))
            // The kernel zero-fills the tail of a file's last page; rounding to the 16 KiB VM
            // page never maps a page that lies wholly past the end of the file.
            total += alignUp(size, hostPageSize)
        }
        guard let reserved = mmap(nil, total, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0), reserved != MAP_FAILED else {
            let e = errno
            closeAll()
            throw ColibriError.io(path: paths[0], operation: "mmap (reserve \(total) bytes)", errno: e)
        }
        for (s, fd) in zip(segments, fds) {
            let want = reserved + s.start
            let p = mmap(want, alignUp(s.size, hostPageSize), PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0)
            guard p == want else {
                let e = errno
                munmap(reserved, total)
                closeAll()
                throw ColibriError.io(path: s.path, operation: "mmap", errno: e)
            }
        }
        self.path = paths[0]
        self.segments = segments
        self.fileSize = segments.last!.end
        self.base = UnsafeRawPointer(reserved)
        self.mappedLength = total
        let storage = MappingStorage(base: reserved, length: total, fds: fds)
        self.storage = storage
        let label = (paths[0] as NSString).lastPathComponent + (paths.count > 1 ? " (+\(paths.count - 1) shards)" : "")
        self.region = MemoryRegion(base: UnsafeRawPointer(reserved), length: total, kind: .fileMapping,
                                   label: label, keepAlive: storage)
    }

    /// Hints the kernel to read `range` ahead (`madvise(MADV_WILLNEED)`), so the first
    /// GPU touch of resident weights does not fault from disk inside a kernel.
    public func prefault(_ range: Range<Int>) {
        let lo = max(0, range.lowerBound) & ~(hostPageSize - 1)
        let hi = min(mappedLength, alignUp(max(lo, range.upperBound), hostPageSize))
        guard hi > lo else { return }
        _ = madvise(UnsafeMutableRawPointer(mutating: base + lo), hi - lo, MADV_WILLNEED)
    }

    /// The bytes `[offset, offset + count)` of the mapping, bounds-checked against the file
    /// that holds them.
    public func bytes(_ offset: Int, _ count: Int) -> UnsafeRawBufferPointer {
        precondition(offset >= 0 && count >= 0 && segments.contains { offset >= $0.start && offset + count <= $0.end },
                     "MappedFile: range out of file")
        return UnsafeRawBufferPointer(start: base + offset, count: count)
    }
}
