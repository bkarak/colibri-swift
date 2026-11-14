import Foundation

/// Owns the mapping and the descriptor. Separate from `MappedFile` so the cached
/// `MemoryRegion` can keep the bytes alive without a reference cycle
/// (`MappedFile -> region -> storage`, `MappedFile -> storage`).
final class MappingStorage: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int
    let fd: Int32

    init(base: UnsafeMutableRawPointer, length: Int, fd: Int32) {
        self.base = base
        self.length = length
        self.fd = fd
    }

    deinit {
        munmap(base, length)
        close(fd)
    }
}

/// A whole file mapped read-only (`mmap(PROT_READ, MAP_SHARED)`), with the mapping
/// length rounded up to `hostPageSize` so Metal can wrap it with `bytesNoCopy`.
/// `MAP_SHARED`, not `MAP_PRIVATE`: the mapping is never written, and making a private
/// mapping resident for the GPU cost about 1.5 s per 17 GB on the first command buffer
/// (Qwen3-30B-A3B, file cache warm) against 0.17 s for a shared one.
/// Dense weights are read through this mapping; routed experts in streaming mode are
/// read with `pread` instead (see Experts/), so their pages never count against RSS.
public final class MappedFile: @unchecked Sendable {
    public let path: String
    /// File size in bytes (the mapping may be longer, zero-padded to a page).
    public let fileSize: Int
    public let base: UnsafeRawPointer
    /// Mapping length, a multiple of `hostPageSize`.
    public let mappedLength: Int
    /// A read-only file descriptor kept open for `pread` (expert streaming, prefault).
    public let fd: Int32

    private let storage: MappingStorage
    /// The whole mapping as a weight region (created once).
    public let region: MemoryRegion

    public init(path: String) throws {
        let fd = open(path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw ColibriError.io(path: path, operation: "open", errno: errno) }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            let e = errno
            close(fd)
            throw ColibriError.io(path: path, operation: "fstat", errno: e)
        }
        let size = Int(st.st_size)
        guard size > 0 else {
            close(fd)
            throw ColibriError.invalidGGUF("\(path) is empty")
        }
        // The kernel zero-fills the tail of the last page; rounding to the 16 KiB VM page
        // never maps a page that lies wholly past the end of the file.
        let length = alignUp(size, hostPageSize)
        guard let p = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), p != MAP_FAILED else {
            let e = errno
            close(fd)
            throw ColibriError.io(path: path, operation: "mmap", errno: e)
        }
        self.path = path
        self.fileSize = size
        self.base = UnsafeRawPointer(p)
        self.mappedLength = length
        self.fd = fd
        let storage = MappingStorage(base: p, length: length, fd: fd)
        self.storage = storage
        self.region = MemoryRegion(base: UnsafeRawPointer(p), length: length, kind: .fileMapping,
                                   label: (path as NSString).lastPathComponent, keepAlive: storage)
    }

    /// Hints the kernel to read `range` ahead (`madvise(MADV_WILLNEED)`), so the first
    /// GPU touch of resident weights does not fault from disk inside a kernel.
    public func prefault(_ range: Range<Int>) {
        let lo = max(0, range.lowerBound) & ~(hostPageSize - 1)
        let hi = min(mappedLength, alignUp(max(lo, range.upperBound), hostPageSize))
        guard hi > lo else { return }
        _ = madvise(UnsafeMutableRawPointer(mutating: base + lo), hi - lo, MADV_WILLNEED)
    }

    /// The bytes `[offset, offset + count)` of the file, bounds-checked.
    public func bytes(_ offset: Int, _ count: Int) -> UnsafeRawBufferPointer {
        precondition(offset >= 0 && count >= 0 && offset + count <= fileSize, "MappedFile: range out of file")
        return UnsafeRawBufferPointer(start: base + offset, count: count)
    }
}
