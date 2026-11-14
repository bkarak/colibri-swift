import Foundation

// GGUF reader. Ported in behaviour from colibri c/gguf.h (parse, sanity limits, getters);
// unlike upstream it maps the file instead of pread-ing tensors. Layout and limits:
// docs/port/gguf-and-quants.md section 1.

/// One tensor-info entry.
public struct GGUFTensorInfo: Sendable, Equatable {
    public var name: String
    /// ggml dims, `ne[0]` first (the contiguous row length). 1...4 entries.
    public var dims: [Int]
    public var type: GGMLType
    /// Offset relative to the data section, a multiple of the file's alignment.
    public var relativeOffset: Int
    /// Byte offset in the model's mapping: `dataStart + relativeOffset` in a single file;
    /// in a split model, the shard's `FileSegment.start` plus that within the shard.
    public var fileOffset: Int
    /// Computed byte size.
    public var byteCount: Int

    public init(name: String, dims: [Int], type: GGMLType, relativeOffset: Int, fileOffset: Int, byteCount: Int) {
        self.name = name
        self.dims = dims
        self.type = type
        self.relativeOffset = relativeOffset
        self.fileOffset = fileOffset
        self.byteCount = byteCount
    }

    /// `dims[i]`, or 1 past the stored rank.
    public func ne(_ i: Int) -> Int { i < dims.count ? dims[i] : 1 }
    public var elementCount: Int { dims.reduce(1, *) }
}

/// Sanity limits for untrusted headers (upstream gguf.h).
enum GGUFLimits {
    static let maxTensors = 1 << 24
    static let maxKeys = 1 << 20
    static let maxString = 1 << 30
    static let maxDim = 1 << 40
    static let maxAlignment = 1 << 20
    static let defaultAlignment = 32
}

/// Bounds-checked little-endian cursor over the mapped header.
struct GGUFByteReader {
    let base: UnsafeRawPointer
    let size: Int
    var pos: Int = 0

    init(base: UnsafeRawPointer, size: Int) {
        self.base = base
        self.size = size
    }

    mutating func need(_ n: Int) throws {
        guard n >= 0, n <= size - pos else {
            throw ColibriError.invalidGGUF("truncated header at byte \(pos) (need \(n) more bytes)")
        }
    }

    mutating func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
        try need(MemoryLayout<T>.size)
        let v = T(littleEndian: base.loadUnaligned(fromByteOffset: pos, as: T.self))
        pos += MemoryLayout<T>.size
        return v
    }

    mutating func readF32() throws -> Float { Float(bitPattern: try read(UInt32.self)) }
    mutating func readF64() throws -> Double { Double(bitPattern: try read(UInt64.self)) }

    mutating func readString() throws -> String {
        let n64 = try read(UInt64.self)
        guard n64 <= UInt64(GGUFLimits.maxString), Int(n64) <= size - pos else {
            throw ColibriError.invalidGGUF("string of \(n64) bytes at \(pos) exceeds the file")
        }
        let n = Int(n64)
        let s = String(decoding: UnsafeRawBufferPointer(start: base + pos, count: n), as: UTF8.self)
        pos += n
        return s
    }

    mutating func readScalar(_ t: GGUFValueType) throws -> GGUFValue {
        switch t {
        case .u8: return .u8(try read(UInt8.self))
        case .i8: return .i8(try read(Int8.self))
        case .u16: return .u16(try read(UInt16.self))
        case .i16: return .i16(try read(Int16.self))
        case .u32: return .u32(try read(UInt32.self))
        case .i32: return .i32(try read(Int32.self))
        case .f32: return .f32(try readF32())
        case .bool: return .bool(try read(UInt8.self) != 0)
        case .string: return .string(try readString())
        case .u64: return .u64(try read(UInt64.self))
        case .i64: return .i64(try read(Int64.self))
        case .f64: return .f64(try readF64())
        case .array: throw ColibriError.invalidGGUF("nested arrays are not supported")
        }
    }

    mutating func readValue(_ t: GGUFValueType) throws -> GGUFValue {
        guard t == .array else { return try readScalar(t) }
        let et32 = try read(UInt32.self)
        guard let et = GGUFValueType(rawValue: et32) else {
            throw ColibriError.invalidGGUF("unknown array element type \(et32) at \(pos)")
        }
        guard et != .array else { throw ColibriError.invalidGGUF("nested arrays are not supported") }
        let count64 = try read(UInt64.self)
        // Every element takes at least one byte (a string at least eight).
        let minElement = et == .string ? 8 : GGUFByteReader.scalarSize(et)
        guard count64 <= UInt64(size - pos) / UInt64(minElement) else {
            throw ColibriError.invalidGGUF("array of \(count64) elements at \(pos) exceeds the file")
        }
        let count = Int(count64)
        var out = [GGUFValue]()
        out.reserveCapacity(count)
        for _ in 0..<count { out.append(try readScalar(et)) }
        return .array(et, out)
    }

    static func scalarSize(_ t: GGUFValueType) -> Int {
        switch t {
        case .u8, .i8, .bool: return 1
        case .u16, .i16: return 2
        case .u32, .i32, .f32: return 4
        case .u64, .i64, .f64: return 8
        case .string: return 8
        case .array: return 0
        }
    }
}

/// A parsed GGUF file: header, metadata, tensor table and the mapping that holds the
/// bytes. Treat every header field as untrusted input: the reader enforces upstream's
/// limits (at most 2^24 tensors, 2^20 keys, 1 GiB strings, no nested arrays, dims in
/// 1...2^40, tensor bytes inside the file, power-of-two alignment <= 1 MiB).
public final class GGUFFile: @unchecked Sendable {
    public let version: UInt32
    public let metadata: GGUFMetadata
    /// Tensors in file order.
    public let tensors: [GGUFTensorInfo]
    public let alignment: Int
    /// Offset of the data section (of the first shard, for a split model).
    public let dataStart: Int
    public let file: MappedFile

    private let index: [String: Int]

    /// Opens and parses `path`. A split model (`<name>-00001-of-0000N.gguf`, llama.cpp's
    /// `gguf-split` layout: `split.count` > 1 in every shard) is opened whole from any of
    /// its shards: every shard is mapped back to back (`MappedFile(paths:)`), metadata comes
    /// from the first, and tensor offsets are offsets into that combined mapping.
    public init(path: String) throws {
        let first = try MappedFile(path: path)
        let h0 = try Self.parseHeader(base: first.base, size: first.fileSize, path: path)
        let count = h0.metadata.int("split.count") ?? 1
        let file: MappedFile
        var headers: [Header]
        if count <= 1 {
            file = first
            headers = [h0]
        } else {
            let paths = try Self.splitPaths(path: path, count: count)
            file = try MappedFile(paths: paths)
            headers = []
            for (i, seg) in file.segments.enumerated() {
                let h = try Self.parseHeader(base: file.base + seg.start, size: seg.size, path: seg.path)
                guard h.metadata.int("split.count") == count, (h.metadata.int("split.no") ?? -1) == i else {
                    throw ColibriError.invalidGGUF("\(seg.path): split.no \(h.metadata.int("split.no").map(String.init) ?? "missing") / split.count \(h.metadata.int("split.count").map(String.init) ?? "missing"); expected \(i) / \(count)")
                }
                headers.append(h)
            }
        }

        var tensors = [GGUFTensorInfo]()
        var index = [String: Int]()
        for (h, seg) in zip(headers, file.segments) {
            for t in h.tensors {
                guard index[t.name] == nil else {
                    throw ColibriError.invalidGGUF("\(seg.path): duplicate tensor name \(t.name)")
                }
                index[t.name] = tensors.count
                var t = t
                t.fileOffset += seg.start
                tensors.append(t)
            }
        }
        let h = headers[0]  // the first shard holds the metadata, whichever shard was opened
        if count > 1, let expected = h.metadata.int("split.tensors.count"), expected != tensors.count {
            throw ColibriError.invalidGGUF("\(path): split.tensors.count is \(expected), the \(count) shards hold \(tensors.count)")
        }

        self.version = h.version
        self.metadata = h.metadata
        self.tensors = tensors
        self.alignment = h.alignment
        self.dataStart = h.dataStart
        self.file = file
        self.index = index
    }

    /// The shard paths of a split model, from the name of any shard
    /// (`<prefix>-%05d-of-%05d.gguf`, llama.cpp's `llama_split_path`).
    static func splitPaths(path: String, count: Int) throws -> [String] {
        let suffix = String(format: "-of-%05d.gguf", count)
        guard path.hasSuffix(suffix) else {
            throw ColibriError.invalidGGUF("\(path) has split.count \(count) but is not named <name>-NNNNN\(suffix)")
        }
        let stem = path.dropLast(suffix.count)
        guard stem.count > 6, stem.suffix(6).first == "-", stem.suffix(5).allSatisfy(\.isNumber) else {
            throw ColibriError.invalidGGUF("\(path) has split.count \(count) but is not named <name>-NNNNN\(suffix)")
        }
        let prefix = stem.dropLast(6)
        return (1...count).map { String(prefix) + String(format: "-%05d-of-%05d.gguf", $0, count) }
    }

    /// One file's header: metadata and its tensor table, offsets relative to that file.
    struct Header {
        var version: UInt32
        var metadata: GGUFMetadata
        var alignment: Int
        var dataStart: Int
        var tensors: [GGUFTensorInfo]
    }

    /// Parses the GGUF header of the `size` bytes at `base` (one file or one shard).
    static func parseHeader(base: UnsafeRawPointer, size: Int, path: String) throws -> Header {
        var r = GGUFByteReader(base: base, size: size)

        let magic = try r.read(UInt32.self)
        guard magic == 0x4655_4747 else { // "GGUF" little-endian
            throw ColibriError.invalidGGUF("\(path): bad magic 0x\(String(magic, radix: 16))")
        }
        let version = try r.read(UInt32.self)
        guard version == 2 || version == 3 else {
            throw ColibriError.unsupported("\(path): GGUF version \(version) (only 2 and 3)")
        }
        let nTensors64 = try r.read(UInt64.self)
        let nKV64 = try r.read(UInt64.self)
        guard nTensors64 <= UInt64(GGUFLimits.maxTensors) else {
            throw ColibriError.invalidGGUF("\(path): \(nTensors64) tensors exceeds the limit")
        }
        guard nKV64 <= UInt64(GGUFLimits.maxKeys) else {
            throw ColibriError.invalidGGUF("\(path): \(nKV64) metadata keys exceeds the limit")
        }

        var md = GGUFMetadata()
        for _ in 0..<Int(nKV64) {
            let key = try r.readString()
            let t32 = try r.read(UInt32.self)
            guard let t = GGUFValueType(rawValue: t32) else {
                throw ColibriError.invalidGGUF("\(path): key \(key) has unknown value type \(t32)")
            }
            md.set(key, try r.readValue(t))
        }

        var alignment = GGUFLimits.defaultAlignment
        if case .u32(let a)? = md["general.alignment"] {
            let a = Int(a)
            guard a > 0, a & (a - 1) == 0, a <= GGUFLimits.maxAlignment else {
                throw ColibriError.invalidGGUF("\(path): general.alignment \(a) is not a power of two <= 1 MiB")
            }
            alignment = a
        }

        struct Raw { var name: String; var dims: [Int]; var type: GGMLType; var offset: Int }
        var raws = [Raw]()
        raws.reserveCapacity(Int(nTensors64))
        var names = Set<String>()
        for _ in 0..<Int(nTensors64) {
            let name = try r.readString()
            let nd = try r.read(UInt32.self)
            guard nd >= 1 && nd <= 4 else {
                throw ColibriError.invalidGGUF("\(path): tensor \(name) has \(nd) dims")
            }
            var dims = [Int]()
            var elements = 1
            for _ in 0..<Int(nd) {
                let d = try r.read(UInt64.self)
                guard d >= 1 && d <= UInt64(GGUFLimits.maxDim) else {
                    throw ColibriError.invalidGGUF("\(path): tensor \(name) has dim \(d)")
                }
                let (p, o) = elements.multipliedReportingOverflow(by: Int(d))
                guard !o else {
                    throw ColibriError.invalidGGUF("\(path): tensor \(name) element count overflows")
                }
                elements = p
                dims.append(Int(d))
            }
            let t32 = try r.read(UInt32.self)
            guard let type = GGMLType(rawValue: t32) else {
                throw ColibriError.unsupported("\(path): tensor \(name) has unknown ggml type \(t32)")
            }
            let off = try r.read(UInt64.self)
            guard off <= UInt64(size), Int(off) % alignment == 0 else {
                throw ColibriError.invalidGGUF("\(path): tensor \(name) offset \(off) is out of range or unaligned")
            }
            guard dims[0] % type.blockSize == 0 else {
                throw ColibriError.invalidGGUF("\(path): tensor \(name) row of \(dims[0]) is not a multiple of \(type)'s block \(type.blockSize)")
            }
            guard names.insert(name).inserted else {
                throw ColibriError.invalidGGUF("\(path): duplicate tensor name \(name)")
            }
            raws.append(Raw(name: name, dims: dims, type: type, offset: Int(off)))
        }

        let dataStart = alignUp(r.pos, alignment)
        var tensors = [GGUFTensorInfo]()
        tensors.reserveCapacity(raws.count)
        for t in raws {
            let rowBytes = t.type.rowBytes(t.dims[0])
            var bytes = rowBytes
            for d in t.dims.dropFirst() {
                let (p, o) = bytes.multipliedReportingOverflow(by: d)
                guard !o else { throw ColibriError.invalidGGUF("\(path): tensor \(t.name) size overflows") }
                bytes = p
            }
            let abs = dataStart + t.offset
            guard abs <= size, bytes <= size - abs else {
                throw ColibriError.invalidGGUF("\(path): tensor \(t.name) [\(abs), +\(bytes)) leaves the file (\(size) bytes)")
            }
            tensors.append(GGUFTensorInfo(name: t.name, dims: t.dims, type: t.type, relativeOffset: t.offset,
                                          fileOffset: abs, byteCount: bytes))
        }
        return Header(version: version, metadata: md, alignment: alignment, dataStart: dataStart, tensors: tensors)
    }

    /// Tensor by name.
    public func tensor(_ name: String) -> GGUFTensorInfo? {
        index[name].map { tensors[$0] }
    }

    /// The raw bytes of a tensor (inside the mapping).
    public func data(_ info: GGUFTensorInfo) -> UnsafeRawBufferPointer {
        file.bytes(info.fileOffset, info.byteCount)
    }

    /// The values of an F32 tensor (small tensors: norms, scales). Other float types
    /// and block types are dequantised.
    public func floats(_ name: String) throws -> [Float] {
        guard let t = tensor(name) else { throw ColibriError.unsupported("missing tensor \(name)") }
        guard Dequant.supports(t.type) else {
            throw ColibriError.unsupported("tensor \(name) has type \(t.type), which has no decoder")
        }
        var out = [Float](repeating: 0, count: t.elementCount)
        out.withUnsafeMutableBufferPointer { o in
            Dequant.dequantizeRow(t.type, file.base + t.fileOffset, into: o.baseAddress!, count: t.elementCount)
        }
        return out
    }

    /// The `<stem>.scale` companion of `<stem>.weight`, if present.
    private func scaleTensorName(_ name: String) -> String {
        name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) + ".scale" : name + ".scale"
    }

    /// A weight view of a 1-D or 2-D tensor (`cols = ne[0]`, `rows = ne[1]`), applying
    /// the optional `<name>.scale` companion tensor. Throws if absent.
    public func weight(_ name: String) throws -> WeightView {
        guard let t = tensor(name) else { throw ColibriError.unsupported("missing tensor \(name)") }
        guard t.ne(2) == 1 && t.ne(3) == 1 else {
            throw ColibriError.unsupported("tensor \(name) has dims \(t.dims); expected 1-D or 2-D")
        }
        var scale: Float = 1
        let sname = scaleTensorName(name)
        if tensor(sname) != nil {
            let s = try floats(sname)
            guard s.count == 1 else {
                throw ColibriError.invalidGGUF("\(sname) has \(s.count) values; expected 1")
            }
            scale = s[0]
        }
        return WeightView(region: file.region, offset: t.fileOffset, type: t.type,
                          cols: t.ne(0), rows: t.ne(1), scale: scale)
    }

    /// A view of a 3-D fused expert tensor `[cols, rows, n]` as `n` stacked matrices.
    public func stackedWeights(_ name: String) throws -> StackedWeights {
        guard let t = tensor(name) else { throw ColibriError.unsupported("missing tensor \(name)") }
        guard t.dims.count == 3 || (t.dims.count == 4 && t.ne(3) == 1) else {
            throw ColibriError.unsupported("tensor \(name) has dims \(t.dims); expected [cols, rows, experts]")
        }
        let stride = t.type.rowBytes(t.ne(0)) * t.ne(1)
        var scales: [Float]? = nil
        let sname = scaleTensorName(name)
        if tensor(sname) != nil {
            let s = try floats(sname)
            guard s.count == t.ne(2) else {
                throw ColibriError.invalidGGUF("\(sname) has \(s.count) values; expected \(t.ne(2))")
            }
            scales = s
        }
        return StackedWeights(region: file.region, baseOffset: t.fileOffset, stride: stride, count: t.ne(2),
                              type: t.type, cols: t.ne(0), rows: t.ne(1), scales: scales)
    }
}
