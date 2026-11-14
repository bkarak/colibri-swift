import Foundation

/// Writes a GGUF v3 file. Used to build tiny synthetic models for tests (the only real
/// model on disk is a dense Qwen3-0.6B, so the MoE families are verified on synthetic
/// files) and for round-trip tests of the reader.
public struct GGUFWriter {
    public var alignment: Int
    public private(set) var metadata: GGUFMetadata
    public private(set) var tensors: [(name: String, dims: [Int], type: GGMLType, bytes: [UInt8])]

    public init(alignment: Int = 32) {
        self.alignment = alignment
        self.metadata = GGUFMetadata()
        self.tensors = []
    }

    public mutating func set(_ key: String, _ value: GGUFValue) { metadata.set(key, value) }

    /// Copies every metadata entry of `source` whose key satisfies `include`, in the
    /// source's order. The typical use: `copyMetadata(from: qwen.metadata) { $0.hasPrefix("tokenizer.") }`
    /// gives a synthetic model the real tokenizer.
    public mutating func copyMetadata(from source: GGUFMetadata, where include: (String) -> Bool) {
        for k in source.keys where include(k) {
            if let v = source[k] { metadata.set(k, v) }
        }
    }

    /// Adds a tensor whose `bytes` are already in `type`'s block format.
    /// `bytes.count` must equal `type.rowBytes(dims[0]) * product(dims[1...])`.
    public mutating func addTensor(_ name: String, dims: [Int], type: GGMLType, bytes: [UInt8]) {
        precondition(!dims.isEmpty && dims.count <= 4, "\(name): 1 to 4 dims")
        precondition(dims[0] % type.blockSize == 0, "\(name): ne[0] \(dims[0]) not a multiple of \(type) block")
        let expected = type.rowBytes(dims[0]) * dims.dropFirst().reduce(1, *)
        precondition(bytes.count == expected, "\(name): \(bytes.count) bytes, expected \(expected)")
        precondition(!tensors.contains { $0.name == name }, "\(name): duplicate tensor")
        tensors.append((name, dims, type, bytes))
    }

    /// Quantises `values` (row-major, `ne[0]` fastest) to `type` and adds the tensor.
    public mutating func addTensor(_ name: String, dims: [Int], type: GGMLType, values: [Float]) throws {
        precondition(values.count == dims.reduce(1, *), "\(name): \(values.count) values for dims \(dims)")
        addTensor(name, dims: dims, type: type, bytes: try Quantize.encode(values, as: type))
    }

    /// Serialises header, metadata, tensor infos, padding and data.
    public func serialize() throws -> [UInt8] {
        guard alignment > 0, alignment & (alignment - 1) == 0, alignment <= GGUFLimits.maxAlignment else {
            throw ColibriError.invalidArgument("GGUFWriter: alignment \(alignment) is not a power of two <= 1 MiB")
        }
        var md = metadata
        if alignment != GGUFLimits.defaultAlignment {
            md.set("general.alignment", .u32(UInt32(alignment)))
        } else if let a = md["general.alignment"], a != .u32(32) {
            throw ColibriError.invalidArgument("GGUFWriter: general.alignment \(a) disagrees with alignment 32")
        }

        var out = [UInt8]()
        out.reserveCapacity(1 << 16 + tensors.reduce(0) { $0 + $1.bytes.count + alignment })
        func put<T: FixedWidthInteger>(_ v: T) {
            withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) }
        }
        func putString(_ s: String) {
            let u = Array(s.utf8)
            put(UInt64(u.count))
            out.append(contentsOf: u)
        }
        func putScalar(_ v: GGUFValue) throws {
            switch v {
            case .u8(let x): put(x)
            case .i8(let x): put(x)
            case .u16(let x): put(x)
            case .i16(let x): put(x)
            case .u32(let x): put(x)
            case .i32(let x): put(x)
            case .u64(let x): put(x)
            case .i64(let x): put(x)
            case .f32(let x): put(x.bitPattern)
            case .f64(let x): put(x.bitPattern)
            case .bool(let x): put(UInt8(x ? 1 : 0))
            case .string(let s): putString(s)
            case .array: throw ColibriError.invalidArgument("GGUFWriter: nested arrays are not supported")
            }
        }

        put(UInt32(0x4655_4747))
        put(UInt32(3))
        put(UInt64(tensors.count))
        put(UInt64(md.keys.count))
        for k in md.keys {
            let v = md[k]!
            putString(k)
            put(v.type.rawValue)
            if case .array(let et, let elems) = v {
                guard et != .array else { throw ColibriError.invalidArgument("GGUFWriter: nested array in \(k)") }
                put(et.rawValue)
                put(UInt64(elems.count))
                for e in elems {
                    guard e.type == et else {
                        throw ColibriError.invalidArgument("GGUFWriter: \(k) mixes \(e.type) into an array of \(et)")
                    }
                    try putScalar(e)
                }
            } else {
                try putScalar(v)
            }
        }

        var offset = 0
        var offsets = [Int]()
        for t in tensors {
            offsets.append(offset)
            putString(t.name)
            put(UInt32(t.dims.count))
            for d in t.dims { put(UInt64(d)) }
            put(t.type.rawValue)
            put(UInt64(offset))
            offset = alignUp(offset + t.bytes.count, alignment)
        }
        out.append(contentsOf: repeatElement(0, count: alignUp(out.count, alignment) - out.count))
        let dataStart = out.count
        for (i, t) in tensors.enumerated() {
            precondition(out.count == dataStart + offsets[i])
            out.append(contentsOf: t.bytes)
            out.append(contentsOf: repeatElement(0, count: alignUp(out.count - dataStart, alignment) - (out.count - dataStart)))
        }
        return out
    }

    public func write(to path: String) throws {
        let bytes = try serialize()
        try Data(bytes).write(to: URL(fileURLWithPath: path))
    }
}
