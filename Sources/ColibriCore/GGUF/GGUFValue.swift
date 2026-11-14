// GGUF metadata value model. Format: docs/port/gguf-and-quants.md section 1.1; getter
// semantics follow colibri c/gguf.h (gguf_get_int / gguf_get_float).

/// GGUF metadata value type ids.
public enum GGUFValueType: UInt32, Sendable {
    case u8 = 0, i8 = 1, u16 = 2, i16 = 3, u32 = 4, i32 = 5, f32 = 6, bool = 7
    case string = 8, array = 9, u64 = 10, i64 = 11, f64 = 12
}

/// A decoded metadata value. Arrays of arrays are refused by the reader, as upstream does.
public enum GGUFValue: Sendable, Equatable {
    case u8(UInt8), i8(Int8), u16(UInt16), i16(Int16), u32(UInt32), i32(Int32)
    case u64(UInt64), i64(Int64), f32(Float), f64(Double), bool(Bool)
    case string(String)
    /// Homogeneous array: element type plus values (never `.array`).
    case array(GGUFValueType, [GGUFValue])

    public var type: GGUFValueType {
        switch self {
        case .u8: return .u8
        case .i8: return .i8
        case .u16: return .u16
        case .i16: return .i16
        case .u32: return .u32
        case .i32: return .i32
        case .u64: return .u64
        case .i64: return .i64
        case .f32: return .f32
        case .f64: return .f64
        case .bool: return .bool
        case .string: return .string
        case .array: return .array
        }
    }

    /// Any integer or bool widened to Int64; nil for other kinds.
    public var intValue: Int64? {
        switch self {
        case .u8(let v): return Int64(v)
        case .i8(let v): return Int64(v)
        case .u16(let v): return Int64(v)
        case .i16(let v): return Int64(v)
        case .u32(let v): return Int64(v)
        case .i32(let v): return Int64(v)
        case .u64(let v): return Int64(exactly: v)
        case .i64(let v): return v
        case .bool(let v): return v ? 1 : 0
        default: return nil
        }
    }

    /// F32/F64, falling back to integers; nil for other kinds.
    public var floatValue: Double? {
        switch self {
        case .f32(let v): return Double(v)
        case .f64(let v): return v
        default: return intValue.map(Double.init)
        }
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var arrayValue: [GGUFValue]? {
        if case .array(_, let a) = self { return a }
        return nil
    }
}

/// The key/value metadata of a GGUF file, with upstream's lenient typed getters: a
/// missing key or a key of the wrong kind yields nil, and the caller supplies the default.
public struct GGUFMetadata: Sendable {
    public private(set) var entries: [String: GGUFValue]
    /// Keys in file order (for `coli-swift info`).
    public private(set) var keys: [String]

    public init(entries: [(String, GGUFValue)] = []) {
        self.entries = [:]
        self.keys = []
        for (k, v) in entries { set(k, v) }
    }

    public mutating func set(_ key: String, _ value: GGUFValue) {
        if entries.updateValue(value, forKey: key) == nil { keys.append(key) }
    }

    public subscript(_ key: String) -> GGUFValue? { entries[key] }

    public func int(_ key: String) -> Int? { entries[key]?.intValue.map { Int($0) } }
    public func float(_ key: String) -> Double? { entries[key]?.floatValue }
    public func string(_ key: String) -> String? { entries[key]?.stringValue }
    public func bool(_ key: String) -> Bool? { entries[key]?.intValue.map { $0 != 0 } }
    public func strings(_ key: String) -> [String]? { entries[key]?.arrayValue?.compactMap(\.stringValue) }
    public func ints(_ key: String) -> [Int]? { entries[key]?.arrayValue?.compactMap { $0.intValue.map { Int($0) } } }

    /// `general.architecture`, the registry key.
    public var architecture: String? { string("general.architecture") }
}
