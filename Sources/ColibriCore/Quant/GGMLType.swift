// Type ids and block geometry ported from colibri c/ggml_quants.h (Apache-2.0), which
// follows ggml's own table. See docs/port/gguf-and-quants.md section 3.

/// A ggml tensor element type, as stored (u32) in a GGUF tensor info.
public enum GGMLType: UInt32, Sendable, CaseIterable, CustomStringConvertible {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_K = 10
    case q3_K = 11
    case q4_K = 12
    case q5_K = 13
    case q6_K = 14
    case q8_K = 15
    case iq2_xxs = 16
    case iq2_xs = 17
    case iq3_xxs = 18
    case iq1_s = 19
    case iq4_nl = 20
    case iq3_s = 21
    case iq2_s = 22
    case iq4_xs = 23
    case i8 = 24
    case i16 = 25
    case i32 = 26
    case i64 = 27
    case f64 = 28
    case iq1_m = 29
    case bf16 = 30
    case tq1_0 = 34
    case tq2_0 = 35
    case mxfp4 = 39
    case nvfp4 = 40
    case q1_0 = 41
    case q2_0 = 42

    /// Elements per block (1 for plain scalar types).
    public var blockSize: Int {
        switch self {
        case .f32, .f16, .bf16, .i8, .i16, .i32, .i64, .f64: return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl, .mxfp4: return 32
        case .nvfp4, .q2_0: return 64
        case .q1_0: return 128
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .q8_K,
             .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq3_s, .iq2_s, .iq4_xs, .iq1_m,
             .tq1_0, .tq2_0: return 256
        }
    }

    /// Bytes per block.
    public var typeSize: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q4_0: return 18
        case .q4_1: return 20
        case .q5_0: return 22
        case .q5_1: return 24
        case .q8_0: return 34
        case .q8_1: return 36
        case .q2_K: return 84
        case .q3_K: return 110
        case .q4_K: return 144
        case .q5_K: return 176
        case .q6_K: return 210
        case .q8_K: return 292
        case .iq2_xxs: return 66
        case .iq2_xs: return 74
        case .iq3_xxs: return 98
        case .iq1_s: return 50
        case .iq4_nl: return 18
        case .iq3_s: return 110
        case .iq2_s: return 82
        case .iq4_xs: return 136
        case .i8: return 1
        case .i16: return 2
        case .i32: return 4
        case .i64, .f64: return 8
        case .iq1_m: return 56
        case .tq1_0: return 54
        case .tq2_0: return 66
        case .mxfp4: return 17
        case .nvfp4: return 36
        case .q1_0, .q2_0: return 18
        }
    }

    /// Bytes of one row of `count` elements. `count` must be a multiple of `blockSize`
    /// (the GGUF reader refuses tensors where it is not).
    public func rowBytes(_ count: Int) -> Int {
        precondition(count % blockSize == 0, "\(self): row of \(count) is not a multiple of \(blockSize)")
        return count / blockSize * typeSize
    }

    /// Types the first port decodes ("now" in docs/port/gguf-and-quants.md section 3).
    /// Everything else is "later": the GGUF reader accepts the file, and a model refuses
    /// to load when it needs a weight of an undecodable type.
    public static let firstPortWeightTypes: Set<GGMLType> = [
        .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0,
        .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_nl, .iq4_xs,
    ]

    /// ggml's name for the type ("Q8_0", "Q4_K", ...).
    public var description: String { Self.names[self] ?? "type\(rawValue)" }

    private static let names: [GGMLType: String] = [
        .f32: "F32", .f16: "F16", .bf16: "BF16", .f64: "F64", .i8: "I8", .i16: "I16", .i32: "I32", .i64: "I64",
        .q4_0: "Q4_0", .q4_1: "Q4_1", .q5_0: "Q5_0", .q5_1: "Q5_1", .q8_0: "Q8_0", .q8_1: "Q8_1",
        .q2_K: "Q2_K", .q3_K: "Q3_K", .q4_K: "Q4_K", .q5_K: "Q5_K", .q6_K: "Q6_K", .q8_K: "Q8_K",
        .iq2_xxs: "IQ2_XXS", .iq2_xs: "IQ2_XS", .iq3_xxs: "IQ3_XXS", .iq1_s: "IQ1_S",
        .iq4_nl: "IQ4_NL", .iq3_s: "IQ3_S", .iq2_s: "IQ2_S", .iq4_xs: "IQ4_XS", .iq1_m: "IQ1_M",
        .tq1_0: "TQ1_0", .tq2_0: "TQ2_0", .mxfp4: "MXFP4", .nvfp4: "NVFP4", .q1_0: "Q1_0", .q2_0: "Q2_0",
    ]
}
