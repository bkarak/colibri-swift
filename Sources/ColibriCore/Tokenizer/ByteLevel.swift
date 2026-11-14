// Ported from colibri c/qwen36.c build_byte_sym (GPT-2's bytes_to_unicode).

/// GPT-2 ByteLevel byte <-> code point map: bytes 33...126, 161...172 and 174...255 map to
/// themselves, the other 68 bytes, in increasing order, to U+0100, U+0101, ...
/// (space -> U+0120 "Ġ", newline -> U+010A "Ċ").
public enum ByteLevel {
    /// byte -> code point.
    private static let forward: [UInt32] = {
        var table = [UInt32](repeating: 0, count: 256)
        var next: UInt32 = 256
        for b in 0..<256 {
            if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) {
                table[b] = UInt32(b)
            } else {
                table[b] = next
                next += 1
            }
        }
        return table
    }()

    /// code point (< 324) -> byte, or -1.
    private static let backward: [Int16] = {
        var table = [Int16](repeating: -1, count: 324)
        for b in 0..<256 { table[Int(forward[b])] = Int16(b) }
        return table
    }()

    /// The code point byte `b` is spelled with in vocabulary pieces.
    public static func scalar(for byte: UInt8) -> Unicode.Scalar {
        Unicode.Scalar(forward[Int(byte)])!
    }

    /// The byte a piece code point stands for, or nil if it is not in the map.
    public static func byte(for scalar: Unicode.Scalar) -> UInt8? {
        let v = Int(scalar.value)
        guard v < backward.count else { return nil }
        let b = backward[v]
        return b < 0 ? nil : UInt8(b)
    }

    /// The byte-level spelling of raw bytes (what a vocabulary piece looks like).
    public static func encode<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        var s = String.UnicodeScalarView()
        for b in bytes { s.append(scalar(for: b)) }
        return String(s)
    }

    /// The raw bytes a byte-level piece stands for. A code point outside the map (not
    /// expected in a byte-level vocabulary) is kept as its own UTF-8.
    public static func decode(_ piece: String) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(piece.utf8.count)
        for u in piece.unicodeScalars {
            if let b = byte(for: u) {
                out.append(b)
            } else {
                out.append(contentsOf: String(u).utf8)
            }
        }
        return out
    }
}
