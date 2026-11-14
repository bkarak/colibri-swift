import Foundation

// Ported from colibri c/route_trace.h (Apache-2.0): the `.coli_usage` expert history.
// Only the history half is ported; the ROUTE_TRACE stream is not.

/// Per-(layer, expert) routing counts persisted as `.coli_usage`, byte-compatible with
/// colibri c/route_trace.h: header records `-1 <layers> <experts>` and
/// `-2 1 <FNV-1a32(engine name)>`, then `<layer> <expert> <count>` lines; exactly three
/// numeric fields per record; dense layers have no row; atomic save (`.tmp` + rename);
/// a history from another engine is refused unless the user named the file.
///
/// Thread-safe: every method takes an internal lock.
public final class RoutingHistory: @unchecked Sendable {
    public let layers: Int
    public let experts: Int
    public let engineName: String
    /// Layers that have routed experts (others are dense and reject records).
    public let moeLayers: Set<Int>

    /// `RT_FORMAT_VERSION`.
    public static let formatVersion = 1
    /// inkling's dense binary layout, `uint32 {magic, layers, experts}` then counts.
    static let iku1Magic: UInt32 = 0x3155_4B49

    private let lock = NSLock()
    /// `[layer][expert]`; empty rows for dense layers.
    private var counts: [[UInt32]]

    public init(layers: Int, experts: Int, engineName: String, moeLayers: Set<Int>) {
        self.layers = layers
        self.experts = experts
        self.engineName = engineName
        self.moeLayers = moeLayers
        self.counts = (0 ..< max(layers, 0)).map { moeLayers.contains($0) ? [UInt32](repeating: 0, count: max(experts, 0)) : [] }
    }

    /// FNV-1a 32-bit of `name`, the engine id in the `-2` header.
    public static func engineID(_ name: String) -> UInt32 {
        var h: UInt32 = 0x811C_9DC5
        for b in name.utf8 {
            h ^= UInt32(b)
            h = h &* 0x0100_0193
        }
        return h
    }

    /// Engines upstream writes this format for, so a mismatch can name the writer.
    public static let knownEngines = ["glm_moe_dsa", "inkling", "olmoe", "kimi_k3", "deepseek_v4", "qwen38",
                                      "glm53", "qwen36", "qwen3moe", "qwen3"]

    static func engineName(of id: UInt32) -> String? { knownEngines.first { engineID($0) == id } }

    /// Counts one routing decision per entry (duplicates count again, as upstream's
    /// `rt_count`). Ids out of range and dense layers are ignored.
    public func record(layer: Int, experts ids: [Int32]) {
        lock.lock(); defer { lock.unlock() }
        guard layer >= 0, layer < counts.count, !counts[layer].isEmpty else { return }
        for id in ids where id >= 0 && Int(id) < experts {
            let (v, o) = counts[layer][Int(id)].addingReportingOverflow(1)
            counts[layer][Int(id)] = o ? .max : v
        }
    }

    public func count(layer: Int, expert: Int) -> UInt32 {
        lock.lock(); defer { lock.unlock() }
        guard layer >= 0, layer < counts.count, expert >= 0, expert < counts[layer].count else { return 0 }
        return counts[layer][expert]
    }

    /// A copy of one layer's counters (empty for a dense layer).
    public func counters(layer: Int) -> [UInt32] {
        lock.lock(); defer { lock.unlock() }
        guard layer >= 0, layer < counts.count else { return [] }
        return counts[layer]
    }

    /// Sum of every counter.
    public var total: UInt64 {
        lock.lock(); defer { lock.unlock() }
        return counts.reduce(0) { $0 + $1.reduce(0) { $0 + UInt64($1) } }
    }

    public var isEmpty: Bool { total == 0 }

    /// The `n` most used experts of `layer`, most used first (ties: lower id). Experts
    /// never routed are not returned, as upstream's `pin_hot_experts` (`freq > 0`).
    public func top(layer: Int, _ n: Int) -> [Int32] {
        let row = counters(layer: layer)
        guard n > 0, !row.isEmpty else { return [] }
        let ranked = row.indices.filter { row[$0] > 0 }.sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }
        return ranked.prefix(n).map { Int32($0) }
    }

    // MARK: Load

    /// Loads `path` and adds its counts to the counters (upstream accumulates). Returns
    /// the number of selections accepted. `explicit` (the user named the file) overrides
    /// an engine mismatch; dimension and version mismatches are always refused. A refused
    /// file adds nothing. Accepts the three layouts upstream reads: headered text, legacy
    /// headerless text, and (only when explicit, or for the inkling engine) IKU1.
    @discardableResult
    public func load(path: String, explicit: Bool) throws -> UInt64 {
        guard let data = FileManager.default.contents(atPath: path) else {
            let e: Int32 = access(path, R_OK) == 0 ? EIO : errno
            throw ColibriError.io(path: path, operation: "read history", errno: e)
        }
        var staged: [(Int, Int, UInt32)] = []
        if data.count >= 4, readLE32(data, 0) == Self.iku1Magic {
            try parseIKU1(data, path: path, explicit: explicit, into: &staged)
        } else {
            try parseText(data, path: path, explicit: explicit, into: &staged)
        }
        lock.lock(); defer { lock.unlock() }
        var total: UInt64 = 0
        for (l, e, c) in staged where l < counts.count && !counts[l].isEmpty && e >= 0 && e < counts[l].count {
            let (v, o) = counts[l][e].addingReportingOverflow(c)
            counts[l][e] = o ? .max : v
            total += UInt64(c)
        }
        return total
    }

    private func readLE32(_ d: Data, _ at: Int) -> UInt32 {
        d.withUnsafeBytes { raw in
            var v: UInt32 = 0
            for i in 0 ..< 4 { v |= UInt32(raw[at + i]) << (8 * i) }
            return v
        }
    }

    private func parseIKU1(_ data: Data, path: String, explicit: Bool, into staged: inout [(Int, Int, UInt32)]) throws {
        if engineName != "inkling" {
            guard explicit else {
                throw ColibriError.unsupported("\(path): history written by inkling, this engine is \(engineName); refusing (name the file explicitly to use it anyway)")
            }
            Self.sayOverride(path: path, writer: "inkling", engine: engineName)
        }
        guard data.count >= 12 else { throw ColibriError.invalidArgument("\(path): truncated IKU1 history") }
        let dl = Int(readLE32(data, 4)), de = Int(readLE32(data, 8))
        guard dl == layers, de == experts else {
            throw ColibriError.unsupported("\(path): history is \(dl) layers x \(de) experts, this engine is \(layers) x \(experts); refusing rather than misreading it")
        }
        var at = 12
        for l in 0 ..< dl {
            guard at + 4 * de <= data.count else { break }
            for e in 0 ..< de {
                let c = readLE32(data, at + 4 * e)
                if c != 0 { staged.append((l, e, c)) }
            }
            at += 4 * de
        }
    }

    /// The `while (fscanf(f, "%d %d %u", &l, &e, &c) == 3)` loop: whitespace-separated
    /// integer triples, stopping at the first field that is not a number.
    private func parseText(_ data: Data, path: String, explicit: Bool, into staged: inout [(Int, Int, UInt32)]) throws {
        let text = String(decoding: data, as: UTF8.self)
        var fields = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" }).makeIterator()
        let myID = Self.engineID(engineName)
        while let a = fields.next(), let b = fields.next(), let c = fields.next(),
              let l = Int(a), let v2 = Int(b), let v3 = Self.parseU32(c) {
            if l == -1 {
                guard v2 == layers, Int(v3) == experts else {
                    throw ColibriError.unsupported("\(path): history is \(v2) layers x \(v3) experts, this engine is \(layers) x \(experts); refusing rather than misreading it")
                }
                continue
            }
            if l == -2 {
                guard v2 <= Self.formatVersion else {
                    throw ColibriError.unsupported("\(path): format version \(v2), this build reads \(Self.formatVersion); refusing rather than misreading it")
                }
                if v3 != myID {
                    let writer = Self.engineName(of: v3) ?? "an unknown engine"
                    guard explicit else {
                        throw ColibriError.unsupported("\(path): written by \(writer), this engine is \(engineName); refusing (name the file explicitly to use it anyway)")
                    }
                    Self.sayOverride(path: path, writer: writer, engine: engineName)
                }
                continue
            }
            if l < 0 { continue }  // room for future header records
            staged.append((l, v2, v3))
        }
    }

    /// `%u` semantics: a decimal that wraps modulo 2^32 (a leading minus negates).
    private static func parseU32(_ s: Substring) -> UInt32? {
        if let v = UInt64(s) { return UInt32(truncatingIfNeeded: v) }
        if let v = Int64(s) { return UInt32(truncatingIfNeeded: v) }
        return nil
    }

    private static let overrideLock = NSLock()
    nonisolated(unsafe) private static var saidOverride = false
    /// Says once per process that a named file from another engine is being honoured.
    private static func sayOverride(path: String, writer: String, engine: String) {
        overrideLock.lock()
        let first = !saidOverride
        saidOverride = true
        overrideLock.unlock()
        guard first else { return }
        let msg = "[USAGE] \(path): written by \(writer), this engine is \(engine)\n"
            + "        honouring it because you named it explicitly\n"
            + "        (placement only; expert ids do not transfer between engines)\n"
        FileHandle.standardError.write(Data(msg.utf8))
    }

    // MARK: Save

    /// The file's bytes: empty for an all-zero history (upstream's PIN=auto tests the
    /// size), otherwise the two header records and one line per non-zero counter.
    public func serialized() -> String {
        lock.lock(); defer { lock.unlock() }
        guard counts.contains(where: { $0.contains { $0 != 0 } }) else { return "" }
        var out = "-1 \(layers) \(experts)\n-2 \(Self.formatVersion) \(Self.engineID(engineName))\n"
        for (l, row) in counts.enumerated() {
            for (e, c) in row.enumerated() where c != 0 { out += "\(l) \(e) \(c)\n" }
        }
        return out
    }

    /// Atomic save: writes `<path>.tmp` (never through a symlink planted there) then
    /// renames it over `path`. On failure the previous history is kept.
    public func save(path: String) throws {
        let bytes = Array(serialized().utf8)
        let tmp = path + ".tmp"
        var fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0o644)
        if fd < 0 && (errno == ELOOP || errno == EINVAL) && unlink(tmp) == 0 {
            fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0o644)
        }
        guard fd >= 0 else { throw ColibriError.io(path: tmp, operation: "open", errno: errno) }
        var done = 0
        var failure: Int32 = 0
        bytes.withUnsafeBytes { raw in
            while done < raw.count {
                let n = write(fd, raw.baseAddress! + done, raw.count - done)
                if n < 0 {
                    if errno == EINTR { continue }
                    failure = errno
                    return
                }
                done += n
            }
        }
        if close(fd) != 0 && failure == 0 { failure = errno }
        if failure != 0 {
            unlink(tmp)
            throw ColibriError.io(path: tmp, operation: "write", errno: failure)
        }
        guard rename(tmp, path) == 0 else {
            let e = errno
            unlink(tmp)
            throw ColibriError.io(path: path, operation: "rename", errno: e)
        }
    }
}
