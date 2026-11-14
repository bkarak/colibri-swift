import Foundation

/// Writes a GGUF as a split model in llama.cpp's `gguf-split` layout: shard `i` is
/// `<prefix>-<i+1 as %05d>-of-<n as %05d>.gguf`, the first shard carries every metadata key,
/// each shard carries `split.no` (u16), `split.count` (u16) and `split.tensors.count`
/// (i32), and the tensors are dealt out in file order in `parts` near-equal runs.
/// Holds every tensor in memory: for tests and small models.
public enum GGUFSplit {
    /// Returns the shard paths in order.
    @discardableResult
    public static func write(_ file: GGUFFile, parts: Int, prefix: String) throws -> [String] {
        guard parts >= 1, parts <= file.tensors.count else {
            throw ColibriError.invalidArgument("cannot split \(file.tensors.count) tensors into \(parts) shards")
        }
        var paths: [String] = []
        let n = file.tensors.count
        for i in 0..<parts {
            var w = GGUFWriter(alignment: file.alignment)
            if i == 0 {
                w.copyMetadata(from: file.metadata) { !$0.hasPrefix("split.") }
            }
            w.set("split.no", .u16(UInt16(i)))
            w.set("split.count", .u16(UInt16(parts)))
            w.set("split.tensors.count", .i32(Int32(n)))
            for t in file.tensors[(i * n / parts)..<((i + 1) * n / parts)] {
                w.addTensor(t.name, dims: t.dims, type: t.type, bytes: Array(file.data(t)))
            }
            let path = prefix + String(format: "-%05d-of-%05d.gguf", i + 1, parts)
            try w.write(to: path)
            paths.append(path)
        }
        return paths
    }
}
