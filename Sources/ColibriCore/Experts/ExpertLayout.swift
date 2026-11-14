// Where a stock GGUF keeps its routed experts. A layer stores them as three fused 3-D
// tensors `blk.L.ffn_{gate,up,down}_exps.weight` of shape `[cols, rows, n_expert]`, so
// expert `e` is a contiguous slice at `e * nbytes / n_expert` of each tensor: one expert
// is three contiguous reads. Ported in behaviour from colibri c/qwen36.c
// (load_expert_merged). Design: docs/port/runtime-and-metal.md section 2.2.

/// One expert matrix kind (gate, up or down) of one layer in the GGUF: a fused 3-D tensor
/// `[cols, rows, count]`, expert `e` at `fileOffset + e * bytesPerExpert`.
public struct ExpertTensorSlice: Sendable, Equatable {
    public var fileOffset: Int
    public var bytesPerExpert: Int
    public var type: GGMLType
    public var cols: Int
    public var rows: Int
    public init(fileOffset: Int, bytesPerExpert: Int, type: GGMLType, cols: Int, rows: Int) {
        self.fileOffset = fileOffset
        self.bytesPerExpert = bytesPerExpert
        self.type = type
        self.cols = cols
        self.rows = rows
    }

    /// File byte range of expert `e`.
    public func range(expert e: Int) -> Range<Int> {
        let start = fileOffset + e * bytesPerExpert
        return start ..< start + bytesPerExpert
    }
}

/// Where one MoE layer's experts are. Gate, up and down may have different types
/// (published Q4_K_M files mix Q4_K and Q6_K), so slot sizes are computed per layer.
public struct ExpertLayerLayout: Sendable, Equatable {
    public var layer: Int
    public var count: Int
    public var gate: ExpertTensorSlice
    public var up: ExpertTensorSlice
    public var down: ExpertTensorSlice
    public var slotBytes: Int { gate.bytesPerExpert + up.bytesPerExpert + down.bytesPerExpert }
    public init(layer: Int, count: Int, gate: ExpertTensorSlice, up: ExpertTensorSlice, down: ExpertTensorSlice) {
        self.layer = layer
        self.count = count
        self.gate = gate
        self.up = up
        self.down = down
    }

    /// One past the last file byte any expert of this layer occupies.
    public var fileEnd: Int {
        max(gate.fileOffset + gate.bytesPerExpert * count,
            up.fileOffset + up.bytesPerExpert * count,
            down.fileOffset + down.bytesPerExpert * count)
    }
}

/// Tensor names of a family's routed experts. Families pass this; the store never
/// knows architecture names.
public struct ExpertTensorNames: Sendable {
    public var gate: @Sendable (Int) -> String
    public var up: @Sendable (Int) -> String
    public var down: @Sendable (Int) -> String
    public init(gate: @escaping @Sendable (Int) -> String, up: @escaping @Sendable (Int) -> String,
                down: @escaping @Sendable (Int) -> String) {
        self.gate = gate
        self.up = up
        self.down = down
    }

    /// llama.cpp's names: `blk.L.ffn_{gate,up,down}_exps.weight`.
    public static let llama = ExpertTensorNames(
        gate: { "blk.\($0).ffn_gate_exps.weight" },
        up: { "blk.\($0).ffn_up_exps.weight" },
        down: { "blk.\($0).ffn_down_exps.weight" })
}

/// The routed experts of a model: one entry per layer, nil for dense layers.
public struct ExpertLayout: Sendable, Equatable {
    public var layers: [ExpertLayerLayout?]
    public init(layers: [ExpertLayerLayout?]) { self.layers = layers }

    /// Reads the layout from the GGUF tensor table. Layers without the named tensors
    /// are dense. Throws when a layer has some but not all three, or shapes disagree.
    public init(file: GGUFFile, layerCount: Int, names: ExpertTensorNames = .llama) throws {
        try self.init(tensors: file.tensors, layerCount: layerCount, names: names)
    }

    /// Same as `init(file:)`, from a tensor table (lets tests build a layout without a
    /// parsed file). Tensor infos are untrusted input: every shape and size is checked.
    public init(tensors: [GGUFTensorInfo], layerCount: Int, names: ExpertTensorNames = .llama) throws {
        guard layerCount >= 0 else { throw ColibriError.invalidArgument("layer count \(layerCount)") }
        var byName: [String: GGUFTensorInfo] = [:]
        byName.reserveCapacity(tensors.count)
        for t in tensors { byName[t.name] = t }

        var layers: [ExpertLayerLayout?] = []
        layers.reserveCapacity(layerCount)
        for l in 0 ..< layerCount {
            let g = byName[names.gate(l)], u = byName[names.up(l)], d = byName[names.down(l)]
            if g == nil && u == nil && d == nil {
                layers.append(nil)
                continue
            }
            guard let g, let u, let d else {
                let have = [(names.gate(l), g), (names.up(l), u), (names.down(l), d)]
                    .filter { $0.1 != nil }.map { $0.0 }
                throw ColibriError.invalidGGUF("layer \(l) has only some expert tensors: \(have.joined(separator: ", "))")
            }
            let gs = try Self.slice(g), us = try Self.slice(u), ds = try Self.slice(d)
            let n = g.ne(2)
            guard u.ne(2) == n, d.ne(2) == n else {
                throw ColibriError.invalidGGUF("layer \(l): expert counts differ (gate \(n), up \(u.ne(2)), down \(d.ne(2)))")
            }
            guard gs.cols == us.cols, gs.rows == us.rows else {
                throw ColibriError.invalidGGUF("layer \(l): gate [\(gs.cols), \(gs.rows)] and up [\(us.cols), \(us.rows)] differ")
            }
            guard ds.cols == gs.rows, ds.rows == gs.cols else {
                throw ColibriError.invalidGGUF("layer \(l): down [\(ds.cols), \(ds.rows)] does not invert gate [\(gs.cols), \(gs.rows)]")
            }
            layers.append(ExpertLayerLayout(layer: l, count: n, gate: gs, up: us, down: ds))
        }
        self.layers = layers
    }

    private static func slice(_ t: GGUFTensorInfo) throws -> ExpertTensorSlice {
        guard t.dims.count == 3 || (t.dims.count == 4 && t.dims[3] == 1) else {
            throw ColibriError.invalidGGUF("\(t.name): expert tensor must be [cols, rows, experts], has dims \(t.dims)")
        }
        let cols = t.dims[0], rows = t.dims[1], n = t.dims[2]
        guard cols > 0, rows > 0, n > 0 else { throw ColibriError.invalidGGUF("\(t.name): empty dims \(t.dims)") }
        guard t.type.blockSize > 0, cols % t.type.blockSize == 0 else {
            throw ColibriError.invalidGGUF("\(t.name): row of \(cols) is not a multiple of the \(t.type) block (\(t.type.blockSize))")
        }
        let (perExpert, o1) = t.type.rowBytes(cols).multipliedReportingOverflow(by: rows)
        let (total, o2) = perExpert.multipliedReportingOverflow(by: n)
        guard !o1, !o2, total == t.byteCount else {
            throw ColibriError.invalidGGUF("\(t.name): \(t.byteCount) bytes, expected \(n) x \(perExpert)")
        }
        guard t.fileOffset >= 0 else { throw ColibriError.invalidGGUF("\(t.name): negative offset") }
        return ExpertTensorSlice(fileOffset: t.fileOffset, bytesPerExpert: perExpert, type: t.type, cols: cols, rows: rows)
    }

    /// Bytes of every expert of every layer.
    public var totalBytes: Int { layers.reduce(0) { $0 + ($1.map { $0.slotBytes * $0.count } ?? 0) } }

    /// Indices of the layers that route.
    public var moeLayers: [Int] { layers.indices.filter { layers[$0] != nil } }

    /// Largest expert count of any layer (the routing-history width).
    public var maxExperts: Int { layers.reduce(0) { max($0, $1?.count ?? 0) } }

    /// One past the last file byte any expert occupies.
    public var fileEnd: Int { layers.reduce(0) { max($0, $1?.fileEnd ?? 0) } }
}
