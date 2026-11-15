import Foundation

// MARK: - The compute contract
//
// Everything a model forward pass computes goes through `Backend`. A family file
// (Models/) never touches raw weight bytes or loops over activations itself; it only
// encodes backend ops. That keeps one forward pass running unchanged on the CPU and on
// Metal, and it is what lets the Metal backend put a whole token (or a whole layer) into
// one command buffer.
//
// Execution model ("record, then commit"):
//
// * Ops are *encoded*, in program order, into the current batch. `beginBatch` opens a
//   batch explicitly (with a label for profiling); encoding an op with no batch open
//   opens one implicitly.
// * `commitBatch()` submits the batch and returns immediately. `synchronize()` waits for
//   every submitted batch. Only after `synchronize()` may the host read a buffer that an
//   op wrote, or write a buffer that a submitted op reads.
// * Ops within a batch observe each other's writes in program order (the backend inserts
//   the barriers). There is no implicit synchronisation with the host.
// * The CPU backend executes each op eagerly when it is encoded; for it `commitBatch` and
//   `synchronize` are cheap no-ops. Code that is correct on Metal is therefore correct on
//   the CPU, but not the other way round: always `synchronize()` before host reads.
//
// Threading: a backend and every buffer it made are confined to one engine thread. The
// only cross-thread entry point is `HostSignal.signal(_:)`, which I/O threads call.
//
// Numerics: activations are Float32 everywhere. Weights stay in their ggml block format
// and are dequantised inside the kernels (the "float path" of docs/port/gguf-and-quants.md
// section 4.2). Reductions run in a fixed order so greedy decoding is reproducible.

/// Element type of an activation buffer or KV cache. Weights are described by `GGMLType`.
public enum ScalarType: Sendable, Equatable {
    case f32
    case f16
    case i32

    /// Bytes per element.
    public var size: Int {
        switch self {
        case .f32, .i32: return 4
        case .f16: return 2
        }
    }
}

// MARK: Memory that holds weights

/// A contiguous, read-only span of host memory that holds weight bytes: the `mmap` of a
/// GGUF file, or one layer's expert arena. Regions are what a backend registers (Metal
/// wraps each one once with `makeBuffer(bytesNoCopy:)`), and weight views address bytes
/// as `(region, offset)` so a GPU kernel can use a buffer offset instead of a pointer.
///
/// `base` and `length` must both be multiples of `hostPageSize` (16 KiB) for zero-copy
/// GPU access. The region does not own the memory; `keepAlive` holds whatever does
/// (a `MappedFile`, an `ExpertArena`), so a view keeps its bytes alive.
public final class MemoryRegion: @unchecked Sendable {
    /// What the memory is. A backend may treat them differently (prefault a file mapping,
    /// expect arenas to be rewritten by I/O between batches).
    public enum Kind: Sendable { case fileMapping, expertArena, other }

    public let base: UnsafeRawPointer
    public let length: Int
    public let kind: Kind
    public let label: String
    public let keepAlive: AnyObject?

    /// Process-unique identity, usable as a dictionary key by backends.
    public var id: ObjectIdentifier { ObjectIdentifier(self) }

    public init(base: UnsafeRawPointer, length: Int, kind: Kind, label: String, keepAlive: AnyObject?) {
        self.base = base
        self.length = length
        self.kind = kind
        self.label = label
        self.keepAlive = keepAlive
    }
}

/// A 2-D weight matrix in ggml layout: `rows` rows of `cols` elements each, row after
/// row, every row block-quantised along `cols` (GGUF `ne = [cols, rows]`). A 1-D tensor
/// (a norm weight) is `rows == 1`.
///
/// `matmul` computes `y[o] = scale * dot(row o, x)`.
public struct WeightView: @unchecked Sendable {
    public var region: MemoryRegion
    /// Byte offset of row 0 inside `region`. GGUF tensor offsets are 32-byte aligned.
    public var offset: Int
    public var type: GGMLType
    /// Elements per row (GGUF `ne[0]`, the input features). A multiple of `type.blockSize`.
    public var cols: Int
    /// Number of rows (GGUF `ne[1]`, the output features).
    public var rows: Int
    /// Multiplier applied to every product (upstream's optional `<name>.scale` tensor).
    public var scale: Float

    public init(region: MemoryRegion, offset: Int, type: GGMLType, cols: Int, rows: Int, scale: Float = 1) {
        self.region = region
        self.offset = offset
        self.type = type
        self.cols = cols
        self.rows = rows
        self.scale = scale
    }

    /// Bytes in one row.
    public var rowBytes: Int { type.rowBytes(cols) }
    /// Bytes in the whole matrix.
    public var byteCount: Int { rowBytes * rows }
    /// Host address of row `r`.
    public func rowPointer(_ r: Int) -> UnsafeRawPointer { region.base + offset + r * rowBytes }
}

/// A stack of equally shaped matrices at a fixed byte stride inside one region, indexed
/// by a small integer: the routed experts of one layer.
///
/// * Resident: the region is the file mapping, `baseOffset` is the fused
///   `ffn_*_exps` tensor, `stride` is the bytes of one expert, index = expert id.
/// * Streaming: the region is the layer's expert arena, `baseOffset` is the gate/up/down
///   offset inside a slot, `stride` is the slot stride, index = slot number.
///
/// Either way `matmulExperts` takes the indices from a device buffer, so the routing
/// result never has to visit the host in resident mode.
public struct StackedWeights: @unchecked Sendable {
    public var region: MemoryRegion
    public var baseOffset: Int
    /// Bytes between matrix `i` and matrix `i + 1`.
    public var stride: Int
    /// Number of addressable matrices (experts, or slots).
    public var count: Int
    public var type: GGMLType
    public var cols: Int
    public var rows: Int
    /// Optional per-matrix multipliers (upstream's per-expert `.scale`), `count` values.
    public var scales: [Float]?

    public init(region: MemoryRegion, baseOffset: Int, stride: Int, count: Int,
                type: GGMLType, cols: Int, rows: Int, scales: [Float]? = nil) {
        self.region = region
        self.baseOffset = baseOffset
        self.stride = stride
        self.count = count
        self.type = type
        self.cols = cols
        self.rows = rows
        self.scales = scales
    }

    /// The single matrix at `index`.
    public func matrix(_ index: Int) -> WeightView {
        WeightView(region: region, offset: baseOffset + index * stride, type: type,
                   cols: cols, rows: rows, scale: scales?[index] ?? 1)
    }
}

// MARK: Activation buffers

/// A backend-allocated buffer of activations, ids or KV entries. On Apple Silicon both
/// backends allocate host-visible (shared) memory, so `contents` is always valid, but it
/// may only be read or written by the host when no submitted batch uses the buffer
/// (see `Backend.synchronize()`).
public final class DeviceBuffer: @unchecked Sendable {
    public let dtype: ScalarType
    /// Number of elements.
    public let count: Int
    public let label: String
    /// Host address of element 0.
    public let contents: UnsafeMutableRawPointer
    /// The backend's native object (an `MTLBuffer` for Metal, nil for the CPU).
    public let handle: AnyObject?
    private let release: (() -> Void)?

    public init(dtype: ScalarType, count: Int, label: String, contents: UnsafeMutableRawPointer,
                handle: AnyObject?, release: (() -> Void)?) {
        self.dtype = dtype
        self.count = count
        self.label = label
        self.contents = contents
        self.handle = handle
        self.release = release
    }

    deinit { release?() }

    public var byteCount: Int { count * dtype.size }

    /// A slice of `count` elements starting at element `offset` (default: everything).
    public func slice(_ offset: Int = 0, _ count: Int? = nil) -> BufferSlice {
        BufferSlice(buffer: self, offset: offset, count: count ?? (self.count - offset))
    }
}

/// A range of elements inside a `DeviceBuffer`; the unit every op reads and writes.
public struct BufferSlice: @unchecked Sendable {
    public var buffer: DeviceBuffer
    /// First element (in elements of `buffer.dtype`, not bytes).
    public var offset: Int
    public var count: Int

    public init(buffer: DeviceBuffer, offset: Int, count: Int) {
        precondition(offset >= 0 && count >= 0 && offset + count <= buffer.count, "slice out of range")
        self.buffer = buffer
        self.offset = offset
        self.count = count
    }

    public var byteOffset: Int { offset * buffer.dtype.size }

    /// Sub-slice relative to this slice.
    public func slice(_ offset: Int, _ count: Int) -> BufferSlice {
        precondition(offset + count <= self.count, "sub-slice out of range")
        return BufferSlice(buffer: buffer, offset: self.offset + offset, count: count)
    }

    /// Host view as Float32. Valid only for `.f32` buffers and only when the host may
    /// access the buffer (after `synchronize()`).
    public var floats: UnsafeMutableBufferPointer<Float> {
        precondition(buffer.dtype == .f32)
        return UnsafeMutableBufferPointer(start: (buffer.contents + byteOffset).assumingMemoryBound(to: Float.self), count: count)
    }

    /// Host view as Int32. Valid only for `.i32` buffers, same access rule as `floats`.
    public var int32s: UnsafeMutableBufferPointer<Int32> {
        precondition(buffer.dtype == .i32)
        return UnsafeMutableBufferPointer(start: (buffer.contents + byteOffset).assumingMemoryBound(to: Int32.self), count: count)
    }
}

// MARK: KV cache

/// Shape of a KV cache. Storage order is `[layer][kvHead][position][dim]`, keys and
/// values in separate buffers, so one (layer, head) is a contiguous `maxTokens x dim`
/// matrix that an attention kernel streams through.
public struct KVCacheLayout: Sendable, Equatable {
    public var layers: Int
    public var kvHeads: Int
    public var keyDim: Int
    public var valueDim: Int
    public var maxTokens: Int
    /// `.f32` (oracle comparisons) or `.f16` (upstream's chat default, half the memory).
    public var dtype: ScalarType

    public init(layers: Int, kvHeads: Int, keyDim: Int, valueDim: Int, maxTokens: Int, dtype: ScalarType) {
        self.layers = layers
        self.kvHeads = kvHeads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.maxTokens = maxTokens
        self.dtype = dtype
    }

    public var keyCount: Int { layers * kvHeads * maxTokens * keyDim }
    public var valueCount: Int { layers * kvHeads * maxTokens * valueDim }
    public var byteCount: Int { (keyCount + valueCount) * dtype.size }
}

/// A KV cache made by `Backend.makeKVCache`. Positions are absolute token positions.
public final class KVCache {
    public let layout: KVCacheLayout
    public let keys: DeviceBuffer
    public let values: DeviceBuffer

    public init(layout: KVCacheLayout, keys: DeviceBuffer, values: DeviceBuffer) {
        self.layout = layout
        self.keys = keys
        self.values = values
    }
}

// MARK: Op parameters

/// RoPE over `rows` rows of `heads * headDim` values each; row `s` is at absolute position
/// `positionStart + s`. NEOX (half-split) pairing: `(j, j + ropeDims/2)` for
/// `j < ropeDims/2`, dimensions past `ropeDims` untouched, angle
/// `Float(pos) * powf(theta, -2 * Float(j) / Float(ropeDims))` computed in Float32 as
/// upstream does (docs/port/models.md section 4).
public struct RopeParams: Sendable {
    public var rows: Int
    public var heads: Int
    public var headDim: Int
    public var ropeDims: Int
    public var theta: Float
    public var positionStart: Int

    public init(rows: Int, heads: Int, headDim: Int, ropeDims: Int, theta: Float, positionStart: Int) {
        self.rows = rows
        self.heads = heads
        self.headDim = headDim
        self.ropeDims = ropeDims
        self.theta = theta
        self.positionStart = positionStart
    }
}

/// Causal attention for `rows` query rows at positions `positionStart ..< positionStart + rows`
/// against cache positions `0 ... pos` of one layer. Query head `h` reads KV head
/// `h / (heads / kvHeads)` (consecutive query heads share a KV head). Scores are
/// `dot(q, k) * scale`, softmax is max-subtracted with a Float32 sum.
public struct AttentionParams: Sendable {
    public var rows: Int
    public var heads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var valueDim: Int
    public var scale: Float
    public var positionStart: Int

    public init(rows: Int, heads: Int, kvHeads: Int, headDim: Int, valueDim: Int, scale: Float, positionStart: Int) {
        self.rows = rows
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.valueDim = valueDim
        self.scale = scale
        self.positionStart = positionStart
    }
}

// MARK: Host <-> device signalling

/// A monotonically increasing counter that the host advances and batches can wait on
/// (an `MTLSharedEvent` on Metal). The streaming expert store signals it when a missed
/// expert has landed in its slot, so the GPU can run the resident experts of a layer
/// while the SSD reads the missing ones (docs/port/runtime-and-metal.md section 6.7).
public protocol HostSignal: AnyObject, Sendable {
    /// The last value passed to `signal`.
    var value: UInt64 { get }
    /// Advances the counter. Thread-safe; called from I/O threads.
    func signal(_ value: UInt64)
}

// MARK: Capabilities

/// What a backend can do and how much memory it may use.
public struct BackendCapabilities: Sendable {
    /// Short name for logs and `--backend` ("cpu", "metal").
    public var name: String
    /// Weight types `matmul` / `matmulExperts` / `embed` accept. A model refuses to load on
    /// a backend that lacks a type it needs (it does not silently fall back).
    public var weightTypes: Set<GGMLType>
    /// Largest single buffer or region the backend can address (Metal `maxBufferLength`).
    public var maxBufferLength: Int
    /// Memory the backend should keep its working set under (Metal
    /// `recommendedMaxWorkingSetSize`; physical memory for the CPU). Feeds the expert
    /// cache budget.
    public var workingSetLimit: Int
    /// True when encoded work runs asynchronously, so overlapping I/O with it pays off.
    public var isAsynchronous: Bool

    public init(name: String, weightTypes: Set<GGMLType>, maxBufferLength: Int, workingSetLimit: Int, isAsynchronous: Bool) {
        self.name = name
        self.weightTypes = weightTypes
        self.maxBufferLength = maxBufferLength
        self.workingSetLimit = workingSetLimit
        self.isAsynchronous = isAsynchronous
    }
}

// MARK: Load-time support check

/// What a model asks of a backend's kernels, collected once at load time. A backend
/// refuses a model it cannot run by throwing from `checkSupport(_:)` before any op is
/// encoded, so `--backend auto` can fall back to the CPU and an unsupported shape never
/// reaches a kernel. (Encode-time checks remain, but release builds use -Ounchecked, so
/// load time is where a limit must be enforced.)
public struct ModelRequirements: Sendable {
    /// A matrix that `embed`, `matmul` or `matmulExperts` will read.
    public struct Matrix: Sendable {
        /// GGUF tensor name (for messages).
        public var name: String
        public var type: GGMLType
        public var cols: Int
        public var rows: Int
        /// Byte offset of matrix 0 inside its region.
        public var offset: Int
        /// Bytes between stacked matrices (routed experts); 0 for a single matrix.
        public var stride: Int

        public init(name: String, type: GGMLType, cols: Int, rows: Int, offset: Int, stride: Int = 0) {
            self.name = name
            self.type = type
            self.cols = cols
            self.rows = rows
            self.offset = offset
            self.stride = stride
        }
    }

    /// Matrices read by the matrix kernels (embedding, projections, LM head, experts).
    public var matrices: [Matrix] = []
    /// Norm weight vectors (read by `rmsNorm`).
    public var norms: [Matrix] = []
    public var heads = 0
    public var kvHeads = 0
    public var headDim = 0
    public var valueDim = 0
    public var ropeDims = 0
    /// `topK` width (the router's expert count); 0 for a dense model.
    public var routerExperts = 0
    /// `topK` k.
    public var expertsUsed = 0

    public init() {}

    /// Throws `ColibriError.unsupported` naming every weight type `capabilities` cannot
    /// decode, with the tensors that use it.
    public func checkWeightTypes(_ capabilities: BackendCapabilities) throws {
        var byType: [GGMLType: [String]] = [:]
        for m in matrices + norms where !capabilities.weightTypes.contains(m.type) {
            byType[m.type, default: []].append(m.name)
        }
        guard !byType.isEmpty else { return }
        let parts = byType.keys.sorted { $0.description < $1.description }.map { t -> String in
            let names = byType[t]!
            let shown = names.prefix(3).joined(separator: ", ")
            let more = names.count > 3 ? " and \(names.count - 3) more" : ""
            return "\(t) (\(shown)\(more))"
        }
        throw ColibriError.unsupported(
            "backend \(capabilities.name) cannot decode weight types: \(parts.joined(separator: "; "))")
    }
}

// MARK: - Backend

/// The compute contract every backend implements. See the comment at the top of this
/// file for the execution model. Shapes are row-major; "rows" is the token batch `S`
/// (1 when decoding, the prompt length or a chunk of it when prefilling).
public protocol Backend: AnyObject {
    var capabilities: BackendCapabilities { get }

    /// Throws `ColibriError.unsupported` when this backend's kernels cannot run a model
    /// with these weights and shapes. Called by the loader before any op is encoded.
    /// Default: the weight-type check (`ModelRequirements.checkWeightTypes`).
    func checkSupport(_ requirements: ModelRequirements) throws

    // MARK: Memory

    /// Makes a weight region addressable by this backend (Metal: wrap it once, zero
    /// copy). Idempotent. Must be called before any view into the region is used.
    func register(_ region: MemoryRegion) throws

    /// Allocates a zero-filled activation buffer of `count` elements.
    func makeBuffer(_ dtype: ScalarType, count: Int, label: String) throws -> DeviceBuffer

    /// Allocates a zero-filled KV cache.
    func makeKVCache(_ layout: KVCacheLayout) throws -> KVCache

    // MARK: Batching

    /// Opens a batch (a Metal command buffer). Optional: encoding an op opens one implicitly.
    func beginBatch(label: String)

    /// Submits the open batch, if any, without waiting.
    func commitBatch()

    /// Commits the open batch and waits for every submitted batch to finish. Throws if a
    /// batch failed on the device.
    func synchronize() throws

    /// Creates a signal that host threads advance and batches can wait on.
    func makeSignal() -> HostSignal

    /// Makes every op encoded after this call wait until `signal.value >= value`. The CPU
    /// backend blocks the calling thread right here.
    func encodeWait(for signal: HostSignal, value: UInt64)

    // MARK: Ops

    /// Embedding lookup: for each `tokens[s]`, dequantises row `tokens[s]` of `table`
    /// into `out[s * table.cols ..< (s + 1) * table.cols]`. `out.count >= tokens.count * table.cols`.
    func embed(table: WeightView, tokens: [Int32], into out: BufferSlice)

    /// Quantised matrix product: for `s < batch`, `y[s][o] = w.scale * dot(w.row(o), x[s])`.
    /// `x` is `[batch][w.cols]`, `y` is `[batch][w.rows]`. `batch == 1` is the decode
    /// matvec; larger batches are prefill (the weight block should be read once per tile,
    /// not once per row).
    func matmul(_ w: WeightView, _ x: BufferSlice, into y: BufferSlice, batch: Int)

    /// Two products of the same input (a SwiGLU gate and up projection): a backend may
    /// fuse them into one pass over `x`. Default: `matmulGroup` of the two.
    func matmulPair(_ w0: WeightView, _ w1: WeightView, _ x: BufferSlice,
                    into y0: BufferSlice, _ y1: BufferSlice, batch: Int)

    /// Products of the same input with several matrices of equal `cols` (q, k and v): a
    /// backend may run them as one pass. Default: one `matmul` each, in order.
    func matmulGroup(_ ws: [WeightView], _ x: BufferSlice, into ys: [BufferSlice], batch: Int)

    /// Indexed (routed) matrix product, llama.cpp's `mul_mv_id`. `indices` holds
    /// `rows * perRow` Int32 values; for row `s` and pick `k`, with `i = indices[s * perRow + k]`:
    ///
    ///     y[s * perRow + k] = w.matrix(i) · (sharedInput ? x[s] : x[s * perRow + k])
    ///
    /// `sharedInput` is true for gate/up (every expert of a token reads the token's
    /// normed hidden state) and false for down (each expert reads its own intermediate).
    /// `y` is `[rows * perRow][w.rows]`. Indices are read on the device, so in resident
    /// mode they can come straight from `topK`.
    func matmulExperts(_ w: StackedWeights, indices: BufferSlice, rows: Int, perRow: Int,
                       _ x: BufferSlice, sharedInput: Bool, into y: BufferSlice)

    /// RMSNorm of `rows` vectors of length `dim`, each multiplied by the same weight
    /// vector (`weight.cols == dim`, `weight.type` F32 in practice):
    /// `out[r][i] = x[r][i] * w[i] / sqrt(mean(x[r]^2) + eps)`. Covers whole-vector norms
    /// (rows = S) and per-head norms (rows = S * heads, dim = headDim). `x` and `out` may
    /// be the same slice. Upstream sums squares in double; the CPU backend should too.
    func rmsNorm(_ x: BufferSlice, weight: WeightView, into out: BufferSlice, rows: Int, dim: Int, eps: Float)

    /// In-place NEOX rotary embedding (see `RopeParams`).
    func ropeNeox(_ x: BufferSlice, _ params: RopeParams)

    /// Writes `rows` positions of keys and values into layer `layer` of `cache` at
    /// positions `positionStart ..< positionStart + rows`. `k` is `[rows][kvHeads * keyDim]`,
    /// `v` is `[rows][kvHeads * valueDim]`; converted to the cache dtype on write.
    func storeKV(k: BufferSlice, v: BufferSlice, cache: KVCache, layer: Int, positionStart: Int, rows: Int)

    /// Causal GQA attention over layer `layer` of `cache` (see `AttentionParams`). `q` is
    /// `[rows][heads * headDim]`, `out` is `[rows][heads * valueDim]`. The current rows'
    /// keys and values must already be stored.
    func attention(q: BufferSlice, cache: KVCache, layer: Int, _ params: AttentionParams, into out: BufferSlice)

    /// In-place softmax of `rows` rows of `cols` values (max-subtracted, Float32 sum).
    func softmax(_ x: BufferSlice, rows: Int, cols: Int)

    /// For each of `rows` rows of `cols` scores, the `k` largest values, ordered by
    /// rank. Ties go to the lower column index (upstream's strict `>` scan). Writes
    /// `ids` (`[rows][k]`, Int32 column indices) and `weights` (`[rows][k]`, the scores;
    /// divided by their row sum when `renormalize`).
    func topK(_ scores: BufferSlice, rows: Int, cols: Int, k: Int, renormalize: Bool,
              ids: BufferSlice, weights: BufferSlice)

    /// `out[i] = silu(gate[i]) * up[i]` with `silu(z) = z / (1 + exp(-z))`, for `i < count`.
    func siluMul(gate: BufferSlice, up: BufferSlice, into out: BufferSlice, count: Int)

    /// `out[i] = a[i] + b[i]` for `i < count`. `out` may alias `a` or `b`.
    func add(_ a: BufferSlice, _ b: BufferSlice, into out: BufferSlice, count: Int)

    /// MoE combine. For each of `rows` rows: `out[r] (+)= sum_k weights[r][k] * x[r][k]`,
    /// summed in rank order `k = 0 ..< terms` (fixed order, no atomics). `x` is
    /// `[rows][terms][dim]`, `weights` `[rows][terms]`. With `accumulate` the sum is added
    /// to `out` (the residual), otherwise it overwrites it.
    func weightedSum(_ x: BufferSlice, weights: BufferSlice, rows: Int, terms: Int, dim: Int,
                     into out: BufferSlice, accumulate: Bool)

    /// `dst[i] = src[i]` for `i < count` (same dtype).
    func copy(_ src: BufferSlice, into dst: BufferSlice, count: Int)
}

extension Backend {
    public var name: String { capabilities.name }

    public func checkSupport(_ requirements: ModelRequirements) throws {
        try requirements.checkWeightTypes(capabilities)
    }

    public func matmulPair(_ w0: WeightView, _ w1: WeightView, _ x: BufferSlice,
                           into y0: BufferSlice, _ y1: BufferSlice, batch: Int) {
        matmulGroup([w0, w1], x, into: [y0, y1], batch: batch)
    }

    public func matmulGroup(_ ws: [WeightView], _ x: BufferSlice, into ys: [BufferSlice], batch: Int) {
        precondition(ws.count == ys.count, "matmulGroup: \(ws.count) matrices, \(ys.count) outputs")
        for (w, y) in zip(ws, ys) { matmul(w, x, into: y, batch: batch) }
    }
}
