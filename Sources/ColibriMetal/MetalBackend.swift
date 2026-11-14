import ColibriCore
import Foundation
import Metal

/// The Metal backend. One device, one queue, pipelines compiled once from
/// `MetalShaders.source`. Weight regions are wrapped zero-copy with
/// `makeBuffer(bytesNoCopy:)`: an expert arena as one buffer, a file mapping as one
/// page-aligned window per tensor the GPU actually reads (see `weightBuffer`).
/// Activation buffers and the KV cache are `.storageModeShared`.
/// Ops are encoded into one command buffer per batch, so a whole token (or layer) is one
/// submit (docs/port/runtime-and-metal.md sections 3 and 6.7). Inside a batch the ops go
/// into one serial compute encoder, which orders each dispatch after the previous one, so
/// an op sees the writes of the ops before it. Waits are passive: CPU spinning steals the
/// shared power budget from the GPU.
///
/// Limits: matrix ops need `cols % 32 == 0` (every quantised type has 32- or 256-element
/// blocks; float weights must follow suit) and 2-byte aligned weights (4-byte for F32),
/// attention needs `headDim % 4 == 0`, `headDim <= 256` and `valueDim <= 256`, `topK`
/// needs `cols <= 2048`, and one tensor (or expert arena) must fit in one `MTLBuffer`
/// (`maxBufferLength`); a file may be larger. `checkSupport(_:)` enforces them when a model loads (it throws, so
/// `--backend auto` falls back to the CPU); every op checks them again when it is encoded
/// with `require`, which -Ounchecked does not remove.
public final class MetalBackend: Backend {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    public let capabilities: BackendCapabilities

    /// Total GPU time (seconds, `gpuEndTime - gpuStartTime`) of the batches waited for by
    /// `synchronize()` since the backend was created. For telemetry and benchmarks.
    public private(set) var gpuSeconds: Double = 0
    /// Benchmark hook: run the decode matvec with `rows` rows per simdgroup and `lanes`
    /// lanes per 32-weight unit (R, L in {1, 2, 4}; `tune_*` kernels exist for Q8_0, Q4_K
    /// and Q6_K only). Nil in normal use.
    public var tuneDecode: (rows: Int, lanes: Int)?
    /// Dense matmuls with at least this many batch rows use the tiled simdgroup-matrix
    /// kernel (`matmul_tiled_*`); smaller batches use the matvec kernels. `Int.max` turns
    /// it off (tests compare the two). `COLIBRI_TILED_MIN_BATCH` overrides the default.
    public var tiledMinBatch: Int = ProcessInfo.processInfo.environment["COLIBRI_TILED_MIN_BATCH"].flatMap(Int.init) ?? 20
    /// Number of command buffers committed since the backend was created.
    public private(set) var batchesCommitted = 0

    /// Registered weight regions: region id -> the buffer that wraps it.
    /// Wrapped regions and the converted norm weights are keyed by the region's
    /// `ObjectIdentifier`, which a later region can reuse once the first is freed: each
    /// entry keeps a weak reference to its region and is only used while that is still
    /// the same object.
    private struct RegionEntry { weak var region: MemoryRegion?; let buffer: MTLBuffer }
    private var regionBuffers: [ObjectIdentifier: RegionEntry] = [:]

    /// File mappings are not wrapped whole. A buffer makes all of its pages resident for
    /// the GPU, so wrapping a whole GGUF reads every byte of it into RAM on the first
    /// command buffer: the routed experts too when they are streamed through arenas,
    /// which defeats the streaming, and about 0.2 s per 17 GB even with a warm file cache.
    /// Instead each tensor gets its own page-aligned window the first time it is bound,
    /// keyed by its byte offset (a tensor is always bound at the same offset).
    private struct Window { let start: Int; let end: Int; let buffer: MTLBuffer }
    private struct FileWindows { weak var region: MemoryRegion?; var byOffset: [Int: Window] }
    private var fileWindows: [ObjectIdentifier: FileWindows] = [:]
    /// Bytes of file mappings wrapped so far (windows can overlap by a page at their ends).
    public private(set) var wrappedFileBytes = 0
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var current: MTLCommandBuffer?
    private var encoder: MTLComputeCommandEncoder?
    private var submitted: [MTLCommandBuffer] = []

    /// Non-F32 (or misaligned) norm weights, converted to an F32 buffer once.
    private struct NormKey: Hashable { let region: ObjectIdentifier; let offset: Int; let type: GGMLType; let count: Int }
    private var normWeights: [NormKey: RegionEntry] = [:]

    // Must match MM_NSG and the template arguments in MetalShaders.
    private static let mmSimdgroups = 4
    private static let prefillTile = 8
    /// Decode matvecs with at least this many rows use the 2-rows-per-simdgroup kernel.
    private static let wideDecodeRows = 16_384
    private static let attentionSimdgroups = 4
    private static let attentionMaxDim = 256
    /// Widest row `topK` handles (one threadgroup per row). Must match the shader.
    private static let topKMaxCols = 2048
    /// Split-KV decode: query batches of at most this many rows split the positions into
    /// chunks of `attentionSplitChunk` (a multiple of 32), at most `attentionMaxSplits`,
    /// so a decode step at long context fills the GPU instead of one threadgroup per head.
    static let attentionSplitRows = 4
    static let attentionSplitChunk = 128
    static let attentionMaxSplits = 64
    /// Partials of split attention (grown on demand; a command buffer that still uses an
    /// older, smaller one keeps it alive).
    private var attentionPartials: MTLBuffer?

    /// The weight types every kernel family (matmul, matmulExperts, embed) decodes. Each
    /// is checked against the CPU backend in ColibriMetalTests.
    public static let supportedWeightTypes: Set<GGMLType> = [
        .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0,
        .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .iq4_nl, .iq4_xs,
    ]

    /// Opens the default GPU and compiles the kernels. Throws `ColibriError.backend` when
    /// there is no Metal device or the shader source does not compile.
    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw ColibriError.backend("metal: no Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw ColibriError.backend("metal: cannot create a command queue")
        }
        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        // IEEE-conforming math (no fast-math reassociation, precise transcendental
        // functions), so the kernels stay close to the CPU reference. The matrix kernels are
        // memory-bound, so this costs nothing measurable there.
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        do {
            self.library = try device.makeLibrary(source: MetalShaders.source, options: options)
        } catch {
            throw ColibriError.backend("metal: shader compile failed: \(error)")
        }
        self.capabilities = BackendCapabilities(
            name: "metal",
            weightTypes: MetalBackend.supportedWeightTypes,
            maxBufferLength: device.maxBufferLength,
            workingSetLimit: Int(device.recommendedMaxWorkingSetSize),
            isAsynchronous: true)
    }

    /// Pipeline for kernel `name`, compiled on first use.
    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let f = library.makeFunction(name: name) else {
            throw ColibriError.backend("metal: no kernel named \(name)")
        }
        let p = try device.makeComputePipelineState(function: f)
        pipelines[name] = p
        return p
    }

    private func kernel(_ name: String) -> MTLComputePipelineState {
        do { return try pipeline(name) } catch { fatalError("metal: \(error)") }
    }

    // MARK: Memory

    public func register(_ region: MemoryRegion) throws {
        if let e = regionBuffers[region.id], e.region === region { return }
        guard Int(bitPattern: region.base) % hostPageSize == 0, region.length % hostPageSize == 0 else {
            throw ColibriError.backend("metal: region \(region.label) is not 16 KiB aligned")
        }
        // A file mapping is wrapped lazily, tensor by tensor (`weightBuffer`).
        if region.kind == .fileMapping { return }
        guard region.length <= device.maxBufferLength else {
            throw ColibriError.backend("metal: region \(region.label) (\(region.length) bytes) exceeds maxBufferLength \(device.maxBufferLength)")
        }
        guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: region.base),
                                             length: region.length, options: .storageModeShared,
                                             deallocator: nil) else {
            throw ColibriError.backend("metal: cannot wrap region \(region.label)")
        }
        buffer.label = region.label
        regionBuffers[region.id] = RegionEntry(region: region, buffer: buffer)
    }

    /// The buffer holding `length` bytes of `region` from `offset`, and the region offset
    /// of that buffer's first byte. Expert arenas (and other regions) are one buffer
    /// starting at 0; a file mapping gets a page-aligned window per tensor.
    public func weightBuffer(_ region: MemoryRegion, offset: Int, length: Int) -> (buffer: MTLBuffer, start: Int) {
        guard region.kind == .fileMapping else { return (buffer(for: region), 0) }
        let end = offset + max(length, 1)
        if let f = fileWindows[region.id], f.region === region, let w = f.byOffset[offset], w.end >= end {
            return (w.buffer, w.start)
        }
        require(offset >= 0 && end <= region.length, "metal: bytes \(offset)..<\(end) outside \(region.label)")
        let start = offset & ~(hostPageSize - 1)
        let stop = min(region.length, (end + hostPageSize - 1) & ~(hostPageSize - 1))
        require(stop - start <= device.maxBufferLength,
                "metal: a tensor of \(stop - start) bytes in \(region.label) exceeds maxBufferLength \(device.maxBufferLength)")
        guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: region.base + start),
                                             length: stop - start, options: .storageModeShared, deallocator: nil) else {
            fatalError("metal: cannot wrap bytes \(start)..<\(stop) of \(region.label)")
        }
        buffer.label = "\(region.label)@\(offset)"
        var f = fileWindows[region.id]
        if f == nil || f!.region !== region { f = FileWindows(region: region, byOffset: [:]) }
        f!.byOffset[offset] = Window(start: start, end: stop, buffer: buffer)
        fileWindows[region.id] = f
        wrappedFileBytes += stop - start
        return (buffer, start)
    }

    /// The Metal buffer wrapping all of `region`, registering it on first use. For a file
    /// mapping this is one window over the whole file; ops use `weightBuffer` instead.
    public func buffer(for region: MemoryRegion) -> MTLBuffer {
        if region.kind == .fileMapping { return weightBuffer(region, offset: 0, length: region.length).buffer }
        if let e = regionBuffers[region.id], e.region === region { return e.buffer }
        do { try register(region) } catch { fatalError("metal: \(error)") }
        return regionBuffers[region.id]!.buffer
    }

    public func makeBuffer(_ dtype: ScalarType, count: Int, label: String) throws -> DeviceBuffer {
        let bytes = max(count * dtype.size, 16)
        guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw ColibriError.backend("metal: cannot allocate \(bytes) bytes for \(label)")
        }
        b.label = label
        memset(b.contents(), 0, bytes)
        return DeviceBuffer(dtype: dtype, count: count, label: label, contents: b.contents(), handle: b, release: nil)
    }

    public func makeKVCache(_ layout: KVCacheLayout) throws -> KVCache {
        guard layout.dtype == .f32 || layout.dtype == .f16 else {
            throw ColibriError.backend("metal: KV cache must be f32 or f16")
        }
        return KVCache(layout: layout,
                       keys: try makeBuffer(layout.dtype, count: layout.keyCount, label: "kv.keys"),
                       values: try makeBuffer(layout.dtype, count: layout.valueCount, label: "kv.values"))
    }

    // MARK: Batching

    public func beginBatch(label: String) {
        if current != nil { commitBatch() }
        guard let cb = queue.makeCommandBuffer() else { fatalError("metal: cannot create a command buffer") }
        cb.label = label
        current = cb
    }

    /// The open command buffer, opening one if needed.
    public var commandBuffer: MTLCommandBuffer {
        if current == nil { beginBatch(label: "implicit") }
        return current!
    }

    /// The open compute encoder of the open batch. Serial dispatch: each dispatch runs
    /// after the previous one has finished, which is the ordering the contract promises.
    private var compute: MTLComputeCommandEncoder {
        if let e = encoder { return e }
        guard let e = commandBuffer.makeComputeCommandEncoder(dispatchType: .serial) else {
            fatalError("metal: cannot create a compute encoder")
        }
        encoder = e
        return e
    }

    private func endEncoding() {
        encoder?.endEncoding()
        encoder = nil
    }

    public func commitBatch() {
        guard let cb = current else { return }
        endEncoding()
        cb.commit()
        submitted.append(cb)
        batchesCommitted += 1
        current = nil
    }

    public func synchronize() throws {
        commitBatch()
        let pending = submitted
        submitted.removeAll()
        var failure: Error?
        for cb in pending {
            cb.waitUntilCompleted()
            if let error = cb.error, failure == nil {
                failure = ColibriError.backend("metal: command buffer \(cb.label ?? "?") failed: \(error)")
            }
            if cb.gpuEndTime > cb.gpuStartTime { gpuSeconds += cb.gpuEndTime - cb.gpuStartTime }
        }
        if let failure { throw failure }
    }

    public func makeSignal() -> HostSignal {
        MetalHostSignal(event: device.makeSharedEvent()!)
    }

    public func encodeWait(for signal: HostSignal, value: UInt64) {
        guard let s = signal as? MetalHostSignal else { fatalError("metal: foreign HostSignal") }
        let cb = commandBuffer
        endEncoding()   // an event wait sits between encoders
        cb.encodeWaitForEvent(s.event, value: value)
    }

    // MARK: Binding helpers

    private func mtl(_ s: BufferSlice) -> MTLBuffer {
        guard let b = s.buffer.handle as? MTLBuffer else {
            fatalError("metal: \(s.buffer.label) was not made by the Metal backend")
        }
        return b
    }

    private func bind(_ e: MTLComputeCommandEncoder, _ s: BufferSlice, _ index: Int) {
        e.setBuffer(mtl(s), offset: s.byteOffset, index: index)
    }

    private func requireF32(_ s: BufferSlice, _ op: String) {
        require(s.buffer.dtype == .f32, "metal \(op): \(s.buffer.label) is not f32")
    }

    /// Binds the region buffer at `offset` rounded down to 256 bytes (setBuffer offsets
    /// must be aligned for the kernel's access type) and returns the remainder, which the
    /// kernel adds to the bound pointer.
    private func bindWeights(_ e: MTLComputeCommandEncoder, _ region: MemoryRegion, offset: Int, length: Int, index: Int) -> UInt64 {
        let (buf, start) = weightBuffer(region, offset: offset, length: length)
        let local = offset - start   // start is page-aligned, so 256-byte alignment carries over
        let aligned = local & ~255
        e.setBuffer(buf, offset: aligned, index: index)
        return UInt64(local - aligned)
    }

    /// Small constant data inline (`setBytes`, up to 4 KiB), larger data in a fresh buffer
    /// that the command buffer keeps alive until it completes.
    private func bindBytes<T: BitwiseCopyable>(_ e: MTLComputeCommandEncoder, _ values: [T], index: Int) {
        let bytes = max(values.count * MemoryLayout<T>.stride, MemoryLayout<T>.stride)
        if values.isEmpty {
            var zero = [UInt8](repeating: 0, count: 16)
            e.setBytes(&zero, length: 16, index: index)
        } else if bytes <= 4096 {
            values.withUnsafeBytes { e.setBytes($0.baseAddress!, length: bytes, index: index) }
        } else {
            let b = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: bytes, options: .storageModeShared)! }
            e.setBuffer(b, offset: 0, index: index)
        }
    }

    private func setParams<T: BitwiseCopyable>(_ e: MTLComputeCommandEncoder, _ value: T, index: Int) {
        withUnsafeBytes(of: value) { e.setBytes($0.baseAddress!, length: MemoryLayout<T>.size, index: index) }
    }

    private func dispatch1D(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ n: Int) {
        guard n > 0 else { return }
        let w = min(256, p.maxTotalThreadsPerThreadgroup)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    private static func kernelSuffix(_ t: GGMLType) -> String { t.description.lowercased() }

    /// An encode-time check that stays in release builds (`precondition` is removed by
    /// -Ounchecked). The conditions are a few integer compares per dispatch.
    @inline(__always)
    private func require(_ condition: Bool, _ message: @autoclosure () -> String = "metal: invalid op arguments",
                         file: StaticString = #fileID, line: UInt = #line) {
        if !condition { fatalError(message(), file: file, line: line) }
    }

    // MARK: Load-time support check

    /// Weight types, matrix widths and alignment, attention head dims and router width
    /// (see the limits in the type's doc comment). Throws `ColibriError.unsupported`.
    public func checkSupport(_ r: ModelRequirements) throws {
        try r.checkWeightTypes(capabilities)
        var problems: [String] = []
        for m in r.matrices {
            if m.cols % 32 != 0 { problems.append("\(m.name): cols \(m.cols) is not a multiple of 32") }
            let a = m.type == .f32 ? 4 : 2
            if m.offset % a != 0 || m.stride % a != 0 {
                problems.append("\(m.name): \(m.type) weights are not \(a)-byte aligned")
            }
        }
        if r.headDim % 4 != 0 || r.headDim > MetalBackend.attentionMaxDim || r.valueDim > MetalBackend.attentionMaxDim {
            problems.append("attention head dim \(r.headDim) / value dim \(r.valueDim) (needs a multiple of 4, at most \(MetalBackend.attentionMaxDim))")
        }
        if r.routerExperts > MetalBackend.topKMaxCols {
            problems.append("router over \(r.routerExperts) experts (top-k supports at most \(MetalBackend.topKMaxCols))")
        }
        guard problems.isEmpty else {
            let shown = problems.prefix(4).joined(separator: "; ")
            let more = problems.count > 4 ? "; and \(problems.count - 4) more" : ""
            throw ColibriError.unsupported("backend metal cannot run this model: \(shown)\(more)")
        }
    }

    private func checkWeightType(_ t: GGMLType, _ op: String) {
        require(MetalBackend.supportedWeightTypes.contains(t), "metal \(op): no kernel for weight type \(t)")
    }

    /// The decoders read 2-byte fields (4-byte for F32) straight from the block bytes, so
    /// every matrix must start at such an address. GGUF offsets are 32-byte aligned, arena
    /// slots 256-byte aligned, and every supported type has an even row size.
    private func checkWeightAlignment(_ t: GGMLType, _ offsets: Int..., op: String) {
        let a = t == .f32 ? 4 : 2
        for o in offsets { require(o % a == 0, "metal \(op): \(t) weights at byte offset \(o) are not \(a)-byte aligned") }
    }

    // MARK: Parameter blocks (layouts match the structs in MetalShaders)

    private struct MMParams { var rowBytes: UInt64; var wDelta: UInt64; var cols: UInt32; var rows: UInt32; var batch: UInt32
                              var xStride: UInt32; var yStride: UInt32; var scale: Float }
    private struct MMIdParams { var rowBytes: UInt64; var stride: UInt64; var wDelta: UInt64
                                var cols: UInt32; var rows: UInt32; var perRow: UInt32; var sharedInput: UInt32
                                var count: UInt32; var hasScales: UInt32 }
    private struct EmbedParams { var rowBytes: UInt64; var wDelta: UInt64; var cols: UInt32; var rows: UInt32
                                 var ntok: UInt32; var scale: Float }
    private struct NormParams { var dim: UInt32; var eps: Float }
    private struct TopKParams { var cols: UInt32; var k: UInt32; var renormalize: UInt32 }
    private struct RopeArgs { var rows: UInt32; var heads: UInt32; var headDim: UInt32; var ropeDims: UInt32
                              var posStart: UInt32; var theta: Float }
    private struct KVStoreArgs { var rows: UInt32; var kvHeads: UInt32; var dim: UInt32; var maxTokens: UInt32
                                 var layer: UInt32; var posStart: UInt32 }
    private struct AttnArgs { var heads: UInt32; var kvHeads: UInt32; var headDim: UInt32; var valueDim: UInt32
                              var maxTokens: UInt32; var layer: UInt32; var posStart: UInt32; var scale: Float
                              var splits: UInt32; var chunk: UInt32 }
    private struct WSumArgs { var rows: UInt32; var terms: UInt32; var dim: UInt32; var accumulate: UInt32 }

    // MARK: Ops

    public func embed(table: WeightView, tokens: [Int32], into out: BufferSlice) {
        guard !tokens.isEmpty else { return }
        checkWeightType(table.type, "embed")
        requireF32(out, "embed")
        require(table.cols % 32 == 0, "metal embed: cols \(table.cols) is not a multiple of 32")
        require(out.count >= tokens.count * table.cols, "embed: output too small")
        require(table.offset >= 0 && table.offset + table.byteCount <= table.region.length, "embed: table outside its region")
        checkWeightAlignment(table.type, table.offset, op: "embed")
        let p = kernel("embed_" + MetalBackend.kernelSuffix(table.type))
        let e = compute
        e.setComputePipelineState(p)
        let delta = bindWeights(e, table.region, offset: table.offset, length: table.byteCount, index: 0)
        bindBytes(e, tokens, index: 1)
        bind(e, out, 2)
        setParams(e, EmbedParams(rowBytes: UInt64(table.rowBytes), wDelta: delta, cols: UInt32(table.cols),
                                 rows: UInt32(table.rows), ntok: UInt32(tokens.count), scale: table.scale), index: 3)
        let w = min(256, p.maxTotalThreadsPerThreadgroup)
        e.dispatchThreads(MTLSize(width: table.cols / 4, height: tokens.count, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: min(w, table.cols / 4), height: 1, depth: 1))
    }

    public func matmul(_ w: WeightView, _ x: BufferSlice, into y: BufferSlice, batch: Int) {
        guard batch > 0, w.rows > 0 else { return }
        checkWeightType(w.type, "matmul")
        requireF32(x, "matmul"); requireF32(y, "matmul")
        require(w.cols % 32 == 0, "metal matmul: cols \(w.cols) is not a multiple of 32")
        require(x.count >= batch * w.cols && y.count >= batch * w.rows, "matmul: slice too small")
        require(w.offset >= 0 && w.offset + w.byteCount <= w.region.length, "matmul: weight outside its region")
        checkWeightAlignment(w.type, w.offset, op: "matmul")
        if batch >= tiledMinBatch {
            // Prefill on the simdgroup matrix units: 64-row x 32-token tiles.
            let e = compute
            e.setComputePipelineState(kernel("matmul_tiled_" + MetalBackend.kernelSuffix(w.type)))
            let delta = bindWeights(e, w.region, offset: w.offset, length: w.byteCount, index: 0)
            bind(e, x, 1)
            bind(e, y, 2)
            setParams(e, MMParams(rowBytes: UInt64(w.rowBytes), wDelta: delta, cols: UInt32(w.cols), rows: UInt32(w.rows),
                                  batch: UInt32(batch), xStride: UInt32(w.cols), yStride: UInt32(w.rows), scale: w.scale), index: 3)
            e.dispatchThreadgroups(MTLSize(width: (w.rows + 63) / 64, height: (batch + 31) / 32, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        let prefill = batch > 1
        let ns = prefill ? MetalBackend.prefillTile : 1
        // Decode: one row per simdgroup keeps a layer's 768..4096-row matrices spread over
        // every GPU core; two rows for tall matrices (output heads). See MetalShaders.
        let wide = prefill || w.rows >= MetalBackend.wideDecodeRows
        var r = wide ? 2 : 1
        var name = "matmul_" + MetalBackend.kernelSuffix(w.type) + (prefill ? "_8" : (wide ? "_1w" : "_1"))
        if !prefill, let t = tuneDecode {
            r = t.rows; name = "tune_" + MetalBackend.kernelSuffix(w.type) + "_r\(t.rows)_l\(t.lanes)"
        }
        let rowsPerGroup = r * MetalBackend.mmSimdgroups
        let p = kernel(name)
        let e = compute
        e.setComputePipelineState(p)
        let delta = bindWeights(e, w.region, offset: w.offset, length: w.byteCount, index: 0)
        bind(e, x, 1)
        bind(e, y, 2)
        setParams(e, MMParams(rowBytes: UInt64(w.rowBytes), wDelta: delta, cols: UInt32(w.cols), rows: UInt32(w.rows),
                              batch: UInt32(batch), xStride: UInt32(w.cols), yStride: UInt32(w.rows), scale: w.scale), index: 3)
        e.dispatchThreadgroups(MTLSize(width: (w.rows + rowsPerGroup - 1) / rowsPerGroup,
                                       height: (batch + ns - 1) / ns, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32 * MetalBackend.mmSimdgroups, height: 1, depth: 1))
    }

    public func matmulExperts(_ w: StackedWeights, indices: BufferSlice, rows: Int, perRow: Int,
                              _ x: BufferSlice, sharedInput: Bool, into y: BufferSlice) {
        let pairs = rows * perRow
        guard pairs > 0, w.rows > 0 else { return }
        checkWeightType(w.type, "matmulExperts")
        requireF32(x, "matmulExperts"); requireF32(y, "matmulExperts")
        require(indices.buffer.dtype == .i32, "matmulExperts: indices must be i32")
        require(w.cols % 32 == 0, "metal matmulExperts: cols \(w.cols) is not a multiple of 32")
        let xRows = sharedInput ? rows : pairs
        require(indices.count >= pairs, "matmulExperts: indices too small")
        require(x.count >= xRows * w.cols && y.count >= pairs * w.rows, "matmulExperts: slice too small")
        require(w.baseOffset >= 0 && w.baseOffset + (w.count - 1) * w.stride + w.type.rowBytes(w.cols) * w.rows <= w.region.length,
                     "matmulExperts: stack outside its region")
        checkWeightAlignment(w.type, w.baseOffset, w.stride, op: "matmulExperts")
        if let s = w.scales { require(s.count >= w.count, "matmulExperts: \(s.count) scales for \(w.count) matrices") }
        let rowsPerGroup = MetalBackend.mmSimdgroups   // one row per simdgroup
        let p = kernel("matmul_id_" + MetalBackend.kernelSuffix(w.type))
        let e = compute
        e.setComputePipelineState(p)
        let delta = bindWeights(e, w.region, offset: w.baseOffset,
                                length: (w.count - 1) * w.stride + w.type.rowBytes(w.cols) * w.rows, index: 0)
        bind(e, x, 1)
        bind(e, y, 2)
        bind(e, indices, 3)
        bindBytes(e, w.scales ?? [], index: 4)
        setParams(e, MMIdParams(rowBytes: UInt64(w.type.rowBytes(w.cols)), stride: UInt64(w.stride), wDelta: delta,
                                cols: UInt32(w.cols), rows: UInt32(w.rows), perRow: UInt32(perRow),
                                sharedInput: sharedInput ? 1 : 0, count: UInt32(w.count),
                                hasScales: w.scales == nil ? 0 : 1), index: 5)
        e.dispatchThreadgroups(MTLSize(width: (w.rows + rowsPerGroup - 1) / rowsPerGroup, height: pairs, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32 * MetalBackend.mmSimdgroups, height: 1, depth: 1))
    }

    /// An F32 view of a norm weight: the region itself when it is F32 and 4-byte aligned,
    /// otherwise a converted copy made once (norm weights are a few KiB).
    private func normWeight(_ w: WeightView, count: Int) -> (MTLBuffer, Int) {
        if w.type == .f32 && w.offset % 4 == 0 {
            let (b, start) = weightBuffer(w.region, offset: w.offset, length: count * 4)
            return (b, w.offset - start)
        }
        let key = NormKey(region: w.region.id, offset: w.offset, type: w.type, count: count)
        if w.region.kind != .expertArena, let e = normWeights[key], e.region === w.region { return (e.buffer, 0) }
        var values = [Float](repeating: 0, count: count)
        values.withUnsafeMutableBufferPointer {
            Dequant.dequantizeRow(w.type, w.rowPointer(0), into: $0.baseAddress!, count: count)
        }
        let b = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max(16, count * 4), options: .storageModeShared)! }
        b.label = "norm.f32"
        if w.region.kind != .expertArena { normWeights[key] = RegionEntry(region: w.region, buffer: b) }
        return (b, 0)
    }

    public func rmsNorm(_ x: BufferSlice, weight: WeightView, into out: BufferSlice, rows: Int, dim: Int, eps: Float) {
        guard rows > 0, dim > 0 else { return }
        require(weight.cols * weight.rows == dim, "rmsNorm: weight has \(weight.cols * weight.rows) values, dim \(dim)")
        require(x.count >= rows * dim && out.count >= rows * dim, "rmsNorm: slice too small")
        requireF32(x, "rmsNorm"); requireF32(out, "rmsNorm")
        let (wb, wo) = normWeight(weight, count: dim)
        let p = kernel("k_rmsnorm")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, x, 0)
        e.setBuffer(wb, offset: wo, index: 1)
        bind(e, out, 2)
        setParams(e, NormParams(dim: UInt32(dim), eps: eps), index: 3)
        let threads = min(p.maxTotalThreadsPerThreadgroup, 1024, max(32, (dim / 4 + 31) / 32 * 32))
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    public func ropeNeox(_ x: BufferSlice, _ params: RopeParams) {
        let width = params.heads * params.headDim
        require(x.count >= params.rows * width, "ropeNeox: slice too small")
        require(params.ropeDims <= params.headDim && params.ropeDims % 2 == 0, "ropeNeox: bad ropeDims \(params.ropeDims)")
        requireF32(x, "ropeNeox")
        let half = params.ropeDims / 2
        guard half > 0, params.rows > 0, params.heads > 0 else { return }
        let p = kernel("k_rope_neox")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, x, 0)
        setParams(e, RopeArgs(rows: UInt32(params.rows), heads: UInt32(params.heads), headDim: UInt32(params.headDim),
                              ropeDims: UInt32(params.ropeDims), posStart: UInt32(params.positionStart),
                              theta: params.theta), index: 1)
        let tw = min(half, 64)
        e.dispatchThreads(MTLSize(width: half, height: params.heads, depth: params.rows),
                          threadsPerThreadgroup: MTLSize(width: tw, height: max(1, min(params.heads, 256 / tw)), depth: 1))
    }

    public func storeKV(k: BufferSlice, v: BufferSlice, cache: KVCache, layer: Int, positionStart: Int, rows: Int) {
        let L = cache.layout
        require(layer >= 0 && layer < L.layers, "storeKV: layer \(layer)")
        require(positionStart >= 0 && positionStart + rows <= L.maxTokens, "storeKV: positions past maxTokens \(L.maxTokens)")
        require(k.count >= rows * L.kvHeads * L.keyDim && v.count >= rows * L.kvHeads * L.valueDim, "storeKV: slice too small")
        requireF32(k, "storeKV"); requireF32(v, "storeKV")
        guard rows > 0 else { return }
        let p = kernel(L.dtype == .f16 ? "store_kv_f16" : "store_kv_f32")
        let e = compute
        e.setComputePipelineState(p)
        for (src, dst, dim) in [(k, cache.keys, L.keyDim), (v, cache.values, L.valueDim)] {
            bind(e, src, 0)
            e.setBuffer(dst.handle as? MTLBuffer, offset: 0, index: 1)
            setParams(e, KVStoreArgs(rows: UInt32(rows), kvHeads: UInt32(L.kvHeads), dim: UInt32(dim),
                                     maxTokens: UInt32(L.maxTokens), layer: UInt32(layer),
                                     posStart: UInt32(positionStart)), index: 2)
            dispatch1D(e, p, rows * L.kvHeads * dim)
        }
    }

    public func attention(q: BufferSlice, cache: KVCache, layer: Int, _ params: AttentionParams, into out: BufferSlice) {
        let L = cache.layout
        require(params.kvHeads == L.kvHeads && params.headDim == L.keyDim && params.valueDim == L.valueDim,
                     "attention: params disagree with the cache layout")
        require(params.heads % params.kvHeads == 0, "attention: heads \(params.heads) not a multiple of kvHeads \(params.kvHeads)")
        require(params.positionStart + params.rows <= L.maxTokens, "attention: positions past maxTokens")
        require(layer >= 0 && layer < L.layers, "attention: layer \(layer)")
        require(q.count >= params.rows * params.heads * params.headDim && out.count >= params.rows * params.heads * params.valueDim,
                     "attention: slice too small")
        require(params.headDim % 4 == 0 && params.headDim <= MetalBackend.attentionMaxDim
                     && params.valueDim <= MetalBackend.attentionMaxDim,
                     "metal attention: headDim \(params.headDim) / valueDim \(params.valueDim) unsupported (need headDim % 4 == 0, both <= 256)")
        requireF32(q, "attention"); requireF32(out, "attention")
        guard params.rows > 0, params.heads > 0 else { return }
        let p = kernel(L.dtype == .f16 ? "attention_f16" : "attention_f32")
        let n = params.positionStart + params.rows
        let chunk = MetalBackend.attentionSplitChunk
        let splits = params.rows <= MetalBackend.attentionSplitRows
            ? min(MetalBackend.attentionMaxSplits, (n + chunk - 1) / chunk) : 1
        let effChunk = splits > 1 ? ((n + splits - 1) / splits + 31) / 32 * 32 : 0
        var args = AttnArgs(heads: UInt32(params.heads), kvHeads: UInt32(params.kvHeads), headDim: UInt32(params.headDim),
                            valueDim: UInt32(params.valueDim), maxTokens: UInt32(L.maxTokens), layer: UInt32(layer),
                            posStart: UInt32(params.positionStart), scale: params.scale,
                            splits: UInt32(splits), chunk: UInt32(effChunk))
        let partials: MTLBuffer?
        if splits > 1 {
            let bytes = params.rows * params.heads * splits * (params.valueDim + 2) * MemoryLayout<Float>.stride
            if (attentionPartials?.length ?? 0) < bytes {
                guard let b = device.makeBuffer(length: max(bytes, 1 << 20), options: .storageModePrivate) else {
                    fatalError("metal: cannot allocate \(bytes) bytes of attention partials")
                }
                b.label = "attention.partials"
                attentionPartials = b
            }
            partials = attentionPartials
        } else {
            partials = nil
            args.chunk = UInt32(n)
        }
        let e = compute
        e.setComputePipelineState(p)
        bind(e, q, 0)
        e.setBuffer(cache.keys.handle as? MTLBuffer, offset: 0, index: 1)
        e.setBuffer(cache.values.handle as? MTLBuffer, offset: 0, index: 2)
        bind(e, out, 3)
        setParams(e, args, index: 4)
        // Unused when splits == 1, but every declared buffer argument must be bound.
        if let partials { e.setBuffer(partials, offset: 0, index: 5) } else { bind(e, out, 5) }
        e.dispatchThreadgroups(MTLSize(width: params.heads, height: params.rows, depth: splits),
                               threadsPerThreadgroup: MTLSize(width: 32 * MetalBackend.attentionSimdgroups, height: 1, depth: 1))
        guard let partials else { return }
        let merge = kernel("k_attention_merge")
        e.setComputePipelineState(merge)
        e.setBuffer(partials, offset: 0, index: 0)
        bind(e, out, 1)
        setParams(e, args, index: 2)
        e.dispatchThreadgroups(MTLSize(width: params.heads, height: params.rows, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(params.valueDim, 128).roundedUp32, height: 1, depth: 1))
    }

    public func softmax(_ x: BufferSlice, rows: Int, cols: Int) {
        require(x.count >= rows * cols, "softmax: slice too small")
        requireF32(x, "softmax")
        guard rows > 0, cols > 0 else { return }
        let p = kernel("k_softmax")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, x, 0)
        setParams(e, UInt32(cols), index: 1)
        let threads = min(p.maxTotalThreadsPerThreadgroup, 256, max(32, (cols + 31) / 32 * 32))
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    public func topK(_ scores: BufferSlice, rows: Int, cols: Int, k: Int, renormalize: Bool,
                     ids: BufferSlice, weights: BufferSlice) {
        require(k >= 1 && k <= cols, "topK: k \(k) outside 1...\(cols)")
        require(cols <= MetalBackend.topKMaxCols, "metal topK: cols \(cols) > \(MetalBackend.topKMaxCols)")
        require(scores.count >= rows * cols && ids.count >= rows * k && weights.count >= rows * k, "topK: slice too small")
        requireF32(scores, "topK"); requireF32(weights, "topK")
        require(ids.buffer.dtype == .i32, "topK: ids must be i32")
        guard rows > 0 else { return }
        let p = kernel("k_topk")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, scores, 0)
        bind(e, ids, 1)
        bind(e, weights, 2)
        setParams(e, TopKParams(cols: UInt32(cols), k: UInt32(k), renormalize: renormalize ? 1 : 0), index: 3)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    public func siluMul(gate: BufferSlice, up: BufferSlice, into out: BufferSlice, count: Int) {
        require(gate.count >= count && up.count >= count && out.count >= count, "siluMul: slice too small")
        requireF32(gate, "siluMul"); requireF32(up, "siluMul"); requireF32(out, "siluMul")
        guard count > 0 else { return }
        let p = kernel("k_silu_mul")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, gate, 0); bind(e, up, 1); bind(e, out, 2)
        setParams(e, UInt32(count), index: 3)
        dispatch1D(e, p, count)
    }

    public func add(_ a: BufferSlice, _ b: BufferSlice, into out: BufferSlice, count: Int) {
        require(a.count >= count && b.count >= count && out.count >= count, "add: slice too small")
        requireF32(a, "add"); requireF32(b, "add"); requireF32(out, "add")
        guard count > 0 else { return }
        let p = kernel("k_add")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, a, 0); bind(e, b, 1); bind(e, out, 2)
        setParams(e, UInt32(count), index: 3)
        dispatch1D(e, p, count)
    }

    public func weightedSum(_ x: BufferSlice, weights: BufferSlice, rows: Int, terms: Int, dim: Int,
                            into out: BufferSlice, accumulate: Bool) {
        require(x.count >= rows * terms * dim && weights.count >= rows * terms && out.count >= rows * dim,
                     "weightedSum: slice too small")
        requireF32(x, "weightedSum"); requireF32(weights, "weightedSum"); requireF32(out, "weightedSum")
        guard rows > 0, dim > 0 else { return }
        let p = kernel("k_weighted_sum")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, x, 0); bind(e, weights, 1); bind(e, out, 2)
        setParams(e, WSumArgs(rows: UInt32(rows), terms: UInt32(terms), dim: UInt32(dim), accumulate: accumulate ? 1 : 0), index: 3)
        dispatch1D(e, p, rows * dim)
    }

    public func copy(_ src: BufferSlice, into dst: BufferSlice, count: Int) {
        require(src.buffer.dtype == dst.buffer.dtype && count <= src.count && count <= dst.count)
        guard count > 0 else { return }
        let size = src.buffer.dtype.size
        let p = kernel(size == 4 ? "k_copy_u32" : "k_copy_u16")
        let e = compute
        e.setComputePipelineState(p)
        bind(e, src, 0); bind(e, dst, 1)
        setParams(e, UInt32(count), index: 2)
        dispatch1D(e, p, count)
    }
}

/// `HostSignal` backed by an `MTLSharedEvent`, so command buffers can wait on it.
public final class MetalHostSignal: HostSignal, @unchecked Sendable {
    public let event: MTLSharedEvent
    private let lock = NSLock()

    init(event: MTLSharedEvent) { self.event = event }

    public var value: UInt64 { event.signaledValue }

    public func signal(_ value: UInt64) {
        lock.lock(); defer { lock.unlock() }
        if value > event.signaledValue { event.signaledValue = value }
    }
}

private extension Int {
    /// The next multiple of 32 (a whole simdgroup).
    var roundedUp32: Int { (self + 31) / 32 * 32 }
}
