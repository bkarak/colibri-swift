// Shared pre-norm transformer blocks, encoded as Backend ops. Equations:
// docs/port/models.md section 4. Families compose these; they differ in config, q/k norm
// scope, FFN kind and routing policy.
//
// Ported in behaviour from colibri c/qwen36.c (layers_forward_control, attention,
// moe_ex) and c/olmoe.c (layers_forward_range, attention, moe_route_row, moe_by_expert).

import Foundation

// MARK: - Weight loading helpers

extension GGUFFile {
    /// `weight(name)` with its shape checked: `cols` inputs, `rows` outputs.
    func checkedWeight(_ name: String, cols: Int, rows: Int) throws -> WeightView {
        let w = try weight(name)
        guard w.cols == cols, w.rows == rows else {
            throw ColibriError.invalidGGUF("\(name) is [\(w.cols), \(w.rows)], expected [\(cols), \(rows)]")
        }
        return w
    }

    /// A norm weight of `count` values (a 1-D tensor).
    func checkedNorm(_ name: String, count: Int) throws -> WeightView {
        let w = try weight(name)
        guard w.cols * w.rows == count else {
            throw ColibriError.invalidGGUF("\(name) has \(w.cols * w.rows) values, expected \(count)")
        }
        return w
    }
}

/// Attention weights of one layer.
public struct AttentionWeights {
    public var norm: WeightView
    public var q: WeightView
    public var k: WeightView
    public var v: WeightView
    public var output: WeightView
    public var qNorm: WeightView?
    public var kNorm: WeightView?

    /// Loads `blk.L.attn_{norm,q,k,v,output,q_norm,k_norm}.weight` with the shapes `config`
    /// implies (`attn_q [D, H*hd]`, `attn_k [D, KV*hd]`, `attn_v [D, KV*vd]`,
    /// `attn_output [H*vd, D]`; q/k norms `[hd]` per head or `[H*hd]`/`[KV*hd]` whole).
    public static func load(_ file: GGUFFile, layer: Int, config c: ModelConfig) throws -> AttentionWeights {
        let p = "blk.\(layer)."
        var w = AttentionWeights(
            norm: try file.checkedNorm(p + "attn_norm.weight", count: c.hidden),
            q: try file.checkedWeight(p + "attn_q.weight", cols: c.hidden, rows: c.heads * c.headDim),
            k: try file.checkedWeight(p + "attn_k.weight", cols: c.hidden, rows: c.kvHeads * c.headDim),
            v: try file.checkedWeight(p + "attn_v.weight", cols: c.hidden, rows: c.kvHeads * c.valueDim),
            output: try file.checkedWeight(p + "attn_output.weight", cols: c.heads * c.valueDim, rows: c.hidden),
            qNorm: nil, kNorm: nil)
        switch c.qkNorm {
        case .none:
            break
        case .perHead:
            w.qNorm = try file.checkedNorm(p + "attn_q_norm.weight", count: c.headDim)
            w.kNorm = try file.checkedNorm(p + "attn_k_norm.weight", count: c.headDim)
        case .wholeVector:
            w.qNorm = try file.checkedNorm(p + "attn_q_norm.weight", count: c.heads * c.headDim)
            w.kNorm = try file.checkedNorm(p + "attn_k_norm.weight", count: c.kvHeads * c.headDim)
        }
        return w
    }

    var all: [WeightView] { [norm, q, k, v, output] + [qNorm, kNorm].compactMap { $0 } }
}

/// Dense SwiGLU FFN weights of one layer.
public struct DenseFFNWeights {
    public var norm: WeightView
    public var gate: WeightView
    public var up: WeightView
    public var down: WeightView

    /// Loads `blk.L.ffn_{norm,gate,up,down}.weight` (`gate`/`up` `[D, F]`, `down` `[F, D]`).
    public static func load(_ file: GGUFFile, layer: Int, config c: ModelConfig) throws -> DenseFFNWeights {
        let p = "blk.\(layer)."
        let up = try file.weight(p + "ffn_up.weight")
        let width = up.rows
        guard up.cols == c.hidden, width > 0 else {
            throw ColibriError.invalidGGUF("\(p)ffn_up.weight is [\(up.cols), \(up.rows)], expected [\(c.hidden), F]")
        }
        return DenseFFNWeights(
            norm: try file.checkedNorm(p + "ffn_norm.weight", count: c.hidden),
            gate: try file.checkedWeight(p + "ffn_gate.weight", cols: c.hidden, rows: width),
            up: up,
            down: try file.checkedWeight(p + "ffn_down.weight", cols: width, rows: c.hidden))
    }

    var all: [WeightView] { [norm, gate, up, down] }
}

/// Dense part of a MoE FFN layer: the pre-FFN norm and the router. The experts come from
/// the `ExpertStore`.
public struct MoEFFNWeights {
    public var norm: WeightView
    /// `ffn_gate_inp.weight`, `[D, E]`, F32 in practice.
    public var router: WeightView

    public static func load(_ file: GGUFFile, layer: Int, config c: ModelConfig) throws -> MoEFFNWeights {
        let p = "blk.\(layer)."
        return MoEFFNWeights(
            norm: try file.checkedNorm(p + "ffn_norm.weight", count: c.hidden),
            router: try file.checkedWeight(p + "ffn_gate_inp.weight", cols: c.hidden, rows: c.experts))
    }

    var all: [WeightView] { [norm, router] }
}

/// Turns router logits into expert ids and weights. The hook a future family most often
/// changes (sigmoid scoring, bias-corrected selection, grouped top-k), so it is a value
/// the family picks, not a hard-coded step.
public protocol RoutingPolicy {
    /// `logits` is `[rows][experts]` (may be overwritten); writes `ids` and `weights`, `[rows][k]`.
    func route(_ backend: Backend, logits: BufferSlice, rows: Int, experts: Int, k: Int,
               ids: BufferSlice, weights: BufferSlice)
}

/// Softmax over all experts, top-k (lower id wins ties), optional renormalisation.
/// qwen3moe: `renormalize = true`; olmoe: `expert_weights_norm`, default false.
public struct SoftmaxTopK: RoutingPolicy {
    public var renormalize: Bool
    public init(renormalize: Bool) { self.renormalize = renormalize }

    public func route(_ backend: Backend, logits: BufferSlice, rows: Int, experts: Int, k: Int,
                      ids: BufferSlice, weights: BufferSlice) {
        backend.softmax(logits, rows: rows, cols: experts)
        backend.topK(logits, rows: rows, cols: experts, k: k, renormalize: renormalize, ids: ids, weights: weights)
    }
}

// MARK: - Scratch

/// Activation buffers for one sequence, sized for `maxBatch` rows.
public final class TransformerScratch {
    public let maxBatch: Int
    /// Residual stream `[maxBatch][D]`.
    public let x: DeviceBuffer
    /// Normed input of the current sublayer `[maxBatch][D]`.
    public let normed: DeviceBuffer
    public let q: DeviceBuffer
    public let k: DeviceBuffer
    public let v: DeviceBuffer
    public let attnOut: DeviceBuffer
    /// FFN intermediates `[maxBatch * max(1, K)][max(F, I)]`.
    public let gate: DeviceBuffer
    public let up: DeviceBuffer
    /// Projection output `[maxBatch * max(1, K)][D]`.
    public let proj: DeviceBuffer
    /// Router logits `[maxBatch][E]` (MoE only).
    public let routerLogits: DeviceBuffer?
    /// Routing ids `[layers][maxBatch][K]`: each layer writes its own region, so the ids of
    /// a whole token can be read back after one synchronise (routing history) without a
    /// host round trip per layer.
    public let expertIds: DeviceBuffer?
    /// Routing weights `[maxBatch][K]`.
    public let expertWeights: DeviceBuffer?
    /// Slot indices for a streaming store `[maxBatch][K]`.
    public let expertSlots: DeviceBuffer?
    /// Logits of the last row `[vocab]`.
    public let logits: DeviceBuffer

    /// `ffnWidth` is the widest dense FFN of any layer (0 if none).
    public init(config c: ModelConfig, maxBatch: Int, ffnWidth: Int? = nil, backend: Backend) throws {
        precondition(maxBatch >= 1)
        self.maxBatch = maxBatch
        let perRow = max(1, c.expertsUsed)
        let width = max(ffnWidth ?? c.ffnDim, c.expertFFNDim, 1)
        func f32(_ n: Int, _ label: String) throws -> DeviceBuffer { try backend.makeBuffer(.f32, count: max(1, n), label: label) }
        x = try f32(maxBatch * c.hidden, "x")
        normed = try f32(maxBatch * max(c.hidden, c.heads * c.valueDim), "normed")
        q = try f32(maxBatch * c.heads * c.headDim, "q")
        k = try f32(maxBatch * c.kvHeads * c.headDim, "k")
        v = try f32(maxBatch * c.kvHeads * c.valueDim, "v")
        attnOut = try f32(maxBatch * c.heads * c.valueDim, "attnOut")
        gate = try f32(maxBatch * perRow * width, "gate")
        up = try f32(maxBatch * perRow * width, "up")
        proj = try f32(maxBatch * perRow * c.hidden, "proj")
        if c.isMoE {
            routerLogits = try f32(maxBatch * c.experts, "routerLogits")
            expertIds = try backend.makeBuffer(.i32, count: c.layers * maxBatch * c.expertsUsed, label: "expertIds")
            expertWeights = try f32(maxBatch * c.expertsUsed, "expertWeights")
            expertSlots = try backend.makeBuffer(.i32, count: maxBatch * c.expertsUsed, label: "expertSlots")
        } else {
            routerLogits = nil
            expertIds = nil
            expertWeights = nil
            expertSlots = nil
        }
        logits = try f32(c.vocab, "logits")
    }

    /// Bytes of every buffer (for the memory budget).
    public static func byteEstimate(config c: ModelConfig, maxBatch: Int, ffnWidth: Int? = nil) -> Int {
        let k = max(1, c.expertsUsed)
        let width = max(ffnWidth ?? c.ffnDim, c.expertFFNDim, 1)
        var n = 2 * maxBatch * max(c.hidden, c.heads * c.valueDim)
        n += maxBatch * (2 * c.heads * c.headDim + c.kvHeads * (c.headDim + c.valueDim))
        n += maxBatch * k * (2 * width + c.hidden) + c.vocab
        if c.isMoE { n += maxBatch * (c.experts + 2 * c.expertsUsed) + c.layers * maxBatch * c.expertsUsed }
        return n * 4
    }
}

// MARK: - Shared blocks

/// The shared blocks. Each encodes ops into the backend's current batch; none
/// synchronises except the streaming MoE path, which must read the routing on the host.
public enum Transformer {
    /// `x += Attention(RMSNorm(x))` for `rows` rows at `state.position`, including the
    /// q/k norm (by `config.qkNorm`), NEOX RoPE, KV store and GQA attention.
    public static func attentionBlock(_ backend: Backend, config c: ModelConfig, weights w: AttentionWeights,
                                      layer: Int, rows: Int, state: InferenceState) {
        let s = state.scratch
        let D = c.hidden, H = c.heads, KV = c.kvHeads, hd = c.headDim, vd = c.valueDim
        let pos = state.position
        let x = s.x.slice(0, rows * D)
        let normed = s.normed.slice(0, rows * D)
        let q = s.q.slice(0, rows * H * hd)
        let k = s.k.slice(0, rows * KV * hd)
        let v = s.v.slice(0, rows * KV * vd)

        backend.rmsNorm(x, weight: w.norm, into: normed, rows: rows, dim: D, eps: c.rmsEps)
        backend.matmul(w.q, normed, into: q, batch: rows)
        backend.matmul(w.k, normed, into: k, batch: rows)
        backend.matmul(w.v, normed, into: v, batch: rows)

        // q/k RMSNorm before RoPE: one head at a time with a shared [hd] weight (Qwen3), or
        // across the whole projected vector (OLMoE).
        if let qn = w.qNorm, let kn = w.kNorm {
            switch c.qkNorm {
            case .perHead:
                backend.rmsNorm(q, weight: qn, into: q, rows: rows * H, dim: hd, eps: c.rmsEps)
                backend.rmsNorm(k, weight: kn, into: k, rows: rows * KV, dim: hd, eps: c.rmsEps)
            case .wholeVector:
                backend.rmsNorm(q, weight: qn, into: q, rows: rows, dim: H * hd, eps: c.rmsEps)
                backend.rmsNorm(k, weight: kn, into: k, rows: rows, dim: KV * hd, eps: c.rmsEps)
            case .none:
                break
            }
        }

        backend.ropeNeox(q, RopeParams(rows: rows, heads: H, headDim: hd, ropeDims: c.ropeDims,
                                       theta: c.ropeTheta, positionStart: pos))
        backend.ropeNeox(k, RopeParams(rows: rows, heads: KV, headDim: hd, ropeDims: c.ropeDims,
                                       theta: c.ropeTheta, positionStart: pos))
        backend.storeKV(k: k, v: v, cache: state.kv, layer: layer, positionStart: pos, rows: rows)

        let attnOut = s.attnOut.slice(0, rows * H * vd)
        backend.attention(q: q, cache: state.kv, layer: layer,
                          AttentionParams(rows: rows, heads: H, kvHeads: KV, headDim: hd, valueDim: vd,
                                          scale: 1 / sqrtf(Float(hd)), positionStart: pos),
                          into: attnOut)
        let proj = s.proj.slice(0, rows * D)
        backend.matmul(w.output, attnOut, into: proj, batch: rows)
        backend.add(x, proj, into: x, count: rows * D)
    }

    /// `x += down(silu(gate f) * up f)`, `f = RMSNorm(x)`.
    public static func denseFFNBlock(_ backend: Backend, config c: ModelConfig, weights w: DenseFFNWeights,
                                     rows: Int, state: InferenceState) {
        let s = state.scratch
        let D = c.hidden, F = w.up.rows
        let x = s.x.slice(0, rows * D)
        let normed = s.normed.slice(0, rows * D)
        let gate = s.gate.slice(0, rows * F)
        let up = s.up.slice(0, rows * F)
        let proj = s.proj.slice(0, rows * D)
        backend.rmsNorm(x, weight: w.norm, into: normed, rows: rows, dim: D, eps: c.rmsEps)
        backend.matmulPair(w.gate, w.up, normed, into: gate, up, batch: rows)
        backend.siluMul(gate: gate, up: up, into: gate, count: rows * F)
        backend.matmul(w.down, gate, into: proj, batch: rows)
        backend.add(x, proj, into: x, count: rows * D)
    }

    /// `x += sum_k w_k * expert_k(RMSNorm(x))`. Resident store: routing stays on the
    /// device. Streaming store: commits the batch up to the router, reads the ids,
    /// acquires leases, encodes hits first and waits on the store's signal for misses
    /// (docs/port/runtime-and-metal.md section 6.7, "streaming v1").
    ///
    /// Leases the streaming path takes are appended to `leases`; the caller releases them
    /// once the batch that used them has completed (after its next `synchronize`).
    public static func moeFFNBlock(_ backend: Backend, config c: ModelConfig, weights w: MoEFFNWeights,
                                   routing: RoutingPolicy, store: ExpertStore, layer: Int, rows: Int,
                                   state: InferenceState, leases: inout [ExpertLease]) throws {
        let s = state.scratch
        let D = c.hidden, E = c.experts, K = c.expertsUsed
        guard let routerLogits = s.routerLogits, let idsBuf = s.expertIds,
              let weightsBuf = s.expertWeights, let slotsBuf = s.expertSlots else {
            preconditionFailure("moeFFNBlock: scratch was built for a dense model")
        }
        let x = s.x.slice(0, rows * D)
        let normed = s.normed.slice(0, rows * D)
        let logits = routerLogits.slice(0, rows * E)
        let ids = idsBuf.slice(layer * s.maxBatch * K, rows * K)
        let rw = weightsBuf.slice(0, rows * K)

        backend.rmsNorm(x, weight: w.norm, into: normed, rows: rows, dim: D, eps: c.rmsEps)
        backend.matmul(w.router, normed, into: logits, batch: rows)
        routing.route(backend, logits: logits, rows: rows, experts: E, k: K, ids: ids, weights: rw)

        let ew = store.weights(layer: layer)
        let I = ew.gate.rows
        let gate = s.gate.slice(0, rows * K * I)
        let up = s.up.slice(0, rows * K * I)
        let proj = s.proj.slice(0, rows * K * D)

        /// Encodes the experts of pairs `p0 ..< p1` (pair = row * K + rank), all inside one
        /// row when `p1 - p0 < rows * K`, reading indices from `indices`.
        func encode(_ indices: BufferSlice, rowStart: Int, rowCount: Int, perRow: Int, pairStart: Int) {
            let n = rowCount * perRow
            let xin = normed.slice(rowStart * D, rowCount * D)
            let idx = indices.slice(pairStart, n)
            let g = gate.slice(pairStart * I, n * I)
            let u = up.slice(pairStart * I, n * I)
            backend.matmulExperts(ew.gate, indices: idx, rows: rowCount, perRow: perRow, xin, sharedInput: true, into: g)
            backend.matmulExperts(ew.up, indices: idx, rows: rowCount, perRow: perRow, xin, sharedInput: true, into: u)
            backend.siluMul(gate: g, up: u, into: g, count: n * I)
            backend.matmulExperts(ew.down, indices: idx, rows: n, perRow: 1, g, sharedInput: false,
                                  into: proj.slice(pairStart * D, n * D))
        }

        if store.isFullyResident {
            // Index the fused file tensors with the router's ids, on the device.
            encode(ids, rowStart: 0, rowCount: rows, perRow: K, pairStart: 0)
        } else {
            // The host needs the ids: one round trip per MoE layer.
            try backend.synchronize()
            try releaseAll(store, &leases)
            let hostIds = Array(ids.int32s)
            let slots = slotsBuf.slice(0, rows * K)
            let capacity = (store as? StreamingExpertStore)?.slotsPerLayer[layer] ?? Int.max
            var r0 = 0
            while r0 < rows {
                // The largest run of rows whose distinct experts fit the layer's slots.
                var seen = Set<Int32>()
                var r1 = r0
                while r1 < rows {
                    let next = seen.union(hostIds[r1 * K ..< (r1 + 1) * K])
                    if next.count > capacity && r1 > r0 { break }
                    seen = next
                    r1 += 1
                }
                if r0 > 0 {
                    // The previous group's leases may hold the slots this group needs.
                    try backend.synchronize()
                    try releaseAll(store, &leases)
                }
                let lease = try store.acquire(layer: layer, experts: Array(hostIds[r0 * K ..< r1 * K]))
                leases.append(lease)
                let sp = slots.int32s
                for (i, slot) in lease.slots.enumerated() { sp[r0 * K + i] = slot }
                if lease.misses.isEmpty {
                    encode(slots, rowStart: r0, rowCount: r1 - r0, perRow: K, pairStart: r0 * K)
                } else {
                    // Resident-first: the hits run while the SSD reads the misses; the
                    // batch then waits on the store's signal, not the host.
                    let missed = Set(lease.misses.map { $0 + r0 * K })
                    for pass in 0..<2 {
                        if pass == 1, let signal = store.signal {
                            backend.encodeWait(for: signal, value: lease.readyValue)
                        } else if pass == 1 {
                            store.waitForMisses(lease)
                        }
                        // Runs of consecutive pairs inside one row with the same hit/miss state.
                        for r in r0..<r1 {
                            var p = r * K
                            while p < (r + 1) * K {
                                guard missed.contains(p) == (pass == 1) else { p += 1; continue }
                                var e = p + 1
                                while e < (r + 1) * K && missed.contains(e) == (pass == 1) { e += 1 }
                                encode(slots, rowStart: r, rowCount: 1, perRow: e - p, pairStart: p)
                                p = e
                            }
                        }
                    }
                }
                r0 = r1
            }
        }
        // Combine in rank order, added to the residual.
        backend.weightedSum(proj, weights: rw, rows: rows, terms: K, dim: D, into: x, accumulate: true)
    }

    /// Ends every lease in `leases` (their batches must have completed). Throws, after
    /// releasing them all, if a read one of them depended on failed (`checkLease`): the
    /// batch then ran on stale expert bytes and its output must not be used.
    static func releaseAll(_ store: ExpertStore, _ leases: inout [ExpertLease]) throws {
        var failure: Error?
        for l in leases {
            if failure == nil {
                do { try store.checkLease(l) } catch { failure = error }
            }
            store.release(l)
        }
        leases.removeAll()
        if let failure { throw failure }
    }

    /// Final RMSNorm and LM head on the last row only, into `state.scratch.logits`.
    public static func outputHead(_ backend: Backend, config c: ModelConfig, norm: WeightView, head: WeightView,
                                  rows: Int, state: InferenceState) {
        let s = state.scratch
        let D = c.hidden
        let last = s.x.slice((rows - 1) * D, D)
        let normed = s.normed.slice(0, D)
        backend.rmsNorm(last, weight: norm, into: normed, rows: 1, dim: D, eps: c.rmsEps)
        backend.matmul(head, normed, into: s.logits.slice(0, c.vocab), batch: 1)
    }
}

// MARK: - The decoder stack the three families share

/// A pre-norm decoder: token embedding, `L` layers of attention plus a dense or MoE FFN,
/// final norm and LM head. Qwen3, Qwen3-MoE and OLMoE are all this stack with different
/// configs and routing; a family with a different layer shape writes its own `forward`
/// from the blocks above instead.
public final class DecoderStack {
    public enum FFN {
        case dense(DenseFFNWeights)
        case moe(MoEFFNWeights)
    }

    public let config: ModelConfig
    public let backend: Backend
    public let tokenEmbedding: WeightView
    public let outputNorm: WeightView
    public let head: WeightView
    public let attention: [AttentionWeights]
    public let ffn: [FFN]
    public let store: ExpertStore?
    public let routing: RoutingPolicy?
    public let maxBatch: Int
    public let maxContext: Int
    public let kvType: ScalarType
    let ffnWidth: Int
    /// Streaming leases whose batch has not been synchronised yet.
    private var pendingLeases: [ExpertLease] = []

    /// Loads every weight view, checks shapes and backend support, registers the file
    /// mapping, and (for MoE layers) builds the expert store.
    public init(file: GGUFFile, config c: ModelConfig, context: LoadContext, engineName: String,
                routing: RoutingPolicy?) throws {
        guard context.maxBatch >= 1 else { throw ColibriError.invalidArgument("maxBatch must be >= 1") }
        guard context.maxContext >= 1 else { throw ColibriError.invalidArgument("context must be >= 1") }
        guard c.heads > 0, c.kvHeads > 0, c.heads % c.kvHeads == 0 else {
            throw ColibriError.unsupported("\(c.heads) heads is not a multiple of \(c.kvHeads) KV heads")
        }
        guard c.ropeDims % 2 == 0, c.ropeDims <= c.headDim else {
            throw ColibriError.unsupported("rope dims \(c.ropeDims) with head dim \(c.headDim)")
        }
        config = c
        backend = context.backend
        maxBatch = context.maxBatch
        maxContext = context.maxContext
        kvType = context.kvType
        self.routing = routing

        tokenEmbedding = try file.checkedWeight("token_embd.weight", cols: c.hidden, rows: c.vocab)
        outputNorm = try file.checkedNorm("output_norm.weight", count: c.hidden)
        head = c.tiedEmbeddings ? tokenEmbedding
            : try file.checkedWeight("output.weight", cols: c.hidden, rows: c.vocab)

        var layout: ExpertLayout?
        if c.isMoE {
            let l = try ExpertLayout(file: file, layerCount: c.layers)
            for (i, ll) in l.layers.enumerated() {
                guard let ll else { continue }
                guard ll.count == c.experts, ll.gate.cols == c.hidden, ll.gate.rows == c.expertFFNDim else {
                    throw ColibriError.invalidGGUF(
                        "layer \(i): experts [\(ll.gate.cols), \(ll.gate.rows), \(ll.count)], expected [\(c.hidden), \(c.expertFFNDim), \(c.experts)]")
                }
            }
            layout = l
        }

        var attn: [AttentionWeights] = []
        var ffns: [FFN] = []
        var width = 0
        for i in 0..<c.layers {
            attn.append(try AttentionWeights.load(file, layer: i, config: c))
            if let layout, layout.layers[i] != nil {
                ffns.append(.moe(try MoEFFNWeights.load(file, layer: i, config: c)))
            } else {
                let d = try DenseFFNWeights.load(file, layer: i, config: c)
                width = max(width, d.up.rows)
                ffns.append(.dense(d))
            }
        }
        attention = attn
        ffn = ffns
        ffnWidth = width

        // Ask the backend whether its kernels can run this model (weight types, matrix
        // widths, head dims, router width). It throws before any op is encoded, so an
        // unsupported model is refused at load (no silent fallback, no garbage output)
        // and `--backend auto` can retry on the CPU.
        let names = Dictionary(file.tensors.map { ($0.fileOffset, $0.name) }, uniquingKeysWith: { a, _ in a })
        func matrix(_ w: WeightView) -> ModelRequirements.Matrix {
            ModelRequirements.Matrix(name: names[w.offset] ?? "tensor at offset \(w.offset)", type: w.type,
                                     cols: w.cols, rows: w.rows, offset: w.offset)
        }
        var req = ModelRequirements()
        req.matrices = [tokenEmbedding, head].map(matrix)
        req.norms = [matrix(outputNorm)]
        for a in attn {
            req.matrices += [a.q, a.k, a.v, a.output].map(matrix)
            req.norms += ([a.norm, a.qNorm, a.kNorm].compactMap { $0 }).map(matrix)
        }
        for f in ffns {
            switch f {
            case .dense(let d):
                req.matrices += [d.gate, d.up, d.down].map(matrix)
                req.norms.append(matrix(d.norm))
            case .moe(let m):
                req.matrices.append(matrix(m.router))
                req.norms.append(matrix(m.norm))
            }
        }
        if let layout {
            for ll in layout.layers.compactMap({ $0 }) {
                for e in [ll.gate, ll.up, ll.down] {
                    req.matrices.append(ModelRequirements.Matrix(
                        name: names[e.fileOffset] ?? "blk.\(ll.layer) experts", type: e.type,
                        cols: e.cols, rows: e.rows, offset: e.fileOffset, stride: e.bytesPerExpert))
                }
            }
        }
        req.heads = c.heads
        req.kvHeads = c.kvHeads
        req.headDim = c.headDim
        req.valueDim = c.valueDim
        req.ropeDims = c.ropeDims
        if c.isMoE {
            req.routerExperts = c.experts
            req.expertsUsed = c.expertsUsed
        }
        try context.backend.checkSupport(req)

        try context.backend.register(file.file.region)

        if let layout {
            var opts = context.expertOptions
            opts.extraResidentBytes += KVCacheLayout.forModel(c, maxTokens: context.maxContext, dtype: context.kvType).byteCount
                + TransformerScratch.byteEstimate(config: c, maxBatch: context.maxBatch, ffnWidth: width)
            store = try ExpertStores.make(file: file, layout: layout, backend: context.backend, options: opts,
                                          engineName: engineName, topK: c.expertsUsed)
        } else {
            store = nil
        }
    }

    deinit {
        if let store { for l in pendingLeases { store.release(l) } }
    }

    public func makeState() throws -> InferenceState {
        let kv = try backend.makeKVCache(KVCacheLayout.forModel(config, maxTokens: maxContext, dtype: kvType))
        let scratch = try TransformerScratch(config: config, maxBatch: maxBatch, ffnWidth: ffnWidth, backend: backend)
        return InferenceState(kv: kv, scratch: scratch)
    }

    /// Runs `tokens` at `state.position`, advances it, and returns the last row's logits
    /// (host-readable: this synchronises).
    public func forward(_ tokens: [Int32], state: InferenceState) throws -> BufferSlice {
        let c = config
        let rows = tokens.count
        guard rows >= 1 else { throw ColibriError.invalidArgument("forward needs at least one token") }
        guard rows <= state.scratch.maxBatch else {
            throw ColibriError.invalidArgument("\(rows) tokens in one forward; the scratch holds \(state.scratch.maxBatch)")
        }
        guard state.position + rows <= state.kv.layout.maxTokens else {
            throw ColibriError.invalidArgument(
                "context full: position \(state.position) + \(rows) tokens exceeds \(state.kv.layout.maxTokens) (--ctx)")
        }
        for t in tokens where t < 0 || Int(t) >= c.vocab {
            throw ColibriError.invalidArgument("token id \(t) outside the vocabulary of \(c.vocab)")
        }

        backend.beginBatch(label: rows == 1 ? "decode" : "prefill[\(rows)]")
        let x = state.scratch.x.slice(0, rows * c.hidden)
        backend.embed(table: tokenEmbedding, tokens: tokens, into: x)
        var residentMoE: [Int] = []
        do {
            for l in 0..<c.layers {
                Transformer.attentionBlock(backend, config: c, weights: attention[l], layer: l, rows: rows, state: state)
                switch ffn[l] {
                case .dense(let w):
                    Transformer.denseFFNBlock(backend, config: c, weights: w, rows: rows, state: state)
                case .moe(let w):
                    guard let store, let routing else { preconditionFailure("MoE layer without a store") }
                    try Transformer.moeFFNBlock(backend, config: c, weights: w, routing: routing, store: store,
                                                layer: l, rows: rows, state: state, leases: &pendingLeases)
                    if store.isFullyResident { residentMoE.append(l) }
                }
            }
            Transformer.outputHead(backend, config: c, norm: outputNorm, head: head, rows: rows, state: state)
            try backend.synchronize()
        } catch {
            try? backend.synchronize()
            if let store { try? Transformer.releaseAll(store, &pendingLeases) }
            throw error
        }
        if let store {
            try Transformer.releaseAll(store, &pendingLeases)
            if let ids = state.scratch.expertIds {
                // Resident routing never visited the host; count it now for the history.
                for l in residentMoE {
                    let s = ids.slice(l * state.scratch.maxBatch * c.expertsUsed, rows * c.expertsUsed)
                    store.recordUsage(layer: l, experts: Array(s.int32s))
                }
            }
            store.endToken()
        }
        state.position += rows
        return state.scratch.logits.slice(0, c.vocab)
    }

    /// Ids each layer routed the last forward's rows to (`[rows][K]` per MoE layer), for
    /// tests and debugging. Valid until the next forward.
    public func lastRouting(rows: Int, state: InferenceState) -> [[Int32]] {
        guard let ids = state.scratch.expertIds else { return [] }
        let K = config.expertsUsed
        return (0..<config.layers).map { l in Array(ids.slice(l * state.scratch.maxBatch * K, rows * K).int32s) }
    }
}

extension KVCacheLayout {
    /// The cache a model needs for `maxTokens` positions.
    public static func forModel(_ c: ModelConfig, maxTokens: Int, dtype: ScalarType) -> KVCacheLayout {
        KVCacheLayout(layers: c.layers, kvHeads: c.kvHeads, keyDim: c.headDim, valueDim: c.valueDim,
                      maxTokens: maxTokens, dtype: dtype)
    }
}
