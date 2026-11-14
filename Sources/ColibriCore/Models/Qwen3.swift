// Qwen3 dense (GGUF architecture "qwen3"). Forward pass ported in behaviour from colibri
// c/qwen36.c (gguf_load_cfg, layers_forward_control, attention); docs/port/models.md.
// Per-head q/k RMSNorm before NEOX RoPE, dense SwiGLU FFN, optionally tied LM head.

import Foundation

public final class Qwen3: ModelArchitecture {
    public static let architectures = ["qwen3"]
    public static let engineName = "qwen3"

    public let config: ModelConfig
    let stack: DecoderStack

    public static func config(from file: GGUFFile) throws -> ModelConfig {
        try QwenHyperparameters.read(file, moe: false)
    }

    public init(file: GGUFFile, config: ModelConfig, context: LoadContext) throws {
        self.config = config
        stack = try DecoderStack(file: file, config: config, context: context, engineName: Self.engineName, routing: nil)
    }

    public func makeState() throws -> InferenceState { try stack.makeState() }

    public func forward(_ tokens: [Int32], state: InferenceState) throws -> BufferSlice {
        try stack.forward(tokens, state: state)
    }

    public func chatTemplate(metadata: GGUFMetadata, thinking: Bool) -> ChatTemplate {
        ChatMLTemplate(thinkingSupported: true, thinking: thinking)
    }

    /// `<|im_end|>`, `<|endoftext|>` and whatever the file names as eos/eot/eom/eog.
    public func stopTokens(tokenizer: Tokenizer, metadata: GGUFMetadata) -> Set<Int32> {
        tokenizer.specialTokens.stop
    }

    public var expertStats: ExpertStoreStats? { nil }
}

/// Hyperparameters of the Qwen3 family (`qwen3.` / `qwen3moe.` keys), as colibri
/// c/qwen36.c gguf_load_cfg reads them (docs/port/models.md section 2).
enum QwenHyperparameters {
    static func read(_ file: GGUFFile, moe: Bool) throws -> ModelConfig {
        let md = file.metadata
        guard let arch = md.architecture else { throw ColibriError.invalidGGUF("no general.architecture") }
        let r = HyperparameterReader(md, prefix: arch)

        let blocks = try r.required("block_count")
        let layers = blocks - r.int("nextn_predict_layers", default: 0)
        let hidden = try r.required("embedding_length")
        let heads = try r.required("attention.head_count")
        let kvHeads = r.intOrMaxOfArray("attention.head_count_kv") ?? heads
        guard layers > 0, hidden > 0, heads > 0, kvHeads > 0 else {
            throw ColibriError.invalidGGUF("\(arch): non-positive layer/width/head counts")
        }
        let headDim = r.int("attention.key_length", default: hidden / heads)
        let valueDim = r.int("attention.value_length", default: headDim)
        guard let embd = file.tensor("token_embd.weight") else { throw ColibriError.unsupported("missing tensor token_embd.weight") }

        try refuseUnsupportedRouting(r, arch: arch)
        warnRopeScaling(r, arch: arch)

        var c = ModelConfig(
            architecture: arch, layers: layers, hidden: hidden, heads: heads, kvHeads: kvHeads,
            headDim: headDim, valueDim: valueDim, vocab: embd.ne(1),
            ropeTheta: r.float("rope.freq_base", default: 10000),
            ropeDims: r.int("rope.dimension_count", default: headDim),
            rmsEps: r.float("attention.layer_norm_rms_epsilon", default: 1e-6),
            contextLength: r.int("context_length", default: 4096),
            tiedEmbeddings: file.tensor("output.weight") == nil,
            qkNorm: file.tensor("blk.\(layers - 1).attn_q_norm.weight") != nil ? .perHead : .none)
        if moe {
            c.experts = try r.required("expert_count")
            c.expertsUsed = try r.required("expert_used_count")
            c.expertFFNDim = try r.required("expert_feed_forward_length")
            // Upstream always renormalises qwen3moe's top-k (c->norm_topk = 1); right for
            // every Qwen3 MoE checkpoint.
            c.normTopK = true
            guard c.experts > 0, c.expertsUsed > 0, c.expertsUsed <= c.experts else {
                throw ColibriError.invalidGGUF("\(arch): \(c.expertsUsed) of \(c.experts) experts")
            }
            if let shared = md.int("\(arch).expert_shared_count"), shared > 0 {
                throw ColibriError.unsupported("\(arch): \(shared) shared experts (not a Qwen3 MoE layout)")
            }
            // Dense layers inside a MoE model (if any) read feed_forward_length.
            c.ffnDim = r.int("feed_forward_length", default: 0)
        } else {
            c.ffnDim = try r.required("feed_forward_length")
        }
        return c
    }

    /// Refuses routing keys the softmax top-k path cannot honour.
    static func refuseUnsupportedRouting(_ r: HyperparameterReader, arch: String) throws {
        // llama.cpp: 1 = softmax, 2 = sigmoid.
        if let g = r.metadata.int("\(arch).expert_gating_func"), g != 1 {
            throw ColibriError.unsupported("\(arch).expert_gating_func = \(g); only softmax (1) is implemented")
        }
        if let s = r.metadata.float("\(arch).expert_weights_scale"), s != 0, s != 1 {
            throw ColibriError.unsupported("\(arch).expert_weights_scale = \(s); only 1 is implemented")
        }
    }

    /// Upstream ignores rope scaling (YaRN) for these families; so does the port, loudly.
    static func warnRopeScaling(_ r: HyperparameterReader, arch: String) {
        let type = r.metadata.string("\(arch).rope.scaling.type")
        let factor = r.metadata.float("\(arch).rope.scaling.factor") ?? 1
        if let type, type != "none", factor > 1 {
            FileHandle.standardError.write(Data(
                "warning: \(arch).rope.scaling.type = \(type) (factor \(factor)) is ignored; positions past the trained context will degrade\n".utf8))
        }
    }
}
