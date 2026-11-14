// OLMoE (GGUF architecture "olmoe": OLMoE-1B-7B). Ported in behaviour from colibri
// c/olmoe.c (layers_forward_range, attention, moe_route_row, moe_by_expert). Whole-vector
// q/k RMSNorm, NEOX RoPE, MoE FFN with softmax top-k NOT renormalised by default
// (`olmoe.expert_weights_norm`), experts summed in rank order. GQA is allowed here even
// though upstream requires kvHeads == heads.

public final class OLMoE: ModelArchitecture {
    public static let architectures = ["olmoe"]
    public static let engineName = "olmoe"

    public let config: ModelConfig
    let stack: DecoderStack

    /// Keys as c/tools/gguf_import.py olmoe_config maps them (docs/port/models.md section 2).
    public static func config(from file: GGUFFile) throws -> ModelConfig {
        let md = file.metadata
        let arch = md.architecture ?? "olmoe"
        let r = HyperparameterReader(md, prefix: arch)
        let layers = try r.required("block_count")
        let hidden = try r.required("embedding_length")
        let heads = try r.required("attention.head_count")
        let kvHeads = r.intOrMaxOfArray("attention.head_count_kv") ?? heads
        guard layers > 0, hidden > 0, heads > 0, kvHeads > 0 else {
            throw ColibriError.invalidGGUF("\(arch): non-positive layer/width/head counts")
        }
        let headDim = r.int("attention.key_length", default: hidden / heads)
        let valueDim = r.int("attention.value_length", default: headDim)
        guard let embd = file.tensor("token_embd.weight") else { throw ColibriError.unsupported("missing tensor token_embd.weight") }
        let ffn: Int
        if let v = md.int("\(arch).expert_feed_forward_length") { ffn = v } else { ffn = try r.required("feed_forward_length") }
        try QwenHyperparameters.refuseUnsupportedRouting(r, arch: arch)
        QwenHyperparameters.warnRopeScaling(r, arch: arch)

        let c = ModelConfig(
            architecture: arch, layers: layers, hidden: hidden, heads: heads, kvHeads: kvHeads,
            headDim: headDim, valueDim: valueDim, vocab: embd.ne(1), ffnDim: 0,
            experts: try r.required("expert_count"), expertsUsed: try r.required("expert_used_count"),
            expertFFNDim: ffn,
            normTopK: r.bool("expert_weights_norm", default: false),
            ropeTheta: r.float("rope.freq_base", default: 10000),
            ropeDims: r.int("rope.dimension_count", default: headDim),
            rmsEps: r.float("attention.layer_norm_rms_epsilon", default: 1e-5),
            contextLength: r.int("context_length", default: 4096),
            tiedEmbeddings: file.tensor("output.weight") == nil,
            qkNorm: file.tensor("blk.\(layers - 1).attn_q_norm.weight") != nil ? .wholeVector : .none)
        guard c.experts > 0, c.expertsUsed > 0, c.expertsUsed <= c.experts else {
            throw ColibriError.invalidGGUF("\(arch): \(c.expertsUsed) of \(c.experts) experts")
        }
        return c
    }

    public init(file: GGUFFile, config: ModelConfig, context: LoadContext) throws {
        self.config = config
        stack = try DecoderStack(file: file, config: config, context: context, engineName: Self.engineName,
                                 routing: SoftmaxTopK(renormalize: config.normTopK))
    }

    public func makeState() throws -> InferenceState { try stack.makeState() }

    public func forward(_ tokens: [Int32], state: InferenceState) throws -> BufferSlice {
        try stack.forward(tokens, state: state)
    }

    public func chatTemplate(metadata: GGUFMetadata, thinking: Bool) -> ChatTemplate { OLMoETemplate() }

    /// `|||IP_ADDRESS|||` (the eos) plus every control token, as the olmoe CHAT path stops
    /// on every token the tokenizer marks special.
    public func stopTokens(tokenizer: Tokenizer, metadata: GGUFMetadata) -> Set<Int32> {
        var stop = tokenizer.specialTokens.stop
        if let id = tokenizer.id(of: "|||IP_ADDRESS|||") { stop.insert(id) }
        for id in 0..<Int32(tokenizer.vocabularySize) where tokenizer.tokenType(id) == .control { stop.insert(id) }
        return stop
    }

    public var expertStats: ExpertStoreStats? { stack.store?.stats }

    /// The expert store (resident or streaming).
    public var expertStore: ExpertStore? { stack.store }
}
