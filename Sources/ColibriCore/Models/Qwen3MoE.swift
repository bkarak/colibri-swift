// Qwen3 mixture of experts (GGUF architecture "qwen3moe": Qwen3-30B-A3B,
// Qwen3-Coder-30B-A3B). Ported in behaviour from colibri c/qwen36.c (moe_ex,
// moe_gq_decode, load_expert_merged). Qwen3 attention; MoE FFN with softmax over all
// experts, top-k, always renormalised; no shared expert.

public final class Qwen3MoE: ModelArchitecture {
    public static let architectures = ["qwen3moe"]
    public static let engineName = "qwen3moe"

    public let config: ModelConfig
    let stack: DecoderStack

    public static func config(from file: GGUFFile) throws -> ModelConfig {
        try QwenHyperparameters.read(file, moe: true)
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

    /// ChatML; the Coder variant (template mentions "interact with a computer to solve
    /// tasks", upstream's own sniff) has no think block.
    public func chatTemplate(metadata: GGUFMetadata, thinking: Bool) -> ChatTemplate {
        let coder = metadata.string("tokenizer.chat_template")?.contains("interact with a computer to solve tasks") ?? false
        return ChatMLTemplate(thinkingSupported: !coder, thinking: thinking)
    }

    public func stopTokens(tokenizer: Tokenizer, metadata: GGUFMetadata) -> Set<Int32> {
        tokenizer.specialTokens.stop
    }

    public var expertStats: ExpertStoreStats? { stack.store?.stats }

    /// The expert store (resident or streaming).
    public var expertStore: ExpertStore? { stack.store }
}
