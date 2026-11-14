// The family plugin surface. A new model family is one file that defines a type
// conforming to `ModelArchitecture`, plus one line in `ModelRegistry.families`.
// What a family supplies: docs/port/models.md section 12.

/// What a model needs from its environment when it loads.
public struct LoadContext {
    public var backend: Backend
    public var expertOptions: ExpertStoreOptions
    /// KV cache capacity in tokens (`--ctx`).
    public var maxContext: Int
    /// KV cache element type (`.f32` for oracle comparisons, `.f16` to halve memory).
    public var kvType: ScalarType
    /// Largest prefill chunk (rows per forward call); scratch buffers are sized for it.
    public var maxBatch: Int

    public init(backend: Backend, expertOptions: ExpertStoreOptions = ExpertStoreOptions(),
                maxContext: Int = 4096, kvType: ScalarType = .f32, maxBatch: Int = 64) {
        self.backend = backend
        self.expertOptions = expertOptions
        self.maxContext = maxContext
        self.kvType = kvType
        self.maxBatch = maxBatch
    }
}

/// The per-sequence mutable state of a forward pass: KV cache, position, scratch.
public final class InferenceState {
    public let kv: KVCache
    public let scratch: TransformerScratch
    /// Tokens already in the KV cache (the next token's absolute position).
    public var position: Int = 0

    public init(kv: KVCache, scratch: TransformerScratch) {
        self.kv = kv
        self.scratch = scratch
    }

    /// Forgets the sequence (the cache contents are overwritten on reuse).
    public func reset() { position = 0 }
}

/// One model family (one or more GGUF `general.architecture` values).
public protocol ModelArchitecture: AnyObject {
    /// `general.architecture` values this family runs.
    static var architectures: [String] { get }

    /// Engine name for the routing history id (`.coli_usage` `-2` header); keep upstream's
    /// so histories are interchangeable ("qwen3moe", "olmoe").
    static var engineName: String { get }

    /// Hyperparameters from the file's metadata and tensor table. Refuses keys the family
    /// cannot honour (e.g. YaRN rope scaling) rather than running wrong.
    static func config(from file: GGUFFile) throws -> ModelConfig

    /// Builds weight views, registers regions with the backend, makes the expert store.
    init(file: GGUFFile, config: ModelConfig, context: LoadContext) throws

    var config: ModelConfig { get }

    /// Allocates a KV cache and scratch for one sequence.
    func makeState() throws -> InferenceState

    /// Runs `tokens` (prefill chunk or one decode token) at `state.position`, advances the
    /// position, and returns the logits of the last row (`vocab` floats). The returned
    /// slice is readable by the host when this returns (the call synchronises).
    func forward(_ tokens: [Int32], state: InferenceState) throws -> BufferSlice

    /// The chat renderer for this file (may sniff `tokenizer.chat_template`).
    func chatTemplate(metadata: GGUFMetadata, thinking: Bool) -> ChatTemplate

    /// Token ids that end generation.
    func stopTokens(tokenizer: Tokenizer, metadata: GGUFMetadata) -> Set<Int32>

    /// Expert cache counters (nil for dense models).
    var expertStats: ExpertStoreStats? { get }

    /// The routed-expert store (nil for dense models). The CLI reads its placement and
    /// counters and saves its routing history; a MoE family built on `DecoderStack`
    /// returns `stack.store`.
    var expertStore: ExpertStore? { get }
}

extension ModelArchitecture {
    public var expertStore: ExpertStore? { nil }
    public var expertStats: ExpertStoreStats? { expertStore?.stats }
}
