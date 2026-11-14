// Hyperparameters from GGUF metadata. Keys, defaults and refusals:
// docs/port/models.md section 2 and docs/port/gguf-and-quants.md section 2.1.

/// How a family applies the q/k RMSNorm before RoPE.
public enum QKNormScope: Sendable, Equatable {
    /// No q/k norm.
    case none
    /// Each head's `headDim` vector separately, one shared `[headDim]` weight (Qwen3).
    case perHead
    /// The whole projected q (and k) vector across all heads (OLMoE).
    case wholeVector
}

/// Everything the shared transformer code needs to know about a model. Families fill it
/// from GGUF metadata; fields a family does not use stay zero.
public struct ModelConfig: Sendable, Equatable {
    /// `general.architecture`.
    public var architecture: String
    public var layers: Int
    /// Hidden size D.
    public var hidden: Int
    public var heads: Int
    public var kvHeads: Int
    /// Query/key head dim. Not necessarily `hidden / heads` (Qwen3-0.6B: 128 vs 64).
    public var headDim: Int
    public var valueDim: Int
    public var vocab: Int
    /// Dense FFN width F (0 for an all-MoE model).
    public var ffnDim: Int
    /// Routed experts E, used per token K, expert width I (0 for dense models).
    public var experts: Int
    public var expertsUsed: Int
    public var expertFFNDim: Int
    /// Renormalise the top-k router weights (qwen3moe: always; olmoe: default false).
    public var normTopK: Bool
    public var ropeTheta: Float
    public var ropeDims: Int
    public var rmsEps: Float
    /// Trained context length (informational; `--ctx` caps the KV cache).
    public var contextLength: Int
    /// No `output.weight`: the LM head reads `token_embd.weight`.
    public var tiedEmbeddings: Bool
    public var qkNorm: QKNormScope

    public init(architecture: String, layers: Int, hidden: Int, heads: Int, kvHeads: Int, headDim: Int,
                valueDim: Int, vocab: Int, ffnDim: Int = 0, experts: Int = 0, expertsUsed: Int = 0,
                expertFFNDim: Int = 0, normTopK: Bool = false, ropeTheta: Float, ropeDims: Int, rmsEps: Float,
                contextLength: Int, tiedEmbeddings: Bool, qkNorm: QKNormScope) {
        self.architecture = architecture
        self.layers = layers
        self.hidden = hidden
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.valueDim = valueDim
        self.vocab = vocab
        self.ffnDim = ffnDim
        self.experts = experts
        self.expertsUsed = expertsUsed
        self.expertFFNDim = expertFFNDim
        self.normTopK = normTopK
        self.ropeTheta = ropeTheta
        self.ropeDims = ropeDims
        self.rmsEps = rmsEps
        self.contextLength = contextLength
        self.tiedEmbeddings = tiedEmbeddings
        self.qkNorm = qkNorm
    }

    public var isMoE: Bool { experts > 0 }
}

/// Reads `<prefix>.<key>` hyperparameters with upstream's lenient getters, but refuses a
/// missing required key instead of guessing.
public struct HyperparameterReader {
    public let metadata: GGUFMetadata
    /// The `general.architecture` value, used as key prefix.
    public let prefix: String

    public init(_ metadata: GGUFMetadata, prefix: String) {
        self.metadata = metadata
        self.prefix = prefix
    }

    public func required(_ key: String) throws -> Int {
        guard let v = metadata.int("\(prefix).\(key)") else {
            throw ColibriError.invalidGGUF("missing required key \(prefix).\(key)")
        }
        return v
    }

    public func int(_ key: String, default value: Int) -> Int { metadata.int("\(prefix).\(key)") ?? value }
    public func float(_ key: String, default value: Float) -> Float { metadata.float("\(prefix).\(key)").map(Float.init) ?? value }
    public func bool(_ key: String, default value: Bool) -> Bool { metadata.bool("\(prefix).\(key)") ?? value }

    /// An integer that may be stored as a per-layer array (Ollama's `head_count_kv`):
    /// returns the maximum element.
    public func intOrMaxOfArray(_ key: String) -> Int? {
        let k = "\(prefix).\(key)"
        if let v = metadata.int(k) { return v }
        return metadata.ints(k)?.max()
    }
}
