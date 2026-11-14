/// Maps GGUF `general.architecture` to a family. Adding a family: write
/// `Models/<Family>.swift` and add its type to `families`. Nothing else changes.
public enum ModelRegistry {
    public static let families: [ModelArchitecture.Type] = [
        Qwen3.self,
        Qwen3MoE.self,
        OLMoE.self,
    ]

    public static func family(for architecture: String) -> ModelArchitecture.Type? {
        families.first { $0.architectures.contains(architecture) }
    }

    /// Every architecture string the registry accepts, for `info` and error messages.
    public static var supportedArchitectures: [String] { families.flatMap { $0.architectures } }

    /// Opens a GGUF, picks its family, loads model and tokenizer.
    public static func load(path: String, context: LoadContext) throws -> LoadedModel {
        try load(file: GGUFFile(path: path), context: context)
    }

    /// Picks the family of an opened GGUF and loads model and tokenizer.
    public static func load(file: GGUFFile, context: LoadContext) throws -> LoadedModel {
        guard let arch = file.metadata.architecture else {
            throw ColibriError.invalidGGUF("no general.architecture key")
        }
        guard let family = family(for: arch) else {
            throw ColibriError.unsupported(
                "architecture '\(arch)'; this build runs \(supportedArchitectures.joined(separator: ", "))")
        }
        let config = try family.config(from: file)
        let tokenizer = try BPETokenizer(metadata: file.metadata)
        guard tokenizer.vocabularySize <= config.vocab else {
            throw ColibriError.invalidGGUF("tokenizer has \(tokenizer.vocabularySize) tokens, the embedding \(config.vocab) rows")
        }
        let model = try family.init(file: file, config: config, context: context)
        return LoadedModel(file: file, model: model, tokenizer: tokenizer)
    }
}

/// A ready-to-run model.
public struct LoadedModel {
    public var file: GGUFFile
    public var model: ModelArchitecture
    public var tokenizer: Tokenizer
    public init(file: GGUFFile, model: ModelArchitecture, tokenizer: Tokenizer) {
        self.file = file
        self.model = model
        self.tokenizer = tokenizer
    }

    public var metadata: GGUFMetadata { file.metadata }
    public var config: ModelConfig { model.config }
    /// Tokens that end a generation.
    public var stopTokens: Set<Int32> { model.stopTokens(tokenizer: tokenizer, metadata: file.metadata) }
    public func chatTemplate(thinking: Bool) -> ChatTemplate { model.chatTemplate(metadata: file.metadata, thinking: thinking) }
}
