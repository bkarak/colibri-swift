// The library facade: load a GGUF, stream text. Everything below it (ColibriCore,
// ColibriMetal) stays available for callers that want tokens, logits or their own loop;
// this file is what an app needs for chat and completion.
//
//     import Colibri
//
//     let engine = try await ColibriEngine.load(modelPath: path)
//     for try await text in engine.respond(to: [ChatMessage(role: .user, content: "Hi")]) {
//         print(text, terminator: "")
//     }
//
// Concurrency: the model, its KV cache and the generator are not thread-safe, so every
// use of them runs on one private serial queue, never on the caller's thread or on Swift's
// cooperative pool (a forward pass blocks for milliseconds to seconds). Calls from any
// thread or task are queued and run one at a time. A stream's consumer that stops
// iterating (or a cancelled task) stops the generation at the next token.

@_exported import ColibriCore
import ColibriMetal
import Foundation

/// How to load a model. The defaults are the CLI's.
public struct EngineConfiguration: Sendable {
    public enum Compute: Sendable, Equatable {
        /// Metal when it can start and load the model, else the CPU.
        case auto
        case metal
        case cpu
    }

    /// CPU matmul activations: int8 (the default, as fast as llama.cpp) or float (upstream's
    /// exact reference, 2-4x slower). Metal ignores it.
    public enum CPUMath: Sendable, Equatable { case int8, float }

    public var compute: Compute = .auto
    public var cpuMath: CPUMath = .int8
    /// CPU worker threads; nil = the performance cores.
    public var threads: Int?
    /// KV cache capacity in tokens: the longest prompt plus reply.
    public var contextLength = 4096
    /// Largest prefill chunk in tokens.
    public var maxBatch = 512
    /// KV cache element type (`.f16` halves its memory).
    public var kvCache: ScalarType = .f32
    /// Routed experts of MoE models: resident, streamed from the file, or chosen by size.
    public var experts: ExpertStoreOptions.Mode = .auto
    /// RAM for streamed experts in GB; nil = derived from free memory.
    public var expertCacheGB: Double?
    /// Streamed experts read with `F_NOCACHE` (from the SSD, not the file cache).
    public var directIO = false
    /// Keep the MoE routing history (`<model>.coli_usage`), which pins the most used
    /// experts of the next load; written after each generation.
    public var saveRoutingHistory = true

    public init() {}
}

/// One generation's settings.
public struct GenerationOptions: Sendable {
    public var maxTokens = 512
    /// Greedy by default; set a temperature (and top-p, top-k, min-p, a seed) to sample.
    public var sampling: SamplerConfig = .greedy
    /// Qwen3's thinking block (ignored by models without one).
    public var thinking = false

    public init(maxTokens: Int = 512, sampling: SamplerConfig = .greedy, thinking: Bool = false) {
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.thinking = thinking
    }
}

public final class ColibriEngine: @unchecked Sendable {
    /// What was loaded and where it runs.
    public let modelPath: String
    public let architecture: String
    /// "metal" or "cpu".
    public let backendName: String
    /// KV cache capacity in tokens.
    public let contextLength: Int

    // Touched only on `queue`.
    private let model: LoadedModel
    private let generator: Generator
    private let configuration: EngineConfiguration
    private var stats: GenerationStats?
    private let queue = DispatchQueue(label: "colibri.engine", qos: .userInitiated)

    /// Loads `modelPath` (any shard of a split model) on the calling thread. Prefer
    /// `load(modelPath:configuration:)` from async code: loading maps the file and, on
    /// Metal, builds the pipelines (0.1-3 s).
    public init(modelPath: String, configuration: EngineConfiguration = EngineConfiguration()) throws {
        let (model, backend) = try Self.loadModel(modelPath, configuration)
        self.model = model
        self.modelPath = modelPath
        self.configuration = configuration
        self.architecture = model.file.metadata.architecture ?? "?"
        self.backendName = backend.capabilities.name
        self.generator = try Generator(loaded: model, sampler: .greedy)
        self.contextLength = generator.capacity
    }

    /// Loads off the caller's thread.
    public static func load(modelPath: String,
                            configuration: EngineConfiguration = EngineConfiguration()) async throws -> ColibriEngine {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try ColibriEngine(modelPath: modelPath, configuration: configuration) })
            }
        }
    }

    // MARK: Generation

    /// The chat reply to `messages` (the whole conversation, rendered with the model's
    /// template), streamed as text. Tokens shared with the previous call's conversation
    /// are reused from the KV cache, so a chat that appends turns pays only for the new
    /// ones. When the conversation does not fit the context, `trimToFit` drops the oldest
    /// turns (keeping system messages and at least the last turn) until it fits with room
    /// for a reply of `options.maxTokens` (at most half the context); otherwise the stream
    /// throws `ColibriError.invalidArgument`.
    public func respond(to messages: [ChatMessage], options: GenerationOptions = GenerationOptions(),
                        trimToFit: Bool = false) -> AsyncThrowingStream<String, Error> {
        stream(options) { m, capacity in
            let template = m.chatTemplate(thinking: options.thinking)
            var ids = m.tokenizer.encode(template.render(messages))
            if trimToFit {
                let room = capacity - min(options.maxTokens, capacity / 2)
                let system = messages.filter { $0.role == .system }
                var rest = messages.filter { $0.role != .system }
                while ids.count > room, rest.count > 1 {
                    rest.removeFirst()
                    while rest.count > 1, rest.first?.role != .user { rest.removeFirst() }
                    ids = m.tokenizer.encode(template.render(system + rest))
                }
            }
            return (ids, true)
        }
    }

    /// A plain continuation of `text` (no chat template), from an empty context.
    public func complete(_ text: String, options: GenerationOptions = GenerationOptions()) -> AsyncThrowingStream<String, Error> {
        stream(options) { m, _ in (m.tokenizer.encode(text), false) }
    }

    /// `respond(to:options:)` collected into one string.
    public func reply(to messages: [ChatMessage], options: GenerationOptions = GenerationOptions()) async throws -> String {
        var text = ""
        for try await piece in respond(to: messages, options: options) { text += piece }
        return text
    }

    /// Tokens `messages` take once rendered with the chat template. Waits for a running
    /// generation (the tokenizer is used on the engine's queue).
    public func tokenCount(of messages: [ChatMessage], thinking: Bool = false) -> Int {
        queue.sync { model.tokenizer.encode(model.chatTemplate(thinking: thinking).render(messages)).count }
    }

    /// Prefill and decode counters of the last finished generation.
    public var lastStats: GenerationStats? { queue.sync { stats } }

    /// Drops the cached conversation (the next call prefills from scratch).
    public func reset() { queue.async { self.generator.reset() } }

    // MARK: Internals

    private static func loadModel(_ path: String, _ c: EngineConfiguration) throws -> (LoadedModel, Backend) {
        var experts = ExpertStoreOptions(mode: c.experts)
        if let gb = c.expertCacheGB { experts.cacheBytes = Int(gb * 1_073_741_824) }
        experts.uncached = c.directIO
        experts.saveHistory = c.saveRoutingHistory
        func context(_ b: Backend) -> LoadContext {
            LoadContext(backend: b, expertOptions: experts, maxContext: c.contextLength, kvType: c.kvCache,
                        maxBatch: c.maxBatch)
        }
        let cpu = { CPUBackend(threads: c.threads, integerActivations: c.cpuMath == .int8) }
        switch c.compute {
        case .cpu:
            let b = cpu()
            return (try ModelRegistry.load(path: path, context: context(b)), b)
        case .metal:
            let b = try MetalBackend()
            return (try ModelRegistry.load(path: path, context: context(b)), b)
        case .auto:
            if let metal = try? MetalBackend(), let m = try? ModelRegistry.load(path: path, context: context(metal)) {
                return (m, metal)
            }
            let b = cpu()
            return (try ModelRegistry.load(path: path, context: context(b)), b)
        }
    }

    /// Runs one generation on `queue`; `prompt` gives the ids and whether they are the
    /// whole conversation (reuse the cache) or a fresh prompt (reset first).
    private func stream(_ options: GenerationOptions,
                        prompt: @escaping @Sendable (LoadedModel, Int) throws -> ([Int32], Bool)) -> AsyncThrowingStream<String, Error> {
        let stop = StopFlag()
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { _ in stop.set() }
            queue.async {
                do {
                    let (ids, conversation) = try prompt(self.model, self.generator.capacity)
                    guard !ids.isEmpty else { throw ColibriError.invalidArgument("empty prompt") }
                    guard ids.count + 1 <= self.generator.capacity else {
                        throw ColibriError.invalidArgument(
                            "prompt of \(ids.count) tokens does not fit the \(self.generator.capacity)-token context")
                    }
                    self.generator.sampler = Sampler(options.sampling)
                    var decoder = StreamingDecoder(tokenizer: self.model.tokenizer)
                    let stopTokens = self.generator.stopTokens
                    let onToken: (Int32) -> Bool = { id in
                        if stop.isSet { return false }
                        if !stopTokens.contains(id) {
                            let text = decoder.push(id)
                            if !text.isEmpty { continuation.yield(text) }
                        }
                        return true
                    }
                    let s: GenerationStats
                    if conversation {
                        s = try self.generator.generate(sequence: ids, maxTokens: options.maxTokens, onToken: onToken)
                    } else {
                        self.generator.reset()
                        s = try self.generator.generate(prompt: ids, maxTokens: options.maxTokens, onToken: onToken)
                    }
                    let tail = decoder.flush()
                    if !tail.isEmpty { continuation.yield(tail) }
                    self.stats = s
                    if self.configuration.saveRoutingHistory { try? self.model.model.expertStore?.saveHistory() }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

/// A conversation over an engine: keeps the messages and appends each reply. The whole
/// conversation is sent each turn (the engine reuses the cached prefix); when it outgrows
/// the context the engine drops the oldest turns for that call (`messages` keeps them).
public final class ChatSession: @unchecked Sendable {
    public let engine: ColibriEngine
    public var options: GenerationOptions
    private let lock = NSLock()
    private var _messages: [ChatMessage]

    public init(engine: ColibriEngine, system: String? = nil, options: GenerationOptions = GenerationOptions()) {
        self.engine = engine
        self.options = options
        self._messages = system.map { [ChatMessage(role: .system, content: $0)] } ?? []
    }

    /// The conversation so far (system message, user turns, assistant replies).
    public var messages: [ChatMessage] { lock.withLock { _messages } }

    /// Sends `text` as the next user turn and streams the reply, which is appended to
    /// `messages` when the stream finishes.
    public func send(_ text: String) -> AsyncThrowingStream<String, Error> {
        let turn = lock.withLock { () -> [ChatMessage] in
            _messages.append(ChatMessage(role: .user, content: text))
            return _messages
        }
        let inner = engine.respond(to: turn, options: options, trimToFit: true)
        return AsyncThrowingStream { continuation in
            let task = Task {
                var reply = ""
                do {
                    for try await piece in inner {
                        reply += piece
                        continuation.yield(piece)
                    }
                    self.lock.withLock {
                        self._messages = turn + [ChatMessage(role: .assistant,
                                                             content: reply.trimmingCharacters(in: .whitespacesAndNewlines))]
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// `send` collected into one string.
    public func reply(_ text: String) async throws -> String {
        var out = ""
        for try await piece in send(text) { out += piece }
        return out
    }

    /// Forgets every turn but the system message.
    public func reset() {
        lock.withLock { _messages = _messages.filter { $0.role == .system } }
    }
}

/// Set from a stream's termination handler, read by the generation loop.
private final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set() { lock.withLock { value = true } }
    var isSet: Bool { lock.withLock { value } }
}
