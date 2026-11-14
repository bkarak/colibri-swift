import Foundation

/// Timing and counts for one generation, printed by `bench` and `--stats`.
public struct GenerationStats: Sendable {
    /// Tokens fed through prefill (the part of the prompt not already in the KV cache).
    public var promptTokens = 0
    /// Tokens sampled.
    public var generatedTokens = 0
    /// Single-token forward passes after the prefill (one fewer than `generatedTokens`
    /// when the last token was not fed back).
    public var decodeSteps = 0
    public var prefillSeconds = 0.0
    /// Wall time of the decode forwards (sampling included).
    public var decodeSeconds = 0.0
    /// Prompt tokens found already in the KV cache (chat turns).
    public var reusedTokens = 0
    public var expert: ExpertStoreStats?
    public init() {}

    public var prefillTokensPerSecond: Double { prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0 }
    public var decodeTokensPerSecond: Double { decodeSeconds > 0 ? Double(decodeSteps) / decodeSeconds : 0 }
}

/// One sampling step, for `--dump` (oracle comparisons): the chosen id and the `k` best
/// logits it was chosen from.
public struct StepTrace: Sendable {
    public var step: Int
    /// Absolute position the logits were computed at (the last fed token's).
    public var position: Int
    public var token: Int32
    public var top: [(id: Int32, logit: Float)]
}

/// The generation loop shared by `generate`, `chat` and `bench`: prefill in chunks of
/// `maxBatch`, then decode one token at a time, sampling on the host from the logits,
/// stopping on a stop token or `maxTokens`.
public final class Generator {
    public let loaded: LoadedModel
    public let state: InferenceState
    public var sampler: Sampler
    public var stopTokens: Set<Int32>
    /// Tokens whose keys and values are in the cache, in position order.
    public private(set) var context: [Int32] = []
    /// When > 0, `onTrace` gets the top `traceTopK` logits of every step.
    public var traceTopK = 0
    public var onTrace: ((StepTrace) -> Void)?
    /// Gets every step's full logits (`vocab` floats) before sampling.
    public var onLogits: ((Int, UnsafeBufferPointer<Float>) -> Void)?

    public init(loaded: LoadedModel, sampler: SamplerConfig) throws {
        self.loaded = loaded
        self.state = try loaded.model.makeState()
        self.sampler = Sampler(sampler)
        self.stopTokens = loaded.stopTokens
    }

    public var maxBatch: Int { state.scratch.maxBatch }
    public var capacity: Int { state.kv.layout.maxTokens }

    /// Forgets the sequence and the sampler's history.
    public func reset() {
        state.reset()
        context.removeAll()
        sampler.reset()
    }

    /// Feeds `prompt` (appended to the current sequence) and generates up to `maxTokens`
    /// tokens. `onToken` gets each id; return false to stop. `ignoreStop` keeps going past
    /// stop tokens (oracle comparisons do).
    public func generate(prompt: [Int32], maxTokens: Int, ignoreStop: Bool = false,
                         onToken: (Int32) -> Bool) throws -> GenerationStats {
        guard !prompt.isEmpty || !context.isEmpty else { throw ColibriError.invalidArgument("empty prompt") }
        var feed = prompt
        if feed.isEmpty {
            // Nothing new: recompute the last cached token to get logits.
            feed = [context.removeLast()]
            state.position -= 1
        }
        return try run(feed: feed, reused: 0, maxTokens: maxTokens, ignoreStop: ignoreStop, onToken: onToken)
    }

    /// Generates after `sequence`, the whole conversation so far: the longest prefix it
    /// shares with the cached tokens is kept, the rest of the cache is overwritten. Chat
    /// turns re-render the full conversation, so the cache is reused up to the first
    /// token where the new rendering differs from what was run. With Qwen3's template
    /// and thinking off that is the start of the previous reply: the generation prompt
    /// ends in an empty `<think>\n\n</think>\n\n` block that the history rendering of
    /// that reply drops (as the HF template does), so each turn re-prefills the previous
    /// reply plus the new user message, not only the new text. Feeding the model what
    /// its template renders is the point; the extra prefill is the reply's length.
    public func generate(sequence: [Int32], maxTokens: Int, ignoreStop: Bool = false,
                         onToken: (Int32) -> Bool) throws -> GenerationStats {
        guard !sequence.isEmpty else { throw ColibriError.invalidArgument("empty prompt") }
        var keep = 0
        while keep < context.count && keep < sequence.count && context[keep] == sequence[keep] { keep += 1 }
        keep = min(keep, sequence.count - 1)  // the last token must run to produce logits
        context.removeSubrange(keep...)
        state.position = keep
        return try run(feed: Array(sequence[keep...]), reused: keep, maxTokens: maxTokens,
                       ignoreStop: ignoreStop, onToken: onToken)
    }

    /// Runs `tokens` through the model at the current position (chunked by `maxBatch`)
    /// and returns the last row's logits, host-readable until the next forward.
    public func prefill(_ tokens: [Int32]) throws -> BufferSlice {
        guard !tokens.isEmpty else { throw ColibriError.invalidArgument("empty prefill") }
        guard state.position + tokens.count <= capacity else {
            throw ColibriError.invalidArgument(
                "prompt of \(tokens.count) tokens at position \(state.position) does not fit the \(capacity)-token context (--ctx)")
        }
        var logits: BufferSlice?
        var i = 0
        while i < tokens.count {
            let chunk = Array(tokens[i ..< min(tokens.count, i + maxBatch)])
            logits = try loaded.model.forward(chunk, state: state)
            context += chunk
            i += chunk.count
        }
        return logits!
    }

    private func run(feed: [Int32], reused: Int, maxTokens: Int, ignoreStop: Bool,
                     onToken: (Int32) -> Bool) throws -> GenerationStats {
        var stats = GenerationStats()
        stats.reusedTokens = reused
        stats.promptTokens = feed.count
        sampler.accept(feed)

        let t0 = DispatchTime.now().uptimeNanoseconds
        var logits = try prefill(feed)
        let t1 = DispatchTime.now().uptimeNanoseconds
        stats.prefillSeconds = Double(t1 - t0) / 1e9

        var step = 0
        while step < maxTokens {
            let host = UnsafeBufferPointer(logits.floats)
            onLogits?(step, host)
            let token = sampler.sample(host)
            if traceTopK > 0, let onTrace {
                onTrace(StepTrace(step: step, position: state.position - 1, token: token,
                                  top: Generator.top(host, traceTopK)))
            }
            stats.generatedTokens += 1
            step += 1
            if !onToken(token) { break }
            if !ignoreStop && stopTokens.contains(token) { break }
            if step == maxTokens || state.position >= capacity { break }
            logits = try loaded.model.forward([token], state: state)
            context.append(token)
            stats.decodeSteps += 1
        }
        stats.decodeSeconds = Double(DispatchTime.now().uptimeNanoseconds - t1) / 1e9
        stats.expert = loaded.model.expertStats
        return stats
    }

    /// The `k` largest logits, best first; ties go to the lower id.
    public static func top(_ logits: UnsafeBufferPointer<Float>, _ k: Int) -> [(id: Int32, logit: Float)] {
        var best: [(id: Int32, logit: Float)] = []
        best.reserveCapacity(k + 1)
        for (i, v) in logits.enumerated() {
            if best.count == k, let last = best.last, !(v > last.logit) { continue }
            var at = best.count
            while at > 0 && v > best[at - 1].logit { at -= 1 }
            best.insert((Int32(i), v), at: at)
            if best.count > k { best.removeLast() }
        }
        return best
    }
}
