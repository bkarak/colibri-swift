import Colibri
import ColibriCore
import Foundation
import Testing

// The facade on a tiny random qwen3 (byte vocabulary with ChatML control tokens): its text
// must be what the layers below give, streams must stop when abandoned, and a chat must
// reuse its cached prefix and trim when it outgrows the context.
@Suite struct ColibriEngineTests {
    static func tinyModel() throws -> (dir: String, path: String) {
        let dir = NSTemporaryDirectory() + "colibri-engine-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = dir + "/tiny-qwen3.gguf"
        try TinyModel.write(TinyModelSpec(architecture: "qwen3", weightType: .q8_0), to: path)
        return (dir, path)
    }

    static func cpu(_ math: EngineConfiguration.CPUMath = .int8, context: Int = 4096) -> EngineConfiguration {
        var c = EngineConfiguration()
        c.compute = .cpu
        c.cpuMath = math
        c.contextLength = context
        c.saveRoutingHistory = false
        return c
    }

    /// The same chat through `Generator` directly.
    static func directReply(path: String, messages: [ChatMessage], maxTokens: Int,
                            integer: Bool) throws -> String {
        let ctx = LoadContext(backend: CPUBackend(integerActivations: integer), maxContext: 4096)
        let m = try ModelRegistry.load(path: path, context: ctx)
        let g = try Generator(loaded: m, sampler: .greedy)
        let ids = m.tokenizer.encode(m.chatTemplate(thinking: false).render(messages))
        var reply: [Int32] = []
        let stop = g.stopTokens
        _ = try g.generate(prompt: ids, maxTokens: maxTokens) { id in
            if !stop.contains(id) { reply.append(id) }
            return true
        }
        return m.tokenizer.decode(reply)
    }

    @Test(arguments: [EngineConfiguration.CPUMath.int8, .float])
    func replyIsWhatTheGeneratorGives(_ math: EngineConfiguration.CPUMath) async throws {
        let fx = try Self.tinyModel()
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        let engine = try await ColibriEngine.load(modelPath: fx.path, configuration: Self.cpu(math))
        #expect(engine.architecture == "qwen3" && engine.backendName == "cpu")
        let messages = [ChatMessage(role: .user, content: "hello there")]
        let text = try await engine.reply(to: messages, options: GenerationOptions(maxTokens: 24))
        let want = try Self.directReply(path: fx.path, messages: messages, maxTokens: 24, integer: math == .int8)
        #expect(text == want)
        #expect(!text.isEmpty)
        #expect(engine.lastStats?.generatedTokens ?? 0 > 0)
    }

    @Test func completionStreamsAndStopsWhenAbandoned() async throws {
        let fx = try Self.tinyModel()
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        let engine = try ColibriEngine(modelPath: fx.path, configuration: Self.cpu())
        var pieces = 0
        for try await _ in engine.complete("abc", options: GenerationOptions(maxTokens: 400)) {
            pieces += 1
            if pieces == 3 { break }
        }
        // `lastStats` waits on the engine's queue for the abandoned generation, which must
        // have stopped soon after the consumer left, far short of its 400 tokens.
        #expect(pieces == 3)
        let ran = try #require(engine.lastStats).generatedTokens
        #expect(ran >= 3 && ran < 50, "abandoned generation ran \(ran) tokens")
        // The engine is usable afterwards.
        let next = try await engine.reply(to: [ChatMessage(role: .user, content: "x")], options: GenerationOptions(maxTokens: 4))
        #expect(!next.isEmpty)
    }

    @Test func chatReusesTheCacheAndTrims() async throws {
        let fx = try Self.tinyModel()
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        let engine = try ColibriEngine(modelPath: fx.path, configuration: Self.cpu(context: 256))
        let chat = ChatSession(engine: engine, system: "be brief", options: GenerationOptions(maxTokens: 16))
        _ = try await chat.reply("first question")
        _ = try await chat.reply("second")
        let reused = engine.lastStats?.reusedTokens ?? 0
        #expect(reused > 0, "turn 2 should reuse turn 1's prefix")
        #expect(chat.messages.map(\.role) == [.system, .user, .assistant, .user, .assistant])
        // Long turns overflow the 256-token context: the engine drops old turns, keeps going.
        for i in 0..<6 { _ = try await chat.reply(String(repeating: "long turn \(i) ", count: 6)) }
        #expect(chat.messages.count == 1 + 2 * 8)
        #expect(engine.lastStats?.promptTokens ?? 0 <= 256)
    }

    @Test func oversizedPromptThrowsWithoutTrimming() async throws {
        let fx = try Self.tinyModel()
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        let engine = try ColibriEngine(modelPath: fx.path, configuration: Self.cpu(context: 64))
        await #expect(throws: ColibriError.self) {
            for try await _ in engine.complete(String(repeating: "x", count: 200)) {}
        }
    }

    @Test func callsFromManyTasksRunOneAtATime() async throws {
        let fx = try Self.tinyModel()
        defer { try? FileManager.default.removeItem(atPath: fx.dir) }
        let engine = try ColibriEngine(modelPath: fx.path, configuration: Self.cpu())
        let replies = try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 0..<4 {
                group.addTask { try await engine.reply(to: [ChatMessage(role: .user, content: "same")],
                                                       options: GenerationOptions(maxTokens: 12)) }
            }
            return try await group.reduce(into: [String]()) { $0.append($1) }
        }
        #expect(Set(replies).count == 1, "greedy replies to the same chat must agree")
    }
}
