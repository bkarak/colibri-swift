import Foundation
import Testing
@testable import ColibriCore

// Checks the Swift engine against upstream colibri's recorded outputs
// (Fixtures/oracle/, written by scripts/oracle/record.py). Three layers:
//
//  1. the generator still writes the exact files upstream was run on (fingerprints);
//  2. the Swift tokenizer turns each fixture prompt into upstream's prompt ids;
//  3. the engine (CPU backend, f32 KV cache) reproduces upstream's GQ_EXACT=1 prefill
//     logits and greedy ids. While the model families are stubs (ModelRegistry.load
//     throws ColibriError.notImplemented) these tests print a skip line and pass.
//
// By default a small subset runs; environment switches:
//   COLIBRI_ORACLE_FULL=1   every tiny case (slower: the generator runs in a debug build)
//   COLIBRI_ORACLE_REAL=1   also run the engine on Qwen3-0.6B (very slow in a debug build;
//                           use `swift test -c release -Xswiftc -enable-testing`)

struct OracleFixture: Decodable {
    struct Model: Decodable {
        var fingerprint: String?
        var bytes: Int
        var architecture: String
        var vocabSize: Int
    }
    struct Step: Decodable {
        var top: [[Double]]  // [[id, logit], ...], best first
        var gap: Float
    }
    struct Prompt: Decodable {
        var name: String?
        var text: String
        var promptIds: [Int32]
        var greedyExact: [Int32]
        var greedyDefault: [Int32]
        var stepsExact: [Step]
        var prefillLogitsExact: [Float]?
        var prefillTopExact: [[Double]]?
    }
    var `case`: String?
    var model: Model
    var nNew: Int
    var prompts: [Prompt]
}

enum Oracle {
    static let env = ProcessInfo.processInfo.environment
    static let full = env["COLIBRI_ORACLE_FULL"] == "1"
    static let real = env["COLIBRI_ORACLE_REAL"] == "1"

    /// Logit tolerance against upstream's float reference (different summation order only).
    static let tinyLogitTolerance: Float = 2e-3
    static let realLogitTolerance: Float = 2e-2
    /// A divergence where upstream's top-2 gap is below this is reported as a near tie.
    static let nearTie: Float = 1e-2

    static let defaultCases = ["qwen3_q8_0_tied", "qwen3moe_q4_K", "olmoe_q4_K_M"]

    static var engineCases: [String] {
        full ? TinyOracleCases.all.filter { !$0.needsRealTokenizer || realModelPresent }.map(\.name) : defaultCases
    }

    static var realModelPresent: Bool { FileManager.default.fileExists(atPath: TinyOracleCases.qwen3TokenizerSource) }

    static func url(_ relative: String) -> URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures/oracle/\(relative)")
    }

    static func load(_ relative: String) throws -> OracleFixture {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return try d.decode(OracleFixture.self, from: Data(contentsOf: url(relative)))
    }

    static func tinyCase(_ name: String) throws -> TinyOracleCase {
        try #require(TinyOracleCases.all.first { $0.name == name }, "no TinyOracleCases entry \(name)")
    }

    static func argmaxFirst(_ x: UnsafeMutableBufferPointer<Float>) -> Int32 {
        var best = 0
        for i in 1..<x.count where x[i] > x[best] { best = i }
        return Int32(best)
    }

    struct Outcome {
        var ids: [Int32] = []
        var prefill: [Float] = []
    }

    /// Greedy decode with the CPU backend; nil while the model code is still a stub.
    static func run(path: String, prompt: [Int32], steps: Int) throws -> Outcome? {
        let backend = CPUBackend()
        let context = LoadContext(backend: backend, maxContext: 512, kvType: .f32)
        let loaded: LoadedModel
        do {
            loaded = try ModelRegistry.load(path: path, context: context)
        } catch ColibriError.notImplemented(let what) {
            print("[oracle] skip: engine not implemented (\(what))")
            return nil
        }
        let state = try loaded.model.makeState()
        var out = Outcome()
        // Prefill in chunks of maxBatch rows; the last chunk's logits are the prompt's.
        var start = 0
        var logits = UnsafeMutableBufferPointer<Float>(start: nil, count: 0)
        while start < prompt.count {
            let end = min(start + context.maxBatch, prompt.count)
            logits = try loaded.model.forward(Array(prompt[start..<end]), state: state).floats
            start = end
        }
        out.prefill = Array(logits)
        for s in 0..<steps {
            let t = argmaxFirst(logits)
            out.ids.append(t)
            if s + 1 < steps { logits = try loaded.model.forward([t], state: state).floats }
        }
        return out
    }

    /// Compares one prompt's outcome with the fixture; returns failures (near ties are printed only).
    static func compare(_ label: String, _ p: OracleFixture.Prompt, _ o: Outcome, tolerance: Float) -> [String] {
        var failures: [String] = []
        if let ref = p.prefillLogitsExact {
            if ref.count != o.prefill.count {
                failures.append("\(label): \(o.prefill.count) logits, upstream \(ref.count)")
            } else {
                var worst: Float = 0, at = 0
                for i in ref.indices where abs(ref[i] - o.prefill[i]) > worst { worst = abs(ref[i] - o.prefill[i]); at = i }
                print("[oracle] \(label): prefill max |swift - upstream| = \(worst) (token \(at))")
                if !(worst <= tolerance) { failures.append("\(label): prefill logit \(at) differs by \(worst) > \(tolerance)") }
            }
        } else if let top = p.prefillTopExact {
            var worst: Float = 0
            for e in top {
                let i = Int(e[0])
                guard i < o.prefill.count else { failures.append("\(label): no logit \(i)"); break }
                worst = max(worst, abs(o.prefill[i] - Float(e[1])))
            }
            print("[oracle] \(label): prefill top-\(top.count) max |swift - upstream| = \(worst)")
            if !(worst <= tolerance) { failures.append("\(label): prefill top logits differ by \(worst) > \(tolerance)") }
        }
        let common = min(o.ids.count, p.greedyExact.count)
        let diverged = (0..<common).first { o.ids[$0] != p.greedyExact[$0] }
        if let i = diverged ?? (o.ids.count < p.greedyExact.count ? o.ids.count : nil) {
            let step = i < p.stepsExact.count ? p.stepsExact[i] : nil
            let swiftID = i < o.ids.count ? String(o.ids[i]) : "none"
            let msg = "\(label): first divergence at step \(i): swift \(swiftID), upstream \(p.greedyExact[i])"
                + (step.map { " (upstream top-2 gap \($0.gap), top \($0.top.prefix(3).map { "\(Int($0[0])):\($0[1])" }.joined(separator: " ")))" } ?? "")
            if let s = step, s.gap < nearTie {
                print("[oracle] near tie, not a failure: \(msg)")
            } else {
                failures.append(msg)
            }
        } else {
            print("[oracle] \(label): \(o.ids.count)/\(p.greedyExact.count) greedy ids match")
        }
        return failures
    }
}

@Suite struct OracleTests {
    @Test func everyCaseHasAFixture() throws {
        for c in TinyOracleCases.all {
            let f = try Oracle.load("tiny/\(c.name).json")
            #expect(f.case == c.name)
            #expect(f.model.architecture == c.spec.architecture)
            #expect(!f.prompts.isEmpty)
            for p in f.prompts {
                #expect(p.greedyExact.count == f.nNew && p.greedyDefault.count == f.nNew && p.stepsExact.count == f.nNew)
                // The recorded steps are self-consistent: the emitted id is the top logit.
                for (id, s) in zip(p.greedyExact, p.stepsExact) { #expect(Int32(s.top[0][0]) == id) }
            }
        }
        let real = try Oracle.load("qwen3-0.6b-q8_0.json")
        #expect(real.prompts.count == 5)
        // models.md section 6/10 golden prompt and answer.
        #expect(real.prompts[0].promptIds == [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271])
        #expect(Array(real.prompts[0].greedyExact.prefix(10)) == [785, 6722, 315, 9625, 374, 3070, 59604, 334, 13, 151645])
    }

    /// The generator must still write the bytes upstream was run on, or the fixtures are stale.
    @Test(arguments: Oracle.full ? TinyOracleCases.all.map(\.name) : Oracle.defaultCases)
    func generatorReproducesFixtureModel(name: String) throws {
        let c = try Oracle.tinyCase(name)
        if c.needsRealTokenizer && !Oracle.realModelPresent { return }
        let f = try Oracle.load("tiny/\(name).json")
        let bytes = try TinyModel.build(c.spec).serialize()
        #expect(bytes.count == f.model.bytes)
        #expect(TinyModel.fingerprint(bytes) == f.model.fingerprint,
                "TinyModel output changed for \(name): re-record with scripts/oracle/record.py --tiny")
    }

    @Test func tokenizerReproducesUpstreamPromptIds() throws {
        for c in TinyOracleCases.all {
            if c.needsRealTokenizer && !Oracle.realModelPresent { continue }
            let f = try Oracle.load("tiny/\(c.name).json")
            let tok = try BPETokenizer(metadata: TinyModel.metadata(c.spec))
            for p in f.prompts { #expect(tok.encode(p.text) == p.promptIds, "\(c.name): \(p.text.debugDescription)") }
        }
        guard Oracle.realModelPresent else { return }
        let real = try Oracle.load("qwen3-0.6b-q8_0.json")
        let tok = try BPETokenizer(metadata: GGUFFile(path: TinyOracleCases.qwen3TokenizerSource).metadata)
        for p in real.prompts { #expect(tok.encode(p.text) == p.promptIds, "\(p.name ?? ""): \(p.text.debugDescription)") }
    }

    @Test(arguments: Oracle.engineCases)
    func engineMatchesUpstreamOnTinyModel(name: String) throws {
        let c = try Oracle.tinyCase(name)
        let f = try Oracle.load("tiny/\(name).json")
        let path = NSTemporaryDirectory() + "colibri-oracle-\(name)-\(getpid()).gguf"
        try TinyModel.write(c.spec, to: path)
        defer { try? FileManager.default.removeItem(atPath: path) }
        var failures: [String] = []
        for (i, p) in f.prompts.enumerated() {
            guard let o = try Oracle.run(path: path, prompt: p.promptIds, steps: f.nNew) else { return }
            failures += Oracle.compare("\(name)#\(i)", p, o, tolerance: Oracle.tinyLogitTolerance)
        }
        for m in failures { Issue.record(Comment(rawValue: m)) }
    }

    @Test(.enabled(if: Oracle.real && Oracle.realModelPresent))
    func engineMatchesUpstreamOnQwen3_06B() throws {
        let f = try Oracle.load("qwen3-0.6b-q8_0.json")
        var failures: [String] = []
        for p in f.prompts {
            guard let o = try Oracle.run(path: TinyOracleCases.qwen3TokenizerSource, prompt: p.promptIds, steps: f.nNew) else { return }
            failures += Oracle.compare("qwen3-0.6b/\(p.name ?? "?")", p, o, tolerance: Oracle.realLogitTolerance)
        }
        for m in failures { Issue.record(Comment(rawValue: m)) }
    }

    /// The real MoE model (18.6 GB, `COLIBRI_QWEN3MOE_GGUF` or the default download path).
    /// Its fixture was recorded in light mode: greedy ids and prefill logits only.
    static var qwen3MoEPath: String {
        ProcessInfo.processInfo.environment["COLIBRI_QWEN3MOE_GGUF"]
            ?? NSHomeDirectory() + "/colibri-models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf"
    }

    @Test(.enabled(if: Oracle.real && FileManager.default.fileExists(atPath: OracleTests.qwen3MoEPath)))
    func engineMatchesUpstreamOnQwen3_30B_A3B() throws {
        let f = try Oracle.load("qwen3-30b-a3b-q4_k_m.json")
        var failures: [String] = []
        for p in f.prompts {
            guard let o = try Oracle.run(path: Self.qwen3MoEPath, prompt: p.promptIds, steps: f.nNew) else { return }
            failures += Oracle.compare("qwen3-30b-a3b/\(p.name ?? "?")", p, o, tolerance: Oracle.realLogitTolerance)
        }
        for m in failures { Issue.record(Comment(rawValue: m)) }
    }

    /// Real OLMoE-1B-7B-0125-Instruct (4.2 GB, `COLIBRI_OLMOE_GGUF` or the default download
    /// path). Upstream's olmoe engine has no token dump, so the fixture's prompt ids come from
    /// the model's own HF tokenizer (scripts/oracle/hf_tokenize.py); the port must match them.
    static var olmoePath: String {
        ProcessInfo.processInfo.environment["COLIBRI_OLMOE_GGUF"]
            ?? NSHomeDirectory() + "/colibri-models/olmoe-1b-7b/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf"
    }

    @Test(.enabled(if: Oracle.real && FileManager.default.fileExists(atPath: OracleTests.olmoePath)))
    func engineMatchesUpstreamOnOLMoE() throws {
        let f = try Oracle.load("olmoe-1b-7b-0125-instruct-q4_k_m.json")
        let tok = try BPETokenizer(metadata: GGUFFile(path: Self.olmoePath).metadata)
        var failures: [String] = []
        for p in f.prompts {
            #expect(tok.encode(p.text) == p.promptIds, "olmoe tokenizer vs HF: \(p.name ?? "?")")
            guard let o = try Oracle.run(path: Self.olmoePath, prompt: p.promptIds, steps: f.nNew) else { return }
            failures += Oracle.compare("olmoe/\(p.name ?? "?")", p, o, tolerance: Oracle.realLogitTolerance)
        }
        for m in failures { Issue.record(Comment(rawValue: m)) }
    }
}
