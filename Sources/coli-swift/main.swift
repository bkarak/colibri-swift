// coli-swift: command-line front end for the Swift port of colibri.

import ColibriCore
import ColibriMetal
import Foundation

/// The CPU backend: int8 activations unless `--float` (`CPUBackend` itself defaults to the
/// float path, the reference the tests compare against).
func makeCPU(_ o: CLIOptions) -> Backend {
    CPUBackend(threads: o.threads, integerActivations: (o.cpuMath ?? .int8) == .int8)
}

/// The backend `--backend` names. `auto` is Metal on Apple Silicon and the CPU
/// otherwise (and for `--int8` or `--float`, CPU-only options); `load` falls back to
/// the CPU when Metal cannot start or cannot load the model.
func makeBackend(_ o: CLIOptions) throws -> Backend {
    switch o.backend {
    case .cpu: return makeCPU(o)
    case .metal:
        if o.cpuMath != nil { throw CLIOptions.ParseError(description: "--int8 and --float are CPU backend options") }
        return try MetalBackend()
    case .auto:
        #if arch(arm64)
        if o.cpuMath != nil { return makeCPU(o) }
        do {
            return try MetalBackend()
        } catch {
            err("coli-swift: Metal unavailable (\(error)); using the CPU backend\n")
            return makeCPU(o)
        }
        #else
        return makeCPU(o)
        #endif
    }
}

// MARK: Output helpers

func err(_ s: String) {
    FileHandle.standardError.write(Data(s.utf8))
}

func out(_ s: String) {
    guard !s.isEmpty else { return }
    FileHandle.standardOutput.write(Data(s.utf8))
}

func bytesString(_ n: Int) -> String {
    let d = Double(n)
    if n >= 1 << 30 { return String(format: "%.2f GiB", d / 1_073_741_824) }
    if n >= 1 << 20 { return String(format: "%.1f MiB", d / 1_048_576) }
    if n >= 1 << 10 { return String(format: "%.1f KiB", d / 1024) }
    return "\(n) B"
}

func idsString(_ ids: [Int32]) -> String { ids.map(String.init).joined(separator: " ") }

func statsLine(_ s: GenerationStats) -> String {
    var line = String(format: "prefill: %d tokens in %.3f s (%.1f tok/s)", s.promptTokens, s.prefillSeconds, s.prefillTokensPerSecond)
    if s.reusedTokens > 0 { line += " [\(s.reusedTokens) reused from cache]" }
    line += String(format: " | decode: %d tokens in %.3f s (%.1f tok/s)", s.decodeSteps, s.decodeSeconds, s.decodeTokensPerSecond)
    return line
}

func expertStore(_ m: LoadedModel) -> ExpertStore? {
    m.model.expertStore
}

/// "resident (...)" or "streamed: N of E slots per layer (cache X of Y)".
func placement(_ store: ExpertStore) -> String {
    let layout = store.layout
    if store.isFullyResident { return "resident (zero-copy, \(bytesString(layout.totalBytes)))" }
    guard let streaming = store as? StreamingExpertStore else { return "streamed through a slot cache" }
    let slots = streaming.slotsPerLayer
    var cacheBytes = 0, experts = 0
    for (l, layer) in layout.layers.enumerated() {
        guard let layer else { continue }
        experts = max(experts, layer.count)
        cacheBytes += slots[l] * ExpertArena.slotStride(for: layer)
    }
    let moeSlots = layout.moeLayers.map { slots[$0] }
    let slotText = Set(moeSlots).count == 1 ? "\(moeSlots[0])" : "\(moeSlots.min() ?? 0)-\(moeSlots.max() ?? 0)"
    return "streamed: \(slotText) of \(experts) slots per layer (cache \(bytesString(cacheBytes)) for \(bytesString(layout.totalBytes)) of experts)"
}

func expertLine(_ e: ExpertStoreStats) -> String {
    String(format: "experts: lookups %d, hit rate %.1f%%, misses %d, evictions %d, read %@ (%.2f GB/s), wait %.3f s",
           e.lookups, e.hitRate * 100, e.misses, e.evictions, bytesString(e.bytesRead), e.readGBPerSecond, e.waitSeconds)
}

func modelSummary(_ m: LoadedModel, backend: Backend, path: String) -> String {
    let c = m.config
    var s = "model: \(path)\n  \(c.architecture): \(c.layers) layers, D \(c.hidden), heads \(c.heads)/\(c.kvHeads), head dim \(c.headDim), vocab \(c.vocab)"
    if c.isMoE {
        s += ", experts \(c.experts) (top \(c.expertsUsed), width \(c.expertFFNDim), renorm \(c.normTopK))"
    } else {
        s += ", FFN \(c.ffnDim)"
    }
    s += "\n  backend \(backend.capabilities.name)"
    if let cpu = backend as? CPUBackend {
        s += cpu.integerActivations ? " (int8 activations)" : " (float path)"
        s += ", \(cpu.threads) threads"
    }
    if let store = expertStore(m) {
        s += ", experts " + placement(store)
    }
    return s + "\n"
}

/// Loads the model named by the options on the chosen backend. With `--backend auto`,
/// a model Metal cannot load (an unsupported weight type, say) is loaded on the CPU.
func load(_ o: CLIOptions) throws -> (LoadedModel, Backend) {
    guard let path = o.modelPath else { throw CLIOptions.ParseError(description: "needs --model PATH") }
    var backend = try makeBackend(o)
    let t0 = Date()
    let m: LoadedModel
    do {
        m = try ModelRegistry.load(path: path, context: o.loadContext(backend))
    } catch where o.backend == .auto && !(backend is CPUBackend) {
        err("coli-swift: \(backend.capabilities.name) cannot load the model (\(error)); using the CPU backend\n")
        backend = makeCPU(o)
        m = try ModelRegistry.load(path: path, context: o.loadContext(backend))
    }
    err(modelSummary(m, backend: backend, path: path))
    err(String(format: "  loaded in %.2f s\n", Date().timeIntervalSince(t0)))
    return (m, backend)
}

func makeGenerator(_ m: LoadedModel, _ o: CLIOptions) throws -> Generator {
    let g = try Generator(loaded: m, sampler: o.samplerConfig)
    if o.dump {
        g.traceTopK = o.dumpTop
        g.onTrace = { t in
            let top = t.top.map { "\($0.id):\(String(format: "%.6f", $0.logit))" }.joined(separator: " ")
            err("step \(t.step) pos \(t.position) id \(t.token) top\(t.top.count) \(top)\n")
        }
    }
    if let path = o.dumpLogits {
        guard FileManager.default.createFile(atPath: path, contents: nil),
              let h = FileHandle(forWritingAtPath: path) else {
            throw CLIOptions.ParseError(description: "--dump-logits: cannot create \(path)")
        }
        g.onLogits = { _, logits in h.write(Data(buffer: logits)) }
    }
    return g
}

/// Runs one generation with streamed text on stdout; returns the generated ids.
func streamGeneration(_ g: Generator, _ m: LoadedModel, _ o: CLIOptions, sequence: [Int32]? = nil,
                      prompt: [Int32]? = nil) throws -> ([Int32], GenerationStats) {
    var decoder = StreamingDecoder(tokenizer: m.tokenizer)
    var ids: [Int32] = []
    let stop = g.stopTokens
    let onToken: (Int32) -> Bool = { id in
        ids.append(id)
        if o.ignoreEOS || !stop.contains(id) { out(decoder.push(id)) }
        return true
    }
    let stats: GenerationStats
    if let sequence {
        stats = try g.generate(sequence: sequence, maxTokens: o.maxTokens, ignoreStop: o.ignoreEOS, onToken: onToken)
    } else {
        stats = try g.generate(prompt: prompt ?? [], maxTokens: o.maxTokens, ignoreStop: o.ignoreEOS, onToken: onToken)
    }
    out(decoder.flush())
    return (ids, stats)
}

// MARK: Commands

func runInfo(_ o: CLIOptions) throws {
    guard let path = o.modelPath else { throw CLIOptions.ParseError(description: "info needs a GGUF path") }
    let f = try GGUFFile(path: path)
    var s = "file: \(path)\n"
    s += "  GGUF v\(f.version), \(bytesString(f.file.fileSize)), alignment \(f.alignment), data at byte \(f.dataStart)\n"
    if f.file.segments.count > 1 {
        s += "  split model, \(f.file.segments.count) shards: \(f.file.segments.map { ($0.path as NSString).lastPathComponent }.joined(separator: ", "))\n"
    }
    s += "  \(f.metadata.keys.count) metadata keys, \(f.tensors.count) tensors\n"
    let arch = f.metadata.architecture ?? "?"
    if let family = ModelRegistry.family(for: arch) {
        s += "\narchitecture: \(arch) -> \(family)\n"
        do {
            let c = try family.config(from: f)
            s += "  layers \(c.layers), hidden \(c.hidden), heads \(c.heads), kv heads \(c.kvHeads), head dim \(c.headDim), value dim \(c.valueDim)\n"
            s += "  vocab \(c.vocab), context \(c.contextLength), rope theta \(c.ropeTheta) (dims \(c.ropeDims)), rms eps \(c.rmsEps)\n"
            s += "  q/k norm \(c.qkNorm), tied embeddings \(c.tiedEmbeddings)\n"
            if c.isMoE {
                s += "  experts \(c.experts), used \(c.expertsUsed), expert FFN \(c.expertFFNDim), renormalise top-k \(c.normTopK)\n"
                if let layout = try? ExpertLayout(file: f, layerCount: c.layers) {
                    s += "  routed expert bytes \(bytesString(layout.totalBytes)) of \(bytesString(f.file.fileSize)) in \(layout.moeLayers.count) MoE layers\n"
                }
            } else {
                s += "  dense FFN \(c.ffnDim)\n"
            }
        } catch {
            s += "  cannot run: \(error)\n"
        }
    } else {
        s += "\narchitecture: \(arch) (not supported; this build runs \(ModelRegistry.supportedArchitectures.joined(separator: ", ")))\n"
    }

    s += "\nmetadata:\n"
    for key in f.metadata.keys {
        guard let v = f.metadata[key] else { continue }
        s += "  \(key) = \(describe(v))\n"
    }

    var byType: [String: (count: Int, bytes: Int)] = [:]
    for t in f.tensors {
        let k = t.type.description
        byType[k, default: (0, 0)].count += 1
        byType[k, default: (0, 0)].bytes += t.byteCount
    }
    s += "\ntensor types: " + byType.sorted { $0.key < $1.key }
        .map { "\($0.key) x\($0.value.count) (\(bytesString($0.value.bytes)))" }.joined(separator: ", ") + "\n"
    s += "\ntensors:\n"
    for t in f.tensors {
        let dims = t.dims.map(String.init).joined(separator: ", ")
        s += "  \(t.name.padding(toLength: max(36, t.name.count), withPad: " ", startingAt: 0)) \(t.type.description.padding(toLength: 6, withPad: " ", startingAt: 0)) [\(dims)]  @\(t.fileOffset)  \(bytesString(t.byteCount))\n"
    }
    out(s)
}

func describe(_ v: GGUFValue) -> String {
    if let a = v.arrayValue {
        let head = a.prefix(4).map(describe).joined(separator: ", ")
        return "[\(a.count) items: \(head)\(a.count > 4 ? ", ..." : "")]"
    }
    if let str = v.stringValue {
        let one = str.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return one.count > 100 ? "\"\(one.prefix(100))...\" (\(str.count) chars)" : "\"\(one)\""
    }
    if let i = v.intValue { return String(i) }
    if let d = v.floatValue { return String(d) }
    return "\(v)"
}

/// The prompt ids for generate/bench: `--prompt-ids`, else the (templated) text.
func promptIDs(_ m: LoadedModel, _ o: CLIOptions, text: String) -> [Int32] {
    if let ids = o.promptIDs { return ids }
    if o.raw { return m.tokenizer.encode(text) }
    var messages: [ChatMessage] = []
    if let sys = o.system { messages.append(ChatMessage(role: .system, content: sys)) }
    messages.append(ChatMessage(role: .user, content: text))
    return m.tokenizer.encode(m.chatTemplate(thinking: o.thinking).render(messages))
}

func runGenerate(_ o: CLIOptions) throws {
    var text = o.prompt
    if text == nil && o.promptIDs == nil {
        let data = FileHandle.standardInput.readDataToEndOfFile()
        text = String(decoding: data, as: UTF8.self)
    }
    let (m, _) = try load(o)
    for id in o.promptIDs ?? [] where Int(id) >= m.config.vocab {
        throw CLIOptions.ParseError(description: "--prompt-ids: \(id) is outside the vocabulary of \(m.config.vocab)")
    }
    let prompt = promptIDs(m, o, text: text ?? "")
    guard !prompt.isEmpty else { throw CLIOptions.ParseError(description: "empty prompt") }
    let g = try makeGenerator(m, o)
    if o.dump { err("prompt_ids (\(prompt.count)): \(idsString(prompt))\n") }
    let (ids, stats) = try streamGeneration(g, m, o, prompt: prompt)
    out("\n")
    if o.dump { err("generated_ids (\(ids.count)): \(idsString(ids))\n") }
    err(statsLine(stats) + "\n")
    if let e = stats.expert, let store = expertStore(m) {
        // Expert cache counters for every MoE run (a resident store reads nothing).
        err(store.isFullyResident ? "experts: \(placement(store)), no reads\n"
                                  : expertLine(e) + "\n  cache: " + placement(store) + "\n")
    }
    saveHistory(m)
}

/// Set once a history save has failed, so chat warns once, not every turn.
var historySaveWarned = false

/// Saves the routing history (`<model>.coli_usage`). Not fatal: the history only steers
/// which experts are pinned next time, so a failed save (a read-only model directory,
/// say) is a warning and the run goes on, as upstream's `rt_save` returns a status.
func saveHistory(_ m: LoadedModel) {
    do {
        try expertStore(m)?.saveHistory()
    } catch {
        if !historySaveWarned {
            err("coli-swift: warning: routing history not saved (\(error)); continuing without it\n")
            historySaveWarned = true
        }
    }
}

func runChat(_ o: CLIOptions) throws {
    let (m, _) = try load(o)
    let template: ChatTemplate = o.raw ? RawTemplate() : m.chatTemplate(thinking: o.thinking)
    let g = try makeGenerator(m, o)
    var messages: [ChatMessage] = []
    if let sys = o.system { messages.append(ChatMessage(role: .system, content: sys)) }
    let base = messages
    err("chat: /reset clears the conversation, /exit or end of input quits\n")
    while true {
        out("> ")
        guard let line = readLine(strippingNewline: true) else { out("\n"); break }
        let input = line.trimmingCharacters(in: .whitespaces)
        if input.isEmpty { continue }
        if input == "/exit" || input == "/quit" { break }
        if input == "/reset" {
            messages = base
            g.reset()
            err("(conversation cleared)\n")
            continue
        }
        messages.append(ChatMessage(role: .user, content: input))
        var ids = m.tokenizer.encode(template.render(messages))
        if ids.count + 1 > g.capacity {
            // Out of context: start over, keeping only the system message and this turn.
            messages = base + [ChatMessage(role: .user, content: input)]
            ids = m.tokenizer.encode(template.render(messages))
            g.reset()
            err("(context full: earlier turns dropped)\n")
            guard ids.count + 1 <= g.capacity else {
                err("(message does not fit the \(g.capacity)-token context)\n")
                messages = base
                continue
            }
        }
        if o.dump { err("prompt_ids (\(ids.count)): \(idsString(ids))\n") }
        let (reply, stats) = try streamGeneration(g, m, o, sequence: ids)
        out("\n")
        let stop = g.stopTokens
        let text = m.tokenizer.decode(reply.filter { !stop.contains($0) })
        messages.append(ChatMessage(role: .assistant, content: text.trimmingCharacters(in: .whitespacesAndNewlines)))
        err(statsLine(stats) + "\n")
        if o.stats, let e = stats.expert { err(expertLine(e) + "\n") }
        saveHistory(m)
    }
}

func runBench(_ o: CLIOptions) throws {
    let (m, backend) = try load(o)
    let metal = backend as? MetalBackend
    var prompt: [Int32]
    if let p = o.promptIDs {
        prompt = p
    } else if let text = o.prompt {
        prompt = promptIDs(m, o, text: text)
    } else {
        // A fixed English text, repeated and cut to --prompt-tokens.
        let filler = "The history of the city goes back more than two thousand years. Merchants, scholars and sailors passed through its harbour, and each left something behind: a word, a recipe, a song. "
        let one = m.tokenizer.encode(filler)
        prompt = []
        while prompt.count < o.promptTokens { prompt += one }
        prompt = Array(prompt.prefix(o.promptTokens))
    }
    let decodeTokens = max(1, o.maxTokens)
    let g = try makeGenerator(m, o)
    guard prompt.count + decodeTokens <= g.capacity else {
        throw CLIOptions.ParseError(description: "prompt \(prompt.count) + decode \(decodeTokens) tokens exceed --ctx \(g.capacity)")
    }
    // Warm-up: touch the weights and compile kernels.
    _ = try g.generate(prompt: Array(prompt.prefix(min(prompt.count, g.maxBatch))), maxTokens: 2, ignoreStop: true) { _ in true }
    err("bench: prompt \(prompt.count) tokens, decode \(decodeTokens) tokens, batch \(g.maxBatch), \(o.runs) run(s)\n")
    var prefill: [Double] = [], decode: [Double] = []
    for run in 0..<o.runs {
        g.reset()
        let gpu0 = metal?.gpuSeconds ?? 0, cb0 = metal?.batchesCommitted ?? 0
        let s = try g.generate(prompt: prompt, maxTokens: decodeTokens, ignoreStop: true) { _ in true }
        prefill.append(s.prefillTokensPerSecond)
        decode.append(s.decodeTokensPerSecond)
        out("run \(run + 1): \(statsLine(s))\n")
        if let metal {
            out(String(format: "  gpu busy %.3f s of %.3f s in %d command buffers\n", metal.gpuSeconds - gpu0,
                       s.prefillSeconds + s.decodeSeconds, metal.batchesCommitted - cb0))
        }
        if o.stats, let e = s.expert { out(expertLine(e) + "\n") }
    }
    if o.runs > 1 {
        func median(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        out(String(format: "median: prefill %.1f tok/s, decode %.1f tok/s\n", median(prefill), median(decode)))
    }
}

// MARK: Entry point

let options: CLIOptions
do {
    options = try CLIOptions.parse(Array(CommandLine.arguments.dropFirst()))
} catch {
    FileHandle.standardError.write(Data("coli-swift: \(error)\n\n\(CLIOptions.usage)\n".utf8))
    exit(2)
}

do {
    switch options.command {
    case .help: print(CLIOptions.usage)
    case .info: try runInfo(options)
    case .generate: try runGenerate(options)
    case .chat: try runChat(options)
    case .bench: try runBench(options)
    }
} catch {
    FileHandle.standardError.write(Data("coli-swift: \(error)\n".utf8))
    exit(1)
}
