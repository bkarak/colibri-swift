import ColibriCore

/// Parsed command line. Hand-rolled (no package dependencies): `coli-swift <command>
/// [positional...] [--option value | --flag]...`, options in any order.
struct CLIOptions {
    enum Command: String, CaseIterable {
        case info, generate, chat, bench, help
    }

    /// `auto` (the default): Metal on Apple Silicon when a device is available, else the
    /// CPU backend. An explicit `metal` fails instead of falling back.
    enum BackendKind: String { case auto, cpu, metal }

    var command: Command = .help
    var positional: [String] = []

    var model: String?
    var backend: BackendKind = .auto
    var threads: Int?
    var temperature: Float = 0
    var topP: Float = 1
    var topK: Int = 0
    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    var maxTokens: Int = 128
    var expertCacheGB: Double?
    var experts: ExpertStoreOptions.Mode = .auto
    var directIO = false
    var prompt: String?
    var context: Int = 4096
    var raw = false
    var thinking = false
    var system: String?
    /// Prompt given as token ids (no tokenizer, no template): oracle comparisons.
    var promptIDs: [Int32]?
    /// Print prompt ids, every step's id and top logits, and the generated ids to stderr.
    var dump = false
    var dumpTop = 5
    /// Appends every step's logits (vocab x float32, little endian) to this file.
    var dumpLogits: String?
    var ignoreEOS = false
    var kvType: ScalarType = .f32
    var maxBatch = 512
    var stats = false
    /// CPU: upstream's default int8-activation path instead of the float path.
    var int8 = false
    var promptTokens = 128
    var runs = 1

    /// The model path: `--model`, else the first positional argument.
    var modelPath: String? { model ?? positional.first }

    var samplerConfig: SamplerConfig {
        SamplerConfig(temperature: temperature, topP: topP, topK: topK, seed: seed)
    }

    static let usage = """
    usage: coli-swift <command> [options]

    commands:
      info <model.gguf>        print header, metadata summary and tensor table
      generate                 complete --prompt (or stdin, or --prompt-ids) and print the result
      chat                     interactive chat with the model's template
      bench                    measure prefill and decode speed

    options:
      --model PATH             GGUF file (or the first positional argument)
      --backend auto|cpu|metal compute backend (default auto: Metal on Apple Silicon, CPU fallback)
      --threads N              CPU worker threads (default: all cores)
      --temp T                 temperature, 0 = greedy (default 0)
      --top-p P                nucleus mass (default 1, off)
      --top-k K                keep the K most likely tokens (default 0, off)
      --seed N                 sampler seed
      --max-tokens N           tokens to generate (default 128)
      --expert-cache-gb G      RAM budget for routed experts; smaller than the experts forces streaming
      --experts auto|resident|stream   expert placement (default auto)
      --direct                 streamed experts read with F_NOCACHE, from the SSD and not the file cache
      --prompt TEXT            prompt for generate/bench
      --ctx N                  KV cache capacity in tokens (default 4096)
      --raw                    no chat template (prompt is passed through)
      --think                  enable thinking (Qwen3)
      --system TEXT            system message (generate/chat)
      --prompt-ids "1 2 3"     prompt as token ids: no tokenizer, no template
      --dump                   stderr: prompt ids, each step's id and top logits, generated ids
      --dump-top N             logits per step for --dump (default 5)
      --dump-logits PATH       write every step's logits to PATH (steps x vocab float32 LE)
      --ignore-eos             keep generating past stop tokens
      --kv f32|f16             KV cache element type (default f32)
      --batch N                largest prefill chunk in tokens (default 512)
      --stats                  bench/chat: print expert cache counters (generate always prints them for MoE models)
      --int8                   cpu: int8 activations (upstream's default) instead of float
      --prompt-tokens N        bench: prompt length when --prompt is not given (default 128)
      --runs N                 bench: measured runs (default 1)
    """

    struct ParseError: Error, CustomStringConvertible {
        let description: String
    }

    static func parse(_ args: [String]) throws -> CLIOptions {
        var o = CLIOptions()
        var i = 0
        func value(_ name: String) throws -> String {
            i += 1
            guard i < args.count else { throw ParseError(description: "\(name) needs a value") }
            return args[i]
        }
        func number<T: LosslessStringConvertible>(_ name: String, _: T.Type) throws -> T {
            let s = try value(name)
            guard let v = T(s) else { throw ParseError(description: "\(name): not a number: \(s)") }
            return v
        }
        if let first = args.first, !first.hasPrefix("-") {
            guard let c = Command(rawValue: first) else { throw ParseError(description: "unknown command \(first)") }
            o.command = c
            i = 1
        }
        while i < args.count {
            let a = args[i]
            switch a {
            case "-h", "--help": o.command = .help
            case "--model": o.model = try value(a)
            case "--backend":
                let v = try value(a)
                guard let b = BackendKind(rawValue: v) else { throw ParseError(description: "--backend: auto, cpu or metal, not \(v)") }
                o.backend = b
            case "--threads": o.threads = try number(a, Int.self)
            case "--temp": o.temperature = try number(a, Float.self)
            case "--top-p": o.topP = try number(a, Float.self)
            case "--top-k": o.topK = try number(a, Int.self)
            case "--seed": o.seed = try number(a, UInt64.self)
            case "--max-tokens": o.maxTokens = try number(a, Int.self)
            case "--expert-cache-gb": o.expertCacheGB = try number(a, Double.self)
            case "--direct": o.directIO = true
            case "--experts":
                let v = try value(a)
                guard let m = ExpertStoreOptions.Mode(rawValue: v) else { throw ParseError(description: "--experts: auto, resident or stream, not \(v)") }
                o.experts = m
            case "--prompt": o.prompt = try value(a)
            case "--ctx": o.context = try number(a, Int.self)
            case "--raw": o.raw = true
            case "--think": o.thinking = true
            case "--system": o.system = try value(a)
            case "--prompt-ids":
                let v = try value(a)
                let parts = v.split(whereSeparator: { $0 == " " || $0 == "," || $0 == "\n" || $0 == "\t" })
                var ids: [Int32] = []
                for p in parts {
                    guard let id = Int32(p), id >= 0 else { throw ParseError(description: "--prompt-ids: not a token id: \(p)") }
                    ids.append(id)
                }
                guard !ids.isEmpty else { throw ParseError(description: "--prompt-ids: no ids") }
                o.promptIDs = ids
            case "--dump": o.dump = true
            case "--dump-top": o.dumpTop = try number(a, Int.self)
            case "--dump-logits": o.dumpLogits = try value(a)
            case "--ignore-eos": o.ignoreEOS = true
            case "--kv":
                let v = try value(a)
                switch v {
                case "f32": o.kvType = .f32
                case "f16": o.kvType = .f16
                default: throw ParseError(description: "--kv: f32 or f16, not \(v)")
                }
            case "--batch": o.maxBatch = try number(a, Int.self)
            case "--stats": o.stats = true
            case "--int8": o.int8 = true
            case "--prompt-tokens": o.promptTokens = try number(a, Int.self)
            case "--runs": o.runs = try number(a, Int.self)
            default:
                if a.hasPrefix("-") { throw ParseError(description: "unknown option \(a)") }
                o.positional.append(a)
            }
            i += 1
        }
        guard o.maxTokens >= 0 else { throw ParseError(description: "--max-tokens must be >= 0") }
        guard o.context >= 1 else { throw ParseError(description: "--ctx must be >= 1") }
        guard o.maxBatch >= 1 else { throw ParseError(description: "--batch must be >= 1") }
        guard o.dumpTop >= 1 else { throw ParseError(description: "--dump-top must be >= 1") }
        guard o.promptTokens >= 1, o.runs >= 1 else { throw ParseError(description: "--prompt-tokens and --runs must be >= 1") }
        if let t = o.threads, t < 1 { throw ParseError(description: "--threads must be >= 1") }
        return o
    }

    var expertOptions: ExpertStoreOptions {
        var e = ExpertStoreOptions(mode: experts)
        if let gb = expertCacheGB { e.cacheBytes = Int(gb * 1_073_741_824) }
        e.uncached = directIO
        return e
    }

    var loadContext: (Backend) -> LoadContext {
        { LoadContext(backend: $0, expertOptions: expertOptions, maxContext: context, kvType: kvType, maxBatch: maxBatch) }
    }
}
