# colibri-swift

A Swift + Metal port of [colibri](https://github.com/JustVugg/colibri) for Apple Silicon.

colibri runs large mixture-of-experts models on machines with less memory than the model:
the dense weights (attention, norms, embeddings, router) stay resident, and the routed
experts are read from disk on demand through a RAM cache sized to what the machine has.
This port keeps that idea and the GGUF path of upstream's `qwen36` and `olmoe` engines,
and runs them on the CPU or on the GPU through Metal.

Ported from upstream commit `0a5504a` (Apache-2.0; see `NOTICE`). Code that is a direct
port says so in its file header ("Ported from colibri c/<file>"). Upstream colibri is also
the correctness oracle: its C engines, built from source, produced the reference outputs
the tests and `scripts/oracle/` compare against.

No package dependencies: Swift 6 toolchain, Foundation, Metal.

## Build

```bash
swift build -c release          # binary: .build/release/coli-swift
swift test                      # unit, oracle and Metal-vs-CPU tests (debug build)
```

Requires macOS 15 or later. Metal needs Apple Silicon; elsewhere the CPU backend runs alone.

## Usage

```bash
M=~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf

# what is in a GGUF, and how this build would run it
coli-swift info $M

# one completion through the model's chat template (greedy by default)
coli-swift generate --model $M --prompt "What is the capital of France?"

# sampling, a system message, Qwen3 thinking on
coli-swift generate --model $M --prompt "Write a haiku about rain." --temp 0.7 --top-p 0.9 --seed 7
coli-swift generate --model $M --system "Answer in Greek." --prompt "Hello" --think

# raw text continuation (no template), prompt on stdin
echo "The quick brown fox" | coli-swift generate --model $M --raw --max-tokens 64

# interactive chat with KV reuse across turns (/reset, /exit); with Qwen3 and thinking
# off each turn re-prefills the previous reply too (its empty think block is dropped from history)
coli-swift chat --model $M

# prefill and decode speed
coli-swift bench --model $M --prompt-tokens 512 --max-tokens 128 --runs 3
```

A MoE model is used the same way. By default the experts stay resident when they fit the
memory budget and are streamed otherwise; `--expert-cache-gb` sets the cache size and
forces streaming when it is smaller than the experts:

```bash
coli-swift generate --model Qwen3-30B-A3B-Q4_K_M.gguf --prompt "..." --expert-cache-gb 8
```

`generate` ends with the expert cache counters on stderr for every MoE model, for example:

```
experts: lookups 1572, hit rate 18.1%, misses 1288, evictions 1276, read 256.6 MiB (4.00 GB/s), wait 0.000 s
  cache: streamed: 4 of 16 slots per layer (cache 2.4 MiB for 9.6 MiB of experts)
```

The routing history is saved next to the model as `<model>.coli_usage` (upstream's
format and behaviour); the next streaming run pins each layer's most used experts from it.

Main options (`coli-swift --help` lists all):

| option | meaning |
|---|---|
| `--backend auto\|cpu\|metal` | `auto` (default): Metal on Apple Silicon, falling back to the CPU when Metal cannot start or cannot load the model |
| `--temp`, `--top-p`, `--top-k`, `--seed` | sampling; `--temp 0` (default) is greedy |
| `--max-tokens N`, `--ctx N`, `--batch N` | tokens to generate, KV capacity, largest prefill chunk |
| `--kv f32\|f16` | KV cache element type (default f32) |
| `--experts auto\|resident\|stream`, `--expert-cache-gb G`, `--direct` | expert placement, cache budget, uncached (`F_NOCACHE`) expert reads |
| `--raw`, `--system`, `--think` | template control |
| `--int8` | CPU only: upstream's default int8-activation matmuls instead of the float path |
| `--dump`, `--dump-logits PATH`, `--prompt-ids`, `--ignore-eos` | oracle and debugging aids |

## Supported models

| GGUF `general.architecture` | models | family file |
|---|---|---|
| `qwen3` | Qwen3 dense (0.6B ... 32B) | `Sources/ColibriCore/Models/Qwen3.swift` |
| `qwen3moe` | Qwen3-30B-A3B, Qwen3-Coder-30B-A3B | `Qwen3MoE.swift` |
| `olmoe` | OLMoE-1B-7B (0924 and 0125, base and Instruct) | `OLMoE.swift` |

Weight types, on both backends: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS. That covers the common `Q4_K_M`, `Q5_K_M`, `Q6_K` and
`Q8_0` files. The other i-quants (IQ1_*, IQ2_*, IQ3_*), the ternary and FP4 types are not
decoded yet; such a file is refused at load time, naming the type and the tensors that use
it. Metal also refuses, at load, shapes its kernels do not handle (attention head dim above
256 or not a multiple of 4, a matrix row length not a multiple of 32, more than 2048 router
experts); with `--backend auto` such a model runs on the CPU instead.

Each family has been run as a real model against upstream: Qwen3-0.6B (Q8_0, dense),
Qwen3-30B-A3B (Q4_K_M, MoE, 128 experts, top 8) and OLMoE-1B-7B-0125-Instruct (Q4_K_M, MoE,
64 experts, top 8, Allen AI's own GGUF). All three were also verified on small random GGUFs
written by `Sources/ColibriCore/Synthetic/TinyModel.swift` with the real tensor layout
(below).

## Verification

Upstream's engines were built from source (`scripts/oracle/build-upstream.sh`) and run in
their float reference mode (`GQ_EXACT=1`, f32 KV cache, greedy, no EOS stop). Their outputs
are fixtures in `Tests/ColibriCoreTests/Fixtures/oracle/`. `scripts/oracle/compare.sh`
runs `coli-swift generate` on every fixture prompt and reports the first divergence.

| check | CPU | Metal |
|---|---|---|
| Qwen3-0.6B Q8_0 vs upstream: 5 prompts (chat, raw English, raw Python, Greek with a system message) x 32 tokens | 5/5 match | 5/5 match |
| Qwen3-30B-A3B Q4_K_M vs upstream: the same 5 prompts x 32 tokens (prefill top-20 logits within 5e-5) | 5/5 match | 5/5 match |
| OLMoE-1B-7B-0125-Instruct Q4_K_M vs upstream: 5 prompts (its own chat template, raw English, raw Python, Greek with a system message) x 32 tokens; prompt ids from the model's HF tokenizer, which the port matches (prefill top-20 logits within 3e-5) | 5/5 match | 5/5 match |
| OLMoE with streamed experts, `--expert-cache-gb 0.5` (8 of 64 slots per layer) | 5/5 match | 5/5 match |
| Qwen3-30B-A3B with streamed experts: caches of 4 and 1 GB (Metal; 31 and 8 of 128 slots per layer) and 2 GB (CPU) | 5/5 match | 5/5 match |
| synthetic qwen3, qwen3moe, olmoe vs upstream: 16 models (F32, Q8_0, Q4_K, Q4_K_M, tied head, Qwen vocabulary) x 4 prompts x 24 tokens | 64/64 match | 64/64 match |
| same, f16 KV cache | | 68/69 match, 1 near-tie (`olmoe_q4_K` prompt 3, step 18: upstream's own top-2 gap 0.0038) |
| same MoE models with streamed experts, cache smaller than the experts (`--expert-cache-gb 0.001`: 4 of 16 or 2-4 of 8 slots per layer) | 64/64 match | 64/64 match (also with `--batch 4`) |
| resident vs streamed experts, 4 synthetic MoE models x 128 tokens | identical | identical |
| Metal vs CPU on Qwen3-0.6B: 5 prompts x 128 tokens, f32 and f16 KV | | identical |
| Metal vs CPU on Qwen3-0.6B: a 2654-token prompt x 96 tokens, f32 and f16 KV | | identical |
| `--int8` vs upstream's default int8 path | Qwen3-0.6B 5/5, synthetic 54/64 (see `docs/port/status.md`) | n/a |
| upstream's golden quant blocks (14 types) through the Metal kernels | | embed bit-exact except BF16 subnormals (flushed to 0); matvec within 1e-5 of magnitude of upstream's `gq_matmul_f32` |
| a model shape Metal does not support (synthetic qwen3, head dim 512), release build | runs, finite logits | refused at load; `--backend auto` falls back to the CPU |
| an expert read that fails mid-run (streamed qwen3moe, file truncated after load) | the forward throws | the forward throws |

On top of that, `swift test` checks every block decoder bit for bit against upstream's,
the tokenizer against upstream's prompt ids, the prefill logits and greedy ids of three
synthetic models against upstream's (all 16 with `COLIBRI_ORACLE_FULL=1`, Qwen3-0.6B with
`COLIBRI_ORACLE_REAL=1`, which also runs Qwen3-30B-A3B when its file is present; tolerance 2e-3, typically about 1e-5), resident against streamed
experts, every Metal kernel against the CPU backend for every weight type, and the
failure modes above.

```bash
scripts/oracle/compare.sh                                   # CPU, all cases
scripts/oracle/compare.sh --backend metal
scripts/oracle/compare.sh --backend metal --tiny-only -- --expert-cache-gb 0.001
scripts/oracle/compare.sh --backend metal --real-only --real-models qwen3-30b-a3b-q4_k_m -- --expert-cache-gb 1
```

The real models the harness knows are listed in `scripts/oracle/real-models.json`; adding
one there and running `record.py --real --real-models <key>` records upstream's outputs
for it (`--light` for big models: one run per prompt instead of one per step).

## Performance

Qwen3-0.6B Q8_0 on an M4 Max (16 cores, 40-core GPU, 64 GB), release build, a 512-token
prompt then 128 generated tokens, median of 3 runs
(`coli-swift bench --prompt-tokens 512 --max-tokens 128 --runs 3 --batch 512`):

| backend | prefill tok/s | decode tok/s |
|---|---|---|
| Metal, f32 KV | 2031 | 220 |
| Metal, f16 KV | 2075 | 218 |
| CPU, float path (the reference) | 345 | 37.5 |
| CPU, `--int8` | 759 | 67.5 |

Qwen3-30B-A3B Q4_K_M (17.3 GiB, 3.3B active parameters per token), same machine and settings
but the default `--batch 64` and 2 runs:

| backend, experts | prefill tok/s | decode tok/s |
|---|---|---|
| Metal, resident (zero-copy, 16.35 GiB of experts) | 222 | 84 |
| CPU, resident | 32 | 10.7 |
| Metal, streamed through an 8 / 4 / 2 GB cache, `--direct` (128 tokens of `generate`) | | 29.9 / 27.0 / 25.4 |

OLMoE-1B-7B-0125-Instruct Q4_K_M (3.9 GiB, 1.3B active parameters per token), same settings:
Metal 640 tok/s prefill and 231 tok/s decode, CPU 81 and 28.9.

The streamed rows were measured with the whole file already in the macOS file cache, so
the reads came from RAM (24-56 GB/s), not the SSD: `F_NOCACHE` does not drop pages that are
already cached. For a disk-bound number run `sudo purge` first. The first prompt of a
fresh process takes 0.27 s for 19 tokens on this model; the GPU maps only the tensors it
reads, so streamed experts are never pulled into memory by the GPU.

With a short prompt Metal decodes Qwen3-0.6B at about 233 tok/s; after a 2048-token prompt, at about
194. The GPU is busy for about 93% of the wall time during decode. There is no comparison
with upstream's speed: its macOS build here has no OpenMP, so it runs on one core.

## Adding a model family

A family is one Swift file in `Sources/ColibriCore/Models/` and one line in
`ModelRegistry.families`. A llama-like decoder (pre-norm, RMSNorm, GQA attention with
NEOX RoPE, SwiGLU or softmax-routed MoE FFN) is mostly configuration: read its
hyperparameters, build a `DecoderStack`, name its chat template and stop tokens.
`docs/port/adding-a-model.md` walks through it, including what to do when the new model
needs something the shared blocks do not have.

## Documentation

- `docs/port/status.md`: what is ported, what is not, known gaps.
- `docs/port/adding-a-model.md`: adding a model family.
- `docs/port/architecture.md`: modules, the Backend contract, one token through a MoE layer.
- `docs/port/models.md`: the three families' forward passes, tokenizer, templates, oracle recipe.
- `docs/port/gguf-and-quants.md`: GGUF and the block formats.
- `docs/port/runtime-and-metal.md`: expert placement and streaming, the Metal design.
- `scripts/oracle/README.md`: the upstream-oracle harness.

## Licence

Apache-2.0, as upstream. See `LICENSE` and `NOTICE`.
