# Port status

What the Swift port covers, how it was checked, and what it does not do yet. Upstream is
colibri at commit `0a5504a`. Measurements were taken on 2026-11-14 on an M4 Max (16 cores,
40-core GPU, 64 GB, macOS 27, Swift 6.4).

## 1. Ported

| area | files | upstream source |
|---|---|---|
| GGUF v2/v3 reader (mmap, metadata, tensor table, merged expert tensors) and writer | `GGUF/` | `c/gguf.h`, `c/st_gguf.h` |
| block decoders and float/int8 dot products: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS | `Quant/` | `c/ggml_quants.h` |
| byte-level BPE tokenizer, `qwen2` and GPT-2 (`olmo`) pre-tokenizers, streaming decode | `Tokenizer/` | `c/qwen36.c`, `c/tok.h` |
| chat templates: ChatML with the Qwen3 think switch (and the Coder variant), OLMoE's Tulu format | `Tokenizer/ChatTemplate.swift` | `c/openai_server.py` renderers |
| sampler: greedy, temperature, top-k, top-p | `Sampling/` | `c/argmax_first.h`, `c/qwen36_sample.h`, `c/sample.h` |
| expert placement: resident (zero-copy views of the mapping) or streamed through a per-layer slot cache; LRU with pins, routing history (`.coli_usage`), memory budget | `Experts/` | `c/olmoe.c`, `c/qwen36.c`, `c/tier.h`, `c/route_trace.h` |
| CPU backend (float reference path and the int8-activation path) | `Backend/CPU/`, `Quant/` | `c/ggml_quants.h` |
| Metal backend: every Backend op, all 15 weight types decoded in the kernels, one command buffer per forward call, split-KV decode attention | `ColibriMetal/` | new: upstream's Metal backend (`c/backend_metal.mm`) decodes its own formats, no ggml block type |
| families `qwen3`, `qwen3moe`, `olmoe` on a shared decoder | `Models/` | `c/qwen36.c` (GGUF path), `c/olmoe.c` |
| generator with chunked prefill and KV reuse across chat turns; CLI `info`, `generate`, `chat`, `bench` | `Models/Generator.swift`, `coli-swift/` | new, after upstream's engines' main loops |
| deterministic tiny random GGUFs for the MoE families | `Synthetic/TinyModel.swift` | new |

## 2. Verification

Reference: upstream built from source (`scripts/oracle/build-upstream.sh`), run in its float
mode (`GQ_EXACT=1`, f32 KV cache, greedy, never stopping at EOS). Fixtures:
`Tests/ColibriCoreTests/Fixtures/oracle/`. Comparison: `scripts/oracle/compare.sh`, which
reports MATCH, NEAR-TIE (diverged where upstream's own top-2 logit gap is under 0.01) or
DIVERGED with the step.

| case | backend | matches upstream | first divergence |
|---|---|---|---|
| Qwen3-0.6B Q8_0, 5 prompts x 32 tokens | CPU | 5/5 | none |
| Qwen3-0.6B Q8_0, 5 prompts x 32 tokens | Metal | 5/5 | none |
| 16 synthetic models x 4 prompts x 24 tokens (qwen3, qwen3moe, olmoe; F32, Q8_0, Q4_K, Q4_K_M mix, tied head, Qwen vocabulary) | CPU | 64/64 | none |
| same | Metal | 64/64 | none |
| same, f16 KV cache | Metal | 68/69 with Qwen3-0.6B | `olmoe_q4_K` prompt 3, step 18: picked upstream's second token, upstream's gap 0.0038 (7.32823 vs 7.32444): a near-tie |
| synthetic MoE models, experts streamed through a cache smaller than the experts (`--expert-cache-gb 0.001`: 4 of 16 slots per layer for qwen3moe, 2 or 4 of 8 for olmoe; hundreds of evictions per run) | CPU | 64/64 (dense cases included, unaffected) | none |
| same | Metal | 64/64; also with `--batch 4` (prefill split into chunks) | none |
| resident vs streamed experts, 4 synthetic MoE models x 128 tokens | CPU, Metal | identical ids | none |
| Metal vs CPU, Qwen3-0.6B, 5 prompts x 128 tokens, f32 and f16 KV | | identical ids | none |
| Metal vs CPU, Qwen3-0.6B, 2654-token prompt x 96 tokens, f32 and f16 KV | | identical ids | none |
| `--int8` vs upstream's default int8-activation path | CPU | Qwen3-0.6B 5/5; synthetic 54/64 | see section 4 |
| upstream's golden quant blocks (14 types) fed straight into the Metal embed and decode matvec (`MetalSupportTests`) | Metal | embed bit-exact for 13 types; BF16: subnormals flushed to 0 (see section 4); matvec within 1e-5 x magnitude of `gq_matmul_f32` | |
| unsupported shape (synthetic qwen3 with head dim 512), release CLI | Metal / auto | refused at load with the limit named; `--backend auto` loads it on the CPU (finite logits) | |
| expert read failure (streamed qwen3moe truncated after load, prefill routing to the lost expert) | CPU, Metal | the forward throws (`FailureModeTests`, `MetalSupportTests`) | |

Commands:

```bash
scripts/oracle/compare.sh                                                # CPU
scripts/oracle/compare.sh --no-build --backend metal
scripts/oracle/compare.sh --no-build --backend metal -- --kv f16
scripts/oracle/compare.sh --no-build --tiny-only -- --expert-cache-gb 0.001
scripts/oracle/compare.sh --no-build --backend metal --tiny-only -- --expert-cache-gb 0.001 --batch 4
scripts/oracle/compare.sh --no-build --reference default                 # --int8
```

`swift test` (162 tests: 136 in ColibriCoreTests, 26 in ColibriMetalTests; the benchmarks
and the real-model oracle case are skipped unless enabled) covers, besides the per-module unit tests: every block decoder
bit for bit against upstream's golden data, the tokenizer against upstream's ids (including
a fuzz set), prefill logits and greedy ids of three synthetic models against upstream
(`COLIBRI_ORACLE_FULL=1` for all 16, `COLIBRI_ORACLE_REAL=1` adds Qwen3-0.6B), resident
against streamed experts byte for byte, every Metal op against the CPU for every weight
type, upstream's golden blocks through the Metal kernels, whole layers on Metal, real
Qwen3-0.6B tensors on Metal against the CPU, Qwen3-0.6B's golden prompt (upstream's 16
greedy tokens) on the CPU, and the failure modes: weight types refused with the tensors
named, Metal shape limits refused at load, a failed expert read failing the forward.
`COLIBRI_ORACLE_FULL=1 COLIBRI_ORACLE_REAL=1 swift test -c release -Xswiftc -enable-testing
--filter OracleTests` passes too: all 16 synthetic models and Qwen3-0.6B, prefill logits
and greedy ids against upstream.

## 3. Performance

Qwen3-0.6B Q8_0, release build, 512-token prompt, 128 generated tokens, median of 3:

| backend | prefill tok/s | decode tok/s |
|---|---|---|
| Metal, f32 KV, `--batch 512` | 2031 | 220 |
| Metal, f32 KV, default `--batch 64` | 1967 | 220 |
| Metal, f16 KV | 2075 | 218 |
| CPU, float path | 345 | 37.5 |
| CPU, `--int8` | 759 | 67.5 |

Metal decode with an 8-token prompt: 233 tok/s; after a 2048-token prompt: 194 tok/s.
Before the split-KV attention kernel (one threadgroup per head) these were 110 tok/s after
512 tokens and 70 after 2048. During decode the GPU is busy about 93% of the wall time, so
the remaining cost is in the kernels, not in host round trips.

No MoE model large enough to measure streaming throughput exists on this machine; the
synthetic models only show that the cache works (counts, evictions), not how fast.

## 4. Known gaps

**Not run on real MoE weights.** `qwen3moe` and `olmoe` are verified only on synthetic
GGUFs that use the real tensor layout. The first real Qwen3-30B-A3B or OLMoE run should
compare CPU against Metal and resident against `--expert-cache-gb` as in section 2.

**`--int8` is not bit-identical to upstream's int8 path.** Prefill logits of the
synthetic models against upstream's default (int8-activation) path, per prompt: about half
match to 1e-6, the rest differ by 0.02-0.13 (Qwen3-0.6B: up to 0.2), with no pattern by
architecture or weight type (`qwen3_q8_0`: prompts 0, 1 and 3 match, prompt 2 does not).
That is what a discontinuous step does: activations are rounded to int8 per block of 32,
the floats before that rounding differ from upstream's by about 1e-6 (summation order),
and now and then a value sits close enough to a rounding boundary that one code changes by
one, which moves the logits by a few hundredths. The float path, the port's reference and
default, has no such step and matches upstream to about 1e-5. The int8 kernels produce
upstream's codes, scales and integer sums exactly (golden data); only their float
accumulation order differs. 10 of 64 synthetic cases diverge from upstream's
int8 greedy ids (for example `qwen3_q4_K_M` prompt 2 at step 0, where upstream's own int8
top-2 gap is 0.0026); Qwen3-0.6B matches on all 5 prompts.

**Metal and files larger than one buffer.** The Metal backend wraps the whole model
mapping as one buffer, so a file larger than `maxBufferLength` (41.7 GB here) cannot load
on Metal; with `--backend auto` the CLI then falls back to the CPU. Qwen3-30B-A3B up to
Q8_0 (about 32 GB) fits. Larger files need the mapping split into overlapping views
(`TODO(E)` in `MetalBackend.register`).

**Expert streaming features present but not wired.** `ExpertStore.prefetch` exists but no
family calls it (upstream's PILOT router lookahead); live repinning (`repinEvery`, upstream
`--repin`) is implemented in the store but has no CLI option. The `wait` counter only
measures host-side waits; on Metal the GPU waits on an event instead, so it reads 0.

**Release builds and checks.** The engine targets build with -Ounchecked in release,
which removes `precondition`. Backend limits are therefore enforced at load
(`Backend.checkSupport` with a `ModelRequirements`, which throws), and the Metal backend's
per-op checks use `fatalError` (kept in release). The CPU backend's per-op `precondition`s
(slice sizes) are still compiled out in release; they guard internal consistency, not
model shapes, and the debug test suite exercises them.

**Expert read failures.** A failed `pread` still advances the store's signal, so a batch
waiting on it never hangs; the forward then throws when it releases the lease that
depended on the read (`ExpertStore.checkLease`), and the logits of that forward are not
returned. The KV rows that forward wrote are left behind but the position does not
advance, so they are overwritten by the next forward. A failed prefetch nobody leased is
reported by the next `acquire`, as before.

**Routing history.** The CLI writes `<model>.coli_usage` after every MoE run and chat
turn. A failed write (a read-only model directory) is a one-time warning, not an error,
as upstream's `rt_save` returns a status instead of aborting.

**Chat reuse with Qwen3, thinking off.** Each turn re-renders the conversation and keeps
the KV cache up to the first token that differs. The generation prompt ends in an empty
`<think>\n\n</think>\n\n` block that the history rendering of that reply drops (as the
HF template does), so the shared prefix ends before the previous reply and every turn
re-prefills that reply plus the new message (measured on Qwen3-0.6B: turn 2 reused 20 of
81 tokens). Reused and fresh logits agree (at most 3.7e-5 on the CPU, 0 on Metal).
Keeping the think block in history would save the prefill but feed the model something
its template never produces, so the port does not.

**BF16 subnormals on Metal.** Metal's f32 arithmetic flushes subnormals to zero, so a BF16
weight below about 1.2e-38 decodes to 0 on the GPU (14 of 2048 values of upstream's
random golden BF16 blocks) and stays subnormal on the CPU and upstream. No practical effect
on a model; every other type decodes bit for bit on Metal too.

**Weight types.** Not decoded: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
TQ1_0, TQ2_0, MXFP4, NVFP4, and upstream's Q1_0/Q2_0. Files using them are refused at load,
naming the type and (up to three of) the tensors that use it.
Upstream's own formats (safetensors with FP8 or int4 experts, `qpack`, canonical `.qc`
dense files) are not read; the port takes GGUF only.

**Upstream engines not ported.** Upstream `c/` has an engine per family; this port has
the GGUF path of `qwen36.c` for `qwen3`/`qwen3moe` and `olmoe.c`. Not ported: the rest of
`qwen36.c` (Qwen3.5/3.6 with Gated DeltaNet linear attention and shared experts), `qwen38.c`
(Qwen3.8, vision), `deepseek_v4.c` / `deepseek_v41.c` (DeepSeek V4), `glm53.c`,
`kimi_k3.c`, `mimo.c`, `inkling.c`, `laya.c`, `gliner_decide.c`, `qwenimage.c` (image
generation), the edge and segment runtimes, and the GGUF models upstream lists beyond
Qwen3 (Mistral-Large-4, Muse Spark Open, Flint, LFM3-VL). `adding-a-model.md` lists what each kind of model needs.

**Other upstream features not ported.**
- The OpenAI-compatible server (`openai_server.py`, `coli serve`) and the MCP server.
- Vision towers and image input.
- MTP heads and speculative decoding (`qwen36_mtp.h`, `spec_draft.h`), and n-gram lookup
  decoding (`COLI_LOOKUP`).
- Grammar and JSON-schema constrained sampling, tool-call parsing.
- KV cache persistence and prefix caching across processes (`kv_persist.h`, `kv_prefix.h`),
  KV quantisation beyond f16 (`kv_fp8.h`, `kv_tq.h`).
- CUDA and Vulkan backends, the cluster mode, the dense chain offload.
- Setup, download and doctor tooling (`start-here.sh`, `setup_*.py`), autotune, telemetry
  and route traces.
- YaRN rope scaling: the port warns and runs unscaled, as upstream does.

**Differences from upstream on purpose.**
- The KV cache defaults to f32 (upstream's oracle mode); upstream's chat rounds K and V to
  f16, which is `--kv f16` here.
- OLMoE with grouped-query attention and a tied head is accepted (upstream requires
  `KV == H` and an `output.weight`), and its context is not capped at 4096.
- Metal is a new backend, not a port of upstream's: upstream's Metal code targets its own
  formats.
