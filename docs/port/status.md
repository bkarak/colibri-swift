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
| Qwen3-4B Q4_K_M (dense, K-quants, tied output), the same 5 prompts x 32 tokens, every step's logits recorded; prefill logits within 5e-5 | CPU, Metal | 5/5, 5/5 | none |
| same, `--int8` vs upstream's default int8 path | CPU | 5/5 | none |
| Qwen3-30B-A3B Q4_K_M, the same 5 prompts x 32 tokens (fixture recorded with `record.py --light`: greedy ids and prefill top-20 only; prefill logits within 5e-5) | CPU | 5/5 | none |
| same | Metal | 5/5 | none |
| OLMoE-1B-7B-0125-Instruct Q4_K_M, 5 OLMoE-template and raw prompts x 32 tokens; prompt ids from the model's HF `tokenizer.json` (`scripts/oracle/hf_tokenize.py`), which the port's tokenizer matches on all 5; every step's logits recorded; prefill logits within 3e-5 | CPU, Metal | 5/5, 5/5 | none |
| same, experts streamed: `--expert-cache-gb 0.5` (8 of 64 slots per layer) | CPU, Metal | 5/5, 5/5 | none |
| same, `--int8` vs upstream's default int8 path | CPU | 3/5 | `raw_fox` step 6 and `raw_code` step 8 (the known int8 gap, section 4) |
| same | Metal | 5/5 | none |
| same, experts streamed: `--expert-cache-gb 4` and `1` (31 and 8 of 128 slots per layer) | Metal | 5/5, 5/5 | none |
| same, experts streamed: `--expert-cache-gb 2` | CPU | 5/5 | none |
| 16 synthetic models x 4 prompts x 24 tokens (qwen3, qwen3moe, olmoe; F32, Q8_0, Q4_K, Q4_K_M mix, tied head, Qwen vocabulary) | CPU | 64/64 | none |
| same | Metal | 64/64 | none |
| same, f16 KV cache | Metal | 68/69 with Qwen3-0.6B | `olmoe_q4_K` prompt 3, step 18: picked upstream's second token, upstream's gap 0.0038 (7.32823 vs 7.32444): a near-tie |
| synthetic MoE models, experts streamed through a cache smaller than the experts (`--expert-cache-gb 0.001`: 4 of 16 slots per layer for qwen3moe, 2 or 4 of 8 for olmoe; hundreds of evictions per run) | CPU | 64/64 (dense cases included, unaffected) | none |
| same | Metal | 64/64; also with `--batch 4` (prefill split into chunks) | none |
| resident vs streamed experts, 4 synthetic MoE models x 128 tokens | CPU, Metal | identical ids | none |
| Metal vs CPU, Qwen3-0.6B, 5 prompts x 128 tokens, f32 and f16 KV | | identical ids | none |
| Metal vs CPU, Qwen3-0.6B, 2654-token prompt x 96 tokens, f32 and f16 KV | | identical ids | none |
| `--int8` vs upstream's default int8-activation path | CPU | Qwen3-0.6B 5/5, Qwen3-4B 5/5, Qwen3-30B-A3B 3/5 (`chat_france` step 5, `chat_haiku` step 29); synthetic 54/64 | see section 4 |
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
(`COLIBRI_ORACLE_FULL=1` for all 16, `COLIBRI_ORACLE_REAL=1` adds the real models present), resident
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
| Metal, f32 KV (default `--batch 512`, tiled prefill kernel) | 5237 | 220 |
| Metal, f16 KV | 5624 | 204 |
| Metal, before the tiled kernel, `--batch 512` / `--batch 64` | 2031 / 1967 | 220 |
| CPU, float path | 345 | 37.5 |
| CPU, `--int8` | 759 | 67.5 |

Metal decode with an 8-token prompt: 233 tok/s; after a 2048-token prompt: 194 tok/s.
Before the split-KV attention kernel (one threadgroup per head) these were 110 tok/s after
512 tokens and 70 after 2048. During decode the GPU is busy about 93% of the wall time, so
the remaining cost is in the kernels, not in host round trips.

OLMoE-1B-7B-0125-Instruct Q4_K_M, same settings: Metal 640 tok/s prefill (2840 with the tiled
kernels, below), 231 decode; CPU
81 and 28.9. A two-turn chat on Metal reused all 139 tokens of turn 1 from the KV cache.

Qwen3-4B Q4_K_M (2.3 GiB, dense), same settings, median of 3: Metal 279 tok/s prefill, 87
decode (f16 KV: 283 and 92); CPU 44.6 and 10.7; CPU `--int8` 54.8 and 15.7. With the tiled
prefill kernel and `--batch 512` (section 3, below): Metal 951 tok/s prefill (f16 KV: 892).
Decode reads about 2.5 GB of weights per token, so 87 tok/s is about 215 GB/s, about 40% of
the M4 Max's 546 GB/s. Prefill uses the decode matvec kernel over groups of tokens, in
fp32; it has no simdgroup-matrix (tiled matrix-multiply) kernel, which is where llama.cpp's
prefill speed comes from.

Qwen3-30B-A3B Q4_K_M, 512-token prompt, 128 generated tokens, `--batch 64`, median of 2:
Metal resident 222 tok/s prefill, 84 decode; CPU resident 32 and 10.7. Streamed on Metal
with `--direct` (128 tokens of `generate`): 29.9 / 27.0 / 25.4 tok/s decode with 8 / 4 / 2 GB
caches. Those reads came from the file cache (24-56 GB/s), not the SSD: `F_NOCACHE` does not
evict pages already cached, and the whole file was. A disk-bound measurement needs
`sudo purge` before the run; it has not been done.

### Against llama.cpp

All four models resident, tok/s (llama.cpp / coli-swift):

| model | Metal prefill: llama.cpp / coli-swift | Metal decode | CPU prefill | CPU decode |
|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 10712 / 1981 (5.4x) | 339 / 215 (1.6x) | 2315 / 365 (6.3x) | 220 / 42.4 (5.2x) |
| Qwen3-4B Q4_K_M | 1625 / 278 (5.8x) | 118 / 87 (1.35x) | 276 / 50.3 (5.5x) | 67.4 / 12.1 (5.6x) |
| OLMoE-1B-7B-0125-Instruct Q4_K_M | 4080 / 635 (6.4x) | 322 / 221 (1.5x) | 550 / 85.1 (6.5x) | 199 / 30.9 (6.4x) |
| Qwen3-30B-A3B Q4_K_M | 1660 / 226 (7.3x) | 128 / 81 (1.6x) | 210 / 35.2 (6.0x) | 69.3 / 11.3 (6.1x) |

llama.cpp v1.0.0 (`b1b770d`), built from source with CMake (Release, Metal and Accelerate,
no curl), `llama-bench -p 512 -n 128 -r 3`, decode measured at depth 512 (`-p 0 -n 128 -d 512`)
to match coli-swift's decode after a 512-token prompt; its defaults otherwise (f16 KV cache,
flash attention auto, `-ngl 99`). CPU: `-ngl 0 -dev none -t 12`. With `-t 16` llama.cpp's CPU
decode collapses (Qwen3-4B 2.8 tok/s, Qwen3-0.6B 8.5) as threads land on the efficiency cores;
12 is its default and the performance-core count. coli-swift: `bench --prompt-tokens 512
--max-tokens 128 --runs 3`, `--batch 64` (the default then) and f32 KV on Metal, the better of
`--threads 12` and `16` on the CPU (within 10% of each other). Same machine, same session,
same GGUF files, 2026-11-14.

Where the gap most likely comes from (from how llama.cpp is built; not profiled here):

- **Metal prefill (5-7x).** llama.cpp switches to tiled matrix-multiply kernels
  (`simdgroup_matrix`, half precision) once a batch has more than a few tokens. coli-swift
  ran the decode matvec over groups of tokens, in fp32. It now has a tiled f32 kernel for
  dense matmuls (below); the routed expert matmuls still use the matvec kernel.
- **Metal decode (1.35-1.6x).** Both are bound by reading the weights. llama.cpp's kernels
  reach about 55% of the memory bandwidth on Qwen3-4B, coli-swift's about 40%; llama.cpp
  also fuses more of each layer (norm, rope, flash attention) into fewer dispatches.
- **CPU (5-7x).** llama.cpp quantises activations to 8 bits and uses the NEON int8
  dot-product and matrix instructions (`sdot`, `smmla`), plus Accelerate for large prompt
  batches. coli-swift's default CPU path dequantises every weight to fp32 to stay
  bit-close to upstream's float reference; its `--int8` path uses `sdot` but still does
  the per-16 scaling in scalar floats, and runs MoE expert rows one token at a time.
  llama.cpp's CPU decode is close to coli-swift's Metal decode. (Measured before the
  `--int8` fix below; the float path is unchanged.)

### Tiled prefill kernel

Dense matmuls with 20 or more batch rows run on `matmul_tiled_<type>`: a threadgroup of 4
simdgroups computes a 64-row x 32-token tile, each 32-weight unit is decoded once into
threadgroup memory (f32, the same block decoders) beside its 32 x 32 activations, and the
products run as 8x8 f32 `simdgroup_matrix` multiply-accumulates. Everything stays f32
(llama.cpp's tiles are half), so only the summation order differs from the matvec
kernels: the oracle comparison is unchanged (84/84 on Metal, also with the tiled kernel
forced on every prompt, `COLIBRI_TILED_MIN_BATCH=2`). Below about 20 rows the matvec
kernel is faster (fewer tiles than GPU cores to fill). The default `--batch` went from 64
to 512 with it, since longer chunks now pay off. 512-token prompt, prefill tok/s:

| model | before (`--batch 64`) | tiled, `--batch 64` | tiled, `--batch 512` (default) | llama.cpp |
|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 2018 | 2504 | 5237 | 10712 |
| Qwen3-4B Q4_K_M | 281 | 675 | 951 | 1625 |
| OLMoE-1B-7B-0125-Instruct Q4_K_M | 619 | | 618; with the routed tiled kernel 2840 | 4080 |
| Qwen3-30B-A3B Q4_K_M | 213 | | 227; with the routed tiled kernel 861 | 1660 |

A version that decodes two units per step (64-wide K, 24 KB of threadgroup memory) was
5-7% slower: fewer threadgroups fit on a core.

**Routed (expert) matmuls.** With 320 or more (token, expert) pairs, `matmulExperts`
encodes two dispatches (llama.cpp's `mul_mm_id` scheme). `k_group_pairs` (one simdgroup
per matrix) lists, in pair order, the pairs routed to each matrix, on the GPU, so the
routing ids never go to the host. Ids out of range get NaN output rows, as before.
`matmul_id_tiled_<type>` then runs the same 64 x 32 f32 tiles over each matrix's pairs.
The grid is (row tiles, token tiles, matrices), and a matrix with more pairs than its
parallel token tiles loops over them. Below about 256 pairs (two to four tokens per
matrix) the per-pair matvec kernel is faster, so the threshold is 320.

Checked:
- **Against the CPU:** every weight type, with a matrix taking 80% of the pairs and NaN
  rows for bad ids (`MetalBackendTests`).
- **Against upstream:** the oracle comparison with both tiled kernels forced on every
  prompt is 84/84, plus every streamed case (tiny MoE models at 0.001 GB, OLMoE at
  0.5 GB, Qwen3-30B-A3B at 2 GB).
- **Long prompts:** Metal equals the CPU on a 520-token OLMoE prompt and a 693-token
  Qwen3-30B-A3B prompt, 40 tokens each.

Prefill, 512-token prompt: OLMoE 618 -> 2840 tok/s, Qwen3-30B-A3B 227 -> 861. Both are
now within 2x of llama.cpp (4080 and 1660).

### Larger than memory: Qwen3-235B-A22B

Qwen3-235B-A22B Q2_K from Unsloth (`Qwen3-235B-A22B-Q2_K-0000{1,2}-of-00002.gguf`, 79.8 GiB:
Q2_K x377, Q3_K x188, Q4_K x94, Q6_K output; 94 layers, 128 experts, top 8), opened through
the split support, experts streamed. 64 GB M4 Max, swap already in use by other processes
(about 11-13 GB throughout).

Correctness: a chat prompt ("Explain in two sentences why the sky is blue.") gives a
coherent answer, and Metal and CPU produce the same 48 greedy tokens (also with Metal's
cache raised to the CPU's 38 GiB). No upstream run: upstream is single-core on macOS, and
at about 1 tok/s on 16 cores that would take hours per prompt.

Speed, `bench --prompt-tokens 512 --max-tokens 32 --runs 3`, against `llama-bench -p 512
-n 32 -r 3` (llama.cpp v1.0.0 `b1b770d`; its decode is at depth 0 with random tokens,
coli-swift's after the 512-token prompt with greedy tokens; tok/s):

| engine, settings | prefill | decode |
|---|---|---|
| coli-swift Metal, `--expert-cache-gb 18` (30 of 128 slots per layer) | 17.3 | 3.6 |
| coli-swift Metal, automatic (43 GiB, 72 slots) | 19.8 | 0.9 |
| coli-swift CPU, automatic (44 GiB, 74 slots) | 3.1 | 1.6 |
| coli-swift CPU, `--int8`, 18 GB | 1.2 | 0.8 |
| llama.cpp Metal, `-ngl 99` | `kIOGPUCommandBufferCallbackErrorOutOfMemory` | |
| llama.cpp Metal, `-ngl 40` | 8.3 | 1.8 |
| llama.cpp Metal, `-ngl 56` | 9.9 | 0.8 |
| llama.cpp CPU, `-ngl 0 -dev none -t 12` | 6.9 | 5.2 |

Decode against the expert cache size, 64-token prompt, 32 tokens, median of 2:

| cache GB | 12 | 18 | 24 | 32 | 43 (auto) |
|---|---|---|---|---|---|
| Metal decode tok/s | 3.6 | 3.8 | 3.5 | 3.0 | 1.9 |
| CPU decode tok/s | 1.6 | 1.9 | 1.7 | 2.0 | (44: 1.6) |

What this shows:
- **The automatic budget is too large on Metal for a streamed model.** It follows
  upstream (88% of available memory). With 43 GiB of slots the system swaps; a sample
  of the decode has the main thread in `waitUntilCompleted` after each MoE layer while
  the GPU reports itself busy only about 40% of the wall time, which points at command
  buffers waiting for cache pages to be made resident. At 18 GB the GPU is busy 92% of
  the time. A smaller cache loses little hit rate because the slot reads come from the
  file cache, which then holds much of what the slots do not.
- **Prefill is where coli-swift leads.** The tiled routed kernel processes each expert's
  tokens together, so a 512-token prompt reads each expert a few times (about 500 GB per
  run at 18 GB, almost all from the file cache at 30 GB/s), not once per token.
- **CPU decode is compute-bound,** 0.5-0.6 s per token at every cache size, consistent
  with the float kernels' speed on the 30B (22B active parameters here against 3.3B).
  llama.cpp's CPU kernels are 5-7x faster on resident models (above), and here that
  makes its CPU decode the fastest configuration of all.
- **`--int8` is slower than the float path on this model** (0.8 against 1.9 tok/s at
  18 GB): unlike Q4_K, its Q2_K and Q3_K blocks have not been profiled on the int8 path.

## 4. Known gaps

**Real models run.** Qwen3-0.6B and Qwen3-4B (qwen3), Qwen3-30B-A3B (qwen3moe) and
OLMoE-1B-7B-0125-Instruct (olmoe), all against upstream (section 2). Qwen3-235B-A22B Q2_K (split, larger than memory) ran Metal against the CPU only (section 3). Other quantisations of them
(only Q8_0 and Q4_K_M files were run) and the 0924 OLMoE release have not been.

**First prompt on Metal (fixed).** The first forward of a process used to wait about
1.6 s on Qwen3-30B-A3B (7 s with part of the file out of the page cache) before the GPU
started: the GPU's 0.07 s of work sat behind making the whole-file buffer resident. Two
causes, both fixed: the file was mapped `MAP_PRIVATE` (1.3-1.7 s to make 17 GB resident,
against 0.17 s for `MAP_SHARED`), and the whole file was one buffer, so even with
streamed experts every expert page was read in and made resident. The mapping is now
`MAP_SHARED` and Metal wraps one page-aligned window per tensor on first use
(`MetalBackend.weightBuffer`), so streamed expert tensors are never wrapped
(`MetalSupportTests.fileMappingIsWrappedPerTensor`). First prompt on Qwen3-30B-A3B, warm
file cache: 0.27 s for 19 tokens (was 2.2 s); with `--expert-cache-gb 2` the first
command buffer waits 0.01 s (was 1.7 s). What remains with resident experts (about 0.16 s)
is the GPU mapping 16 GB of experts it really uses.

**Free memory differs from upstream on purpose.** `MemoryBudget.availableBytes` is
physical memory minus anonymous, wired and compressed pages (Activity Monitor's "memory
used"), so the file cache counts as available. Upstream's free + inactive + purgeable made
a second CPU run of Qwen3-30B-A3B, right after a resident Metal run had cached the file,
stream its experts through a 12.8 GB cache on a machine with 64 GB.

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

**`--int8` on K-quants (fixed).** The integer path was slower than the float path on
every type but Q8_0 (Qwen3-4B Q4_K_M: 13 tok/s prefill against 47). The cause was its
16-wide int8 dot: LLVM unrolled the loop into scalar multiply-adds (`ldursb` + `smaddl`
per value) instead of vectorising it; Q8_0's 32-wide loop did get `sdot`. It is now one
NEON `sdot` per 16 values (`colibri_dot_i8_16x2`, the C target `CColibriKernels`), and a
batch unpacks each weight row once instead of once per token. Integer sums are exact and
the float arithmetic is untouched, so outputs are bit-identical (checked per type in
`QuantTests.unpackedRowDotIsBitIdentical`, and the `--int8` oracle comparison gives the
same 14 divergences at the same steps as before). CPU `--int8`, 512-token prompt then 128
tokens, before / after:

| model | prefill tok/s | decode tok/s |
|---|---|---|
| Qwen3-4B Q4_K_M | 13.0 / 54.8 | 9.2 / 15.7 |
| OLMoE-1B-7B-0125-Instruct Q4_K_M | 46.1 / 140.7 | 24.1 / 40.5 |
| Qwen3-30B-A3B Q4_K_M | 18.8 / 60.8 | 9.2 / 13.3 |
| Qwen3-0.6B Q8_0 (unaffected) | 663 / 640 | 68.3 / 67.9 |

`--int8` is now faster than the float path on all four. Still left: MoE expert matmuls
run one token at a time on the CPU (each pair is its own product), and the per-16 float
scaling is scalar.

**Metal and large files.** A file mapping is wrapped one tensor at a time, so a file may
be larger than `maxBufferLength` (41.7 GB here); only a single tensor (a stacked expert
tensor in resident mode) must fit in one buffer. No file over 41.7 GB has been run.

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
