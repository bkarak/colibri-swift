# Oracle harness

Checks the Swift port against upstream colibri (commit `0a5504a`), the C engines run as a
correctness oracle. Upstream is built natively from a copy of its checkout (never in the
checkout itself), run on the same GGUF files as the port, and its outputs are kept as
fixtures in `Tests/ColibriCoreTests/Fixtures/oracle/`. Background and upstream flags:
`docs/port/models.md` section 10.

Everything here uses the system `python3` with the standard library only (run as
`python3 -I`), clang through upstream's Makefile, and SwiftPM. Nothing is installed.

| file | what it does |
|---|---|
| `build-upstream.sh` | copies upstream `c/` to `$ORACLE_DIR/c` (default `${TMPDIR:-/tmp}/colibri-oracle`) and runs `make qwen36 olmoe` (single-threaded, no libomp). `--force` rebuilds. |
| `record.py` | writes the tiny models, runs upstream on them and on the real models, and writes the fixtures. Only needed when the generator, the prompts or upstream change, or for a new real model. |
| `real-models.json` | the real GGUFs the harness knows: fixture key, architecture, upstream engine, default path and the env var that overrides it. A model whose file is absent is skipped. |
| `compare.sh` | builds `coli-swift` (release), writes the tiny models if missing, and runs `compare.py`. |
| `compare.py` | runs `coli-swift generate` on every fixture prompt and reports, per case, MATCH or the first divergence. |

## Synthetic models

The only real model on disk is dense (Qwen3-0.6B), so the MoE families are checked on tiny
random GGUFs from `Sources/ColibriCore/Synthetic/TinyModel.swift`. They are deterministic
(SplitMix64 plus a sum-of-uniforms Gaussian, exact float arithmetic only) and are written
by a test hook, so no extra executable target is needed:

```bash
COLIBRI_TINY_OUT=${TMPDIR:-/tmp}/colibri-oracle/tiny \
  swift test --scratch-path .build-oracle --filter TinyModelTests/writeOracleModels
```

That writes `<case>.gguf` for every entry of `TinyOracleCases.all`, plus `manifest.json`
with each file's FNV-1a 64 fingerprint. Shapes (every matrix input width is a multiple of
256, so K-quants work everywhere; norms and routers are F32):

| arch | L | D | H / KV / hd | FFN |
|---|---|---|---|---|
| qwen3 | 2 | 256 | 4 / 2 / 128 (attention width 512 != D) | dense 512 |
| qwen3moe | 3 | 256 | 8 / 2 / 64 (512 != D) | 16 experts, top-4, I 256, renormalised |
| olmoe | 2 | 256 | 4 / 4 / 64 | 8 experts, top-2, I 256, not renormalised |

Cases (16): `{qwen3moe,olmoe,qwen3}_{f32,q8_0,q4_K,q4_K_M}` (q4_K_M = Q4_K with Q6_K for
`attn_v`, the down projections and `output`), `qwen3_q8_0_tied`, `qwen3moe_q8_0_tied`,
and two with the real Qwen3 tokenizer copied in (V = 151936):
`qwen3moe_q8_0_qwenvocab`, `olmoe_q4_K_qwenvocab`. The other cases use the 256 GPT-2
byte symbols plus a few control tokens and no merges, so ASCII text is one token per
byte. The qwenvocab cases are skipped when `~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`
(or `$COLIBRI_QWEN3_GGUF`) is absent.

## How upstream is run

- **qwen3, qwen3moe**: `SNAP=<file.gguf> COLI_LOOKUP=0 GQ_EXACT=1 DUMP=<f32> ./qwen36 0 4 ref.json`,
  where `ref.json` is `{"prompt_ids": [...], "full_ids": prompt + n zeros}`. Oracle mode:
  greedy, float32 KV cache, never stops at EOS; stdout has `C engine : <ids> Text : ...`.
  `DUMP` holds only the logits of the last step, so `record.py` runs it once per step
  count (1..n) to get every step's logits (the runs are deterministic; it checks that
  each run's ids are a prefix of the next).
- **olmoe**: `python3 -I c/tools/gguf_import.py <file.gguf> <dir>` (what `coli import`
  runs; link mode, nothing copied), then `SNAP=<dir> GQ_EXACT=1 DUMP=<f32> ./olmoe <E> 8 ref.json`.
  Here `DUMP` holds every step's logits.
- **prompt ids**: `SNAP=<gguf> TOK_ONLY=1 ./qwen36 0 4 prompt.txt` for the Qwen tokenizer;
  for the byte-vocabulary olmoe models they are the bytes, with `|||IP_ADDRESS|||` as 256.
- Each prompt is also run without `GQ_EXACT` (upstream's default int8-activation path) for
  `greedy_default`.

## Real models

`record.py --real --real-models <key>` runs upstream on the five fixed prompts of every listed
model present (`Fixtures/oracle/<key>.json`); `compare.py --real-only --real-models <key>`
checks coli-swift against it. Upstream's macOS build is single-threaded, so recording every
step's logits (one run per step count) is slow for a large model: `--light` records the
greedy ids from one run plus the prefill logits, and runs the prompts in parallel (`--jobs`;
each qwen36 process held about 8.7 GB on Qwen3-30B-A3B, so keep jobs x that under RAM). A
divergence then cannot be told from a near tie without re-recording that prompt in full.
Qwen3-30B-A3B Q4_K_M took about 6 minutes with `--light --jobs 5`.

A model entry can name its prompt set (`"prompts": "qwen"` or `"olmoe"`, in `record.py`)
and, for an engine without a token-dump mode (olmoe), a file of prompt ids under
`prompt-ids/` made by the model's own Hugging Face tokenizer with `hf_tokenize.py` (in a
throwaway Docker container; the command is in its header). The fixture then holds those
ids, and `compare.py` fails a run as TOKENIZER when the port encodes the text differently.
OLMoE's olmoe engine dumps every step's logits in one run, so it records in full (about 5
minutes) without `--light`.

## Fixtures

`Fixtures/oracle/tiny/<case>.json` (24 new tokens, 4 prompts each) and
`Fixtures/oracle/qwen3-0.6b-q8_0.json` (32 new tokens, 5 prompts: the models.md golden
chat prompt, a haiku chat prompt, a raw English continuation, a raw Python continuation,
and a Greek chat prompt with a system message). Per prompt:

| key | content |
|---|---|
| `text`, `prompt_ids` | the raw prompt text (templates already rendered) and upstream's ids for it |
| `greedy_exact` | greedy ids with `GQ_EXACT=1`: the reference for the port's float math |
| `greedy_default` | greedy ids with upstream's default int8 activations |
| `steps_exact` | per generated token: upstream's top-5 `[id, logit]` and the top-2 gap |
| `prefill_logits_exact` | every prefill logit (byte-vocabulary models) |
| `prefill_top_exact` / `prefill_top_default` | the top-20 prefill logits (large vocabularies) |

Each tiny fixture also stores the model's fingerprint; `OracleTests` regenerates the model
and fails if the generator no longer writes the same bytes.

## Comparing

In the test suite (`Tests/ColibriCoreTests/OracleTests.swift`, CPU backend through the
library): fixture sanity, generator fingerprints, tokenizer vs upstream prompt ids, and
prefill logits (tolerance 2e-3 tiny, 2e-2 top-20 real) plus greedy ids for 3 tiny cases.
`COLIBRI_ORACLE_FULL=1` runs all 16 tiny cases; `COLIBRI_ORACLE_REAL=1` adds every real model present
(use a release build: `swift test -c release -Xswiftc -enable-testing --filter OracleTests`).

Through the CLI, any backend:

```bash
scripts/oracle/compare.sh                         # cpu, all 16 tiny cases + Qwen3-0.6B
scripts/oracle/compare.sh --backend metal
scripts/oracle/compare.sh --tiny-only -- --experts stream     # forced expert streaming
scripts/oracle/compare.sh --ids                   # feed upstream's prompt ids, skip the tokenizer
scripts/oracle/compare.sh --reference default     # coli-swift --int8 vs upstream's int8 path
scripts/oracle/compare.sh --cases olmoe_q8_0 --verbose
```

`compare.py` runs `coli-swift generate --model M --backend B --raw --prompt TEXT --temp 0
--max-tokens N --ignore-eos --kv f32 --dump` and reads `prompt_ids (n): ...` and
`generated_ids (n): ...` from stderr. Statuses: `MATCH`; `NEAR-TIE` (diverged where
upstream's own top-2 gap is under `--near-tie`, default 0.01, so summation order can flip
it); `DIVERGED` (step, both ids, upstream's top-3 and the rank of the Swift pick);
`TOKENIZER` (prompt ids differ); `ERROR`, `MISSING`, `STALE` (model fingerprint differs
from the fixture). Exit 1 unless every run is MATCH or NEAR-TIE. Runs on the same model
file are serialised, because coli-swift saves the routing history next to the model on
exit.

## Re-recording

```bash
scripts/oracle/build-upstream.sh --force          # if upstream moved
python3 -I scripts/oracle/record.py               # tiny + real, about 7 minutes
python3 -I scripts/oracle/record.py --tiny --skip-generate --cases olmoe_q8_0
```

Re-recording is deterministic: running it again gives byte-identical fixtures.
