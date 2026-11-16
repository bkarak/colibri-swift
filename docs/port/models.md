# Model families: qwen3, qwen3moe, olmoe

What the Swift port has to compute for the three GGUF architectures in scope, read from upstream colibri at commit `0a5504a`. It covers the forward pass (exact equations, eps, tensor names and shapes), the tokenizer, chat templates, stop tokens, sampling defaults, and a recipe for running the upstream C engines as an oracle. Everything in the oracle section was run on this machine (M4 Max, macOS 27) on 2026-11-14.

Upstream sources, by section:

| Topic | Upstream file (functions) |
|---|---|
| qwen3 / qwen3moe hyperparameters | `c/qwen36.c` `gguf_load_cfg` (l. 3552), `gg_name` (l. 2834), `gg_tensor` (l. 2894) |
| qwen3 / qwen3moe forward | `c/qwen36.c` `layers_forward_control` (l. 6495), `attention` (l. 4854), `rope_head_partial` (l. 4803), `rmsnorm_row` (l. 2313), `load_norm_n` (l. 3520), `moe_ex` (l. 5646), `moe_gq_decode` (l. 5538), `qwen_shared_experts_cpu` (l. 5043), `step_ex_control` (l. 6701), `q36_embed_row` (l. 6643) |
| olmoe hyperparameters | `c/tools/gguf_import.py` `olmoe_config` (l. 919) -> `c/olmoe.c` `load_cfg` (l. 646); names `c/st_gguf.h` (l. 312) |
| olmoe forward | `c/olmoe.c` `layers_forward_range` (l. 1959), `attention` (l. 1601), `rope_head` (l. 1589), `moe_route_row` (l. 1657), `moe_expert_rows_gx` (l. 1724), `moe_by_expert` (l. 1790), `step` (l. 2149) |
| tokenizer (qwen) | `c/qwen36.c` `load_tokenizer_gguf` (l. 714), `pretok_end` (l. 321), `bpe_piece` (l. 489), `encode_text` (l. 540), `decode_id_to_bytes` (l. 861) |
| tokenizer (olmoe) | `c/tok.h` `tok_load`, `pretok_chunk_gpt2` (l. 584), `bpe_piece` (l. 267), `tok_encode` (l. 730), `tok_decode` (l. 763); `c/tok_unicode.h` |
| sampling | `c/sample.h` (olmoe), `c/qwen36_sample.h` + `c/argmax_first.h` (qwen36 serve) |
| chat templates | `c/openai_server.py` `render_chat_qwen` (l. 1852), `render_chat_qwen3_coder` (l. 2764), `render_chat_olmoe` (l. 1790); `c/olmoe.c` `fmt_user_turn` (l. 2562) |

Upstream is Apache-2.0. Swift files that port these functions directly should carry a `Ported from colibri c/<file>` comment.

---

## 1. Conventions

**GGUF shapes.** ggml writes `ne[0]` first: the number of elements in one row, i.e. the *input* features. A linear layer `W` stored as `[in, out]` in GGUF is `out` rows of `in` values each, row-major, and computes

    y[o] = sum_i W[o][i] * x[i]          o in [0, out), i in [0, in)

A 3-D expert tensor `[in, out, n_expert]` is `n_expert` such matrices back to back. Expert `e` starts at byte `e * out * row_bytes(type, in)` of the tensor.

**Block types in the files we can test.** Qwen3-0.6B-Q8_0.gguf has only type 0 (F32: all norms) and type 8 (Q8_0: every matrix, including `token_embd`). A Q8_0 block is 34 bytes: an IEEE half `d` then 32 `int8` values `q`; value `= float(d) * q`. Routers (`ffn_gate_inp`) stay F32 in llama.cpp quantizations. The full type catalogue is in `gguf-and-quants.md`.

**Names below.** `D` hidden size, `H` query heads, `KV` key/value heads, `hd` head dimension, `E` routed experts, `K` experts used per token, `I` expert FFN width, `F` dense FFN width, `V` vocab, `L` layers, `eps` RMSNorm epsilon, `theta` RoPE base, `t` token position (0-based, absolute).

---

## 2. Hyperparameters from the GGUF

### qwen3 and qwen3moe (upstream reads the GGUF directly)

Prefix `P = general.architecture` (`qwen3.` or `qwen3moe.`). Defaults are upstream's when the key is missing.

| Field | Key | Default / rule |
|---|---|---|
| `L` | `P.block_count` minus `P.nextn_predict_layers` | `nextn` default 0 |
| `D` | `P.embedding_length` | required |
| `H` | `P.attention.head_count` | required |
| `KV` | `P.attention.head_count_kv` | if it is an array (Ollama), the max element |
| `hd` | `P.attention.key_length` | `D / H` |
| v head dim | `P.attention.value_length` | `hd` (upstream assumes equal to `hd` in the attention) |
| `theta` | `P.rope.freq_base` | 10000 (real Qwen3 files carry 1e6) |
| `eps` | `P.attention.layer_norm_rms_epsilon` | 1e-6 |
| rope dims | `P.rope.dimension_count` | `hd` (full rotary) |
| `E`, `K`, `I` (qwen3moe) | `P.expert_count`, `P.expert_used_count`, `P.expert_feed_forward_length` | required for MoE |
| `F` (qwen3) | `P.feed_forward_length` | required; `E = K = 0` |
| `V` | `token_embd.weight` `ne[1]` | |
| q/k norm present | `blk.{L-1}.attn_q_norm.weight` exists | |
| tied embeddings | no `output.weight` in the file | `lm_head` reads `token_embd.weight` |

Keys upstream does **not** read for these archs and the port should at least refuse or warn on: `P.rope.scaling.*` (YaRN context extension), `P.expert_weights_norm`, `P.expert_weights_scale`, `P.expert_gating_func`. Upstream always renormalises the top-k for qwen3moe (`c->norm_topk = 1`, and the flag is never read on the routing path), which is right for every Qwen3 MoE checkpoint.

### olmoe (upstream goes through `gguf_import.py` -> `config.json`)

Prefix `olmoe.`:

| Field | Key | Default |
|---|---|---|
| `L` | `block_count` | required |
| `D` | `embedding_length` | required |
| `H` | `attention.head_count` | required |
| `KV` | `attention.head_count_kv` | `H` (the engine requires `KV == H`, see section 4) |
| `hd` | none | `D / H` |
| `E`, `K` | `expert_count`, `expert_used_count` | required |
| `I` | `expert_feed_forward_length`, else `feed_forward_length` | required |
| `norm_topk` | `expert_weights_norm` | **false** (llama.cpp writes no such key for OLMoE; the official checkpoints have `norm_topk_prob: false`) |
| `eps` | `attention.layer_norm_rms_epsilon` | 1e-5 |
| `theta` | `rope.freq_base` | 10000 |
| `V` | `token_embd.weight` `ne[1]` | |

Not read: `clip_qkv` (null in OLMoE-1B-7B, so it does not matter).

### Reference checkpoints

| | Qwen3-0.6B (verified from the GGUF on disk) | Qwen3-30B-A3B (HF config, not on disk) | OLMoE-1B-7B-0924 (HF config, not on disk) |
|---|---|---|---|
| arch | `qwen3` | `qwen3moe` | `olmoe` |
| `L` / `D` | 28 / 1024 | 48 / 2048 | 16 / 2048 |
| `H` / `KV` / `hd` | 16 / 8 / 128 (`H*hd` = 2048 != `D`) | 32 / 4 / 128 | 16 / 16 / 128 |
| FFN | dense `F` = 3072 | `E`=128, `K`=8, `I`=768 | `E`=64, `K`=8, `I`=1024 |
| `theta` / `eps` | 1e6 / 1e-6 (stored as f32 9.99999997e-07) | 1e6 / 1e-6 | 1e4 / 1e-5 |
| norm_topk | n/a | true | false |
| `V` | 151936 | 151936 | 50304 |
| tied | yes (no `output.weight`) | no | no |
| context | 40960 | 40960 | 4096 |

Qwen3-Coder-30B-A3B is also `qwen3moe` with the same layer shapes; take `theta` and context length from its GGUF rather than from this table.

---

## 3. Tensor names

Global: `token_embd.weight [D, V]`, `output_norm.weight [D]` (F32), `output.weight [D, V]` (absent when tied).

Per layer `blk.N.`:

| Tensor | qwen3 | qwen3moe | olmoe |
|---|---|---|---|
| `attn_norm.weight` | `[D]` | `[D]` | `[D]` |
| `attn_q.weight` | `[D, H*hd]` | `[D, H*hd]` | `[D, D]` |
| `attn_k.weight` | `[D, KV*hd]` | `[D, KV*hd]` | `[D, D]` |
| `attn_v.weight` | `[D, KV*hd]` | `[D, KV*hd]` | `[D, D]` |
| `attn_output.weight` | `[H*hd, D]` | `[H*hd, D]` | `[D, D]` |
| `attn_q_norm.weight` | `[hd]` (one per head, shared by all heads) | `[hd]` | **`[D]`** (whole q vector) |
| `attn_k_norm.weight` | `[hd]` | `[hd]` | **`[KV*hd]`** (= `[D]`) |
| `ffn_norm.weight` | `[D]` | `[D]` | `[D]` |
| `ffn_gate.weight` / `ffn_up.weight` | `[D, F]` | | |
| `ffn_down.weight` | `[F, D]` | | |
| `ffn_gate_inp.weight` (router) | | `[D, E]` | `[D, E]` |
| `ffn_gate_exps.weight` / `ffn_up_exps.weight` | | `[D, I, E]` | `[D, I, E]` |
| `ffn_down_exps.weight` | | `[I, D, E]` | `[I, D, E]` |

llama.cpp's qwen3 and olmoe converters do **not** permute q/k rows for rotary (both are NEOX-style models), so the rows are used as stored. (Upstream un-permutes only for `muse-spark`.) Upstream also honours an optional `<name>.scale` F32 companion tensor (one factor per tensor, or per expert) as a multiplier on the product; standard llama.cpp files do not have it.

---

## 4. Forward pass

One skeleton serves all three archs: pre-norm residual blocks.

```
x_t      = embed(token_t)                                    # dequantized row of token_embd, no scaling
for each layer l:
    a    = RMSNorm(x; attn_norm_l)
    x    = x + Attention_l(a)
    f    = RMSNorm(x; ffn_norm_l)
    x    = x + FFN_l(f)                                       # dense SwiGLU (qwen3) or MoE (qwen3moe, olmoe)
logits   = W_out · RMSNorm(x_last; output_norm)               # W_out = output.weight, or token_embd.weight when tied
```

There is no logit scaling, no softcap, no embedding scale, no attention output gate, no shared expert and no biases in any of the three.

### RMSNorm

```
RMSNorm(x; w)[i] = x[i] * r * w[i],    r = 1 / sqrt( (sum_j x[j]^2) / n + eps )
```

Upstream accumulates `sum x^2` in **double**, divides by `n` in double, converts to float, adds `eps` in float, then `1/sqrtf`. The same `eps` (the model's) is used for every norm, q/k norms included. (qwen36 stores weights as `w - 1` and multiplies by `1 + w`; for these archs that is the plain `x * r * w` above, so the port should just use `w`.)

### Attention, qwen3 and qwen3moe

```
q = W_q a   (H*hd)      k = W_k a   (KV*hd)      v = W_v a   (KV*hd)
for each query head h:  q_h = RMSNorm(q_h; attn_q_norm)  (n = hd)   then  q_h = RoPE(q_h, t)
for each kv head g:     k_g = RMSNorm(k_g; attn_k_norm)  (n = hd)   then  k_g = RoPE(k_g, t)
append k_g, v_g to the cache at position t
for each query head h:  g = h / (H / KV)                  # integer division: consecutive heads share a KV head
    s_u   = (q_h · k_g,u) / sqrt(hd)       for u = 0..t   # causal, no mask beyond u <= t
    p     = softmax(s)                                    # max-subtracted, expf, float sum
    ctx_h = sum_u p_u v_g,u
out = W_o concat_h(ctx_h)                                 # input H*hd, output D
```

Order matters: norm first, then RoPE, per head.

### Attention, olmoe

```
q = W_q a   (D)        k = W_k a   (D)        v = W_v a   (D)
q = RMSNorm(q; attn_q_norm)   over the WHOLE vector, n = D
k = RMSNorm(k; attn_k_norm)   over the WHOLE vector, n = KV*hd = D
for each head h: q_h = RoPE(q_h, t), k_h = RoPE(k_h, t)
standard causal MHA, scale 1/sqrt(hd), head h reads KV head h
out = W_o concat(ctx)
```

The olmoe engine sizes K and V as `D` and indexes the cache by query head, so it only runs `KV == H` (true of every OLMoE release; `load_t` refuses other shapes). The port can do GQA generally.

### RoPE (NEOX / half-split, all three archs)

For a head vector `x` of length `hd`, `R` = rope dims (= `hd` here), `h2 = R/2`:

```
for j in 0 ..< h2:
    inv  = powf(theta, -2.0f * j / R)        # float
    ang  = (float)t * inv                     # float
    c, s = cosf(ang), sinf(ang)
    (x[j], x[j+h2]) = (x[j]*c - x[j+h2]*s,  x[j+h2]*c + x[j]*s)
```

Pairs are `(j, j + R/2)`, not adjacent pairs. Upstream computes the angle in float; at large `t` a double-precision angle will drift from upstream's last bits (and from llama.cpp, which builds `theta^(-2j/R)` by repeated multiplication). Token ids should not care; if logits are compared at long positions, expect it.

### Dense FFN (qwen3)

```
FFN(f) = W_down ( silu(W_gate f) ⊙ (W_up f) ),     silu(z) = z / (1 + exp(-z))
```

### MoE FFN (qwen3moe, olmoe)

```
r      = W_router f                          # E logits; router is F32 in real files
p      = softmax(r)                          # over all E, max-subtracted, expf, float sum
idx    = top-K of p:  K passes, each takes the largest p_e not already taken,
         compared with strict '>' from e = 0 upward, so the LOWER expert id wins a tie
w_k    = p[idx_k]
if norm_topk:  w_k = w_k / sum_k w_k         # qwen3moe: always. olmoe: per expert_weights_norm, default false
FFN(f) = sum_k  w_k * W_down[idx_k] ( silu(W_gate[idx_k] f) ⊙ (W_up[idx_k] f) )
```

Summation order of the K expert outputs (matters only for last-bit comparisons): qwen36 adds them in ascending expert-id order; olmoe adds them in rank order (`k = 0..K-1`).

Routing (softmax, top-k, renormalise) is the hook a future family will most often change (sigmoid scoring, bias-corrected selection, grouped top-k, shared experts). Keep it a per-family function, not a hard-coded step.

### Final norm and head

`RMSNorm` with `output_norm.weight` on the last position only (prefill computes logits for the last row only), then `W_out` (`[D, V]`). For a tied file `W_out` is `token_embd.weight`, which is Q8_0 in Qwen3-0.6B, so the head is a Q8_0 matmul.

### KV cache

Per layer: K and V `[KV, max_t, hd]`. Rows are written at the step's positions before scores are computed. Prefill is one forward of `S` rows at positions `0..S-1`; decode is `S = 1` at `t = len - 1`.

Upstream stores K and V as **float16** by default in chat and serve, and rounds the current step's K and V to f16 before using them. Its oracle modes (a `.json` reference argument) switch to **float32** (`Q36_KV=f32` and `KV16=0` force it). The port can start with an f32 cache; f16 is a memory optimisation to add later, gated by the same oracle.

---

## 5. Numerics: what upstream actually computes

Two upstream behaviours decide how close a float32 Swift implementation can get to the oracle.

1. **Activations are quantized to int8 by default.** For every ggml block-quantized matrix, upstream's CPU path (`gq_act_set`, `ggml_quants.h`) quantizes the activation vector per 32 values (`d = amax/127`, `q = lrintf(x * 127/amax)` clamped to ±127) and runs an integer dot product against the weight blocks. That is ggml's own Q8 activation scheme and it is lossy. `GQ_EXACT=1` switches to the float reference: dequantize each weight row to f32, then a float dot product (`gq_matmul_f32`). Measured on Qwen3-0.6B-Q8_0, prefill logits of the 19-token prompt below: **max |default - GQ_EXACT| = 0.55** (top logit 29.95 vs 30.21), same argmax, and TF-NLL 0.018421 vs 0.016134 nats. A Swift port that dequantizes to f32 should be compared against `GQ_EXACT=1`.
2. **F32 tensors never go through the int8 path.** A synthetic GGUF written all in F32 is computed in plain float by both engines (dense and experts), which makes F32 tiny models the tightest oracle for the MoE archs.

`Q36_ILV16=0` / `OLM_ILV16=0` turn off upstream's 16-row interleaving. That changes speed, not results; `GQ_EXACT=1` implies it.

---

## 6. Tokenizer

### GGUF keys

| Key | Use |
|---|---|
| `tokenizer.ggml.model` | must be `gpt2` (byte-level BPE); upstream refuses anything else |
| `tokenizer.ggml.pre` | pre-tokenizer: `qwen2` for Qwen3, `olmo` for OLMoE |
| `tokenizer.ggml.tokens` | id -> piece (byte-level unicode string) |
| `tokenizer.ggml.token_type` | 1 normal, 3 CONTROL (special), 4 USER_DEFINED (added, not special), 5 UNUSED |
| `tokenizer.ggml.merges` | `"a b"` strings; rank = index in the array |
| `tokenizer.ggml.bos/eos/padding/eot/eom_token_id`, `eog_token_ids` | stops (section 8) |
| `tokenizer.ggml.add_bos_token` | False for Qwen3; OLMoE prompts start with the bos text in the template instead |
| `tokenizer.chat_template` | Jinja; upstream never executes it (hand-written renderers, section 7) |

Qwen3-0.6B: 151936 tokens = 151643 normal + 20 CONTROL + 6 USER_DEFINED (`<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>` `<think>` `</think>`, ids 151657/8, 151665-8) + 267 UNUSED (`[PAD151669]`...). 151387 merges. `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645.

### Byte map (GPT-2 ByteLevel)

Bytes 33..126, 161..172 and 174..255 map to the code point of the same value; the remaining 68 bytes, in increasing order, map to 256, 257, ... A piece is the UTF-8 of the mapped code points (space -> `Ġ` U+0120, `\n` -> `Ċ` U+010A). Decoding reverses it.

### Encoding

1. **Added tokens first.** Every token of type 3 or 4 is matched literally, longest match first, anywhere in the text (`normalized: false`, no lstrip/rstrip). Text between matches is pre-tokenized as its own input, so a regex rule never reaches across an added token (upstream #1653: `X.<|im_end|>` must be 3 tokens).
2. **Pre-tokenize** each ordinary span (regex below), in order, leftmost match.
3. **BPE** each piece: map its bytes to symbols, then repeatedly merge the adjacent pair with the lowest merge rank until no pair is in the merges table; emit the vocab id of each final symbol.
4. No BOS is added for either family (Qwen `add_bos_token` false; OLMoE's template writes `|||IP_ADDRESS|||` as text, which then matches as an added token).

**Qwen2 pre-tokenizer** (`pre = "qwen2"`; HF Qwen2Tokenizer):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
```

Digits are split one by one. `\s+(?!\S)` leaves the last whitespace of a run for the next piece, which takes it as its optional leading char (`"  x"` -> `" "`, `" x"`). Combining marks are **not** letters in HF (upstream's exact mode documents `"x" U+0303 U+0304 "y"` as three pieces).

**GPT-2 pre-tokenizer** (`pre = "olmo"`; OLMoE's tokenizer.json is a bare ByteLevel with `use_regex: true`, which applies GPT-2's pattern):

```
's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
```

Case-sensitive contractions, one optional leading space (not any non-letter), unbounded digit runs, no newline tail on punctuation, no `\s*[\r\n]+` rule.

The pre-tokenizer should be chosen by `tokenizer.ggml.pre`, not by architecture: a future family is likelier to bring a new pre-tokenizer than a new BPE. llama.cpp's `llama-vocab.cpp` lists the regex for every `pre` name; upstream's `gguf_import.py` `PRE_SPLITS` has the ones colibri runs (cl100k, o200k, deepseek, flint, tekken, kimi).

Where upstream deviates from HF (the port should follow HF, and test against upstream only on ASCII and common scripts):

- qwen36 classifies characters with hand-written ranges (`uclass`), not full Unicode tables, and lets marks (U+0300.. etc.) join letter runs. Swift's `Unicode.Scalar.Properties` (`generalCategory`) gives the real classes.
- `tok.h` (olmoe) emits a piece as one token whenever the whole piece is in the vocab, before trying merges (`ignore_merges` behaviour), for every family. For GPT-2-style BPE that differs from HF in rare cases. The port should only do this when the family's tokenizer sets `ignore_merges` (o200k/cl100k ones do; Qwen2 and OLMo do not).
- `gguf_import.py` writes a plain ByteLevel (GPT-2 regex) `tokenizer.json` for `pre = "qwen2"`. Harmless upstream because qwen36 reads the GGUF's vocab directly and uses its own Qwen splitter, but do not copy that mapping.

### Decoding

Map each piece's code points back through the byte map and concatenate bytes; buffer incomplete UTF-8 across tokens when streaming. qwen36 decodes CONTROL tokens (type 3) to nothing (except Muse Spark's `<|start|>`, `<|message|>`, `<|eom|>`) and USER_DEFINED ones (`<think>`) to their text. tok.h decodes every added token to its literal text. The port should decode type 3 to nothing for display and keep `<think>`/`</think>` visible (or strip reasoning in the CLI on purpose).

### Golden ids (upstream, verified)

Prompt text (file bytes, no trailing newline after the last `\n\n`):

```
<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

-> `151644 872 198 3838 374 279 6722 315 9625 30 151645 198 151644 77091 198 151667 271 151668 271` (19 tokens; `TOK_ONLY=1`, see section 9).

---

## 7. Chat templates

Upstream renders templates in Python (`openai_server.py`); it does not run the GGUF's Jinja. The port can do the same: a small Swift renderer per family, chosen by architecture plus a sniff of `tokenizer.chat_template`.

**Qwen3 (qwen3, qwen3moe instruct models)**, `render_chat_qwen`, which matches the GGUF's Jinja for plain chats:

```
[<|im_start|>system\n{system}<|im_end|>\n]          # only when a system message is given
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n{reply}<|im_end|>\n          # history; past replies keep no <think> block
...
<|im_start|>assistant\n                              # generation prompt
<think>\n                                            # thinking ON (the renderer's enable_thinking defaults to false)
<think>\n\n</think>\n\n                              # thinking OFF
```

The GGUF Jinja itself adds nothing after `<|im_start|>assistant\n` when thinking is on and `<think>\n\n</think>\n\n` when `enable_thinking` is false; upstream's `<think>\n` prefill for the on case is its own choice. Message content is `.strip()`ped by upstream. Tool calls use `<tool_call>{"name":...,"arguments":...}</tool_call>`; out of scope for the first port.

**Qwen3-Coder** (also `qwen3moe`; detected upstream by the template text `interact with a computer to solve tasks`): ChatML without any `<think>`, its own XML tool block (`render_chat_qwen3_coder`). For plain chat it is the Qwen3 layout without the think lines.

**OLMoE-Instruct**, `render_chat_olmoe` (bos_token == eos_token == `|||IP_ADDRESS|||`, a PII-scrub artefact the tokenizer uses as BOS/EOS):

```
|||IP_ADDRESS|||
<|system|>\n{system}\n                               # optional
<|user|>\n{user}\n
<|assistant|>\n{reply}|||IP_ADDRESS|||\n            # history; the '\n' only if more messages follow
<|assistant|>\n                                      # generation prompt
```

`<|user|>`, `<|assistant|>` and `<|system|>` are ordinary text in this tokenizer, not added tokens. The CHAT=1 mode of the olmoe engine (`fmt_user_turn`) writes the first turn as `|||IP_ADDRESS|||<|user|>\n{msg}\n<|assistant|>\n` and later turns as `|||IP_ADDRESS|||\n<|user|>\n{msg}\n<|assistant|>\n`.

A base (non-instruct) model gets raw text, no template. The CLI should have a `--raw` / no-template switch for the oracle comparisons below, which all use pre-rendered prompts.

---

## 8. Stop tokens

qwen36 collects `tokenizer.ggml.eog_token_ids` (all), then `eos_token_id`, `eot_token_id`, `eom_token_id`. `gguf_import.py` adds `<|endoftext|>` and `<|im_end|>` when present. For Qwen3: stop on 151645 `<|im_end|>` and 151643 `<|endoftext|>`. For OLMoE: `|||IP_ADDRESS|||` (the eos) plus, in the olmoe CHAT path, every token the tokenizer marks `special: true`. The oracle modes never stop: they generate exactly `n_new` tokens past EOS (the golden run below emits 151645 then keeps going).

---

## 9. Sampling defaults

| Path | Rule |
|---|---|
| upstream CLI / oracle (both engines) | greedy: first index of the maximum, strict `>` scan from id 0 (`generate`) |
| qwen36 serve (`qwen36_sample.h`) | `temp <= 0` -> `argmax_first`; else softmax(`logit/temp`), top-p nucleus (sorted by prob, ties by id), C `rand()` |
| gateway defaults (`openai_server.py`) | `temperature` = `COLI_TEMP` or **0.7**, `top_p` **0.9**; no top-k, no repetition penalty |
| olmoe CHAT (`sample.h`) | `TEMP` default **0.7**, `NUCLEUS` **0.95**, xorshift64* seeded `0x9E3779B97F4A7C15`, argmax skips NaN |

Not upstream, but what the Qwen3 model card recommends: thinking mode temp 0.6, top_p 0.95, top_k 20; non-thinking temp 0.7, top_p 0.8, top_k 20. The port's sampler should take temp/top-p/top-k/seed and default to greedy when `temp == 0`, so oracle runs are reproducible.

---

## 10. Oracle recipe (upstream C engines)

### Build (done once, native, no libomp needed)

```bash
# UPSTREAM: a checkout of https://github.com/JustVugg/colibri at 0a5504a
# ORACLE_DIR: scripts/oracle's working directory (default ${TMPDIR:-/tmp}/colibri-oracle)
mkdir -p "$ORACLE_DIR"
cp -R "$UPSTREAM/c" "$ORACLE_DIR/c"
cd "$ORACLE_DIR/c" && make qwen36 olmoe
# clang -O3 ... qwen36.c -o qwen36 -lm   (single-threaded, ~6 s for both)
```

The binaries land in `$ORACLE_DIR/c/{qwen36,olmoe}` (under `/tmp` by default, which a reboot can clear); `scripts/oracle/build-upstream.sh` does the same.

### qwen3 / qwen3moe: `qwen36` takes the raw GGUF

```
SNAP=<file.gguf> [env] ./qwen36 [cache/layer] [expert_bits] [prompt.txt | ref.json] [tokenizer.json]
```

- `SNAP` may be the `.gguf` itself (no `coli import` needed) or an imported directory with `gguf.json`.
- `cache/layer`: `0` = auto. `expert_bits`: must be 2..8, ignored for GGUF (pass `4`).
- Third argument ending in `.json`: **oracle mode**. File `{"prompt_ids":[...], "full_ids":[...]}`; it generates `len(full_ids) - len(prompt_ids)` tokens greedily (never teacher-forced, never stops at EOS) and prints to stdout `C engine : <ids> Text : <decoded>` and `Matching tokens: m/n` against `full_ids`. KV cache is float32. Exit code 1 unless all match. Put any filler (zeros) in the continuation when you only want ids.
- Any other third argument: a **text prompt file**, encoded with the GGUF tokenizer (specials matched in the text; no template, no BOS added). `N_NEW` sets the count (default 64). Streams text to stdout. KV cache float16.

| Env | Effect |
|---|---|
| `TOK_ONLY=1` | text mode: print the prompt's token ids on stdout and exit |
| `ENC_DEBUG=1` | text mode: prompt ids on stderr |
| `COLI_LOOKUP=0` | turn off prompt-lookup speculative decoding (on by default; output is still the argmax, but keep oracles simple) |
| `GQ_EXACT=1` | float reference matmul (dequantize rows to f32) instead of int8 activations. **Use for logit comparisons.** |
| `DUMP=<path>` | write the logits that chose the **last** generated token, `V` float32 little-endian. With `n_new = 1` that is the prefill's last-position logits. |
| `DUMP_LAYERS=<path>` | per step (file rewritten each step, so the last step wins): for each layer, 3 rows of `D` float32 for the step's last position: attention sublayer output, residual after attention, residual after FFN. Size `L*3*D*4`. |
| `PPL=1` (with ref.json) | teacher-forced NLL over `full_ids[np:]`, prints `TF-NLL: %.6f nats/token over N tokens` |
| `Q36_KV=f32` | float32 KV outside oracle mode |
| `N_NEW`, `NOSTREAM` | text mode count; `NOSTREAM` stops the streamed print |

### olmoe: needs an imported directory

```bash
python3 "$ORACLE_DIR/c/tools/gguf_import.py" <file.gguf> <out_dir>   # what `coli import` runs; stdlib only, link mode
SNAP=<out_dir> [env] ./olmoe [cache/layer=16] [bits=8] ref.json
```

`gguf_import.py` writes `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and `gguf.json` (absolute path to the GGUF; nothing copied). The engine maps GGUF names to HF names through `st_gguf.h`. Without `CHAT=1` or `SERVE=1` the only mode is the ref.json oracle, same file format, same `C engine : ...` / `Matching tokens` output, float32 KV (`KV16=1` for f16). `cache/layer` must be at least `K`; use `E` to hold every expert.

| Env | Effect |
|---|---|
| `DUMP=<path>` | **every** step's logits, `n_new * V` float32 (the prefill's first) |
| `PPL=1` | `TF-NLL: %.4f nats/token ...` |
| `GQ_EXACT=1` | float reference for block-quantized matrices |
| `CHAT=1`, `TEMP`, `NUCLEUS`, `MAX_NEW`, `CTX` (<= 4096) | interactive chat (reads `tokenizer.json` from `SNAP`) |

### Synthetic models for the MoE archs

`mk_tiny_gguf.py` (a stdlib-only throwaway written for this doc and not kept; `Sources/ColibriCore/Synthetic/TinyModel.swift` does the job now) wrote a tiny random GGUF for `qwen3`, `qwen3moe` or `olmoe`: `D`=64, `L`=2, `H`=4, `KV`=2 (qwen) or 4 (olmoe), `hd`=32 (qwen, so `H*hd` != `D`) or 16 (olmoe), `E`=8, `K`=2, `I`=32, `F`=96, a 256-byte-symbol vocab plus a few control tokens (`V` = 261 qwen, 259 olmoe), no merges. Weights are Gaussian with a fixed seed; norms 1 ± 0.1; router F32.

```bash
python3 -I mk_tiny_gguf.py qwen3moe tiny_qwen3moe.gguf --type q8_0      # or --type f32, --seed N, --tied (qwen3)
```

It should move into the repo (for example under `Tools/`) so the test suite can regenerate the same files.

### Results on this machine

| Model | Command | Output |
|---|---|---|
| Qwen3-0.6B-Q8_0 | `SNAP=~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf COLI_LOOKUP=0 ./qwen36 0 4 ref.json` with the 19 golden prompt ids and 16 filler | `785 6722 315 9625 374 3070 59604 334 13 151645 198 151643 33975 25 3555 374` = "The capital of France is **Paris**." then `<|im_end|>` `\n` `<|endoftext|>` "Human: What is". Identical with `GQ_EXACT=1`. 0.7 s. |
| same, prefill logits (`n_new = 1`, `DUMP`) | default / `GQ_EXACT=1` | top-3 `785:29.9455 49000:23.3235 63593:19.5680` / `785:30.2135 49000:23.4182 63593:19.7609` |
| same, TF-NLL over the 10 tokens `785 ... 151645` | `PPL=1` default / `GQ_EXACT=1` | 0.018421 / 0.016134 nats/token |
| tiny qwen3 Q8_0 (seed 1) | `SNAP=tiny_qwen3.gguf COLI_LOOKUP=0 ./qwen36 0 4 ref_tiny.json`, prompt `1..10`, 8 new | `99 46 99 46 254 99 9 94` (F32 file: `99 46 99 46 254 99 155 94`) |
| tiny qwen3 tied Q8_0 | same | `43 144 6 6 30 43 144 54` |
| tiny qwen3moe Q8_0 and F32 | same | `175 188 93 246 36 13 246 36` |
| tiny olmoe Q8_0 and F32 | `gguf_import.py` then `SNAP=<dir> ./olmoe 8 8 ref_tiny.json` | `16 16 16 16 78 140 183 16`; F32 TF-NLL 7.2177 |

The tiny-model ids are only reproducible with the same generator script and seed. They show the engines accept the synthetic files; they are not a statement about model quality.

### Suggested comparison protocol for the port

1. Tokenizer: `TOK_ONLY=1` on a corpus of prompt files (ASCII, code, multi-space, newlines, digits, CJK, emoji, accented text) -> exact id equality.
2. Prefill logits: `n_new = 1`, `GQ_EXACT=1`, `DUMP` -> compare max abs diff and argmax. Use F32 tiny files for the tightest tolerance (both are plain float math), Q8_0 tiny files and Qwen3-0.6B for the quantized path.
3. Per-layer residuals: `DUMP_LAYERS` on the prefill, cosine per row, to localise a mismatch.
4. Greedy ids: ref.json mode, 16-64 tokens, `COLI_LOOKUP=0`. Near-ties can flip a token between float orders; check the top-2 gap before calling it a bug.
5. Long-context drift: `PPL=1` on a few hundred tokens, compare TF-NLL to 1e-3.

---

## 11. Upstream behaviours worth knowing

- qwen3moe top-k is always renormalised; OLMoE's is not (unless the file says `expert_weights_norm`). Getting this wrong still produces fluent text, so only the oracle catches it.
- OLMoE's q/k norm spans the whole projected vector (`[D]`), Qwen3's spans one head (`[hd]`, shared across heads). Both apply before RoPE.
- Qwen3-0.6B's attention width is `H*hd` = 2048 against `D` = 1024: `attn_q` and `attn_output` are not square. Do not assume `hd = D/H`.
- Default upstream numbers are int8-activation numbers; `GQ_EXACT=1` is the float reference (0.55 max logit difference on Qwen3-0.6B).
- Upstream float32 KV only in oracle mode; chat/serve round K and V to f16, including the current step's.
- The oracle modes never stop at EOS and are greedy only.
- The olmoe engine caps context at 4096 (fixed `sc[4096]` score buffer) and requires `KV == H`; qwen36 caps at 8192 by default (`Q36_MAXT`, hard limit 262144).
- Neither engine reads YaRN rope scaling keys; long-context Qwen3 GGUFs that carry them would run unscaled.
- Upstream does not execute the GGUF Jinja template; renderers are hand-written per family.

## 12. What a family plugin has to supply

From the differences above, the per-family surface for the Swift registry is small:

| Hook | qwen3 | qwen3moe | olmoe |
|---|---|---|---|
| hyperparameters from GGUF keys (prefix, defaults) | `qwen3.` | `qwen3moe.` | `olmoe.` |
| q/k norm scope | per head `[hd]` | per head `[hd]` | whole vector `[H*hd]`, `[KV*hd]` |
| RoPE style | NEOX, full head | NEOX, full head | NEOX, full head |
| FFN | dense SwiGLU | MoE, softmax, top-K, renorm | MoE, softmax, top-K, no renorm (by default) |
| tied head allowed | yes | yes | upstream exits without `output.weight`; the port may allow it |
| pre-tokenizer (`tokenizer.ggml.pre`) | `qwen2` | `qwen2` | `olmo` (GPT-2) |
| chat template | ChatML + think switch | ChatML (+ Coder variant) | Tulu-style `<|user|>` with `|||IP_ADDRESS|||` |
| stops | `<|im_end|>`, `<|endoftext|>` | same | `|||IP_ADDRESS|||` |

Everything else (embedding, RMSNorm, attention core, KV cache, head, sampler, expert streaming) is shared.
