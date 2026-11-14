# GGUF and the ggml block formats

What the Swift port has to read, and how to turn every byte of it into numbers. Written from
upstream colibri at commit 0a5504a: `c/gguf.h` (the reader), `c/ggml_quants.h` (the block
decoders and the integer dot product), `c/ggml_grids.h` (the codebooks), `c/st_gguf.h` and
`c/ggml_view.h` (how engines see GGUF tensors), `c/qwen36.c` and `c/tools/gguf_import.py` (the
metadata keys the qwen3 / qwen3moe / olmoe engines read). Line numbers below refer to that
commit. The facts marked *checked* were confirmed against the one real model on disk,
`~/colibri-models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`.

Everything in a GGUF is little-endian. All multi-byte reads in this document are
little-endian, unaligned (use `loadUnaligned(fromByteOffset:as:)` or assemble bytes; GGUF does
not align anything inside a block).

---

## 1. File layout

```
offset 0   magic        4 bytes  "GGUF"  (0x47 0x47 0x55 0x46)
       4   version      u32      2 or 3 (upstream refuses anything else; v1 used u32 counts)
       8   n_tensors    u64
      16   n_kv         u64
      24   n_kv  metadata entries        (section 1.1)
      ..   n_tensors tensor-info entries (section 1.3)
      ..   padding to `alignment`
  data_start = align_up(end_of_tensor_infos, alignment)
      ..   tensor data; each tensor at data_start + its relative offset
```

v2 and v3 have the same little-endian layout; v3 only exists to mark big-endian files
possible. Upstream treats them identically.

### 1.1 Metadata entry

```
key        string           (u64 byte length, then that many UTF-8 bytes, no NUL)
value_type u32              (table below)
value      depends on type
```

| id | type    | payload |
|----|---------|---------|
| 0  | U8      | 1 byte |
| 1  | I8      | 1 byte |
| 2  | U16     | 2 bytes |
| 3  | I16     | 2 bytes |
| 4  | U32     | 4 bytes |
| 5  | I32     | 4 bytes |
| 6  | F32     | 4 bytes |
| 7  | BOOL    | 1 byte (0 = false, anything else true) |
| 8  | STRING  | u64 length + bytes |
| 9  | ARRAY   | u32 element type, u64 count, then `count` elements back to back |
| 10 | U64     | 8 bytes |
| 11 | I64     | 8 bytes |
| 12 | F64     | 8 bytes |

Array elements are scalars of the element type packed tightly, or strings (each with its own
u64 length). The GGUF spec allows arrays of arrays; upstream refuses them (`gguf_scalar_size`
returns 0 for type 9 and the parse fails). No model we target uses nested arrays, so refusing is
fine.

Getter semantics upstream (`gguf_get_int` / `gguf_get_float`, gguf.h:156-171), worth copying:
an integer getter accepts any of U8/U16/U32/U64/BOOL/I8/I16/I32/I64 and widens to Int64; a float
getter accepts F32/F64 and falls back to the integer getter for integer types; a missing key or a
key of the wrong kind returns the caller's default. Per-layer integer arrays (Ollama's
`head_count_kv`) may arrive in any integer width: read them with a width switch
(`q36_kv_arr_int`, qwen36.c:3532).

### 1.2 Alignment

`general.alignment` (U32, must be a power of two) sets the alignment; default **32**. Upstream
only honours it when its type is U32, and refuses 0, a non-power-of-two or anything above 1 MiB.
After the last tensor info, the reader rounds its position up to the alignment: that is
`data_start`. Every tensor's relative offset must itself be a multiple of the alignment.

*Checked*: Qwen3-0.6B-Q8_0 has no `general.alignment` key, the tensor infos end at byte
5,951,108, and every offset is a multiple of 32 (tensors are packed back to back, each padded to
32).

### 1.3 Tensor info

```
name     string
n_dims   u32                1..4
ne[d]    u64 x n_dims       ggml order: ne[0] is the contiguous (fastest-varying) length
type     u32                ggml type id (section 3)
offset   u64                relative to data_start, multiple of alignment
```

Missing dims are 1. Upstream checks each `ne[d]` is in 1...2^40 and the product does not overflow.
Byte size (not stored in the file, computed):

```
row_bytes(type, ne0) = (ne0 / blck(type)) * bytes(type)    -- refuse if ne0 % blck != 0
nbytes               = row_bytes(type, ne0) * ne1 * ne2 * ne3
```

and the reader refuses a tensor whose `[data_start + offset, + nbytes)` leaves the file.

**Shape convention.** A 2-D weight with `ne = [in, out]` is `out` rows of `in` elements each,
row after row; row `o` starts at byte `o * row_bytes(type, in)`. This is PyTorch's
`nn.Linear.weight` of shape `[out, in]` in row-major order, so `y[o] = dot(row o, x)`. Block
quantisation always runs along `ne[0]` (along a row), never across rows. A 3-D tensor
`ne = [in, out, E]` is E such matrices back to back.

Duplicate tensor names are refused. Upstream's sanity limits: at most 2^24 tensors, 2^20 keys,
strings at most 1 GiB. Treat the header as untrusted input and keep these checks.

### 1.4 Split models

`NAME-00001-of-0000N.gguf` (five digits each side): each part is a complete GGUF (own header,
own alignment, own data section). Upstream derives the part list from the file name (part 1
must be the one named), reads the metadata from part 1 only, and concatenates the tensor tables;
each tensor remembers its part. The `split.*` keys are ignored. For the port this is only needed
for big MoE files; the reader should be structured so a tensor is (file, absolute offset, nbytes).

### 1.5 Reading tensor bytes

Upstream never maps the file: it `pread`s exactly the byte range it needs (gguf.h:333). For the
Swift port on macOS, `mmap` of the whole file (PROT_READ, MAP_SHARED: a private mapping costs about 1.5 s per 17 GB to make GPU-resident) is the natural choice for
the resident dense weights; a Metal buffer can wrap a page-aligned mapping with
`makeBuffer(bytesNoCopy:)` and address tensors by buffer offset. Tensor offsets are only 32-byte
aligned, not page aligned, so wrap the whole mapping (or page-rounded windows), not individual
tensors. Routed experts are streamed with `pread` into cache slots (section 2.4).

---

## 2. Metadata and tensors per architecture

Keys are `<arch>.<name>` where `<arch>` is the value of `general.architecture`.

### 2.1 Hyperparameter keys

Default values are what upstream uses when the key is absent (qwen36.c:3583-3602,
gguf_import.py:187-235 and 919-945).

| key (`<arch>.` prefix) | type | qwen3 | qwen3moe | olmoe | meaning |
|---|---|---|---|---|---|
| `block_count` | U32 | required | required | required | number of layers |
| `embedding_length` | U32 | required | required | required | hidden size D |
| `attention.head_count` | U32 | required | required | required | query heads H |
| `attention.head_count_kv` | U32 (or per-layer array) | required | required | default = head_count | KV heads |
| `attention.key_length` | U32 | default D/H | default D/H | (D/H) | head_dim. **Qwen3 sets it and it is NOT D/H** (0.6B: D=1024, H=16, key_length=128) |
| `attention.value_length` | U32 | default key_length | default key_length | (D/H) | value head dim |
| `feed_forward_length` | U32 | required | optional (dense FFN, unused) | fallback for expert width | dense MLP width |
| `expert_feed_forward_length` | U32 | n/a | required | preferred, else `feed_forward_length` | per-expert MLP width |
| `expert_count` | U32 | n/a | required | required | routed experts E |
| `expert_used_count` | U32 | n/a | required | required | top-k |
| `expert_weights_norm` | BOOL | n/a | not written; treat as true | default false | renormalise top-k weights |
| `rope.freq_base` | F32 | default 1e6 (importer) / 1e4 (engine) | same | default 1e4 | RoPE theta |
| `rope.dimension_count` | U32 | default head_dim | default head_dim | default head_dim | rotary dims (full rotation for all three) |
| `attention.layer_norm_rms_epsilon` | F32 | default 1e-6 | default 1e-6 | default 1e-5 | RMSNorm eps |
| `context_length` | U32 | informational | informational | default 4096 | trained context |

Read the defaults loosely: in practice every converter writes these keys; the port should
refuse a file that lacks `block_count`, `embedding_length`, `attention.head_count`, and for MoE
`expert_count` / `expert_used_count` / the expert width, rather than guess.

Router semantics (not in the file; upstream hard-codes them): qwen3moe is softmax over all E
logits, top-k, then renormalise the k weights to sum 1 (`norm_topk_prob = true`). OLMoE is
softmax, top-k, **no** renormalisation (`norm_topk_prob = false`; llama.cpp's olmoe files do not
write `expert_weights_norm`, and upstream's importer writes false, gguf_import.py:936). Both use
SiLU-gated experts: `down(silu(gate x) * up x)`.

RoPE (also not in the file): llama.cpp runs qwen3, qwen3moe and olmoe with the NEOX rotation
(pairs `(i, i + rot/2)`, HF's `rotate_half`), and its converters store q/k exactly as the HF
checkpoint has them (st_gguf.h:310 says so for olmoe: "no rotary permutation"). Do not apply
the LLaMA-style q/k un-permutation.

*Checked* Qwen3-0.6B-Q8_0 metadata: `qwen3.block_count` 28, `context_length` 40960,
`embedding_length` 1024, `feed_forward_length` 3072, `attention.head_count` 16,
`attention.head_count_kv` 8, `rope.freq_base` 1000000.0 (F32), `attention.layer_norm_rms_epsilon`
1e-6 (F32), `attention.key_length` 128, `attention.value_length` 128, all integers U32.
`general.file_type` 7 (= mostly Q8_0), `general.quantization_version` 2.

Known shapes of the MoE targets (from their HF configs, not checked on a file; verify when one
is on disk): Qwen3-30B-A3B / Qwen3-Coder-30B-A3B: 48 layers, D 2048, H 32, KV 4, head_dim 128,
E 128, top-8, expert width 768, vocab 151936, untied output. OLMoE-1B-7B: 16 layers, D 2048,
H 16, KV 16, head_dim 128, E 64, top-8, expert width 1024, vocab 50304, eps 1e-5, theta 1e4.

### 2.2 Tokenizer keys

| key | type | notes |
|---|---|---|
| `tokenizer.ggml.model` | STRING | `"gpt2"` = byte-level BPE. Upstream refuses anything else |
| `tokenizer.ggml.pre` | STRING | pre-tokenizer regex family. Qwen3: `"qwen2"` (*checked*). OLMoE: expected `"olmo"` (GPT-2 regex; unverified, no file on disk) |
| `tokenizer.ggml.tokens` | ARRAY of STRING | id -> piece, in GPT-2 byte-to-unicode spelling (`Ġ` = space). 151,936 for Qwen3 (*checked*) |
| `tokenizer.ggml.token_type` | ARRAY of I32 | 1 normal, 2 unknown, 3 control (special, split out, not decoded as text), 4 user-defined (split out, decoded as text, e.g. `<think>`), 5 unused, 6 byte |
| `tokenizer.ggml.merges` | ARRAY of STRING | `"a b"` (one ASCII space between the two pieces); rank = index. 151,387 for Qwen3 (*checked*) |
| `tokenizer.ggml.bos_token_id` | U32 | Qwen3: 151643 |
| `tokenizer.ggml.eos_token_id` | U32 | Qwen3: 151645 (`<|im_end|>`) |
| `tokenizer.ggml.padding_token_id` | U32 | Qwen3: 151643 |
| `tokenizer.ggml.eot_token_id`, `eom_token_id` | U32 | optional extra stop ids |
| `tokenizer.ggml.eog_token_ids` | ARRAY of int | optional, every end-of-generation id |
| `tokenizer.ggml.add_bos_token` | BOOL | Qwen3: false |
| `tokenizer.chat_template` | STRING | Jinja template |

Stop set upstream builds (qwen36.c:3545, gguf_import.py:1263-1277): eos, eot, eom, every
eog id, plus `<|endoftext|>` and `<|im_end|>` when they are in the vocabulary.

Pre-tokenizer regexes (llama.cpp's, by `tokenizer.ggml.pre`):

- `qwen2`: `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  (note `\p{N}` alone: digits are split one at a time; upstream tok.h calls this `ndig1`).
- `olmo` / GPT-2: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.

Caution: upstream's importer has no entry for `qwen2` in `PRE_SPLITS` and so writes a bare
ByteLevel (GPT-2 regex) pre-tokenizer into the imported `tokenizer.json`; when qwen36 reads the
GGUF directly (`load_tokenizer_gguf`, qwen36.c:714) it uses its own Qwen splitter. Port the
latter behaviour, not the importer's.

### 2.3 Tensor names and shapes

Global tensors (all three architectures):

| name | ne | type in practice |
|---|---|---|
| `token_embd.weight` | `[D, vocab]` | quantised (Q8_0 in the 0.6B file). Row `t` is token t's embedding: dequantise one row on lookup |
| `output_norm.weight` | `[D]` | F32 |
| `output.weight` | `[D, vocab]` | optional. **Absent means tied**: the LM head reads `token_embd.weight` (qwen36.c:2897). Absent in Qwen3-0.6B (*checked*) |

Per layer `blk.<L>.` (L = 0 ... block_count-1). `Hd` = head_dim, `KV` = head_count_kv, `F` = dense
FFN width, `Fe` = expert width, `E` = experts.

| name suffix | ne | qwen3 | qwen3moe | olmoe |
|---|---|---|---|---|
| `attn_norm.weight` | `[D]` | yes | yes | yes |
| `attn_q.weight` | `[D, H*Hd]` | yes | yes | yes |
| `attn_k.weight` | `[D, KV*Hd]` | yes | yes | yes |
| `attn_v.weight` | `[D, KV*Hd]` | yes | yes | yes |
| `attn_output.weight` | `[H*Hd, D]` | yes | yes | yes |
| `attn_q_norm.weight` | qwen*: `[Hd]` per head; olmoe: `[H*Hd]` whole q | yes | yes | yes |
| `attn_k_norm.weight` | qwen*: `[Hd]` per head; olmoe: `[KV*Hd]` whole k | yes | yes | yes |
| `ffn_norm.weight` | `[D]` | yes | yes | yes |
| `ffn_gate.weight` | `[D, F]` | yes | - | - |
| `ffn_up.weight` | `[D, F]` | yes | - | - |
| `ffn_down.weight` | `[F, D]` | yes | - | - |
| `ffn_gate_inp.weight` | `[D, E]` (router, E rows of D) | - | yes | yes |
| `ffn_gate_exps.weight` | `[D, Fe, E]` | - | yes | yes |
| `ffn_up_exps.weight` | `[D, Fe, E]` | - | yes | yes |
| `ffn_down_exps.weight` | `[Fe, D, E]` | - | yes | yes |

`ffn_norm` is the pre-FFN RMSNorm (HF `post_attention_layernorm`); qwen36.c maps it under
`g_gguf_plain` (qwen36.c:2889). Norm weights are stored as is (multiply by `w`, not `1 + w`).
The qk-norm difference matters: Qwen3 RMS-normalises each head's Hd-vector separately with
one shared `[Hd]` weight; OLMoE RMS-normalises the whole projected q (and k) vector across all
heads before splitting into heads (olmoe.c:873 loads both as `D` long).

*Checked* layer 0 of Qwen3-0.6B: `attn_q` `[1024, 2048]` Q8_0, `attn_k`/`attn_v` `[1024, 1024]`,
`attn_output` `[2048, 1024]`, `attn_q_norm`/`attn_k_norm` `[128]` F32, `ffn_gate`/`ffn_up`
`[1024, 3072]`, `ffn_down` `[3072, 1024]`, norms F32; 310 tensors, 197 Q8_0 and 113 F32.

Optional, upstream reads if present: `blk.<L>.ffn_{gate,up,down}_exps.scale`, F32 with E values,
a per-expert multiplier on that matrix (gg_scale, qwen36.c:2907). Not in llama.cpp's own
qwen3moe/olmoe files; support is cheap (multiply the expert's output).

Tensors with `blk.<L>` for L >= block_count - `nextn_predict_layers` are MTP/NextN heads; skip.

### 2.4 Merged expert tensors: addressing one expert

A routed-expert tensor `[in, out, E]` of type t holds expert e as one contiguous slice:

```
expert_bytes = row_bytes(t, in) * out           ( = nbytes / E )
expert e     = bytes [e * expert_bytes, (e + 1) * expert_bytes) of the tensor
row r of e   = e * expert_bytes + r * row_bytes(t, in)
```

So one expert's gate, up and down are three separate reads (three tensors, possibly three
different types: published files mix e.g. Q4_K gate/up with Q6_K down). That is the streaming
unit for colibri's core idea: experts live on disk, a RAM cache of slots holds
`expert_bytes(gate) + expert_bytes(up) + expert_bytes(down)` per slot. Upstream's
`--mode copy` (the `colibri.layout = "expert-packed-v1"` container, st_gguf.h:377) rewrites each
layer's experts as one tensor `blk.<L>.ffn_exps_colibri` of `[stride, E]` bytes, gate | up | down
contiguous in a 4 KiB-aligned slot, with keys `colibri.expert_types` (U32 x 3L),
`colibri.expert_bytes` (U64 x 3L), `colibri.expert_stride` (U64 x L), `colibri.expert_dims`
(U64 x 6L). The port does not need this to start; reading the three slices in place is lossless.

Upstream also disables kernel readahead on the expert files (`POSIX_FADV_RANDOM`; on macOS the
analogue is `fcntl(fd, F_RDAHEAD, 0)`), because readahead past each slice doubled the bytes read.

---

## 3. ggml types

Type ids as stored in tensor infos (ggml_quants.h:86). `blck` = elements per block,
`bytes` = bytes per block.

| id | name | blck | bytes | bits/weight | port scope |
|---|---|---|---|---|---|
| 0 | F32 | 1 | 4 | 32 | now |
| 1 | F16 | 1 | 2 | 16 | now |
| 2 | Q4_0 | 32 | 18 | 4.5 | now |
| 3 | Q4_1 | 32 | 20 | 5.0 | now |
| 6 | Q5_0 | 32 | 22 | 5.5 | now |
| 7 | Q5_1 | 32 | 24 | 6.0 | now |
| 8 | Q8_0 | 32 | 34 | 8.5 | now |
| 9 | Q8_1 | 32 | 36 | - | activation-only in ggml, never in files |
| 10 | Q2_K | 256 | 84 | 2.625 | now |
| 11 | Q3_K | 256 | 110 | 3.4375 | now |
| 12 | Q4_K | 256 | 144 | 4.5 | now |
| 13 | Q5_K | 256 | 176 | 5.5 | now |
| 14 | Q6_K | 256 | 210 | 6.5625 | now |
| 15 | Q8_K | 256 | 292 | - | activation-only in ggml |
| 16 | IQ2_XXS | 256 | 66 | 2.0625 | later |
| 17 | IQ2_XS | 256 | 74 | 2.3125 | later |
| 18 | IQ3_XXS | 256 | 98 | 3.0625 | later |
| 19 | IQ1_S | 256 | 50 | 1.5625 | later |
| 20 | IQ4_NL | 32 | 18 | 4.5 | now |
| 21 | IQ3_S | 256 | 110 | 3.4375 | later |
| 22 | IQ2_S | 256 | 82 | 2.5625 | later |
| 23 | IQ4_XS | 256 | 136 | 4.25 | now |
| 24-27 | I8, I16, I32, I64 | 1 | 1/2/4/8 | - | not weights |
| 28 | F64 | 1 | 8 | - | not used |
| 29 | IQ1_M | 256 | 56 | 1.75 | later |
| 30 | BF16 | 1 | 2 | 16 | now |
| 34 | TQ1_0 | 256 | 54 | 1.6875 | later |
| 35 | TQ2_0 | 256 | 66 | 2.0625 | later |
| 39 | MXFP4 | 32 | 17 | 4.25 | later |
| 40 | NVFP4 | 64 | 36 | 4.5 | later |
| 41 | Q1_0 | 128 | 18 | - | later (colibri/newer ggml) |
| 42 | Q2_0 | 64 | 18 | - | later (colibri/newer ggml) |

Ids 4 and 5 (old Q4_2/Q4_3) and 31-33, 36-38 are retired or unused; refuse them. Types 65 and 66
(UD-TQ1_0, UD-Q1_0) are unknown to upstream too.

### 3.1 Scalar conversions

- **f16 -> f32**: IEEE binary16. Exact. Swift's `Float(Float16(bitPattern: h))` is exact on Apple
  Silicon. Subnormals, +-0, Inf and NaN all convert exactly.
- **bf16 -> f32**: `Float(bitPattern: UInt32(h) << 16)`. Exact.
- In the formulas below, `f16(p)` means "read u16 at byte p and convert", and every `(float)`
  is an exact int -> Float conversion.

### 3.2 Exactness contract

Upstream's decoders reproduce gguf-py's `dequantize` bit for bit: the arithmetic is in Float32,
in the order written below, **with no fused multiply-add** (the file is compiled with
`fp contract(off)`). Two consequences for the port:

- CPU Swift: write the formulas literally in `Float`. Swift does not contract `a * b - c` into an
  FMA on its own (it only fuses through `.addingProduct` / `fma`), so a literal port is bit-exact.
  Unit-test against values produced by the upstream C decoder.
- Metal: the shader compiler contracts and reorders under the default fast-math mode, so GPU
  dequantisation can differ from the CPU in the last bit. Compare GPU against CPU with a
  tolerance (or compile with `mathMode = .safe` / `fastMathEnabled = false` when checking).

Notation used in every block: `b` is the block's byte array (offsets from the block start),
`y[0..blck)` the output, `>>` and `&` on unsigned bytes, integer arithmetic exact, `*`/`+`/`-`
between Floats rounded to Float32 at each step, evaluated left to right as parenthesised.

### 3.3 Float types

- **F32**: 4 bytes per element, IEEE binary32.
- **F16**: 2 bytes per element, `y[i] = f16(2*i)`.
- **BF16**: 2 bytes per element, `y[i] = bf16(2*i)`.

### 3.4 Legacy 32-element blocks

**Q8_0** (34 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 32 | `qs` int8 x 32 |

`y[j] = Float(Int8(qs[j])) * d` for j in 0..<32.

**Q4_0** (18 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 16 | `qs` u8 x 16 (two 4-bit codes each) |

For j in 0..<16:
`y[j] = d * Float(Int(qs[j] & 0x0F) - 8)`,
`y[j + 16] = d * Float(Int(qs[j] >> 4) - 8)`.
Low nibbles are elements 0..15, high nibbles are elements 16..31 (not interleaved pairs).

**Q4_1** (20 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 2 | `m` f16 |
| 4 | 16 | `qs` |

`y[j] = d * Float(qs[j] & 0x0F) + m`, `y[j + 16] = d * Float(qs[j] >> 4) + m`.

**Q5_0** (22 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 4 | `qh` u32 (bit k = 5th bit of element k) |
| 6 | 16 | `qs` |

For j in 0..<16:
`lo = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4)`,
`hi = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)`,
`y[j] = d * Float(lo - 16)`, `y[j + 16] = d * Float(hi - 16)`.

**Q5_1** (24 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 2 | `m` f16 |
| 4 | 4 | `qh` u32 |
| 8 | 16 | `qs` |

`lo`, `hi` as Q5_0; `y[j] = d * Float(lo) + m`, `y[j + 16] = d * Float(hi) + m`.

**IQ4_NL** (18 bytes): Q4_0's layout with a non-linear codebook.

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 16 | `qs` |

`KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
(`GGML_KVALUES_IQ4NL`, ggml_grids.h:2301).
`y[j] = d * Float(KV[qs[j] & 0x0F])`, `y[j + 16] = d * Float(KV[qs[j] >> 4])`.

### 3.5 K-quants (256-element super-blocks)

Shared helper for Q4_K and Q5_K, the 6-bit scale and min of sub-block g (0..<8) from the 12
packed bytes `s` (`gq_scale_min_k4`, ggml_quants.h:271):

```
if g < 4:  sc = s[g] & 0x3F;                          mn = s[g + 4] & 0x3F
else:      sc = (s[g + 4] & 0x0F) | ((s[g - 4] >> 2) & 0x30)
           mn = (s[g + 4] >> 4)   | ((s[g]     >> 2) & 0x30)
```

**Q2_K** (84 bytes)

| offset | size | field |
|---|---|---|
| 0 | 16 | `scales` u8 x 16: low nibble = scale, high nibble = min, one per 16 elements |
| 16 | 64 | `qs` 2-bit codes |
| 80 | 2 | `d` f16 |
| 82 | 2 | `dmin` f16 |

For i in 0..<256: `c = i >> 7` (128-element half), `s = (i >> 5) & 3` (bit-pair),
`j = i & 31`, `g = i >> 4`;
`q = (qs[c*32 + j] >> (2*s)) & 3`;
`dl = d * Float(scales[g] & 0x0F)`; `ml = dmin * Float(scales[g] >> 4)`;
`y[i] = dl * Float(q) - ml`.

**Q3_K** (110 bytes)

| offset | size | field |
|---|---|---|
| 0 | 32 | `hmask` (high bit of each element) |
| 32 | 64 | `qs` low 2 bits |
| 96 | 12 | `scales` 16 x 6-bit, packed |
| 108 | 2 | `d` f16 |

Scales, for k in 0..<16:
`l = k < 8 ? (scales[k] & 0x0F) : (scales[k - 8] >> 4)`;
`h = (scales[8 + (k & 3)] >> (2 * (k >> 2))) & 3`;
`dl[k] = d * Float((l | (h << 4)) - 32)`.

Elements, for i in 0..<256: `c = i >> 7`, `sh = (i >> 5) & 3`, `j = i & 31`;
`ql = (qs[c*32 + j] >> (2*sh)) & 3`;
`hb = (hmask[i & 31] >> (i >> 5)) & 1`;
`q = ql - (hb == 0 ? 4 : 0)`  (range -4..3);
`y[i] = dl[i >> 4] * Float(q)`.

**Q4_K** (144 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 2 | `dmin` f16 |
| 4 | 12 | `scales` (8 scales + 8 mins, 6 bits, see helper) |
| 16 | 128 | `qs` |

For g in 0..<8 (32-element sub-blocks): `(sc, mn) = scale_min_k4(scales, g)`;
`d1 = d * Float(sc)`; `m1 = dmin * Float(mn)`;
`q = qs + (g >> 1) * 32`; `shift = (g & 1) * 4`;
for j in 0..<32: `y[g*32 + j] = d1 * Float((q[j] >> shift) & 0x0F) - m1`.
(Sub-blocks 2k and 2k+1 share the same 32 bytes: low nibbles, then high nibbles.)

**Q5_K** (176 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 2 | `dmin` f16 |
| 4 | 12 | `scales` |
| 16 | 32 | `qh` (bit g of qh[j] = 5th bit of element g*32+j) |
| 48 | 128 | `qs` |

As Q4_K with `v = ((q[j] >> shift) & 0x0F) | (((qh[j] >> g) & 1) << 4)`,
`y[g*32 + j] = d1 * Float(v) - m1`.

**Q6_K** (210 bytes)

| offset | size | field |
|---|---|---|
| 0 | 128 | `ql` low 4 bits |
| 128 | 64 | `qh` high 2 bits |
| 192 | 16 | `scales` int8 x 16 (signed), one per 16 elements |
| 208 | 2 | `d` f16 |

For i in 0..<256: `c = i >> 7`, `r = i & 127`;
`lo = (ql[c*64 + (r & 63)] >> (4 * (r >> 6))) & 0x0F`;
`hi = (qh[c*32 + (r & 31)] >> (2 * (r >> 5))) & 3`;
`q = (lo | (hi << 4)) - 32`  (range -32..31);
`y[i] = (d * Float(Int8(scales[i >> 4]))) * Float(q)`.

**IQ4_XS** (136 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d` f16 |
| 2 | 2 | `scales_h` u16 (2 high bits per sub-block) |
| 4 | 4 | `scales_l` u8 x 4 (4 low bits per sub-block) |
| 8 | 128 | `qs` 16 bytes per 32-element sub-block |

For ib in 0..<8:
`ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((scales_h >> (2*ib)) & 3) << 4)`;
`dl = d * Float(ls - 32)`; `q = qs + 16*ib`;
for j in 0..<16: `y[ib*32 + j] = dl * Float(KV[q[j] & 0x0F])`,
`y[ib*32 + j + 16] = dl * Float(KV[q[j] >> 4])` (KV = the IQ4_NL table).

---

## 4. How upstream computes matrix-vector products

There are two paths (`gq_matmul`, ggml_quants.h:3065; `coli_ggml_matmul`, ggml_view.h:66).

### 4.1 Float reference path

Used for F32 / F16 / BF16 matrices, any width not a multiple of 32, and every type when
`GQ_EXACT=1` is set. Each weight row is dequantised to Float32 (section 3) once and dotted in
f32 with every activation row (`gq_matmul_f32`). Bit-exact dequant; the dot's summation order
depends on the SIMD path.

### 4.2 Integer path (the default for every block type)

**Activation quantisation, once per call, per 32 elements** (`gq_quant_act_ref`,
ggml_quants.h:1184). For each block b of 32 activations `v`:

```
amax  = max |v[i]|
dx[b] = amax / 127                        (Float32 division)
id    = amax > 0 ? 127 / amax : 0
xq[i] = clamp(round_half_even(v[i] * id), -127, 127)     (lrintf; -128 never used)
xs[2b]   = sum of xq[0..16)               (Int32)
xs[2b+1] = sum of xq[16..32)
```

So activations become a Q8_0-like form with an f32 scale (not ggml's Q8_K/Q8_1), plus per-16
code sums for the types with a min term. This is the only approximation of the integer path.

**Weights in integer form** (`gq_ublk`, ggml_quants.h:817 and 994-1164). Every block type is
unpacked to `w8[i]` (Int8), and per group of 16 elements a scale `sw[g]` and a min `mw[g]` (Float),
such that `sw[g] * Float(w8[i]) - mw[g]` reproduces the dequantised value bit for bit:

| type | w8 | sw (per 16) | mw (per 16) |
|---|---|---|---|
| Q8_0 | qs | d | 0 |
| Q4_0 | nibble - 8 | d | 0 |
| Q4_1 | nibble | d | **-m** |
| Q5_0 | 5-bit - 16 | d | 0 |
| Q5_1 | 5-bit | d | **-m** |
| IQ4_NL | KV[nibble] | d | 0 |
| Q2_K | 2-bit code | `d * (sc & 15)` | `dmin * (sc >> 4)` |
| Q3_K | ql - 4*(1-hb) | `d * (scale6 - 32)` | 0 |
| Q4_K | nibble | `d * sc` (two groups share a 32-sub-block value) | `dmin * mn` |
| Q5_K | 5-bit | `d * sc` | `dmin * mn` |
| Q6_K | 6-bit - 32 | `d * scales[g]` | 0 |
| IQ4_XS | KV[nibble] | `d * (ls - 32)` | 0 |

**Dot product** of one weight row with one activation row of n elements
(`gq_dot_i8`, ggml_quants.h:1239):

```
for each 32-block b:
    s0 = sum_{i in 0..<16}  w8[32b+i] * xq[32b+i]      (exact Int32)
    s1 = sum_{i in 16..<32} w8[32b+i] * xq[32b+i]
    acc += dx[b] * (sw[2b] * Float(s0) + sw[2b+1] * Float(s1))
if type has a min (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K):
    m = sum_b dx[b] * (mw[2b] * Float(xs[2b]) + mw[2b+1] * Float(xs[2b+1]))
    acc -= m
y = acc
```

The Int32 inner sums are exact (|w8| <= 127, |xq| <= 127, 16 terms). The float accumulation
order differs between upstream's AVX2, NEON and scalar builds, so different backends agree to
rounding, not bit for bit; upstream verifies the integer path by perplexity/NLL, not by bits.

Q8_0 skips the unpack (`gq_dot_q8_0`): per block, `Float(sum qs*xq) * (d * dx[b])`.

**What upstream runs on Apple Silicon.** The fast row-interleaved (`*_R16`, ids 64+ internal
only) and eight-row (`gq_rows_f8`) kernels are x86-only (AVX2/AVX-512). On an ARM build,
`gq_row_all` (ggml_quants.h:1819) does: Q8_0 -> `gq_dot_q8_0` with NEON `sdot`; every other type
-> unpack the row with `gq_ublk`, then `gq_dot_i8` with NEON `sdot`. The M4 has `sdot` and `i8mm`.
A native build of upstream on this Mac is therefore a clean oracle for the scalar-equivalent
integer path.

**Implication for the port.** Two valid designs, both matching upstream's numerics class:

1. Float path: dequantise to f32 (or f16) on the fly in the GEMV and MAC in f32. Simplest,
   bit-exact weights, no activation rounding; on Metal this is the usual approach for decode
   (bandwidth-bound) and is what most Metal GGUF kernels do.
2. Integer path: quantise activations per 32 as above and use int8 dot products (CPU: NEON
   `sdot` via SIMD intrinsics is not exposed in Swift without C; Metal has no int8 dot
   instruction, so this buys little on the GPU).

Recommended: float path on Metal; on the CPU backend, start with dequantise-row + f32 dot
(matches `GQ_EXACT=1` upstream, which is the cleanest oracle), and add the integer path only if
CPU decode speed matters. Either way, test the CPU reference against `GQ_EXACT=1` upstream
first, then the Metal kernels against the CPU with a tolerance.

### 4.3 Things in upstream the port can ignore

- `GQ_*_R16` (ids 64-72): an in-memory layout with rows interleaved by 16 built at load time for
  AVX kernels; not a file format. `GQ_EXACT=1` or `<ENGINE>_ILV16=0` disables it.
- `gq_rows_i8`: re-quantises any type to int8 per 32 for the Vulkan tier.
- `c/quant.h` and `c/idot.h`: colibri's own non-GGUF formats (per-row int8/int4/int3/int2,
  FP8 e4m3, an E8 lattice, MXFP4 as safetensors checkpoints lay it out) and their kernels, used
  when an engine reads an HF checkpoint it quantised itself. Not used for GGUF. Note that
  quant.h's MXFP4 nibble order (values 2i, 2i+1 in byte i) differs from GGUF's (value j in the
  low nibble of byte j, j+16 in the high nibble); `coli_mxfp4_unblock` (ggml_view.h:43)
  converts.

---

## 5. Later: i-quants, ternary and FP4 types

Not in the first port. Everything needed is in upstream: the block decoders in
`c/ggml_quants.h`, the integer unpack in `gq_ublk` (same file, scalar cases at lines 1054-1163), and the
codebooks in `c/ggml_grids.h` (generated from gguf-py, MIT). Layouts, for orientation:

| type | bytes | layout | decoder | tables |
|---|---|---|---|---|
| IQ2_XXS | 66 | d f16 @0; qs 64 @2: per 32-sub-block two u32: a0 = 4 grid indices (8 bits), a1 = 4 x 7-bit sign indices + 4-bit scale (a1 >> 28); `db = d * (0.5 + s) * 0.25` | `gq_blk_iq2_xxs` :317 | `GGML_IQ2XXS_GRID` (256 x 8), `GGML_KSIGNS_IQ2XS` (128) |
| IQ2_XS | 74 | d @0; qs u16 x 32 @2 (9-bit grid index, 7-bit sign index); scales 4-bit x 16 @66; `db = d * (0.5 + s) * 0.25` | `gq_blk_iq2_xs` :333 | `GGML_IQ2XS_GRID` (512 x 8) |
| IQ2_S | 82 | d @0; qs 32 @2; signs 32 @34; qh 8 @66 (2 high index bits each); scales 8 @74 | `gq_blk_iq2_s` :349 | `GGML_IQ2S_GRID` (1024 x 8) |
| IQ3_XXS | 98 | d @0; qs 64 @2 (grid indices, 4 values each); scales+signs u32 x 8 @66; `db = d * (0.5 + s) * 0.5` | `gq_blk_iq3_xxs` :365 | `GGML_IQ3XXS_GRID` (256 x 4) |
| IQ3_S | 110 | d @0; qs 64 @2; qh 8 @66; signs 32 @74; scales 4 @106; `db = d * (1 + 2s)` | `gq_blk_iq3_s` :383 | `GGML_IQ3S_GRID` (512 x 4) |
| IQ1_S | 50 | d @0; qs 32 @2; qh u16 x 8 @34 (3-bit scale, delta sign, 4 x 3 high index bits); `y = d*(2s+1) * (grid + delta)`, delta = +-0.125 | `gq_blk_iq1_s` :401 | `GGML_IQ1S_GRID` (2048 x 8, values -1/0/1) |
| IQ1_M | 56 | qs 32 @0; qh 16 @32; scales u16 x 4 @48 (the f16 d is assembled from their top nibbles) | `gq_blk_iq1_m` :415 | `GGML_IQ1S_GRID` |
| MXFP4 | 17 | e8m0 exponent @0; qs 16 @1 (value j low nibble of byte j, j+16 high nibble); `y = e8m0_half(e) * KV_MXFP4[nib]`, KV_MXFP4 = [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12] (e2m1 doubled), `e8m0_half(x) = x < 2 ? bits(0x00200000 << x) : bits((x-1) << 23)` | `gq_blk_mxfp4` :454 | `GGML_KVALUES_MXFP4` |
| NVFP4 | 36 | 4 x ue4m3 scales @0 (one per 16, halved); qs 32 @4 | `gq_blk_nvfp4` :493 | `GGML_KVALUES_MXFP4` |
| TQ1_0 | 54 | base-3 packed trits (qs 48 @0, qh 4 @48), d f16 @52 | `gq_blk_tq1_0` :462 | - |
| TQ2_0 | 66 | 2-bit codes 64 @0 (q - 1), d f16 @64 | `gq_blk_tq2_0` :477 | - |
| Q1_0 | 18 | d @0; 128 sign bits @2: `+-d` | `gq_blk_q1_0` :504 | - |
| Q2_0 | 18 | d @0; 64 x 2-bit @2: `(q - 1) * d` | `gq_blk_q2_0` :509 | - |

When porting the grid types, generate the Swift tables from `ggml_grids.h` mechanically
(they are plain byte arrays; `GGML_IQ2XXS_GRID[i*8 + j]` is element j of codebook entry i), keep
the sign table `GGML_KSIGNS_IQ2XS` (128 bytes: bit 7 is the parity of the low 7), and check
every decoder against the upstream C one on random blocks, as section 6 does for the first set.

---

## 6. Verification of this document

Done on 2026-11-14 on the M4 Max, with throwaway scripts (not in the repo):

- **Dequant formulas, bit-exact.** A tiny C harness including upstream's `c/ggml_quants.h`
  (`clang -O2 -ffp-contract=off`, calling `gq_dequant_row`) decoded random blocks (seeded, f16
  scale fields kept finite by clearing bit 14) of F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1,
  IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS: 2,048 values for each 32-element type and
  16,384 for each K-type. An independent Python transcription of sections 3.3-3.5, rounding to
  Float32 after every operation, matched every value bit for bit (0 mismatches).
- **Integer path formula.** For the same twelve block types, the section 4.2 recipe (activation
  per 32 to int8, exact integer sums, scales) in Python agreed with upstream's `gq_matmul` (NEON
  `sdot` build) to a worst relative difference of 6.6e-6 on one 2,048-wide row, i.e. float
  summation order only. Against an exact f32 dot of the same dequantised weights, the int8
  activation rounding moved single random-row results by 0.1% to 7% (random data with heavy
  cancellation; upstream validates this path by NLL on real models, not per dot).
- **Header facts** marked *checked* were read from `Qwen3-0.6B-Q8_0.gguf` with a standalone
  header dumper.

The Swift port should keep the same test: random blocks through the upstream C decoder as
golden data, bit-compared with the Swift decoder.
