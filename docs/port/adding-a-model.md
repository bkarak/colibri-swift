# Adding a model family

A model family is the code that runs one or more GGUF `general.architecture` values. The
three in this port (`qwen3`, `qwen3moe`, `olmoe`) are each one file in
`Sources/ColibriCore/Models/` of 50 to 120 lines, because they share one decoder
(`DecoderStack` in `Transformer.swift`) and differ only in configuration, routing, chat
template and stop tokens. This note says how to add a fourth, how to tell whether a new
model fits the shared decoder, and how to check the result.

## 1. Does the model fit the shared decoder?

`DecoderStack` is a pre-norm decoder:

```
x = embed(tokens)
for each layer:
    h = rmsNorm(x, attn_norm)
    q, k, v = h @ attn_q, attn_k, attn_v               (no biases)
    q, k = rmsNorm per head or over the whole vector     (optional, QKNormScope)
    q, k = NEOX RoPE                                     (half-split pairs)
    x += attention(q, k, v) @ attn_output               (causal, GQA)
    h = rmsNorm(x, ffn_norm)
    x += dense SwiGLU(h)  or  MoE(h): softmax-routed top-k SwiGLU experts, summed in rank order
logits = rmsNorm(x, output_norm) @ output (or token_embd when tied)
```

with the GGUF tensor names llama.cpp writes (`blk.N.attn_q.weight`,
`blk.N.ffn_gate_exps.weight`, ...). If the new model is this, the family is configuration
only (section 2). If it differs, the differences decide the work (section 4). Some
examples, from the llama.cpp converters' conventions:

| `general.architecture` | fits as is? | what is missing |
|---|---|---|
| `qwen3`, `qwen3moe` (any size) | yes, already supported | |
| `olmoe` | yes, already supported | |
| `qwen2`, `qwen2moe` | no | q/k/v biases; `qwen2moe` also a shared expert with a sigmoid gate |
| `llama`, `mistral`, `granite` | no | interleaved (non-NEOX) RoPE; Llama 3 RoPE frequency scaling; SentencePiece tokenizer for Llama 2 and Mistral |
| `olmo2` | no | norms after attention and FFN instead of before |
| `gemma2`, `gemma3` | no | `(1 + w)` RMSNorm, GELU, embedding scaling, logit soft-capping, sliding window |
| `deepseek2`, `glm4moe` and upstream colibri's other engines | no | MLA attention or other routing; see `status.md` |

The tokenizer must be byte-level BPE (`tokenizer.ggml.model = gpt2`) with a pre-tokenizer
this port has (`tokenizer.ggml.pre` = `qwen2`, or `olmo`/`gpt-2`/`default`). Other files
are refused at load with the key's value.

## 2. A family that fits: one file and one line

Copy `Qwen3MoE.swift` (MoE) or `Qwen3.swift` (dense) to `Models/<Family>.swift` and change:

1. **`architectures`**: the `general.architecture` strings this type runs.
2. **`engineName`**: written into the routing history (`.coli_usage`). Use upstream
   colibri's engine name if it has one, so histories stay interchangeable.
3. **`config(from:)`**: fill a `ModelConfig` from the metadata. `HyperparameterReader`
   prefixes keys with the architecture and refuses a missing required key:

   ```swift
   let r = HyperparameterReader(file.metadata, prefix: arch)
   let layers = try r.required("block_count")
   let kvHeads = r.intOrMaxOfArray("attention.head_count_kv") ?? heads
   let theta = r.float("rope.freq_base", default: 10000)
   ```

   Set `qkNorm` (`.none`, `.perHead` or `.wholeVector`), `normTopK` (renormalise the
   top-k router weights) and `tiedEmbeddings` (`file.tensor("output.weight") == nil`).
   Refuse what the stack cannot honour rather than running it wrong; the Qwen helpers
   `QwenHyperparameters.refuseUnsupportedRouting` and `warnRopeScaling` are examples.
4. **`init`**: build the stack and pick the routing.

   ```swift
   stack = try DecoderStack(file: file, config: config, context: context,
                            engineName: Self.engineName,
                            routing: config.isMoE ? SoftmaxTopK(renormalize: config.normTopK) : nil)
   ```
5. **`chatTemplate(metadata:thinking:)`**: an existing `ChatTemplate` (`ChatMLTemplate`,
   `OLMoETemplate`, `RawTemplate`) or a new struct in the family's file. Templates are
   hand-written renderers of the model's Jinja template, as upstream does; the port does
   not execute Jinja.
6. **`stopTokens(tokenizer:metadata:)`**: usually `tokenizer.specialTokens.stop` (eos,
   eot, eom, eog from the file) plus the template's end-of-turn token.
7. **`expertStore`**: a MoE family returns `stack.store` (the CLI prints the cache counters
   and saves the routing history through it). Dense families leave it out.

Then add the type to `ModelRegistry.families` in `Models/Registry.swift`:

```swift
public static let families: [ModelArchitecture.Type] = [
    Qwen3.self,
    Qwen3MoE.self,
    OLMoE.self,
    MyFamily.self,
]
```

Nothing else changes. Both backends, expert streaming, the generator, the sampler and the
CLI work on any family. `coli-swift info model.gguf` shows which family a file maps to and
the config it reads, without loading the weights, which is the first thing to try.

## 3. A family that does not fit

The shared pieces are separate, so a family can be partly custom:

- **Routing**: `RoutingPolicy` is a value the family passes in. Sigmoid scoring, a bias
  added for selection only, or grouped top-k is a new type in the family's file
  implementing `route(_:logits:rows:experts:k:ids:weights:)` with backend ops.
- **Layer shape**: the blocks `Transformer.attentionBlock`, `denseFFNBlock`,
  `moeFFNBlock` and `outputHead` are public. A family with, say, post-norms or a shared
  expert writes its own `forward` that calls them in a different order, with its own
  weight structs loaded through `GGUFFile.checkedWeight` and `checkedNorm`. Keep the
  `DecoderStack.forward` structure: one `beginBatch` per call, the MoE block's leases
  (released through `Transformer.releaseAll`, which throws if an expert read failed),
  one `synchronize` at the end. Before encoding anything, call
  `backend.checkSupport(_:)` with a `ModelRequirements` listing the matrices, norms, head
  dims and router width, as `DecoderStack.init` does: it is how the Metal backend refuses
  a shape its kernels do not handle (head dim above 256, say) at load, so `--backend auto`
  falls back to the CPU. Release builds use -Ounchecked, so the per-op preconditions are
  not a safety net.
- **A new operation** (biases, interleaved RoPE, GELU, soft-capping, sliding-window
  attention): add it to the `Backend` protocol in `Backend/Backend.swift`, implement it in
  `CPUBackend` and `MetalBackend` (plus a kernel in `MetalShaders.swift`), put any limit of
  the kernel into `MetalBackend.checkSupport` (and the fields it needs into
  `ModelRequirements`), and add a
  Metal-against-CPU test in `Tests/ColibriMetalTests/MetalBackendTests.swift`. This is the
  one change that touches shared files; keep the op general (the CPU version is the
  reference, the Metal one must match it to float rounding).
- **A pre-tokenizer**: a type conforming to `PreTokenizer` in `Tokenizer/PreTokenizer.swift`
  and a line in `PreTokenizerRegistry` (`all` and `make`). A SentencePiece vocabulary
  (`tokenizer.ggml.model = llama`) needs a second `Tokenizer` implementation.
- **A weight type**: a decoder in `Quant/Dequant.swift` (and `Dot.swift`), a kernel decoder
  in `MetalShaders.swift`, and the type added to both backends' `weightTypes`. Golden data
  for the decoder comes from upstream's `ggml_quants.h` (`Tests/ColibriCoreTests/Fixtures/quant`).

## 4. Checking a new family

In order of cost:

1. `coli-swift info model.gguf`: the config is what the model card says.
2. A generation that reads well: `coli-swift generate --model model.gguf --prompt "..."`.
   Fluent output is necessary, not sufficient: wrong routing renormalisation, for one,
   still reads fine.
3. CPU and Metal agree: run `generate --backend cpu --dump` and `--backend metal --dump`
   with `--temp 0 --ignore-eos --max-tokens 128` and compare the `generated_ids` lines.
   They should be identical, or diverge only where the top two logits (printed per step by
   `--dump`) are within about 1e-3.
4. Placement does not change the output (MoE): compare `--experts resident` with
   `--expert-cache-gb` set below the expert size (`info` prints it); the ids must be
   identical.
5. A reference. If upstream colibri runs the architecture, `scripts/oracle/record.py`
   shows how to get its greedy ids and logits in float mode (`GQ_EXACT=1`), and
   `compare.py` how to compare. Otherwise llama.cpp or the Hugging Face model in float32
   can serve, run in a throwaway environment.
6. Without a real file, extend `Sources/ColibriCore/Synthetic/TinyModel.swift` (shapes and
   tensor names for the new architecture) to write a small random GGUF, and test the
   family on it in `Tests/ColibriCoreTests/ModelTests.swift`, which already has a host-side
   reference forward pass for the shared decoder.
