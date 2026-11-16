# Architecture of the Swift port

How colibri-swift is put together, what each module promises the others, and who owns
which file while the port is built by parallel workstreams. The three design notes next to
this file hold the facts this one relies on: `gguf-and-quants.md` (file format and block
decoders), `models.md` (forward passes, tokenizer, oracle recipe) and `runtime-and-metal.md`
(expert streaming, Metal plan, measurements on this machine).

Status: implemented and verified against upstream; `status.md` has the verification table,
benchmarks and the gaps, `adding-a-model.md` the steps for a new family. This note began as
the skeleton's design (sections 5 and 6 describe how the parallel workstreams split the
work) and is kept current where the code moved: the routing ids are kept per layer
(`TransformerScratch.expertIds` is `[layers][maxBatch][K]`, so one read-back after a token
gives the whole token's routing), `ModelArchitecture.expertStore` exposes a MoE family's
store to the CLI, and Metal decode attention splits long contexts across threadgroups
(`k_attention` partials merged by `k_attention_merge`).

## 1. Package and module map

```
Package.swift                      tools 6.0, macOS 15+, no dependencies
Sources/
  Colibri/                         library facade: ColibriEngine, ChatSession (load a GGUF,
                                   stream text; Swift 6 mode; re-exports ColibriCore)
  CColibriKernels/                 C: NEON int8 rows kernel (rows_i8.c), CPU worker pool (pool.c)
  ColibriCore/                     library: everything except Metal
    Support/      Errors.swift, Concurrency.swift (UnsafeSendable, alignUp, hostPageSize)
    GGUF/         GGUFValue.swift (value model, GGUFMetadata), GGUFFile.swift (reader),
                  MappedFile.swift (mmap, 16 KiB-rounded), GGUFWriter.swift (+ Quantize)
    Quant/        GGMLType.swift (ids, block geometry), Dequant.swift (row decoders, HalfFloat),
                  Dot.swift (quantised row x f32 vector)
    Backend/      Backend.swift (THE compute contract and its value types)
      CPU/        CPUBackend.swift (+ BlockingHostSignal), Parallel.swift
    Tokenizer/    Tokenizer.swift (protocol, BPETokenizer, StreamingDecoder), ByteLevel.swift,
                  PreTokenizer.swift (qwen2, gpt2, registry), ChatTemplate.swift
    Sampling/     Sampler.swift (SamplerConfig, Xorshift64Star, Sampler)
    Experts/      ExpertStore.swift (protocol, lease, stats, options, factory),
                  ExpertLayout.swift, ResidentExpertStore.swift, StreamingExpertStore.swift
                  (+ ExpertArena), IOPool.swift, RoutingHistory.swift, Tier.swift, MemoryBudget.swift
    Models/       ModelArchitecture.swift (family protocol, LoadContext, InferenceState),
                  ModelConfig.swift (+ HyperparameterReader), Registry.swift, Transformer.swift
                  (shared blocks, RoutingPolicy, TransformerScratch), Generator.swift,
                  Qwen3.swift, Qwen3MoE.swift, OLMoE.swift
    Synthetic/    TinyModel.swift (deterministic tiny random GGUFs for tests and the oracle)
  ColibriMetal/                    library: MetalBackend.swift, MetalShaders.swift (MSL string)
  coli-swift/                      executable: main.swift, CLIOptions.swift
Tests/
  ColibriCoreTests/                Swift Testing; Fixtures/ copied into the bundle
  ColibriMetalTests/
scripts/oracle/                    upstream-oracle harness
docs/port/                         design notes (this file and the three above)
```

Dependency direction: `coli-swift -> ColibriMetal -> ColibriCore`. Inside ColibriCore:
`Models -> {Backend, Experts, Tokenizer, Sampling, GGUF}`, `Experts -> {Backend, GGUF}`,
`Backend -> Quant`, `GGUF -> {Backend (MemoryRegion, WeightView), Quant}`. Tokenizer and
Sampling depend only on GGUF metadata. Nothing in ColibriCore imports Metal.

Choices worth knowing:

- **Swift 5 language mode** for all targets. The engine passes raw pointers into
  `concurrentPerform` closures and shares caches with I/O threads under explicit locks;
  strict concurrency would need `@unchecked Sendable` on nearly every hot type and add no
  safety. `UnsafeSendable<T>` exists for the places that cross threads.
- **`-Ounchecked` in release** for ColibriCore and ColibriMetal (an `unsafeFlags` setting,
  allowed because this is a root package). It comes after SwiftPM's `-O` on the command
  line, so it wins (checked with `swift build -c release -v`). Debug builds and the tests keep
  every bounds and overflow check.
- **Metal shaders as a Swift string** compiled at runtime with `makeLibrary(source:)`, as
  upstream does. SwiftPM does not compile `.metal` files without Xcode's build system. The
  skeleton test compiles the library and runs a batch through the backend.
- **Separate ColibriMetal target** (rather than `ColibriCore/Metal/` as
  `runtime-and-metal.md` 6.1 sketched): ColibriCore builds and tests without Metal, and the
  Metal workstream owns a whole target. The expert-streaming files the same note put under
  `Runtime/` live in `Experts/`.
- **Swift Testing** (`import Testing`), which ships with the toolchain.

## 2. The Backend contract

`Sources/ColibriCore/Backend/Backend.swift` is the only way a forward pass computes. A
family file never loops over activations or touches weight bytes; it encodes ops.

### 2.1 Value types

| type | what it is |
|---|---|
| `MemoryRegion` | a read-only, 16 KiB-aligned span of host memory that holds weights: the file mapping, or one layer's expert arena. Backends register regions (Metal wraps each once with `bytesNoCopy`). |
| `WeightView` | a 2-D ggml matrix inside a region: `(region, offset, type, cols, rows, scale)`. `cols` = GGUF `ne[0]` (inputs), `rows` = `ne[1]` (outputs); `y[o] = scale * dot(row o, x)`. |
| `StackedWeights` | `count` equally shaped matrices at a fixed `stride`: a layer's experts. Resident: the fused `ffn_*_exps` tensor, index = expert id. Streaming: the arena, index = slot. |
| `DeviceBuffer` / `BufferSlice` | backend-allocated activation memory (shared memory on both backends) and an element range of it. |
| `KVCacheLayout` / `KVCache` | `[layer][kvHead][position][dim]`, separate key and value buffers, f32 or f16. |
| `RopeParams`, `AttentionParams` | op parameters. Row `s` of a batch is at position `positionStart + s`. |
| `HostSignal` | a counter the host advances and batches can wait on (`MTLSharedEvent` on Metal). The streaming store signals it when a missed expert lands. |
| `BackendCapabilities` | name, decodable weight types, max buffer length, working-set limit, sync or async. |
| `ModelRequirements` | what a loaded model asks of the kernels (matrices with name, type, width and alignment; norms; head dims; router width). `Backend.checkSupport` throws on anything the backend cannot run, at load, before any op is encoded: the default checks weight types (naming the tensors), Metal adds its shape limits. |

### 2.2 Execution model

Ops are recorded in program order into the current batch (`beginBatch` opens one, or the
first op does). `commitBatch()` submits without waiting; `synchronize()` waits for every
submitted batch. The host may read a buffer an op wrote, or write one a submitted op reads,
only after `synchronize()`. Inside a batch, each op sees the writes of the ops before it.
The CPU backend executes eagerly, so code that is correct on Metal is correct on the CPU,
but not the reverse. This is what lets Metal put a whole dense token, or a resident MoE
token, into one command buffer (~116 us per round trip measured here, so per-op command
buffers are out of the question).

### 2.3 Ops

| op | shape contract |
|---|---|
| `embed(table:tokens:into:)` | dequantise row `tokens[s]` into `out[s]` |
| `matmul(_:_:into:batch:)` | `y[s] = W x[s]`, `x [batch][cols]`, `y [batch][rows]`; batch 1 = decode |
| `matmulPair` | two matrices, same input (gate and up); default = two `matmul`s, Metal may fuse |
| `matmulExperts(_:indices:rows:perRow:_:sharedInput:into:)` | `y[s*K+k] = W[indices[s*K+k]] x[s or s*K+k]`; indices read on the device |
| `rmsNorm(_:weight:into:rows:dim:eps:)` | per-row RMSNorm; per-head (rows = S*H) or whole-vector (rows = S) |
| `ropeNeox(_:_:)` | in place, half-split pairs `(j, j + ropeDims/2)` |
| `storeKV(k:v:cache:layer:positionStart:rows:)` | write the batch's K and V |
| `attention(q:cache:layer:_:into:)` | causal GQA, head `h` reads KV head `h / (H/KV)` |
| `softmax(_:rows:cols:)` | in place |
| `topK(_:rows:cols:k:renormalize:ids:weights:)` | rank order, lower index wins ties |
| `siluMul`, `add`, `copy` | elementwise |
| `weightedSum(_:weights:rows:terms:dim:into:accumulate:)` | MoE combine in rank order, optionally added to the residual |

Adding an op: add the requirement to `Backend.swift`, a default in the extension if it can
be composed from existing ops, and a stub in both `CPUBackend` and `MetalBackend` in the
same change (see section 5).

## 3. Model families and the registry

`ModelArchitecture` (Models/ModelArchitecture.swift) is the plugin surface: the
`general.architecture` strings it runs, an engine name for the routing history,
`config(from:)`, `init(file:config:context:)`, `makeState()`, `forward(_:state:)`,
`chatTemplate`, `stopTokens`, `expertStats`. `ModelRegistry.families` lists the types. A new
family is one file in `Models/` and one line in that list; it composes the shared blocks in
`Transformer.swift` (`attentionBlock`, `denseFFNBlock`, `moeFFNBlock`, `outputHead`) and
picks a `RoutingPolicy` (`SoftmaxTopK(renormalize:)` today; sigmoid or grouped routing
would be a new policy type in the family's file). If it needs a new chat template it defines
a `ChatTemplate` type in its own file; a new pre-tokenizer is the one case that touches
Tokenizer/ (one type plus one `PreTokenizerRegistry` line).

## 4. One decoded token through a MoE layer

Qwen3-30B-A3B shapes (D 2048, H 32, KV 4, hd 128, E 128, K 8, I 768), `rows = 1`,
position `t`. Everything below is encoded into one open batch unless stated.

```
x  [1][D] residual (TransformerScratch.x)

attention block (Transformer.attentionBlock)
  rmsNorm(x, attn_norm)                         -> normed [D]
  matmul q, k, v                                -> q [H*hd], k [KV*hd], v [KV*hd]
  rmsNorm(q, q_norm, rows: H, dim: hd)           per-head (OLMoE: rows 1, dim H*hd)
  rmsNorm(k, k_norm, rows: KV, dim: hd)
  ropeNeox(q), ropeNeox(k)                      position t
  storeKV(k, v, layer, t)
  attention(q, cache, layer)                    -> attnOut [H*hd]
  matmul attn_output                            -> proj [D]
  add(x, proj) -> x

MoE block (Transformer.moeFFNBlock)
  rmsNorm(x, ffn_norm)                          -> normed [D]
  matmul router (ffn_gate_inp, F32)             -> routerLogits [E]
  routing.route: softmax, topK(k: 8, renormalize: qwen3moe true / olmoe false)
                                                -> expertIds [8], expertWeights [8]

  resident store (isFullyResident; the default when experts fit, true for every model in scope):
    w = store.weights(layer)                    StackedWeights over the fused file tensors
    matmulExperts(w.gate, indices: expertIds, perRow 8, normed, sharedInput) -> gate [8][I]
    matmulExperts(w.up,   ...)                                               -> up   [8][I]
    siluMul(gate, up) -> gate
    matmulExperts(w.down, indices: expertIds, gate, sharedInput: false)      -> proj [8][D]
    weightedSum(proj, expertWeights, terms 8, into: x, accumulate: true)
    (no host round trip; after the token, the ids are read back for store.recordUsage)

  streaming store (experts larger than the budget, or forced with --expert-cache-gb / --experts stream):
    commitBatch + synchronize                   the host needs the ids (one extra round trip per layer)
    lease = store.acquire(layer, ids)           hits pinned; misses queued on the IOPool (demand first)
    write lease.slots into expertSlots          (shared memory, host write is fine after sync)
    w = store.weights(layer)                    StackedWeights over the layer's ExpertArena
    encode hit experts' matmulExperts ...       committed before the misses finish (resident-first)
    encodeWait(for: store.signal, value: lease.readyValue)
    encode missed experts' matmulExperts ...    GPU waits on the event, not the CPU
    weightedSum(...)
    store.release(lease) after the batch that used it has completed
    optional PILOT: store.prefetch(layer + 1, predicted ids) from this layer's hidden state

next layer ... then outputHead: rmsNorm(x, output_norm), matmul(output or token_embd) -> logits [V]
synchronize; Sampler picks the token from logits on the host
```

Invariants: placement never changes the math (resident and forced streaming must give
byte-identical tokens), experts are combined in rank order, and a slot is never evicted
while a lease holds it.

## 5. Ownership (during the port)

Historical: how the port was split while it was built. The repository now has one owner.
Workstreams owned disjoint files. A workstream edits only what it owns; a change to someone
else's file goes through the orchestrator. Shared surfaces have exactly one owner.

| workstream | owns (paths relative to the repo root) |
|---|---|
| **A** GGUF + Quant + CPU backend | `Package.swift`; `Sources/ColibriCore/Support/**`; `Sources/ColibriCore/GGUF/**`; `Sources/ColibriCore/Quant/**`; `Sources/ColibriCore/Backend/Backend.swift` (the contract); `Sources/ColibriCore/Backend/CPU/**`; tests `Tests/ColibriCoreTests/GGUF*Tests.swift`, `Quant*Tests.swift`, `CPUBackend*Tests.swift`; fixtures `Tests/ColibriCoreTests/Fixtures/quant/**` |
| **B** Tokenizer + Sampling | `Sources/ColibriCore/Tokenizer/**`; `Sources/ColibriCore/Sampling/**`; tests `Tokenizer*Tests.swift`, `ChatTemplate*Tests.swift`, `Sampler*Tests.swift`; fixtures `Fixtures/tokenizer/**` |
| **C** Experts | `Sources/ColibriCore/Experts/**`; tests `Expert*Tests.swift`, `RoutingHistory*Tests.swift`, `MemoryBudget*Tests.swift` |
| **D** Models + CLI | `Sources/ColibriCore/Models/**`; `Sources/coli-swift/**`; tests `Model*Tests.swift`, `Registry*Tests.swift`, `Generator*Tests.swift`, `SkeletonTests.swift` |
| **E** Metal backend | `Sources/ColibriMetal/**`; `Tests/ColibriMetalTests/**` |
| **F** Oracle harness | `scripts/oracle/**`; `Sources/ColibriCore/Synthetic/**`; `Tests/ColibriCoreTests/Fixtures/README.md` and `Fixtures/oracle/**`; tests `Oracle*Tests.swift`, `TinyModel*Tests.swift` |
| orchestrator | `docs/port/**`, `README.md`, `.gitignore`, `LICENSE`, `NOTICE`; commits |

Rules that keep the shared surface small:

- **The contract files are A's**: `Backend.swift`, `Support/Errors.swift`,
  `Package.swift`. `ColibriError`'s cases are generic on purpose (put detail in the
  message), so nobody should need a new case. If D or E needs a new op or a signature
  change, they ask the orchestrator; the change lands as one edit that touches
  `Backend.swift`, `CPUBackend.swift` (A) and `MetalBackend.swift` (E) together, with stubs
  where the implementation is not ready.
- **Cross-workstream calls go through the declared types only**: D calls
  `GGUFFile.weight/stackedWeights` (A), `ExpertLayout`/`ExpertStores.make` (C),
  `BPETokenizer`/`ChatTemplate`/`Sampler` (B). C calls `Backend.register` and
  `makeSignal` (A's contract). Nobody reaches into another workstream's internals.
- **Tests are per-owner files** in the two shared test targets, named by the prefixes in
  the table. Shared test helpers go in the owner's own file, `internal`, or in
  `Synthetic/` (F) when several workstreams need them.
- **Fixtures**: one subdirectory per owner under `Tests/ColibriCoreTests/Fixtures/`. The
  directory is copied as a whole (`resources: [.copy("Fixtures")]`), so adding a file needs
  no `Package.swift` edit. No `.gguf` files are committed; tiny models come from
  `TinyModel` at test time, the real model from `~/colibri-models/`.
- **Builds**: each workstream uses its own scratch path
  (`swift build --scratch-path .build-<label>`, same for `swift test`) to avoid SwiftPM
  lock contention.

Order of dependency for running code (compiling never blocks, every stub links):
A's `GGUFFile`, `Dequant` and `CPUBackend` unblock D's Qwen3-0.6B oracle test and E's
kernel-versus-CPU tests; F's `TinyModel` needs A's `GGUFWriter` and `Quantize`; C's
streaming tests need A's `MappedFile` and `GGUFFile`. B depends only on `GGUFMetadata`, which
is already complete.

## 6. What the skeleton verified (historical; see `status.md` for the current state)

- `swift build` (debug and `-c release`) succeeds with no warnings.
- `swift test`: block geometry of `GGMLType`, the registry maps `qwen3`/`qwen3moe`/`olmoe`
  to their families, `Tier` hysteresis, CPU buffer allocation and `copy`; on Metal the MSL
  string compiles at runtime and a blit batch round-trips through `beginBatch` /
  `synchronize`.
- `coli-swift --help` prints usage; unknown commands exit 2; stubbed commands exit 1 with
  the TODO message.
