# Runtime and Metal: expert placement, streaming, kernels

Design note for the Swift port. It describes how upstream colibri (commit
`0a5504a`) places and streams routed experts and how its Metal backend batches
work, then decides what the Swift port keeps on a 64 GB M4 Max and how it is
built. Upstream sources read: `c/expert_store.h`, `c/expert_store_registry.*`,
`c/tier.h`, `c/pin_pool.h`, `c/park.h`, `c/xtier.h`, `c/route_trace.h`,
`c/backend_metal.{h,mm}`, `docs/CACHE_ROUTE.md`, `docs/routing-telemetry.md`,
`docs/tuning.md`, `docs/metal.md`, `docs/metal_implementation.md`,
`docs/METAL-M5MAX-PERF-REPORT.md`. Several of those files do not contain the
mechanism their names suggest, so the following were also read:
`c/olmoe.c` (slot cache, eviction, PILOT), `c/qwen36.c` (GGUF expert reads for
`qwen3`/`qwen3moe`), `c/colibri.c` (slab allocation, Metal overlap, PIPE),
`c/st.h` and `c/compat.h` (read path, macOS shims), `c/gguf.h`, and
`c/shaders/blockdec.glsl` (the only upstream GPU decoder for ggml block types).

## 1. What upstream actually has, file by file

| file | what it is | relevance to the port |
|---|---|---|
| `expert_store.h` | lookup/release/prefetch lease contract for an expert store (DeepSeek-V4 only) | **keep the contract shape**, not the code |
| `expert_store_registry.*` | name to open-fn table, selected by `COLI_EXPERT_STORE`, C constructors register backends | idea only: a store is a protocol and the engine never branches on the backend |
| `tier.h` | promotion rule (`hot > cold + cold/4 + 4`), LFRU score, heat decay `>>1` | **port verbatim** (tiny, pure) |
| `pin_pool.h` | up to 8 snapshots of prompt prefixes (ids + last logits) for serve-mode reuse | not expert placement; skip for the CLI |
| `park.h` | park a long conversation's KV rows while a short request runs | serve-mode feature; skip |
| `xtier.h` | picks the CUDA or Vulkan VRAM expert tier | no separate VRAM on Apple Silicon; skip |
| `route_trace.h` | `.coli_usage` history format, `ROUTE_TRACE` stream, atomic save | **port the format byte-compatibly** |
| `backend_metal.mm` | runtime-compiled MSL, zero-copy slab registry, batched expert SwiGLU, fused GLM MLA attention | **design reference**; its kernels do not cover ggml block types (see section 5) |

The streaming mechanism itself lives in the engines: `olmoe.c` (`LCache`,
`Slot`, `expert_get`/`expert_put`, `olmoe_victim`, `kv_room_fit`,
`pin_hot_experts`, `pilot_prefetch`), `qwen36.c` (`load_expert_merged` for
GGUF), and `colibri.c` (page-aligned slabs registered with Metal, PIPE I/O pool,
resident-first GPU submit).

## 2. How upstream places and streams experts

### 2.1 Placement

- **Dense weights are resident** (embeddings, attention, norms, router, shared
  expert, lm_head). Routed experts are not; they are read on demand.
- **Per-layer slot cache** (`LCache`): each layer owns `cap` slots. A slot holds
  one expert's gate, up and down bytes in the file's own block format (GGUF
  path) plus the type of each. `slot_by_expert[eid]` is an O(1) index.
- **Slot states** are encoded in `eid`: `>= 0` ready, `-1` empty, `-(eid+2)`
  loading. A second request for an expert that is in flight waits on a
  condition variable instead of starting a duplicate read
  (`slot_in_flight`, `pthread_cond_wait`).
- **`busy` refcount**: from `expert_get` to `expert_put` a slot cannot be
  evicted. Upstream added this after the prefetch thread evicted a slot that the
  forward pass was multiplying with.
- **Eviction** (`olmoe_victim`): least recently used among slots that are not
  pinned, not loading and not busy. If everything is pinned, take the oldest
  pinned one. If everything is loading, sleep 1 ms and rescan (never steal a
  slot mid-read: two writers would race on one buffer).

### 2.2 Read path

- **`pread`, never `mmap`, for experts.** `st.h` explains why: mmap pages count
  against RSS and the process's peak becomes the model size, not dense + cache.
  `gguf.h` also reads tensor bytes with `pread` ("nothing is mapped").
- **GGUF layout.** A stock GGUF stores the experts of a layer as three fused
  3-D tensors `blk.L.ffn_{gate,up,down}_exps.weight` of shape
  `[in, out, n_expert]`. Expert `e` is a contiguous slice at
  `e * nbytes/n_expert` in each, so **one expert = three contiguous reads**
  (`qwen36.c load_expert_merged`). colibri's own container
  (`blk.L.ffn_exps_colibri`) packs gate|up|down per expert at a fixed stride so
  it is one read; we cannot rely on that, since we read stock GGUFs.
- **Direct I/O.** `DIRECT=1` opens a twin fd with `O_DIRECT` (Linux) or
  `F_NOCACHE` (macOS, `compat_open_direct`). O_DIRECT needs aligned buffer,
  offset and length; upstream avoids a bounce copy by having the caller reserve
  `off % 4096` bytes of slack *before* the payload pointer, reading the aligned
  window in place, and reading a sub-block tail with a buffered fd
  (`st_read_range_rep`). On macOS `F_NOCACHE` has no alignment requirement.
- **Advice.** `posix_fadvise(WILLNEED)` maps to `fcntl(F_RDADVISE)` on macOS;
  `DONTNEED` is a no-op on macOS (XNU has no ranged drop). `EXPERT_DROP=1`
  restores DONTNEED on Linux for RAM-tight boxes; default keeps pages cached.
- **Alignment for Metal.** Every slab that the GPU reads is
  `posix_memalign(16384)` with length rounded up to 16 KiB, then wrapped once
  with `newBufferWithBytesNoCopy`. Reads land directly in that slab, so the GPU
  sees the expert with no copy.

### 2.3 Prefetch and overlap with compute

Three separate mechanisms, all opt-in except PIPE:

1. **Resident-first submit (Metal).** In `colibri.c moe()` the experts already
   in RAM are submitted to the GPU (`coli_metal_moe_block_begin`, asynchronous
   commit) *before* the missed experts are read, so the reads overlap GPU
   compute. A second block for the missed experts follows, then
   `moe_block_end` waits and scatter-adds.
2. **PIPE** (default on in `colibri.c`, 8 workers): missed experts are handed
   to a bounded I/O pool; the matmul of resident experts proceeds and joins
   the cold results before the layer completes. Measured -18% disk service
   time. Also a readahead (`WILLNEED`) of the *next* group of 64 experts.
3. **PILOT** (router lookahead, `olmoe.c pilot_prefetch`): apply layer `L+1`'s
   RMSNorm and router to layer `L`'s hidden state, blend with an EMA of past
   logits (`SMOOTH`, 0.3), take candidates until cumulative probability reaches
   `CONF_LIMIT` (0.92) but at least top-k (`WIDE` multiplies the count), and
   enqueue reads on a single worker thread (ring of 4096). Measured recall on
   GLM-5.2: 71.6% of the true top-8 versus 41.3% for "same experts as the last
   token". Hint-only PILOT can be net negative on disk-saturated hosts.

### 2.4 Cache policy: LRU, pins, LFRU repin

- **Default**: per-layer LRU.
- **HOT / PIN**: after `WARMUP` tokens (or at start from history) pin the top-N
  experts per layer by frequency. olmoe caps pins at `cap - 8` (minimum 4) so
  the LRU keeps room.
- **`PIN=auto` / AUTOPIN**: pin from the persistent `.coli_usage` history at
  startup, capped so automatic pins never eat the no-pin LRU capacity. Explicit
  `PIN`/`PIN_GB` win.
- **Live repin** (`--repin N`, `tier.h`): every N tokens at a safe boundary,
  swap at most four pinned experts for hotter unpinned ones. The LFRU score is
  `heat << 8 | recency` with recency `max(0, 255 - age)`, so recency only
  breaks ties. Promotion needs `hot > cold + cold/4 + 4` (25% plus 4 hysteresis
  against ping-pong); session heat decays by `>> 1`; the persistent history is
  never decayed.
- **Placement never changes the math.** Upstream states this repeatedly (pins
  "decide which bytes are resident, never what the model computes"). The two
  lossy levers, `CACHE_ROUTE` (substitute resident experts at ranks `J..K-1`
  inside a top-M window) and `DEGRADE_ZERO` (zero experts under gate 0.03 when a
  prefetch deadline is missed; +2.9% ppl on OLMoE) are opt-in and documented as
  experimental.
- **Measured coverage**: OLMoE has no hot experts (nearly straight coverage
  curve, upstream #864), so pinning buys little there; Qwen3.6 pins 8 GB of
  experts to cover 80.5% of its own history (57.7% on a held-out session for
  Qwen3.8).

### 2.5 RAM budget accounting

`olmoe.c` model init:

```
budget  = RAM_GB or --ram, else resident + 0.88 * available
room    = budget - resident_dense - 0.5 GB (activations)
slots   = floor(room / slot_bytes / n_layers)   (at least 1 per layer)
```

`kv_room_fit` then shares that room with the KV cache: before a forward writes
positions the KV has not reached, every layer's cap shrinks to what is left,
evicting LRU (pinned last). macOS `available` comes from `host_statistics64`
(free + inactive + purgeable); if it reads under 2% of total it is treated as
unmeasured. Upstream warns that on Metal the knee is lower than on CPU, because
registered buffers share unified memory: on a 128 GB M5 Max `--ram 110` was
safe and `--ram 120` pushed the machine into memory compression.

### 2.6 Persistence of routing history (`.coli_usage`)

Text, one sparse record per line, written next to the model, every turn:

```
-1 <n_layers> <n_experts>
-2 <format_version=1> <engine_id = FNV-1a32(engine name)>
<layer> <expert> <count>
```

Rules that are easy to break: every record has exactly three numeric fields
(old readers are `while(fscanf("%d %d %u")==3)` loops guarded by `l>=0`); the
hash sits in the third field (it exceeds `INT_MAX`); dense layers have no
counter row (`rt_drop_row`) so records naming them are rejected; an empty
history is a zero-byte file; save is atomic (`<path>.tmp` then `rename`);
`USAGE_SAVE=0` is read-only. A history whose engine id differs is refused
unless the user named it explicitly (`PIN=<file>`); dimension mismatches are
always refused. `ROUTE_TRACE=<path>` writes `<call> <row> <layer> id:gate ...`
per routing decision for offline analysis.

## 3. The Metal backend upstream

- **One device, one queue, pipelines compiled once** from an MSL string at
  runtime (`newLibraryWithSource`), so no Xcode toolchain is needed.
- **Zero-copy everywhere.** `wrap()` uses `newBufferWithBytesNoCopy` when the
  pointer and the length are both 16 KiB multiples, otherwise it falls back to
  a *copy* (`newBufferWithBytes`), which silently detaches the GPU from later
  writes. The slot API (`coli_metal_slot_register`) refuses unaligned memory
  for that reason. Resident dense weights go through `qalloc`, which rounds to
  16 KiB and registers.
- **Slab registry + resolve.** Registered slabs live in a vector under a mutex;
  `resolve(ptr)` finds the slab and returns `gpuAddress + (ptr - base)`. The
  batched expert kernel (`moe_gemv`) is *bindless*: it receives tables of
  `ulong` GPU addresses per expert (weights and scales), and the encoder calls
  `useResource:` on every distinct slab, or (opt-in `COLI_METAL_RESSET`, macOS
  15+) relies on one `MTLResidencySet` attached to the queue.
- **Batching.** Per MoE block, one command buffer: gate GEMV and up GEMV for
  all experts' rows, a barrier, SiLU-mul, a barrier, down GEMV; results are
  scatter-added on the CPU after `waitUntilCompleted`. GLM's whole decode layer
  (norm, MLA attention, residual, post-norm, shared expert, router, top-8) is
  another single command buffer. Training ops and early kernels still do one
  CB per op with fresh buffers.
- **Kernel shape.** One simdgroup (32 lanes) per output row, 4 rows per
  128-thread threadgroup, lanes stride across the row with 8-value
  (`uchar4`/`float4`) loads, `simd_sum` at the end. Measured 1.5-2.1x over one
  threadgroup per row (358-389 GB/s on engine-like shapes). Dequant happens in
  registers; accumulation is f32.
- **Quant formats supported on GPU**: f32, int8 per-row, int4 per-row, int2,
  grouped int4, fp8 e4m3 with 128x128 block scales, bf16, E8/IQ3 lattice, MLX
  affine Q4/Q8. **No ggml block type** (Q8_0, Q4_K, Q6_K, ...). See section 5.
- **Fallback.** Every GPU entry point returns 0 on any problem (unregistered
  pointer, bad fmt, command-buffer error) and the caller reruns that block on
  the CPU.
- **Measured lessons** (M5 Max, GLM-5.2, single runs, warm cache):
  - CPU active spin (OpenMP hot team) cost **-39%** because CPU and GPU share
    one power budget; passive waits restored GPU clocks. Never spin on Apple
    Silicon.
  - With passive waits the GPU starves between dispatches; PIPE with 8 I/O
    workers hid it: 1.90 to **2.24 tok/s**.
  - `DIRECT=1` (F_NOCACHE) was ~2x faster than buffered reads. The report says
    buffered reads "break the zero-copy slab registration"; in the code both
    modes `pread` into the same registered slab, so the more likely cause is
    the copy through the unified buffer cache plus the duplicate pages
    competing with the 98 GB working set.
  - Prefill GEMMs on GPU change the accumulation order; greedy output can flip
    on near-tie logits. With PIPE on, runs are not bit-reproducible.
  - `docs/metal.md` quotes "~5 ms submit latency"; `metal_implementation.md`
    measures ~150 us per command-buffer round trip. The second matches this
    machine (section 4).

## 4. Facts about this machine (measured 2026-11-14)

Probe programs in the session scratchpad, built with `swiftc -O`:

| quantity | value |
|---|---|
| GPU | Apple M4 Max, Metal 3, family apple9 |
| CPU | 16 cores: 12 performance + 4 efficiency |
| physical memory | 68.7 GB (64 GiB) |
| `recommendedMaxWorkingSetSize` | **55.7 GB** |
| `maxBufferLength` | **41.7 GB** |
| page size | 16384 |
| `maxThreadgroupMemoryLength` | 32 KiB |
| empty command buffer commit + wait, p50 / p10 / p90 | **116 / 110 / 372 us** |
| `makeBuffer(bytesNoCopy:)` over `mmap` of the whole Qwen3-0.6B GGUF (length rounded up to 16 KiB) | works |
| `pread` 1 MiB random, F_NOCACHE, 1 thread / 8 threads | **3.0 / 7.5 GB/s** |
| same, offset not 16 KiB aligned (GGUF slices are 32-aligned) | 2.7 / 8.0 GB/s |
| same, buffered, file in page cache | 4-6 / 47-64 GB/s |

The I/O numbers come from one run on a 640 MB file. F_NOCACHE still went to the
SSD even though the file was cached (the buffered run shows RAM speed). Queue
depth matters much more than alignment: 8 concurrent reads give 2.5x.

What the target models need (expert bytes from the GGUF block sizes: Q8_0 34 B
per 32, Q4_K 144 B per 256, Q6_K 210 B per 256):

| model | experts total | one expert | routed bytes per decoded token |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M (48 L x 128 E, top-8, D 2048, I 768) | ~17 GB | 2.6-3.0 MB | ~1.1 GB |
| Qwen3-30B-A3B Q8_0 | ~31 GB | 5.0 MB | ~1.9 GB |
| OLMoE-1B-7B Q8_0 (16 L x 64 E, top-8, D 2048, I 1024) | ~6.9 GB | 6.7 MB | ~0.86 GB |
| Qwen3-0.6B Q8_0 (dense, 28 L) | none | - | ~0.6 GB of dense weights |

## 5. Which pieces matter on a 64 GB Apple Silicon machine

1. **All three target architectures fit in RAM.** Qwen3-30B-A3B even at Q8_0
   (~32 GB file) is below both the 41.7 GB max buffer and the 55.7 GB GPU
   working set. The fast path is therefore *fully resident*: mmap the file,
   wrap it with `bytesNoCopy`, and run the entire token on the GPU, routing
   included. Streaming is still built, because it is the project's core idea and
   it is what lets a later, larger family run (Qwen3-235B-A22B, ~133 GB at
   Q4_K_M). For our models it is exercised by forcing a small budget, and the
   test is that forced streaming gives byte-identical tokens to resident mode.
2. **Streaming bound.** Every expert missing at 7.5 GB/s gives ~7 tok/s for
   Qwen3-30B Q4_K_M. The resident bound is memory bandwidth (410-546 GB/s
   depending on the GPU bin, ~1.4 GB per token): a ceiling in the hundreds, in
   practice whatever the kernels reach.
3. **Keep**: per-layer slot cache with busy refcounts and an in-flight state;
   F_NOCACHE `pread` straight into 16 KiB-aligned, GPU-registered memory;
   8-deep I/O parallelism; resident-first GPU submit with misses overlapped;
   PILOT lookahead (cheap: D x E router); `.coli_usage` history plus pin-at-start;
   `tier.h` repin; the RAM budget formula, with the additional ceiling of
   `recommendedMaxWorkingSetSize`; passive waits only.
4. **Drop**: NUMA, io_uring, O_DIRECT bounce logic (F_NOCACHE does not need
   it), VRAM tiers and `xtier.h` (one memory pool, so upstream's two-level
   "VRAM then RAM" residency collapses into one), OpenMP tuning, `park.h`,
   `pin_pool.h`, the store registry's C-constructor machinery, `DEGRADE_ZERO`
   and `CACHE_ROUTE` (lossy; could come back later as opt-ins).
5. **Write new**: Metal kernels for ggml block types. Upstream's Metal backend
   cannot read a GGUF quant block at all, `olmoe.c` has no Metal path, and
   `qwen36.c` (which runs `qwen3` and `qwen3moe` GGUFs) uses Metal only for an
   MLX-affine "qpack" store. The only upstream GPU decoder for ggml blocks is
   the Vulkan `shaders/blockdec.glsl`, which is the reference for section 7.

## 6. Recommended Swift design

### 6.1 Module layout (suggested; ownership is decided by the orchestrator)

```
Sources/ColibriCore/Runtime/
  MappedFile.swift         mmap of the whole GGUF, page-rounded, madvise
  ExpertLayout.swift       per-layer offsets/sizes/types of the fused expert tensors
  ExpertStore.swift        protocol + ExpertKey/Lease/Stats
  ResidentExpertStore.swift
  StreamingExpertStore.swift   slot cache, LRU, pins, in-flight, budget
  ExpertArena.swift        per-layer page-aligned slot memory (+ MTLBuffer)
  IOPool.swift             F_NOCACHE pread workers
  RoutingHistory.swift     .coli_usage read/write, heat, LFRU repin (port of tier.h/route_trace.h)
  MemoryBudget.swift       host_statistics64, budget formula
  Parallel.swift           concurrentPerform helpers
Sources/ColibriCore/Metal/
  MetalContext.swift       device, queue, pipeline cache, scratch buffers
  GPUWeights.swift         bytesNoCopy views of the MappedFile, split by maxBufferLength
  Kernels.metal.swift      the MSL source as a string, compiled at runtime
  MetalForward.swift       per-family encoders built from the kernel set
```

The MSL ships as a Swift string literal compiled with `makeLibrary(source:)`,
as upstream does: SwiftPM does not compile `.metal` files without Xcode, and
this keeps the package free of a build plugin.

### 6.2 Core types

```swift
struct ExpertKey: Hashable, Sendable { var layer: Int32; var expert: Int32 }

/// Where one layer's experts live in the file (stock GGUF: three fused tensors).
struct ExpertLayerLayout: Sendable {
    var gate, up, down: TensorSlice      // fileOffset of expert 0, bytesPerExpert, ggml type, rows, cols
    var slotBytes: Int { gate.bytesPerExpert + up.bytesPerExpert + down.bytesPerExpert }
}

/// A GPU/CPU-addressable view of one expert. Valid until release().
struct ExpertView {
    var buffer: MTLBuffer?               // nil on the CPU-only backend
    var base: UnsafeRawPointer           // CPU address of gate; up and down follow at fixed offsets
    var gateOffset, upOffset, downOffset: Int   // byte offsets inside `buffer`
}

protocol ExpertStore: AnyObject {
    /// Pins the slots for these keys for the duration of one MoE block.
    /// Hits return immediately; misses are started and reported in the lease.
    func acquire(_ keys: [ExpertKey]) -> ExpertLease
    func waitForMisses(_ lease: ExpertLease)      // blocks until every miss is published
    func release(_ lease: ExpertLease)
    func prefetch(_ keys: [ExpertKey])            // advisory, never evicts a busy slot
    var stats: ExpertStoreStats { get }
}
```

The lease contract is upstream's `expert_store.h` contract: release exactly
once, never evict a leased slot, prefetch holds no lease. Families never see
which store is active; the CLI picks it (`--experts resident|stream`,
`--ram GB`).

- **`ResidentExpertStore`**: no slots. The view of expert `e` is
  `(fileBuffer, sliceOffset + e * bytesPerExpert)`. `acquire` only counts.
  This is the default when the whole file fits the budget.
- **`StreamingExpertStore`**: one `ExpertArena` per MoE layer, `cap` slots each,
  slot stride rounded up to 16 KiB. Each slot keeps `eid`, a state
  (`empty | loading | ready`), `busy`, `pinned` and `lastUsed`. All of a
  layer's slot table sits behind one `Mutex<LayerCacheState>` (Synchronization
  module, macOS 15+). Waiters for an in-flight expert block on a per-layer
  `NSCondition`; they never start a second read. Eviction follows
  `olmoe_victim`; if every slot is loading, wait for a publish and never
  steal.

### 6.3 Memory: one allocation per layer, wrapped once

```swift
final class ExpertArena {
    let raw: UnsafeMutableRawPointer      // posix_memalign(16384, cap * stride)
    let buffer: MTLBuffer?                // device.makeBuffer(bytesNoCopy: raw, length:, options: .storageModeShared)
    let stride: Int                       // roundUp(slotBytes + 16384 /*pad*/, 16384)
}
```

Unlike upstream, which keeps one registered slab per slot and passes bindless
GPU address tables, the port uses **one buffer per layer and slot indices**:
the kernel computes `base + slot * stride`. That means no per-slab
registration, no `useResource:` loop whose length grows with the cache, no
residency set, and no `resolve()` lookups. With the 16 KiB pad in front of
each slot, an aligned F_NOCACHE window can be read in place, as in upstream's
`st_read_range_rep`. Measured alignment costs ~10% for a single thread and
nothing at depth 8, so v1 can read unaligned and keep the pad for later.

For the resident store, `GPUWeights` maps the whole GGUF once
(`mmap(PROT_READ, MAP_SHARED)`, length rounded up to 16 KiB) and wraps it with
`bytesNoCopy`. Files over `maxBufferLength` (41.7 GB here) are split into
overlapping views, each `maxBufferLength` long and overlapping by the largest
tensor, so every tensor sits wholly inside one view (llama.cpp does the same).
GGUF data offsets are 32-aligned, which satisfies Metal's buffer-offset rule.
Kernels must not assume 4-byte alignment *inside* a block (Q8_0 is 34 bytes):
read through `packed_` vector types or byte loads.

### 6.4 Budget

```
gpuCeiling = device.recommendedMaxWorkingSetSize          // 55.7 GB here
hostBudget = --ram, else 0.85 * physical - other processes' use (host_statistics64)
budget     = min(gpuCeiling, hostBudget)
room       = budget - denseResident - kvCache(maxContext) - scratch - 0.5 GB
mode       = file fits in room ? resident : streaming
slots[L]   = floor(room * slotBytes[L] / sum(slotBytes) / slotBytes[L])  // equal count per layer, >= topK
```

Slot sizes differ by layer in `Q4_K_M` files (some `ffn_down` are Q6_K), so
the budget is computed in bytes per layer, not as a single slot size (upstream
issue #856). `kv_room_fit` (shrinking the cache as the KV grows) is not
needed in v1: the KV is allocated at startup for `--ctx`.

### 6.5 I/O pool and prefetch

- `IOPool`: N = 8 dedicated `Thread`s (not Swift concurrency: a blocking
  `pread` on the cooperative pool starves it, and not `concurrentPerform`,
  which belongs to compute) draining a `Mutex`-guarded deque of jobs. A job
  makes three `pread`s on an `F_NOCACHE` fd (gate, up, down slices into the
  slot), then publishes: under the layer mutex it sets the state to
  `ready(eid)`, broadcasts the condition, and signals the store's
  `MTLSharedEvent` with a monotonically increasing value (see 6.7).
- Demand reads go to the front of the deque, prefetches to the back. A
  prefetch is dropped when its target is already ready or loading, or when the
  only evictable slots belong to experts acquired for the current token.
- **PILOT**: after layer `L`'s attention, the CPU computes
  `router[L+1](rmsnorm_{L+1}(h))` from the (shared-memory) hidden state. That
  is 2048 x 128 floats, ~0.1 ms on one core. Blend with the EMA, take the
  candidates, prefetch. Only useful in streaming mode.
- Upstream's speculative "next group of 64" readahead does not apply (top-8
  never forms a second group).

### 6.6 CPU compute: `DispatchQueue.concurrentPerform`

- One flat `concurrentPerform` per operation, over **chunks of output rows**
  (e.g. `rows / (activeProcessorCount * 4)`, at least 16 rows). Small chunks
  let the 12 P-cores take work from the 4 E-cores, so no barrier waits for the
  slowest core (upstream's reason for P-core-only OpenMP teams).
- In MoE, flatten `(expert, row-chunk)` pairs into one iteration space so the
  layer has one fork/join instead of one per expert.
- Never nest `concurrentPerform`. Never spin; `concurrentPerform` blocks
  passively.
- Swift 6 strict concurrency: pass raw pointers through a small
  `struct UnsafeSendable<T>: @unchecked Sendable` wrapper; Metal objects stay
  confined to the engine thread.
- Reduction order is fixed (per-row lanes, experts summed in top-k order), so
  greedy decode is reproducible run to run. Upstream lost that under PIPE.

### 6.7 Metal: kernels, batching, command buffers

Command-buffer round trips cost ~116 us here. A Qwen3-30B decode step has 48
layers, so one CB per layer adds ~5.6 ms per token (a 180 tok/s ceiling from
overhead alone) and one CB per op is out of the question. Plan:

| mode | command buffers per token | how routing is resolved |
|---|---|---|
| dense (`qwen3`) | **1** | n/a |
| MoE, resident store | **1** | GPU: router + top-k kernel writes expert ids; expert kernels index the fused tensor by `id * bytesPerExpert` (llama.cpp's `mul_mv_id` idea). The CPU never sees the routing until the token is done; telemetry reads the ids buffer afterwards. |
| MoE, streaming v1 | **2 per layer** | CB-A: attention + router + top-k, commit, wait. CPU: acquire leases, then commit CB-B for hit experts **before** queueing miss reads (upstream's resident-first order). Encode the missed experts into CB-B after an `encodeWaitForEvent(storeEvent, value: v)`, where `v` is the event value the last miss read will signal. One commit per layer, and the GPU does the hits while the SSD works. |
| MoE, streaming v2 | **1** | Whole token in one CB with an event handshake per layer: GPU `encodeSignalEvent(routerDone, 2L+1)`. An `MTLSharedEventListener` on the CPU reads the ids, acquires slots, writes a small slot-table buffer, issues reads, and signals `2L+2` once the table is written; the expert kernels wait on `2L+2`, and miss reads signal a separate store event. Do this only after v1 has been measured. |

Scratch buffers (activations, KV cache, logits, ids, slot tables) are allocated
once with `.storageModeShared` and reused. KV is f16 in shared memory.
Hazard tracking can stay on (the default); the arena and file buffers are
read-only to the GPU.

**Kernel set** (all f32 accumulate, dequant in registers):

| kernel | notes |
|---|---|
| `rmsnorm` (rows, per-head variant for Qwen3 q/k-norm) | one threadgroup per row, `simd_sum` + threadgroup reduction |
| `rope_neox` | Qwen3/OLMoE use the NEOX (half-split) layout |
| `matvec_<type>` | one simdgroup per output row, 4 rows/threadgroup; S rows (S <= 8) share the weight load |
| `matvec_id_<type>` | same, the weight base taken from `ids[k]` (resident) or `slotTable[k]` (streaming) |
| `gate_up_silu_<type>` | gate and up rows of the same output index in one simdgroup, writes `silu(g)*u`; saves a dispatch and a barrier per layer |
| `down_combine` | down projection per selected expert into `[k][D]`, then a fixed-order weighted sum + residual (no float atomics, reproducible) |
| `router_topk` | softmax, top-k for E <= 256, optional renormalise (Qwen3-MoE `norm_topk_prob = true`; OLMoE false) |
| `attn_decode` | GQA, one threadgroup per (kv-head group), online softmax over the f16 KV; split-K across threadgroups past ~2k positions |
| `matmul_<type>` (prefill) | tiled `simdgroup_float8x8` GEMM, dequantize a tile into threadgroup memory (32 KiB available), llama.cpp `mul_mm` style; v1 can loop the S-row matvec |
| `argmax` / `logits` | lm_head reuses `matvec`; Qwen3-0.6B ties it to `token_embd` |

**Dequant-in-kernel per type.** The unit of work is a 32-weight unit, as in
`blockdec.glsl`. A lane takes units `lane, lane+32, ...` along the row and
dots each with the activation, and `simd_sum` combines the lanes.

| type | block | per-unit decode |
|---|---|---|
| F32 / F16 / BF16 | - | `float4` / `half4` loads, `bf16` as `uint << 16` |
| Q8_0 | 34 B / 32 | `d = half` at +0; 32 `int8` at +2, read as 8 `packed_char4`; `d * dot` |
| Q4_0 | 18 B / 32 | `d` half; 16 bytes of nibbles; `(q - 8) * d` |
| Q4_K | 144 B / 256 | `d, dmin` halves; 12 bytes of 6-bit scales/mins (`get_scale_min_k4`: groups 0-3 direct, 4-7 split across the high bits); 128 nibble bytes. Unit `g` = 32 weights: `d*sc*dot(q, x) - dmin*m*sum(x)`, which keeps the mins out of the inner loop |
| Q5_K | 176 B / 256 | as Q4_K plus `qh` (32 bytes) giving bit 4 |
| Q6_K | 210 B / 256 | `ql` 128 B, `qh` 64 B, 16 int8 scales, `d` half at +208; 6-bit value `- 32`; two 16-scale halves per unit |
| Q2_K / Q3_K / IQ* | | later; ship a CPU fallback first |

Each kernel is checked against the Swift CPU reference decoder on random
blocks (maximum absolute error per type, the way upstream's `metal-test`
does), and the CPU reference against upstream's `ggml_quants.h` through the
oracle build.

### 6.8 Telemetry and persistence

- `RoutingHistory` is a port of `route_trace.h`: counters `[layer][expert]`
  (no row for dense layers), the `-1`/`-2` header, FNV-1a engine id, an atomic
  save (`.tmp` + `rename`) after each CLI turn and at exit (also on SIGINT),
  `USAGE_SAVE=0`, and the identity-mismatch refusal, which a path the user
  named overrides. Engine names: `qwen3moe` and `olmoe` (upstream's `olmoe`
  hash, so histories are interchangeable with upstream's OLMoE engine).
- At startup, when streaming: pin the top experts per layer from the history,
  at most `cap - topK` per layer, and load them on the I/O pool before the
  first token.
- `--repin N`: `tier_pick_lfru` with heat and last-access arrays, at most 4
  swaps per boundary, at token boundaries only.
- Footer stats: hits, misses, prefetch hits, bytes read, disk time versus
  foreground wait time (upstream reports both, so overlap is visible), GPU
  time from `cb.gpuEndTime - cb.gpuStartTime`.

## 7. Build order and verification

1. CPU backend + `ResidentExpertStore` (plain mmap): Qwen3-0.6B greedy tokens
   match the oracle. Upstream `qwen36.c` accepts `general.architecture =
   qwen3`, so it is a valid oracle for the one real model on disk.
2. Metal `matvec_<type>` + dense forward, 1 CB per token: token-exact with the
   CPU path on Qwen3-0.6B (greedy). Where they differ, check that it is a
   near-tie (print the top-2 margin, as upstream's `DEBUG_LOGITS`).
3. MoE on synthetic tiny GGUFs (`qwen3moe`, `olmoe`): CPU versus upstream
   oracle, then the GPU-routed resident path.
4. `StreamingExpertStore` with a forced tiny budget (`--ram` or
   `--expert-slots 1`): **tokens byte-identical to resident mode**, plus
   counters that show evictions and in-flight waits happened. Add a stress test
   with cap = topK and PILOT on, the configuration that exposed upstream's
   busy-slot race.
5. Streaming v1 overlap (resident-first CB + event-gated misses), measured
   against the synchronous version.
6. `.coli_usage` round trip, plus a test against a literal copy of the old
   `fscanf` reader loop, as upstream does.

## 8. Open questions and risks

- `bytesNoCopy` over a file mapping makes the GPU fault pages in from disk on
  first touch. For the resident path, prefault with `madvise(MADV_WILLNEED)`
  (or a CPU touch loop on the I/O threads) before the first token, or the
  first decode step pays disk latency inside a kernel.
- File-backed pages read through Metal count toward the GPU working set
  (55.7 GB). A Q8_0 Qwen3-30B (32 GB) plus KV fits; anything larger has to
  stream.
- The v2 event handshake has not been tried by upstream; MTLSharedEvent
  listener latency on this machine is unmeasured.
- The `iogpu.wired_limit_mb` sysctl raises the GPU working-set ceiling. It is a
  system setting: the user can change it, the port must not.
