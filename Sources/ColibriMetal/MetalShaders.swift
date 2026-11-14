// The Metal Shading Language source, compiled at runtime with
// `MTLDevice.makeLibrary(source:options:)` as upstream does (c/backend_metal.mm). SwiftPM
// does not compile .metal files without Xcode's build system, and a string keeps the
// package free of build plugins. Kernel plan: docs/port/runtime-and-metal.md section 6.7.
//
// Conventions:
// * Weights are read straight from the GGUF block bytes through `device const uchar *`.
//   Blocks have no 4-byte alignment inside (Q8_0 is 34 bytes), so multi-byte fields are
//   read as `half` / `ushort` (every block size and field offset is even) or byte by byte.
// * The block decoders (`DQ_<type>`) follow Sources/ColibriCore/Quant/Dequant.swift, which
//   is bit-exact against upstream colibri's c/ggml_quants.h. Each one is built for a
//   32-weight "unit" of a row and returns 4 weights at a time with `get(j)`, j in 0..<8.
//   One exception: Metal's f32 arithmetic flushes subnormals to zero, so a BF16 weight
//   below about 1.2e-38 (the only type whose decode can be subnormal) becomes 0 on the
//   GPU and stays subnormal on the CPU. MetalSupportTests checks this against upstream's
//   golden blocks.
// * Matrix kernels: one simdgroup computes R output rows; L lanes share each 32-weight
//   unit (32 / L weights per lane), so a simdgroup step covers 32 / L units and the
//   activation loads of a step are contiguous; `simd_sum` reduces. Accumulation is f32.
//   A prefill kernel decodes each weight group once for a tile of 8 batch rows.
// * Host-side constants that must match: MM_NSG simdgroups per matmul threadgroup, the R
//   of each matmul variant (see INSTANTIATE), ATT_NSG for attention.
// * Kernel names: `matmul_<type>_1` (decode, R = 1), `matmul_<type>_1w` (decode, R = 2),
//   `matmul_<type>_8` (small prefill batches), `matmul_tiled_<type>` (prefill batches of
//   `MetalBackend.tiledMinBatch` or more), `matmul_id_<type>` (and `matmul_id_tiled_<type>`
//   after `group_pairs` for routed prefill), `embed_<type>`, with <type> the
//   lower-cased ggml name (`q8_0`, `q4_k`, ...). MetalBackend looks them up by name.

enum MetalShaders {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

#define MM_NSG 4
#define ATT_NSG 4
#define ATT_MAXD 256

// Smoke-test kernel used by the skeleton test.
kernel void colibri_copy_f32(device const float *src [[buffer(0)]],
                             device float *dst       [[buffer(1)]],
                             constant uint &n        [[buffer(2)]],
                             uint i                  [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

// ---------------------------------------------------------------------------------------
// Block decoders
// ---------------------------------------------------------------------------------------

inline float ld_h(device const uchar *p) { return float(*(device const half *)p); }
inline uint ld_u16(device const uchar *p) { return uint(*(device const ushort *)p); }
// Four bytes at a 2-byte aligned address (every block field the decoders read is at an
// even offset), as one little-endian word.
inline uint ld_w(device const uchar *p) {
    device const ushort *s = (device const ushort *)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
inline float4 u8x4(uint w) { return float4(as_type<uchar4>(w)); }
inline float4 s8x4(uint w) { return float4(as_type<char4>(w)); }
inline int4 i8x4(uint w) { return int4(as_type<uchar4>(w)); }
// Bits 0..3 of x moved to bit 0 of bytes 0..3.
inline uint spread4(uint x) { return (x & 1) | ((x & 2) << 7) | ((x & 4) << 14) | ((x & 8) << 21); }

constant float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };

inline float4 iq4nl4(uint w) {   // w: four nibbles in the low halves of its bytes
    return float4(kvalues_iq4nl[w & 15], kvalues_iq4nl[(w >> 8) & 15],
                  kvalues_iq4nl[(w >> 16) & 15], kvalues_iq4nl[(w >> 24) & 15]);
}

// gq_scale_min_k4: the 6-bit scale and min of sub-block g of a Q4_K / Q5_K block.
inline void scale_min_k4(uint g, device const uchar *s, thread uint &sc, thread uint &mn) {
    if (g < 4) {
        sc = s[g] & 63; mn = s[g + 4] & 63;
    } else {
        sc = (s[g + 4] & 0xF) | ((s[g - 4] >> 6) << 4);
        mn = (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    }
}

struct DQ_F32 {
    DQ_F32() {}
    device const uchar *p;
    DQ_F32(device const uchar *row, uint u) : p(row + u * 128) {}
    float4 get(uint j) const { return float4(*(device const packed_float4 *)(p + 16 * j)); }
};

struct DQ_F16 {
    DQ_F16() {}
    device const uchar *p;
    DQ_F16(device const uchar *row, uint u) : p(row + u * 64) {}
    float4 get(uint j) const { return float4(*(device const packed_half4 *)(p + 8 * j)); }
};

struct DQ_BF16 {
    DQ_BF16() {}
    device const uchar *p;
    DQ_BF16(device const uchar *row, uint u) : p(row + u * 64) {}
    float4 get(uint j) const {
        uint4 v = uint4(*(device const packed_ushort4 *)(p + 8 * j));
        return as_type<float4>(v << 16);
    }
};

struct DQ_Q8_0 {   // 34 B: d, 32 x int8
    DQ_Q8_0() {}
    device const uchar *b; float d;
    DQ_Q8_0(device const uchar *row, uint u) : b(row + u * 34) { d = ld_h(b); }
    float4 get(uint j) const { return s8x4(ld_w(b + 2 + 4 * j)) * d; }
};

struct DQ_Q4_0 {   // 18 B: d, 16 B nibbles (low = 0..15, high = 16..31)
    DQ_Q4_0() {}
    device const uchar *b; float d;
    DQ_Q4_0(device const uchar *row, uint u) : b(row + u * 18) { d = ld_h(b); }
    float4 get(uint j) const {
        uint w = (ld_w(b + 2 + 4 * (j & 3)) >> ((j >> 2) * 4)) & 0x0F0F0F0Fu;
        return d * (u8x4(w) - 8.0f);
    }
};

struct DQ_Q4_1 {   // 20 B: d, m, 16 B nibbles
    DQ_Q4_1() {}
    device const uchar *b; float d, m;
    DQ_Q4_1(device const uchar *row, uint u) : b(row + u * 20) { d = ld_h(b); m = ld_h(b + 2); }
    float4 get(uint j) const {
        uint w = (ld_w(b + 4 + 4 * (j & 3)) >> ((j >> 2) * 4)) & 0x0F0F0F0Fu;
        return d * u8x4(w) + m;
    }
};

struct DQ_Q5_0 {   // 22 B: d, qh (u32 at +2), 16 B nibbles
    DQ_Q5_0() {}
    device const uchar *b; float d; uint qh;
    DQ_Q5_0(device const uchar *row, uint u) : b(row + u * 22) { d = ld_h(b); qh = ld_w(b + 2); }
    float4 get(uint j) const {
        uint w = (ld_w(b + 6 + 4 * (j & 3)) >> ((j >> 2) * 4)) & 0x0F0F0F0Fu;
        w |= spread4((qh >> (4 * j)) & 15) << 4;
        return d * (u8x4(w) - 16.0f);
    }
};

struct DQ_Q5_1 {   // 24 B: d, m, qh (u32 at +4), 16 B nibbles
    DQ_Q5_1() {}
    device const uchar *b; float d, m; uint qh;
    DQ_Q5_1(device const uchar *row, uint u) : b(row + u * 24) { d = ld_h(b); m = ld_h(b + 2); qh = ld_w(b + 4); }
    float4 get(uint j) const {
        uint w = (ld_w(b + 8 + 4 * (j & 3)) >> ((j >> 2) * 4)) & 0x0F0F0F0Fu;
        w |= spread4((qh >> (4 * j)) & 15) << 4;
        return d * u8x4(w) + m;
    }
};

struct DQ_IQ4_NL { // 18 B: d, 16 B codebook nibbles
    DQ_IQ4_NL() {}
    device const uchar *b; float d;
    DQ_IQ4_NL(device const uchar *row, uint u) : b(row + u * 18) { d = ld_h(b); }
    float4 get(uint j) const {
        return d * iq4nl4(ld_w(b + 2 + 4 * (j & 3)) >> ((j >> 2) * 4));
    }
};

struct DQ_Q2_K {   // 84 B / 256: scales[16], qs[64], d, dmin
    DQ_Q2_K() {}
    device const uchar *b; uint g; float d, dmin;
    DQ_Q2_K(device const uchar *row, uint u) : b(row + (u >> 3) * 84), g(u & 7) {
        d = ld_h(b + 80); dmin = ld_h(b + 82);
    }
    float4 get(uint j) const {
        uchar sc = b[2 * g + (j >> 2)];
        float dl = d * float(sc & 15), ml = dmin * float(sc >> 4);
        uint w = (ld_w(b + 16 + (g >> 2) * 32 + 4 * j) >> (2 * (g & 3))) & 0x03030303u;
        return dl * u8x4(w) - ml;
    }
};

struct DQ_Q3_K {   // 110 B / 256: hmask[32], qs[64], scales[12], d
    DQ_Q3_K() {}
    device const uchar *b; uint g; float d;
    DQ_Q3_K(device const uchar *row, uint u) : b(row + (u >> 3) * 110), g(u & 7) { d = ld_h(b + 108); }
    float4 get(uint j) const {
        device const uchar *s = b + 96;
        uint k = 2 * g + (j >> 2);
        uint l = k < 8 ? (s[k] & 15) : (s[k - 8] >> 4);
        uint h = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        float dl = d * float(int(l | (h << 4)) - 32);
        uint ql = (ld_w(b + 32 + (g >> 2) * 32 + 4 * j) >> (2 * (g & 3))) & 0x03030303u;
        uint hb = (ld_w(b + 4 * j) >> g) & 0x01010101u;
        return dl * float4(i8x4(ql | (hb << 2)) - 4);
    }
};

struct DQ_Q4_K {   // 144 B / 256: d, dmin, scales[12], qs[128]
    DQ_Q4_K() {}
    device const uchar *b; uint g; float d1, m1;
    DQ_Q4_K(device const uchar *row, uint u) : b(row + (u >> 3) * 144), g(u & 7) {
        uint sc, mn; scale_min_k4(g, b + 4, sc, mn);
        d1 = ld_h(b) * float(sc); m1 = ld_h(b + 2) * float(mn);
    }
    float4 get(uint j) const {
        uint w = (ld_w(b + 16 + (g >> 1) * 32 + 4 * j) >> ((g & 1) * 4)) & 0x0F0F0F0Fu;
        return d1 * u8x4(w) - m1;
    }
};

struct DQ_Q5_K {   // 176 B / 256: d, dmin, scales[12], qh[32], qs[128]
    DQ_Q5_K() {}
    device const uchar *b; uint g; float d1, m1;
    DQ_Q5_K(device const uchar *row, uint u) : b(row + (u >> 3) * 176), g(u & 7) {
        uint sc, mn; scale_min_k4(g, b + 4, sc, mn);
        d1 = ld_h(b) * float(sc); m1 = ld_h(b + 2) * float(mn);
    }
    float4 get(uint j) const {
        uint w = (ld_w(b + 48 + (g >> 1) * 32 + 4 * j) >> ((g & 1) * 4)) & 0x0F0F0F0Fu;
        w |= ((ld_w(b + 16 + 4 * j) >> g) & 0x01010101u) << 4;
        return d1 * u8x4(w) - m1;
    }
};

struct DQ_Q6_K {   // 210 B / 256: ql[128], qh[64], scales[16] (int8), d
    DQ_Q6_K() {}
    device const uchar *b; uint g; float d;
    DQ_Q6_K(device const uchar *row, uint u) : b(row + (u >> 3) * 210), g(u & 7) { d = ld_h(b + 208); }
    float4 get(uint j) const {
        float dl = d * float(((device const char *)b)[192 + 2 * g + (j >> 2)]);
        uint c = g >> 2, r0 = (g & 3) * 32, t0 = 4 * j;
        uint lo = (ld_w(b + c * 64 + ((r0 + t0) & 63)) >> (4 * ((g & 3) >> 1))) & 0x0F0F0F0Fu;
        uint hi = (ld_w(b + 128 + c * 32 + t0) >> (2 * (g & 3))) & 0x03030303u;
        return dl * float4(i8x4(lo | (hi << 4)) - 32);
    }
};

struct DQ_IQ4_XS { // 136 B / 256: d, scales_h (u16), scales_l[4], qs[128]
    DQ_IQ4_XS() {}
    device const uchar *b; uint ib; float dl;
    DQ_IQ4_XS(device const uchar *row, uint u) : b(row + (u >> 3) * 136), ib(u & 7) {
        uint sh = ld_u16(b + 2);
        uint ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        dl = ld_h(b) * float(int(ls) - 32);
    }
    float4 get(uint j) const {
        return dl * iq4nl4(ld_w(b + 8 + 16 * ib + 4 * (j & 3)) >> ((j >> 2) * 4));
    }
};

// ---------------------------------------------------------------------------------------
// Quantised matrix products
// ---------------------------------------------------------------------------------------

struct MMParams {
    ulong rowBytes;     // bytes of one weight row
    ulong wDelta;       // byte offset of row 0 from the bound weight pointer
    uint cols, rows, batch;
    uint xStride, yStride;
    float scale;
};

// acc[r][s] += dot(row (row0 + r), x[s]) over this lane's share of the row. LPU lanes
// share a 32-weight unit (32 / LPU weights each), so one simdgroup step covers 32 / LPU
// units; rows past `rows` re-read the last row and are dropped by the caller.
template<typename DQ, int NS, int R, int LPU>
inline void mv_accumulate(device const uchar *W, ulong rowBytes, uint row0, uint rows, uint cols,
                          device const float *X, uint xStride, uint ns, ushort lane,
                          thread float (&acc)[R][NS]) {
    const uint nunits = cols / 32;
    const uint GPL = 8 / LPU;           // float4 groups per lane per unit
    const uint part = lane % LPU;
    device const uchar *rowp[R];
    for (int r = 0; r < R; ++r) rowp[r] = W + (ulong)min(row0 + r, rows - 1) * rowBytes;
    for (uint u = lane / LPU; u < nunits; u += 32 / LPU) {
        DQ q[R];
        for (int r = 0; r < R; ++r) q[r] = DQ(rowp[r], u);
        for (uint jj = 0; jj < GPL; ++jj) {
            const uint j = part * GPL + jj;
            float4 w[R];
            for (int r = 0; r < R; ++r) w[r] = q[r].get(j);
            for (int s = 0; s < NS; ++s) {
                if ((uint)s < ns) {
                    const float4 x4 = float4(*(device const packed_float4 *)(X + (ulong)s * xStride + u * 32 + 4 * j));
                    for (int r = 0; r < R; ++r) acc[r][s] += dot(w[r], x4);
                }
            }
        }
    }
}

// y[s][o] = scale * dot(W row o, x[s]); grid (ceil(rows / (R * MM_NSG)), ceil(batch / NS)).
template<typename DQ, int NS, int R, int LPU, int TAG = 0>   // TAG: distinct tuning copies
kernel void k_matmul(device const uchar *W   [[buffer(0)]],
                     device const float *X   [[buffer(1)]],
                     device float *Y         [[buffer(2)]],
                     constant MMParams &p    [[buffer(3)]],
                     uint3 tg                [[threadgroup_position_in_grid]],
                     ushort sg               [[simdgroup_index_in_threadgroup]],
                     ushort lane             [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * MM_NSG + sg) * R;
    if (row0 >= p.rows) return;
    const uint s0 = tg.y * NS;
    const uint ns = min((uint)NS, p.batch - s0);
    float acc[R][NS];
    for (int r = 0; r < R; ++r) for (int s = 0; s < NS; ++s) acc[r][s] = 0.0f;
    mv_accumulate<DQ, NS, R, LPU>(W + p.wDelta, p.rowBytes, row0, p.rows, p.cols,
                             X + (ulong)s0 * p.xStride, p.xStride, ns, lane, acc);
    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < NS; ++s) {
            float v = simd_sum(acc[r][s]);
            if (lane == 0 && row0 + r < p.rows && (uint)s < ns) {
                Y[(ulong)(s0 + s) * p.yStride + row0 + r] = p.scale * v;
            }
        }
    }
}

// Prefill: a tiled product on the simdgroup matrix units (llama.cpp's mul_mm, in f32).
// A threadgroup of 4 simdgroups computes a 64-row x 32-token tile of Y. For each
// 32-weight unit of the rows, 128 threads decode the tile's 64 x 32 weights once into
// threadgroup memory (the DQ decoders, f32) next to the 32 x 32 activations; each
// simdgroup then multiplies a 32 x 16 block as 8x8 f32 simdgroup matrices. Sums are f32
// throughout; only their order differs from the matvec kernels. The tile is staged in
// threadgroup memory and written with bounds checks (rows past `rows` re-read the last
// row, tokens past `batch` read zeros; both are dropped).
// Grid (ceil(rows / MMT_BM), ceil(batch / MMT_BN)), 128 threads.
#define MMT_BM 64
#define MMT_BN 32
template<typename DQ>
kernel void k_matmul_tiled(device const uchar *W   [[buffer(0)]],
                           device const float *X   [[buffer(1)]],
                           device float *Y         [[buffer(2)]],
                           constant MMParams &p    [[buffer(3)]],
                           uint3 tg                [[threadgroup_position_in_grid]],
                           ushort tid              [[thread_index_in_threadgroup]],
                           ushort sg               [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMT_BM * 32];   // weights [row][k]; later the output [token][row]
    threadgroup float sb[MMT_BN * 32];   // activations [token][k]
    const uint row0 = tg.x * MMT_BM, s0 = tg.y * MMT_BN;
    const uint lr = tid >> 1, lj = (tid & 1) * 4;      // weight loader: row, first float4 group
    device const uchar *wrow = W + p.wDelta + (ulong)min(row0 + lr, p.rows - 1) * p.rowBytes;
    const uint xs = tid >> 2, xj = (tid & 3) * 2;      // activation loader: token, first group
    const bool xin = s0 + xs < p.batch;
    device const float *xrow = X + (ulong)min(s0 + xs, p.batch - 1) * p.xStride;
    const uint sr = (sg & 1) * 32, st = (sg >> 1) * 16; // this simdgroup's rows and tokens
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint nunits = p.cols / 32;
    for (uint u = 0; u < nunits; ++u) {
        DQ q(wrow, u);
        for (uint jj = 0; jj < 4; ++jj) {
            *(threadgroup float4 *)(sa + lr * 32 + 4 * (lj + jj)) = q.get(lj + jj);
        }
        for (uint jj = 0; jj < 2; ++jj) {
            float4 v = xin ? float4(*(device const packed_float4 *)(xrow + u * 32 + 4 * (xj + jj))) : float4(0.0f);
            *(threadgroup float4 *)(sb + xs * 32 + 4 * (xj + jj)) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (int i = 0; i < 4; ++i) simdgroup_load(a[i], sa + (sr + 8 * i) * 32 + k, 32);
            for (int j = 0; j < 2; ++j) simdgroup_load(b[j], sb + (st + 8 * j) * 32 + k, 32, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            simdgroup_store(acc[i][j], sa + (st + 8 * j) * MMT_BM + sr + 8 * i, MMT_BM, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < MMT_BM * MMT_BN; e += 128) {
        const uint s = e / MMT_BM, r = e % MMT_BM;
        if (s0 + s < p.batch && row0 + r < p.rows) Y[(ulong)(s0 + s) * p.yStride + row0 + r] = p.scale * sa[e];
    }
}

struct MMIdParams {
    ulong rowBytes;
    ulong stride;       // bytes between matrix i and i + 1
    ulong wDelta;
    uint cols, rows, perRow, sharedInput, count, hasScales;
};

// Routed product (llama.cpp's mul_mv_id): pair p = s * perRow + k uses matrix ids[p].
// Grid (ceil(rows / MM_NSG), rows * perRow): one row per simdgroup.
template<typename DQ, int LPU>
kernel void k_matmul_id(device const uchar *W     [[buffer(0)]],
                        device const float *X     [[buffer(1)]],
                        device float *Y           [[buffer(2)]],
                        device const int *ids     [[buffer(3)]],
                        device const float *scales [[buffer(4)]],
                        constant MMIdParams &p    [[buffer(5)]],
                        uint3 tg                  [[threadgroup_position_in_grid]],
                        ushort sg                 [[simdgroup_index_in_threadgroup]],
                        ushort lane               [[thread_index_in_simdgroup]]) {
    const int R = 1;
    const uint row0 = (tg.x * MM_NSG + sg) * R;
    if (row0 >= p.rows) return;
    const uint pair = tg.y;
    const int e = ids[pair];
    device float *y = Y + (ulong)pair * p.rows;
    if (e < 0 || (uint)e >= p.count) {   // a bad index is a bug upstream of us: make it loud
        if (lane == 0) for (int r = 0; r < R; ++r) if (row0 + r < p.rows) y[row0 + r] = NAN;
        return;
    }
    const uint xrow = p.sharedInput ? pair / p.perRow : pair;
    float acc[R][1];
    for (int r = 0; r < R; ++r) acc[r][0] = 0.0f;
    mv_accumulate<DQ, 1, R, LPU>(W + p.wDelta + (ulong)e * p.stride, p.rowBytes, row0, p.rows, p.cols,
                            X + (ulong)xrow * p.cols, p.cols, 1, lane, acc);
    const float scale = p.hasScales ? scales[e] : 1.0f;
    for (int r = 0; r < R; ++r) {
        float v = simd_sum(acc[r][0]);
        if (lane == 0 && row0 + r < p.rows) y[row0 + r] = scale * v;
    }
}

// Routed prefill, step 1 (llama.cpp's mul_mm_id grouping): one simdgroup per matrix e
// writes the pairs routed to e, in pair order, to list[e * cap ...] and their number to
// counts[e]. Simdgroup e == count handles out-of-range ids: their output rows become NaN
// (as k_matmul_id does), so a bad index stays loud. Grid count + 1, 32 threads.
struct GroupParams { uint pairs, count, cap, rows; };

kernel void k_group_pairs(device const int *ids    [[buffer(0)]],
                          device uint *counts      [[buffer(1)]],
                          device int *list         [[buffer(2)]],
                          device float *Y          [[buffer(3)]],
                          constant GroupParams &p  [[buffer(4)]],
                          uint e                   [[threadgroup_position_in_grid]],
                          ushort lane              [[thread_index_in_simdgroup]]) {
    uint base = 0;
    for (uint p0 = 0; p0 < p.pairs; p0 += 32) {
        const uint q = p0 + lane;
        const bool valid = q < p.pairs;
        const int id = valid ? ids[q] : 0;
        const bool bad = valid && (id < 0 || (uint)id >= p.count);
        if (e < p.count) {
            const uint hit = (valid && !bad && (uint)id == e) ? 1u : 0u;
            const uint off = simd_prefix_exclusive_sum(hit);
            if (hit) list[(ulong)e * p.cap + base + off] = (int)q;
            base += simd_sum(hit);
        } else if (bad) {
            for (uint r = 0; r < p.rows; ++r) Y[(ulong)q * p.rows + r] = NAN;
        }
    }
    if (e < p.count && lane == 0) counts[e] = base;
}

// Routed prefill, step 2: k_matmul_tiled over each matrix's list of pairs. Threadgroup
// (x, y, e) computes rows x * 64 ..< +64 of matrix e for token tiles y, y + tilesY, ...
// of e's pairs (tile = 32 pairs), reading input row pair / perRow (sharedInput) or pair,
// writing output row pair. Same f32 tiles and simdgroup products as k_matmul_tiled.
// Grid (ceil(rows / 64), tilesY, count), 128 threads.
struct MMTIdParams {
    ulong rowBytes, stride, wDelta;
    uint cols, rows, perRow, sharedInput, count, hasScales, cap, tilesY;
};

template<typename DQ>
kernel void k_matmul_id_tiled(device const uchar *W     [[buffer(0)]],
                              device const float *X     [[buffer(1)]],
                              device float *Y           [[buffer(2)]],
                              device const uint *counts [[buffer(3)]],
                              device const int *list    [[buffer(4)]],
                              device const float *scales [[buffer(5)]],
                              constant MMTIdParams &p   [[buffer(6)]],
                              uint3 tg                  [[threadgroup_position_in_grid]],
                              ushort tid                [[thread_index_in_threadgroup]],
                              ushort sg                 [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[MMT_BM * 32];
    threadgroup float sb[MMT_BN * 32];
    threadgroup int tp[MMT_BN];          // the tile's pair indices (-1 past the list)
    const uint e = tg.z, n = counts[e];
    const uint row0 = tg.x * MMT_BM;
    if (tg.y * MMT_BN >= n) return;
    const float scale = p.hasScales ? scales[e] : 1.0f;
    const uint lr = tid >> 1, lj = (tid & 1) * 4;
    device const uchar *wrow = W + p.wDelta + (ulong)e * p.stride + (ulong)min(row0 + lr, p.rows - 1) * p.rowBytes;
    const uint xs = tid >> 2, xj = (tid & 3) * 2;
    const uint sr = (sg & 1) * 32, st = (sg >> 1) * 16;
    const uint nunits = p.cols / 32;
    device const int *mine = list + (ulong)e * p.cap;
    for (uint t0 = tg.y * MMT_BN; t0 < n; t0 += p.tilesY * MMT_BN) {
        if (tid < MMT_BN) tp[tid] = t0 + tid < n ? mine[t0 + tid] : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int pr = tp[xs];
        const bool xin = pr >= 0;
        device const float *xrow = X + (ulong)(xin ? (p.sharedInput ? (uint)pr / p.perRow : (uint)pr) : 0) * p.cols;
        simdgroup_float8x8 acc[4][2];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint u = 0; u < nunits; ++u) {
            DQ q(wrow, u);
            for (uint jj = 0; jj < 4; ++jj) {
                *(threadgroup float4 *)(sa + lr * 32 + 4 * (lj + jj)) = q.get(lj + jj);
            }
            for (uint jj = 0; jj < 2; ++jj) {
                float4 v = xin ? float4(*(device const packed_float4 *)(xrow + u * 32 + 4 * (xj + jj))) : float4(0.0f);
                *(threadgroup float4 *)(sb + xs * 32 + 4 * (xj + jj)) = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 32; k += 8) {
                simdgroup_float8x8 a[4], b[2];
                for (int i = 0; i < 4; ++i) simdgroup_load(a[i], sa + (sr + 8 * i) * 32 + k, 32);
                for (int j = 0; j < 2; ++j) simdgroup_load(b[j], sb + (st + 8 * j) * 32 + k, 32, ulong2(0, 0), true);
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(acc[i][j], sa + (st + 8 * j) * MMT_BM + sr + 8 * i, MMT_BM, ulong2(0, 0), true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint x = tid; x < MMT_BM * MMT_BN; x += 128) {
            const uint s = x / MMT_BM, r = x % MMT_BM;
            const int ps = tp[s];
            if (ps >= 0 && row0 + r < p.rows) Y[(ulong)ps * p.rows + row0 + r] = scale * sa[x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct EmbedParams {
    ulong rowBytes;
    ulong wDelta;
    uint cols, rows, ntok;
    float scale;
};

// out[s][4i ..< 4i + 4] = row tokens[s], grid (cols / 4, ntok).
template<typename DQ>
kernel void k_embed(device const uchar *W     [[buffer(0)]],
                    device const int *tokens  [[buffer(1)]],
                    device float *out         [[buffer(2)]],
                    constant EmbedParams &p   [[buffer(3)]],
                    uint2 gid                 [[thread_position_in_grid]]) {
    const uint i = gid.x, s = gid.y;
    if (i >= p.cols / 4 || s >= p.ntok) return;
    const int t = tokens[s];
    device float *o = out + (ulong)s * p.cols + 4 * i;
    if (t < 0 || (uint)t >= p.rows) { o[0] = o[1] = o[2] = o[3] = NAN; return; }
    DQ q(W + p.wDelta + (ulong)t * p.rowBytes, i >> 3);
    float4 v = q.get(i & 7) * p.scale;
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}

typedef decltype(k_matmul<DQ_Q8_0, 1, 4, 4>) matmul_t;
typedef decltype(k_matmul_id<DQ_Q8_0, 4>) matmul_id_t;
typedef decltype(k_embed<DQ_Q8_0>) embed_t;
typedef decltype(k_matmul_tiled<DQ_Q8_0>) matmul_tiled_t;
typedef decltype(k_matmul_id_tiled<DQ_Q8_0>) matmul_id_tiled_t;

// Per type: L = lanes per 32-weight unit. Decode uses R = 1 row per simdgroup ("_1",
// more threadgroups for the 768..4096-row matrices of a layer) or R = 2 ("_1w", for
// output heads and other tall matrices); prefill R = 2 over 8-row batch tiles. Chosen
// from a sweep on an M4 Max (MetalTuneTests): the K-quants and IQ4_XS prefer one lane per
// unit (their scale decode is amortised over 32 weights), the 32-blocks two.
#define INSTANTIATE(NAME, DQ, L) \
template [[host_name("matmul_" #NAME "_1")]] kernel matmul_t k_matmul<DQ, 1, 1, L>; \
template [[host_name("matmul_" #NAME "_1w")]] kernel matmul_t k_matmul<DQ, 1, 2, L>; \
template [[host_name("matmul_" #NAME "_8")]] kernel matmul_t k_matmul<DQ, 8, 2, L>; \
template [[host_name("matmul_tiled_" #NAME)]] kernel matmul_tiled_t k_matmul_tiled<DQ>; \
template [[host_name("matmul_id_" #NAME)]] kernel matmul_id_t k_matmul_id<DQ, L>; \
template [[host_name("matmul_id_tiled_" #NAME)]] kernel matmul_id_tiled_t k_matmul_id_tiled<DQ>; \
template [[host_name("embed_" #NAME)]] kernel embed_t k_embed<DQ>;

INSTANTIATE(f32, DQ_F32, 2)
INSTANTIATE(f16, DQ_F16, 2)
INSTANTIATE(bf16, DQ_BF16, 2)
INSTANTIATE(q8_0, DQ_Q8_0, 2)
INSTANTIATE(q4_0, DQ_Q4_0, 2)
INSTANTIATE(q4_1, DQ_Q4_1, 2)
INSTANTIATE(q5_0, DQ_Q5_0, 2)
INSTANTIATE(q5_1, DQ_Q5_1, 2)
INSTANTIATE(iq4_nl, DQ_IQ4_NL, 2)
INSTANTIATE(q2_k, DQ_Q2_K, 1)
INSTANTIATE(q3_k, DQ_Q3_K, 1)
INSTANTIATE(q4_k, DQ_Q4_K, 1)
INSTANTIATE(q5_k, DQ_Q5_K, 1)
INSTANTIATE(q6_k, DQ_Q6_K, 1)
INSTANTIATE(iq4_xs, DQ_IQ4_XS, 1)

// Decode-matvec variants for the shape sweep in MetalTuneTests (MetalBackend.tuneDecode):
// `tune_<type>_r<R>_l<L>`. TAG = 1 keeps them distinct from the production instances.
#define TUNE1(NAME, DQ, R, L) \
template [[host_name("tune_" #NAME "_r" #R "_l" #L)]] kernel matmul_t k_matmul<DQ, 1, R, L, 1>;
#define TUNE(NAME, DQ) \
TUNE1(NAME, DQ, 1, 1) TUNE1(NAME, DQ, 1, 2) TUNE1(NAME, DQ, 1, 4) \
TUNE1(NAME, DQ, 2, 1) TUNE1(NAME, DQ, 2, 2) TUNE1(NAME, DQ, 2, 4) \
TUNE1(NAME, DQ, 4, 1) TUNE1(NAME, DQ, 4, 2) TUNE1(NAME, DQ, 4, 4)
TUNE(q8_0, DQ_Q8_0)
TUNE(q4_k, DQ_Q4_K)
TUNE(q6_k, DQ_Q6_K)

// ---------------------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------------------

// Sum / max over a threadgroup; every thread gets the result. `buf` holds >= 32 floats.
inline float tg_sum(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? buf[lane] : 0.0f;
    return simd_sum(v);
}

inline float tg_max(float v, threadgroup float *buf, ushort sg, ushort lane, ushort nsg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? buf[lane] : -INFINITY;
    return simd_max(v);
}

struct NormParams { uint dim; float eps; };

// One threadgroup per row: out = x * w / sqrt(mean(x^2) + eps). x and out may alias.
kernel void k_rmsnorm(device const float *x     [[buffer(0)]],
                      device const float *w     [[buffer(1)]],
                      device float *out         [[buffer(2)]],
                      constant NormParams &p    [[buffer(3)]],
                      uint row                  [[threadgroup_position_in_grid]],
                      uint tid                  [[thread_position_in_threadgroup]],
                      uint ntg                  [[threads_per_threadgroup]],
                      ushort sg                 [[simdgroup_index_in_threadgroup]],
                      ushort lane               [[thread_index_in_simdgroup]],
                      ushort nsg                [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    device const float *xr = x + (ulong)row * p.dim;
    device float *orow = out + (ulong)row * p.dim;
    float ss = 0.0f;
    for (uint i = tid; i < p.dim; i += ntg) { float v = xr[i]; ss += v * v; }
    ss = tg_sum(ss, buf, sg, lane, nsg);
    const float rinv = 1.0f / precise::sqrt(ss / float(p.dim) + p.eps);
    for (uint i = tid; i < p.dim; i += ntg) orow[i] = xr[i] * rinv * w[i];
}

// One threadgroup per row: upstream's softmax_row (max from -1e30, exp, sum, divide).
kernel void k_softmax(device float *x           [[buffer(0)]],
                      constant uint &cols       [[buffer(1)]],
                      uint row                  [[threadgroup_position_in_grid]],
                      uint tid                  [[thread_position_in_threadgroup]],
                      uint ntg                  [[threads_per_threadgroup]],
                      ushort sg                 [[simdgroup_index_in_threadgroup]],
                      ushort lane               [[thread_index_in_simdgroup]],
                      ushort nsg                [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[32];
    device float *r = x + (ulong)row * cols;
    float m = -1e30f;
    for (uint i = tid; i < cols; i += ntg) if (r[i] > m) m = r[i];
    m = tg_max(m, buf, sg, lane, nsg);
    float s = 0.0f;
    for (uint i = tid; i < cols; i += ntg) { float e = precise::exp(r[i] - m); r[i] = e; s += e; }
    s = tg_sum(s, buf, sg, lane, nsg);
    for (uint i = tid; i < cols; i += ntg) r[i] = r[i] / s;
}

struct TopKParams { uint cols, k, renormalize; };

// One simdgroup per row. Strict '>' scan from -1e30, lowest index wins ties; when nothing
// beats -1e30 (all NaN / -inf) the first untaken column is taken, as the CPU backend does.
kernel void k_topk(device const float *scores   [[buffer(0)]],
                   device int *ids              [[buffer(1)]],
                   device float *weights        [[buffer(2)]],
                   constant TopKParams &p       [[buffer(3)]],
                   uint row                     [[threadgroup_position_in_grid]],
                   ushort lane                  [[thread_index_in_simdgroup]]) {
    device const float *r = scores + (ulong)row * p.cols;
    device int *ir = ids + (ulong)row * p.k;
    device float *wr = weights + (ulong)row * p.k;
    ulong taken = 0;   // bit c: column lane + 32 c (cols <= 2048)
    float sum = 0.0f;
    for (uint kk = 0; kk < p.k; ++kk) {
        float bv = -1e30f; int bi = INT_MAX; int firstFree = INT_MAX;
        uint c = 0;
        for (uint e = lane; e < p.cols; e += 32, ++c) {
            if ((taken >> c) & 1) continue;
            if (firstFree == INT_MAX) firstFree = int(e);
            float v = r[e];
            if (v > bv) { bv = v; bi = int(e); }
        }
        const float m = simd_max(bv);
        int best;
        if (m > -1e30f) best = simd_min(bv == m ? bi : INT_MAX);
        else best = simd_min(firstFree);
        if ((uint)best % 32 == lane) taken |= 1ul << ((uint)best / 32);
        if (lane == 0) {
            float v = r[best];
            ir[kk] = best;
            wr[kk] = v;
            sum += v;
        }
    }
    if (p.renormalize && lane == 0) {
        for (uint kk = 0; kk < p.k; ++kk) wr[kk] = wr[kk] / sum;
    }
}

// ---------------------------------------------------------------------------------------
// Rotary embedding, KV cache, attention
// ---------------------------------------------------------------------------------------

struct RopeArgs { uint rows, heads, headDim, ropeDims, posStart; float theta; };

// grid (ropeDims / 2, heads, rows); pairs (j, j + ropeDims / 2), in place.
kernel void k_rope_neox(device float *x         [[buffer(0)]],
                        constant RopeArgs &p    [[buffer(1)]],
                        uint3 gid               [[thread_position_in_grid]]) {
    const uint hd2 = p.ropeDims / 2;
    const uint j = gid.x, h = gid.y, r = gid.z;
    if (j >= hd2 || h >= p.heads || r >= p.rows) return;
    const float inv = precise::pow(p.theta, -2.0f * float(j) / float(p.ropeDims));
    const float ang = float(p.posStart + r) * inv;
    const float c = precise::cos(ang), s = precise::sin(ang);
    device float *v = x + ((ulong)r * p.heads + h) * p.headDim;
    const float a = v[j], b = v[j + hd2];
    v[j] = a * c - b * s;
    v[j + hd2] = b * c + a * s;
}

struct KVStoreArgs { uint rows, kvHeads, dim, maxTokens, layer, posStart; };

template<typename T>
kernel void k_store_kv(device const float *src  [[buffer(0)]],
                       device T *dst            [[buffer(1)]],
                       constant KVStoreArgs &p  [[buffer(2)]],
                       uint gid                 [[thread_position_in_grid]]) {
    const uint total = p.rows * p.kvHeads * p.dim;
    if (gid >= total) return;
    const uint d = gid % p.dim, g = (gid / p.dim) % p.kvHeads, s = gid / (p.dim * p.kvHeads);
    dst[((ulong)(p.layer * p.kvHeads + g) * p.maxTokens + p.posStart + s) * p.dim + d] = T(src[gid]);
}

template [[host_name("store_kv_f32")]] kernel void k_store_kv<float>(device const float *, device float *, constant KVStoreArgs &, uint);
template [[host_name("store_kv_f16")]] kernel void k_store_kv<half>(device const float *, device half *, constant KVStoreArgs &, uint);

struct AttnArgs { uint heads, kvHeads, headDim, valueDim, maxTokens, layer, posStart; float scale; uint splits, chunk; };

// Causal GQA attention. One threadgroup (ATT_NSG simdgroups) per (head, query row, split).
// Each simdgroup takes chunks of 32 positions (lane = position for the scores, lane =
// output dimension for the value sum) with an online softmax; the simdgroups merge at the
// end. With p.splits == 1 the threadgroup covers every position and writes the output.
// With p.splits > 1 (decode at long context: few heads, many positions) split z covers
// positions [z * chunk, (z + 1) * chunk) and writes a partial (max, sum, unnormalised
// values) to `part`, which k_attention_merge combines.
template<typename T>
kernel void k_attention(device const float *q   [[buffer(0)]],
                        device const T *K       [[buffer(1)]],
                        device const T *V       [[buffer(2)]],
                        device float *out       [[buffer(3)]],
                        constant AttnArgs &p    [[buffer(4)]],
                        device float *part      [[buffer(5)]],
                        uint3 tg                [[threadgroup_position_in_grid]],
                        uint tid                [[thread_index_in_threadgroup]],
                        ushort sg               [[simdgroup_index_in_threadgroup]],
                        ushort lane             [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[ATT_MAXD / 4];
    threadgroup float sm[ATT_NSG], sl[ATT_NSG];
    threadgroup float so[ATT_NSG * ATT_MAXD];
    const uint h = tg.x, s = tg.y, z = tg.z;
    const uint hd = p.headDim, vd = p.valueDim;
    {
        device const float *qv = q + ((ulong)s * p.heads + h) * hd;
        threadgroup float *qs = (threadgroup float *)qs4;
        for (uint i = tid; i < hd; i += ATT_NSG * 32) qs[i] = qv[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint g = h / (p.heads / p.kvHeads);
    const uint n = p.posStart + s + 1;
    const uint lo = z * p.chunk;
    const uint hi = p.splits == 1 ? n : min(n, lo + p.chunk);
    device const T *Kh = K + ((ulong)(p.layer * p.kvHeads + g) * p.maxTokens) * hd;
    device const T *Vh = V + ((ulong)(p.layer * p.kvHeads + g) * p.maxTokens) * vd;
    float m = -INFINITY, l = 0.0f;
    float o[ATT_MAXD / 32];
    for (int c = 0; c < ATT_MAXD / 32; ++c) o[c] = 0.0f;
    for (uint c0 = lo + sg * 32; c0 < hi; c0 += ATT_NSG * 32) {
        const uint t = c0 + lane;
        float score = -INFINITY;
        if (t < hi) {
            device const vec<T, 4> *kr = (device const vec<T, 4> *)(Kh + (ulong)t * hd);
            float acc = 0.0f;
            for (uint i = 0; i < hd / 4; ++i) acc += dot(float4(kr[i]), qs4[i]);
            score = acc * p.scale;
        }
        const float nm = max(m, simd_max(score));
        const float corr = precise::exp(m - nm);
        const float pe = t < hi ? precise::exp(score - nm) : 0.0f;
        l = l * corr + simd_sum(pe);
        for (int c = 0; c < ATT_MAXD / 32; ++c) o[c] *= corr;
        const uint cnt = min(32u, hi - c0);
        for (uint tt = 0; tt < cnt; ++tt) {
            const float pt = simd_shuffle(pe, (ushort)tt);
            device const T *vr = Vh + (ulong)(c0 + tt) * vd;
            for (int c = 0; c < ATT_MAXD / 32; ++c) {
                const uint i = lane + 32 * c;
                if (i < vd) o[c] += pt * float(vr[i]);
            }
        }
        m = nm;
    }
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    for (int c = 0; c < ATT_MAXD / 32; ++c) {
        const uint i = lane + 32 * c;
        if (i < vd) so[sg * ATT_MAXD + i] = o[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -INFINITY;
    for (int k = 0; k < ATT_NSG; ++k) M = max(M, sm[k]);
    float f[ATT_NSG];
    float L = 0.0f;
    for (int k = 0; k < ATT_NSG; ++k) {
        f[k] = sl[k] > 0.0f ? precise::exp(sm[k] - M) : 0.0f;
        L += sl[k] * f[k];
    }
    if (p.splits == 1) {
        device float *ov = out + ((ulong)s * p.heads + h) * vd;
        for (uint i = tid; i < vd; i += ATT_NSG * 32) {
            float acc = 0.0f;
            for (int k = 0; k < ATT_NSG; ++k) acc += so[k * ATT_MAXD + i] * f[k];
            ov[i] = acc / L;
        }
    } else {
        // partial: [max, sum, vd unnormalised values]; an empty range leaves (-inf, 0, 0...)
        device float *pv = part + (((ulong)s * p.heads + h) * p.splits + z) * (vd + 2);
        if (tid == 0) { pv[0] = M; pv[1] = L; }
        for (uint i = tid; i < vd; i += ATT_NSG * 32) {
            float acc = 0.0f;
            for (int k = 0; k < ATT_NSG; ++k) acc += so[k * ATT_MAXD + i] * f[k];
            pv[2 + i] = acc;
        }
    }
}

template [[host_name("attention_f32")]] kernel void k_attention<float>(
    device const float *, device const float *, device const float *, device float *,
    constant AttnArgs &, device float *, uint3, uint, ushort, ushort);
template [[host_name("attention_f16")]] kernel void k_attention<half>(
    device const float *, device const half *, device const half *, device float *,
    constant AttnArgs &, device float *, uint3, uint, ushort, ushort);

// Combines k_attention's per-split partials: one threadgroup per (head, row), splits in
// order (a fixed order, so the result does not depend on scheduling).
kernel void k_attention_merge(device const float *part [[buffer(0)]],
                              device float *out        [[buffer(1)]],
                              constant AttnArgs &p     [[buffer(2)]],
                              uint2 tg                 [[threadgroup_position_in_grid]],
                              uint2 tid2               [[thread_position_in_threadgroup]],
                              uint2 nt2                [[threads_per_threadgroup]]) {
    const uint h = tg.x, s = tg.y, vd = p.valueDim, tid = tid2.x, nt = nt2.x;
    device const float *pv = part + (((ulong)s * p.heads + h) * p.splits) * (vd + 2);
    float M = -INFINITY;
    for (uint z = 0; z < p.splits; ++z) M = max(M, pv[z * (vd + 2)]);
    float L = 0.0f;
    for (uint z = 0; z < p.splits; ++z) {
        device const float *pz = pv + z * (vd + 2);
        if (pz[1] > 0.0f) L += pz[1] * precise::exp(pz[0] - M);
    }
    device float *ov = out + ((ulong)s * p.heads + h) * vd;
    for (uint i = tid; i < vd; i += nt) {
        float acc = 0.0f;
        for (uint z = 0; z < p.splits; ++z) {
            device const float *pz = pv + z * (vd + 2);
            if (pz[1] > 0.0f) acc += pz[2 + i] * precise::exp(pz[0] - M);
        }
        ov[i] = acc / L;
    }
}

// ---------------------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------------------

kernel void k_silu_mul(device const float *g    [[buffer(0)]],
                       device const float *u    [[buffer(1)]],
                       device float *o          [[buffer(2)]],
                       constant uint &n         [[buffer(3)]],
                       uint i                   [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float gv = g[i];
    o[i] = (gv / (1.0f + precise::exp(-gv))) * u[i];
}

kernel void k_add(device const float *a         [[buffer(0)]],
                  device const float *b         [[buffer(1)]],
                  device float *o               [[buffer(2)]],
                  constant uint &n              [[buffer(3)]],
                  uint i                        [[thread_position_in_grid]]) {
    if (i < n) o[i] = a[i] + b[i];
}

struct WSumArgs { uint rows, terms, dim, accumulate; };

// out[r][i] (+)= sum_k w[r][k] * x[r][k][i], k in rank order.
kernel void k_weighted_sum(device const float *x   [[buffer(0)]],
                           device const float *w   [[buffer(1)]],
                           device float *o         [[buffer(2)]],
                           constant WSumArgs &p    [[buffer(3)]],
                           uint gid                [[thread_position_in_grid]]) {
    const uint r = gid / p.dim, i = gid % p.dim;
    if (r >= p.rows) return;
    float acc = 0.0f;
    for (uint k = 0; k < p.terms; ++k) acc += w[r * p.terms + k] * x[((ulong)r * p.terms + k) * p.dim + i];
    const ulong oi = (ulong)r * p.dim + i;
    o[oi] = p.accumulate ? o[oi] + acc : acc;
}

kernel void k_copy_u32(device const uint *s     [[buffer(0)]],
                       device uint *d           [[buffer(1)]],
                       constant uint &n         [[buffer(2)]],
                       uint i                   [[thread_position_in_grid]]) {
    if (i < n) d[i] = s[i];
}

kernel void k_copy_u16(device const ushort *s   [[buffer(0)]],
                       device ushort *d         [[buffer(1)]],
                       constant uint &n         [[buffer(2)]],
                       uint i                   [[thread_position_in_grid]]) {
    if (i < n) d[i] = s[i];
}
"""#
}
