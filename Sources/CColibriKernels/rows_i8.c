// Integer-path matrix-vector rows (IntegerDot.swift's `dotQ8`) for the common block types,
// four weight rows at a time. Per row the result is bit for bit `dotQ8`'s:
//
// - the int8 codes are the ones `IntegerForm.unpack` produces, decoded here in NEON
//   registers straight from the packed block instead of through a scratch buffer;
// - the integer sums per 16 codes are exact (`sdot`, then a pairwise add tree);
// - the float steps per 32 codes are `dotQ8`'s, in its order, one row per vector lane:
//       v = dx * (sw0 * s0 + sw1 * s1)          (acc for even pairs, accB for odd)
//       m = fmaf(dx, fmaf(mw0, xs0, mw1 * xs1), m)
//       y = scale * ((acc + accB) - m)
//   and for Q8_0 `dotQ8_0Int`'s  acc[b & 1] += float(s) * (d * dx).
//
// Contraction is off for this file: a separate multiply and add must stay separate, and
// the fused steps are spelled with vfmaq.
#pragma clang fp contract(off)

#include "colibri_kernels.h"
#include <string.h>

#if defined(__ARM_FEATURE_DOTPROD)

enum {
    T_Q4_0 = 2, T_Q4_1 = 3, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
};

/// Right shift of every byte by a run-time count.
static inline __attribute__((always_inline)) uint8x16_t shr_u8(uint8x16_t v, int n) { return vshlq_u8(v, vdupq_n_s8((int8_t)-n)); }

/// Block geometry: bytes, codes, pairs of 16-groups (32 codes) per block.
static int geometry(int type, int *bytes, int *codes) {
    switch (type) {
    case T_Q4_0: *bytes = 18; *codes = 32; return 1;
    case T_Q4_1: *bytes = 20; *codes = 32; return 1;
    case T_Q8_0: *bytes = 34; *codes = 32; return 1;
    case T_Q2_K: *bytes = 84; *codes = 256; return 1;
    case T_Q3_K: *bytes = 110; *codes = 256; return 1;
    case T_Q4_K: *bytes = 144; *codes = 256; return 1;
    case T_Q5_K: *bytes = 176; *codes = 256; return 1;
    case T_Q6_K: *bytes = 210; *codes = 256; return 1;
    default: return 0;
    }
}

static inline __attribute__((always_inline)) int has_min(int type) {
    return type == T_Q4_1 || type == T_Q2_K || type == T_Q4_K || type == T_Q5_K;
}

/// The half-float at byte `off` of four blocks, as a float vector (exact conversion).
static inline __attribute__((always_inline)) float32x4_t half4(const uint8_t *const b[4], size_t off) {
    // Lane loads straight from the blocks: no stack round trip (a vector load of four
    // just-stored halves defeats store-to-load forwarding).
    uint16x4_t h = vdup_n_u16(0);
    h = vld1_lane_u16((const uint16_t *)(b[0] + off), h, 0);
    h = vld1_lane_u16((const uint16_t *)(b[1] + off), h, 1);
    h = vld1_lane_u16((const uint16_t *)(b[2] + off), h, 2);
    h = vld1_lane_u16((const uint16_t *)(b[3] + off), h, 3);
    return vcvt_f32_f16(vreinterpret_f16_u16(h));
}

/// 4 rows x 8 bytes -> lanes: out[0] = groups 0-3, out[1] = groups 4-7, each byte 4g+k
/// is group g of row k.
static inline __attribute__((always_inline)) void transpose4x8(uint8x8_t a0, uint8x8_t a1, uint8x8_t a2, uint8x8_t a3, uint8x16_t out[2]) {
    uint8x8x2_t z01 = vzip_u8(a0, a1), z23 = vzip_u8(a2, a3);
    uint16x4x2_t lo = vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
    uint16x4x2_t hi = vzip_u16(vreinterpret_u16_u8(z01.val[1]), vreinterpret_u16_u8(z23.val[1]));
    out[0] = vreinterpretq_u8_u16(vcombine_u16(lo.val[0], lo.val[1]));
    out[1] = vreinterpretq_u8_u16(vcombine_u16(hi.val[0], hi.val[1]));
}

/// 4 rows x 16 bytes -> out[q] holds groups 4q..4q+3, byte 4(g-4q)+k = group g of row k.
static inline __attribute__((always_inline)) void transpose4x16(uint8x16_t a0, uint8x16_t a1, uint8x16_t a2, uint8x16_t a3, uint8x16_t out[4]) {
    transpose4x8(vget_low_u8(a0), vget_low_u8(a1), vget_low_u8(a2), vget_low_u8(a3), out);
    transpose4x8(vget_high_u8(a0), vget_high_u8(a1), vget_high_u8(a2), vget_high_u8(a3), out + 2);
}

/// Groups (4 consecutive, 4 rows each) as floats: unsigned or signed bytes.
static inline __attribute__((always_inline)) void widen_u(uint8x16_t v, float32x4_t f[4]) {
    uint16x8_t a = vmovl_u8(vget_low_u8(v)), b = vmovl_u8(vget_high_u8(v));
    f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a))); f[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a)));
    f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b))); f[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b)));
}
static inline __attribute__((always_inline)) void widen_s(uint8x16_t u, float32x4_t f[4]) {
    int8x16_t v = vreinterpretq_s8_u8(u);
    int16x8_t a = vmovl_s8(vget_low_s8(v)), b = vmovl_s8(vget_high_s8(v));
    f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))); f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a)));
    f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))); f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b)));
}

/// Q4_K/Q5_K 6-bit scales and mins of one block (llama.cpp's mask form of `scaleMinK4`):
/// bytes 0-7 of `sc`, 0-7 of `mn`.
static inline __attribute__((always_inline)) void k4_scales(const uint8_t *s, uint8x8_t *sc, uint8x8_t *mn) {
    uint32_t u0, u1, u2;
    memcpy(&u0, s, 4); memcpy(&u1, s + 4, 4); memcpy(&u2, s + 8, 4);
    const uint64_t t0 = u0 & 0x3f3f3f3fu;
    const uint64_t t1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
    const uint64_t t2 = u1 & 0x3f3f3f3fu;
    const uint64_t t3 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
    *sc = vcreate_u8(t0 | (t1 << 32));
    *mn = vcreate_u8(t2 | (t3 << 32));
}

/// Q3_K's sixteen 6-bit scales minus 32 (llama.cpp's mask form), as signed bytes.
static inline __attribute__((always_inline)) uint8x16_t q3_scales(const uint8_t *s) {
    uint32_t a0, a1, a2;
    memcpy(&a0, s, 4); memcpy(&a1, s + 4, 4); memcpy(&a2, s + 8, 4);
    const uint64_t t0 = (a0 & 0x0f0f0f0fu) | (((a2 >> 0) & 0x03030303u) << 4);
    const uint64_t t1 = (a1 & 0x0f0f0f0fu) | (((a2 >> 2) & 0x03030303u) << 4);
    const uint64_t t2 = ((a0 >> 4) & 0x0f0f0f0fu) | (((a2 >> 4) & 0x03030303u) << 4);
    const uint64_t t3 = ((a1 >> 4) & 0x0f0f0f0fu) | (((a2 >> 6) & 0x03030303u) << 4);
    return vsubq_u8(vcombine_u8(vcreate_u8(t0 | (t1 << 32)), vcreate_u8(t2 | (t3 << 32))), vdupq_n_u8(32));
}

/// Per-16 scales `SW[g]` and mins `MW[g]` of one block of four rows, lane k = row k:
/// `IntegerForm.unpack`'s `d * Float(code)` per lane.
static inline __attribute__((always_inline)) void block_scales4(int type, const uint8_t *const b[4], float32x4_t SW[16], float32x4_t MW[16]) {
    switch (type) {
    case T_Q4_0: {
        SW[0] = SW[1] = half4(b, 0);
        return;
    }
    case T_Q4_1: {
        SW[0] = SW[1] = half4(b, 0);
        MW[0] = MW[1] = vnegq_f32(half4(b, 2));
        return;
    }
    case T_Q4_K:
    case T_Q5_K: {
        const float32x4_t d = half4(b, 0), dmin = half4(b, 2);
        uint8x8_t sc[4], mn[4];
        #pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) k4_scales(b[k] + 4, &sc[k], &mn[k]);
        uint8x16_t ts[2], tm[2];
        transpose4x8(sc[0], sc[1], sc[2], sc[3], ts);
        transpose4x8(mn[0], mn[1], mn[2], mn[3], tm);
        float32x4_t fs[8], fm[8];
        widen_u(ts[0], fs); widen_u(ts[1], fs + 4);
        widen_u(tm[0], fm); widen_u(tm[1], fm + 4);
        #pragma clang loop unroll(full)
        for (int g = 0; g < 8; g++) {
            SW[2 * g] = SW[2 * g + 1] = vmulq_f32(d, fs[g]);
            MW[2 * g] = MW[2 * g + 1] = vmulq_f32(dmin, fm[g]);
        }
        return;
    }
    case T_Q6_K: {
        const float32x4_t d = half4(b, 208);
        uint8x16_t t[4];
        transpose4x16(vld1q_u8(b[0] + 192), vld1q_u8(b[1] + 192), vld1q_u8(b[2] + 192), vld1q_u8(b[3] + 192), t);
        #pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {
            float32x4_t f[4];
            widen_s(t[q], f);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) SW[4 * q + i] = vmulq_f32(d, f[i]);
        }
        return;
    }
    case T_Q2_K: {
        const float32x4_t d = half4(b, 80), dmin = half4(b, 82);
        uint8x16_t r[4], ts[4], tm[4];
        #pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) r[k] = vld1q_u8(b[k]);
        const uint8x16_t lo = vdupq_n_u8(0x0F);
        transpose4x16(vandq_u8(r[0], lo), vandq_u8(r[1], lo), vandq_u8(r[2], lo), vandq_u8(r[3], lo), ts);
        transpose4x16(vshrq_n_u8(r[0], 4), vshrq_n_u8(r[1], 4), vshrq_n_u8(r[2], 4), vshrq_n_u8(r[3], 4), tm);
        #pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {
            float32x4_t fs[4], fm[4];
            widen_u(ts[q], fs); widen_u(tm[q], fm);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) {
                SW[4 * q + i] = vmulq_f32(d, fs[i]);
                MW[4 * q + i] = vmulq_f32(dmin, fm[i]);
            }
        }
        return;
    }
    case T_Q3_K: {
        const float32x4_t d = half4(b, 108);
        uint8x16_t t[4];
        transpose4x16(q3_scales(b[0] + 96), q3_scales(b[1] + 96), q3_scales(b[2] + 96), q3_scales(b[3] + 96), t);
        #pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {
            float32x4_t f[4];
            widen_s(t[q], f);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) SW[4 * q + i] = vmulq_f32(d, f[i]);
        }
        return;
    }
    }
}

/// Codes of 16-group `g` (16 consecutive weights) of a 256-code K block.
static inline __attribute__((always_inline)) int8x16_t group_codes_k(int type, const uint8_t *b, int g) {
    const int i0 = g * 16, c = i0 >> 7, r0 = i0 & 127;
    switch (type) {
    case T_Q6_K: {
        uint8x16_t lo = vandq_u8(shr_u8(vld1q_u8(b + c * 64 + (r0 & 63)), 4 * (r0 >> 6)), vdupq_n_u8(0x0F));
        uint8x16_t hi = vandq_u8(shr_u8(vld1q_u8(b + 128 + c * 32 + (r0 & 31)), 2 * (r0 >> 5)), vdupq_n_u8(3));
        return vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(lo, vshlq_n_u8(hi, 4)), vdupq_n_u8(32)));
    }
    case T_Q2_K: {
        const int sh = (i0 >> 5) & 3, j0 = i0 & 31;
        return vreinterpretq_s8_u8(vandq_u8(shr_u8(vld1q_u8(b + 16 + c * 32 + j0), 2 * sh), vdupq_n_u8(3)));
    }
    case T_Q3_K: {
        const int sh = (i0 >> 5) & 3, j0 = i0 & 31;
        uint8x16_t ql = vandq_u8(shr_u8(vld1q_u8(b + 32 + c * 32 + j0), 2 * sh), vdupq_n_u8(3));
        uint8x16_t hb = vandq_u8(shr_u8(vld1q_u8(b + j0), i0 >> 5), vdupq_n_u8(1));
        // ql - ((hb ^ 1) << 2)
        uint8x16_t sub = vshlq_n_u8(veorq_u8(hb, vdupq_n_u8(1)), 2);
        return vreinterpretq_s8_u8(vsubq_u8(ql, sub));
    }
    }
    return vdupq_n_s8(0);
}

/// Codes of pair `p` (32 consecutive weights, groups 2p and 2p+1) of one block.
static inline __attribute__((always_inline)) void pair_codes(int type, const uint8_t *b, int p, int8x16_t *a, int8x16_t *c) {
    switch (type) {
    case T_Q4_0: {
        uint8x16_t q = vld1q_u8(b + 2);
        *a = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(q, vdupq_n_u8(0x0F)), vdupq_n_u8(8)));
        *c = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(q, 4), vdupq_n_u8(8)));
        return;
    }
    case T_Q4_1: {
        uint8x16_t q = vld1q_u8(b + 4);
        *a = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
        *c = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
        return;
    }
    case T_Q8_0:
        *a = vld1q_s8((const int8_t *)b + 2);
        *c = vld1q_s8((const int8_t *)b + 18);
        return;
    case T_Q4_K:
    case T_Q5_K: {
        const int five = type == T_Q5_K;
        const uint8_t *q = b + (five ? 48 : 16) + (p >> 1) * 32;
        uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
        uint8x16_t v0, v1;
        if (p & 1) {
            v0 = vshrq_n_u8(q0, 4); v1 = vshrq_n_u8(q1, 4);
        } else {
            v0 = vandq_u8(q0, vdupq_n_u8(0x0F)); v1 = vandq_u8(q1, vdupq_n_u8(0x0F));
        }
        if (five) {
            uint8x16_t h0 = vandq_u8(shr_u8(vld1q_u8(b + 16), p), vdupq_n_u8(1));
            uint8x16_t h1 = vandq_u8(shr_u8(vld1q_u8(b + 32), p), vdupq_n_u8(1));
            v0 = vorrq_u8(v0, vshlq_n_u8(h0, 4)); v1 = vorrq_u8(v1, vshlq_n_u8(h1, 4));
        }
        *a = vreinterpretq_s8_u8(v0); *c = vreinterpretq_s8_u8(v1);
        return;
    }
    default:
        *a = group_codes_k(type, b, 2 * p);
        *c = group_codes_k(type, b, 2 * p + 1);
        return;
    }
}

/// Sums of the four lanes of each of a0..a3, in lanes 0..3.
static inline __attribute__((always_inline)) int32x4_t hsum4(int32x4_t a0, int32x4_t a1, int32x4_t a2, int32x4_t a3) {
    return vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
}

int colibri_rows_i8_supported(int type) {
    int bytes, codes;
    return geometry(type, &bytes, &codes);
}

/// The generic loop, specialised per type by the switch in `colibri_rows_i8` (the type
/// is a compile-time constant in each call).
static inline __attribute__((always_inline)) void rows_k(
    const int type, const uint8_t *base, size_t rowBytes, int nrows,
    const int8_t *xq, const float *dx, const int32_t *xs, int cols, float scale, float *y) {
    int bytes, codes;
    geometry(type, &bytes, &codes);
    const int ppb = codes / 32, nb = cols / codes;
    const int hm = has_min(type);
    const int32x4_t z = vdupq_n_s32(0);
    for (int o = 0; o < nrows; o += 4) {
        const uint8_t *r[4];
        #pragma clang loop unroll(full)
        for (int k = 0; k < 4; k++) r[k] = base + (size_t)(o + k < nrows ? o + k : nrows - 1) * rowBytes;
        float32x4_t acc = vdupq_n_f32(0), accB = vdupq_n_f32(0), m = vdupq_n_f32(0);
        float32x4_t SW[16], MW[16];
        for (int blk = 0; blk < nb; blk++) {
            const size_t off = (size_t)blk * bytes;
            const uint8_t *const rb[4] = { r[0] + off, r[1] + off, r[2] + off, r[3] + off };
            if (type != T_Q8_0) block_scales4(type, rb, SW, MW);
            // Fully unrolled (ppb is 1 or 8 per type): shift counts, the even/odd choice
            // and the SW/MW indices become constants.
#pragma clang loop unroll(full)
            for (int p = 0; p < ppb; p++) {
                const int P = blk * ppb + p;
                const int8x16_t x0 = vld1q_s8(xq + 32 * P), x1 = vld1q_s8(xq + 32 * P + 16);
                int32x4_t A[4], B[4];
                #pragma clang loop unroll(full)
                for (int k = 0; k < 4; k++) {
                    int8x16_t wa, wb;
                    pair_codes(type, rb[k], p, &wa, &wb);
                    A[k] = vdotq_s32(z, wa, x0);
                    B[k] = vdotq_s32(z, wb, x1);
                }
                const int32x4_t S0 = hsum4(A[0], A[1], A[2], A[3]);
                const int32x4_t S1 = hsum4(B[0], B[1], B[2], B[3]);
                if (type == T_Q8_0) {
                    // dotQ8_0Int: float(s) * (d * dx), accumulators by block parity.
                    float32x4_t v = vmulq_f32(vcvtq_f32_s32(vaddq_s32(S0, S1)), vmulq_n_f32(half4(rb, 0), dx[P]));
                    if (blk & 1) accB = vaddq_f32(accB, v); else acc = vaddq_f32(acc, v);
                    continue;
                }
                const float32x4_t t0 = vmulq_f32(SW[2 * p], vcvtq_f32_s32(S0));
                const float32x4_t t1 = vmulq_f32(SW[2 * p + 1], vcvtq_f32_s32(S1));
                const float32x4_t v = vmulq_n_f32(vaddq_f32(t0, t1), dx[P]);
                if (p & 1) accB = vaddq_f32(accB, v); else acc = vaddq_f32(acc, v);
                if (hm) {
                    const float xs0 = (float)xs[2 * P], xs1 = (float)xs[2 * P + 1];
                    const float32x4_t inner = vfmaq_n_f32(vmulq_n_f32(MW[2 * p + 1], xs1), MW[2 * p], xs0);
                    m = vfmaq_n_f32(m, inner, dx[P]);
                }
            }
        }
        float out[4];
        if (type == T_Q8_0) {
            vst1q_f32(out, vmulq_n_f32(vaddq_f32(acc, accB), scale));
        } else {
            vst1q_f32(out, vmulq_n_f32(vsubq_f32(vaddq_f32(acc, accB), m), scale));
        }
        const int n = nrows - o < 4 ? nrows - o : 4;
        for (int k = 0; k < n; k++) y[o + k] = out[k];
    }
}

void colibri_rows_i8(int type, const uint8_t *base, size_t rowBytes, int nrows,
                     const int8_t *xq, const float *dx, const int32_t *xs, int cols, float scale, float *y) {
    switch (type) {
    case T_Q4_0: rows_k(T_Q4_0, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q4_1: rows_k(T_Q4_1, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q8_0: rows_k(T_Q8_0, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q2_K: rows_k(T_Q2_K, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q3_K: rows_k(T_Q3_K, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q4_K: rows_k(T_Q4_K, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q5_K: rows_k(T_Q5_K, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    case T_Q6_K: rows_k(T_Q6_K, base, rowBytes, nrows, xq, dx, xs, cols, scale, y); return;
    }
}

#else

int colibri_rows_i8_supported(int type) { (void)type; return 0; }
void colibri_rows_i8(int type, const uint8_t *base, size_t rowBytes, int nrows,
                     const int8_t *xq, const float *dx, const int32_t *xs, int cols, float scale, float *y) {}

#endif
