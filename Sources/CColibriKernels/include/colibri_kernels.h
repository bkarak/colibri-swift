// Integer inner loops that Swift's optimiser does not vectorise. The 16-wide int8 dot
// of the integer activation path (IntegerDot.swift) compiles to scalar multiply-adds in
// Swift (LLVM unrolls the 16-trip loop before it can vectorise it); here it is one NEON
// `sdot` per 16 values. Integer sums are exact in any order, so results are unchanged.
// Float arithmetic in C needs contraction off (clang fuses C float expressions into FMAs
// by default): rows_i8.c sets `#pragma clang fp contract(off)`.
#ifndef COLIBRI_KERNELS_H
#define COLIBRI_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

/// `*s0 = w[0..<16] . x[0..<16]`, `*s1 = w[16..<32] . x[16..<32]`, exact in int32.
static inline void colibri_dot_i8_16x2(const int8_t *w, const int8_t *x, int32_t *s0, int32_t *s1) {
#if defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t z = vdupq_n_s32(0);
    *s0 = vaddvq_s32(vdotq_s32(z, vld1q_s8(w), vld1q_s8(x)));
    *s1 = vaddvq_s32(vdotq_s32(z, vld1q_s8(w + 16), vld1q_s8(x + 16)));
#else
    int32_t a = 0, b = 0;
    for (int i = 0; i < 16; i++) {
        a += (int32_t)w[i] * (int32_t)x[i];
        b += (int32_t)w[16 + i] * (int32_t)x[16 + i];
    }
    *s0 = a;
    *s1 = b;
#endif
}

/// 1 when `colibri_rows_i8` handles ggml type id `type` (Q4_0, Q4_1, Q8_0, Q2_K..Q6_K).
int colibri_rows_i8_supported(int type);

/// `y[o] = scale * dotQ8(row o, activations)` for `o < nrows`, rows `rowBytes` apart from
/// `base`; `xq`, `dx`, `xs` are one activation row of `Q8Activations` (`cols` codes, a
/// scale per 32, a code sum per 16). Bit-identical to IntegerDot.swift's `dotQ8`.
void colibri_rows_i8(int type, const uint8_t *base, size_t rowBytes, int nrows,
                     const int8_t *xq, const float *dx, const int32_t *xs, int cols, float scale, float *y);

/// A chunk of a parallel loop: `fn(ctx, i)` for chunk `i`.
typedef void (*colibri_chunk_fn)(void *ctx, int chunk);

/// Runs `fn(ctx, i)` for every `i < chunks` on a spinning worker pool (pool.c) plus the
/// calling thread, and returns when all have finished. The pool grows to `threads - 1`
/// workers. Serial when `threads <= 1`, from a worker, or while another loop runs.
void colibri_parallel_for(int chunks, int threads, colibri_chunk_fn fn, void *ctx);

#endif
