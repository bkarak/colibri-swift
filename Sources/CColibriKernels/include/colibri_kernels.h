// Integer inner loops that Swift's optimiser does not vectorise. The 16-wide int8 dot
// of the integer activation path (IntegerDot.swift) compiles to scalar multiply-adds in
// Swift (LLVM unrolls the 16-trip loop before it can vectorise it); here it is one NEON
// `sdot` per 16 values. Integer sums are exact in any order, so results are unchanged.
// Only integer code lives here: float arithmetic stays in Swift, where it is not fused
// into FMAs (clang contracts C float expressions by default).
#ifndef COLIBRI_KERNELS_H
#define COLIBRI_KERNELS_H

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

#endif
