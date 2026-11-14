/* Golden data for the Swift block decoders and dot products, produced by upstream
 * colibri's own c/ggml_quants.h (commit 0a5504a). Build and run (see README.md):
 *
 *   clang -O2 -ffp-contract=off -I <colibri>/c gen_golden.c -o gen_golden -lm
 *   ./gen_golden <output dir>
 *
 * For each type: random blocks (seeded xorshift; the f16/bf16 scale fields get bit 14
 * cleared so they stay finite), forming a matrix of O = 4 rows by I = 512 columns, a
 * random activation x in [-1, 1), and upstream's results:
 *   deq     gq_dequant_row over the whole matrix   (bit-exact reference)
 *   y_int   gq_matmul, the default integer path    (int8 activations, NEON sdot)
 *   y_exact gq_matmul_f32, the GQ_EXACT=1 path     (dequantise + f32 dot)
 * File <NAME>.golden, little-endian:
 *   u32 type, u32 I, u32 O, u32 nbytes, u8 bytes[nbytes], f32 deq[O*I], f32 x[I],
 *   f32 y_int[O], f32 y_exact[O]
 */
#include <stdio.h>
#include "ggml_quants.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static void clear14(uint8_t *p) { p[1] &= (uint8_t)~0x40; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gen_golden <dir>\n"); return 2; }
    struct { int t; const char *name; int f16[2]; } T[] = {
        {GQ_F16, "F16", {-2, -1}}, {GQ_BF16, "BF16", {-2, -1}},
        {GQ_Q8_0, "Q8_0", {0, -1}}, {GQ_Q4_0, "Q4_0", {0, -1}}, {GQ_Q4_1, "Q4_1", {0, 2}},
        {GQ_Q5_0, "Q5_0", {0, -1}}, {GQ_Q5_1, "Q5_1", {0, 2}}, {GQ_IQ4_NL, "IQ4_NL", {0, -1}},
        {GQ_Q2_K, "Q2_K", {80, 82}}, {GQ_Q3_K, "Q3_K", {108, -1}}, {GQ_Q4_K, "Q4_K", {0, 2}},
        {GQ_Q5_K, "Q5_K", {0, 2}}, {GQ_Q6_K, "Q6_K", {208, -1}}, {GQ_IQ4_XS, "IQ4_XS", {0, -1}},
    };
    const int I = 512, O = 4;
    for (size_t k = 0; k < sizeof T / sizeof T[0]; k++) {
        const int t = T[k].t, bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
        const int64_t rb = gq_row_bytes(t, I), nbytes = rb * O;
        uint8_t *W = malloc(nbytes);
        for (int64_t i = 0; i < nbytes; i++) W[i] = (uint8_t)(xr() >> 24);
        if (T[k].f16[0] == -2) { for (int64_t i = 0; i < nbytes; i += 2) clear14(W + i); }
        else for (int64_t b = 0; b < nbytes / by; b++)
            for (int f = 0; f < 2; f++) if (T[k].f16[f] >= 0) clear14(W + b * by + T[k].f16[f]);
        (void)bl;
        float *deq = malloc(sizeof(float) * I * O), *x = malloc(sizeof(float) * I);
        float yi[4], ye[4];
        for (int o = 0; o < O; o++) gq_dequant_row(t, W + o * rb, deq + o * I, I);
        for (int i = 0; i < I; i++) x[i] = (float)((double)(xr() >> 11) / 9007199254740992.0 * 2.0 - 1.0);
        if (gq_matmul(yi, x, W, t, 1, I, O)) { fprintf(stderr, "gq_matmul failed for %s\n", T[k].name); return 1; }
        if (gq_matmul_f32(ye, x, W, t, 1, I, O)) { fprintf(stderr, "gq_matmul_f32 failed for %s\n", T[k].name); return 1; }
        char path[1024]; snprintf(path, sizeof path, "%s/%s.golden", argv[1], T[k].name);
        FILE *f = fopen(path, "wb"); if (!f) { perror(path); return 1; }
        uint32_t hdr[4] = {(uint32_t)t, (uint32_t)I, (uint32_t)O, (uint32_t)nbytes};
        fwrite(hdr, 4, 4, f); fwrite(W, 1, nbytes, f); fwrite(deq, 4, (size_t)I * O, f);
        fwrite(x, 4, I, f); fwrite(yi, 4, O, f); fwrite(ye, 4, O, f); fclose(f);
        free(W); free(deq); free(x);
    }
    return 0;
}
