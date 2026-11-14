# Quant golden data (workstream A)

`<TYPE>.golden` files hold random blocks of each ggml type the port decodes, and what
upstream colibri's `c/ggml_quants.h` (commit 0a5504a) computes from them: the bit-exact
dequantisation, the integer-path matvec (`gq_matmul`, int8 activations, NEON `sdot`) and
the float-path matvec (`gq_matmul_f32`, = `GQ_EXACT=1`). The layout is in the header
comment of `gen_golden.c`; `QuantTests.swift` reads them.

Regenerate (native clang, no extra tools):

```
clang -O2 -ffp-contract=off -I <colibri>/c gen_golden.c -o /tmp/gen_golden -lm
env -u GQ_EXACT /tmp/gen_golden .
```

The generator is seeded, so the files are identical on every run.
