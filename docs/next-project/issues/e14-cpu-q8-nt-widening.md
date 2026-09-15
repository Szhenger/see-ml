---
title: "Core plane E14: GemmNTQ8 has run scalar since #66 — the int8 cast inside the lane loop defeats vectorization (7× below the f32 form); widen each block first, bit-identical"
labels: bug,efficiency,core-plane,sev:medium
plane: Core plane
origin: Field report
priority: P1
---
## Algorithmic root cause

#66 (merged in #99) rewrote `BlockedNT` (`runtime/executor/gemm.cc`) so
the K reduction runs in eight fixed lanes, and the f32 `GemmNT` gained
3–5× as intended. The same template serves `GemmNTQ8`, and there the
lane loop reads

```cpp
acc0[l] += a * static_cast<float>(b0[k + l]);
```

With `BType = int8_t`, clang refuses to vectorize a multiply-accumulate
whose operand is a widening cast in the same statement, so the whole loop
runs scalar and the four `acc[8]` arrays are worked in memory. The q8
form — the dX backward through every frozen projection under
`--quantize-base`, the frontier configuration — is therefore *slower*
after #66 than before it, while every f32 fixture got faster. The
bf16 form is unaffected (its widening is a shift).

Measured 2026-09-15 on the merged v1.3.0 tree (Apple M5, Apple clang 21,
`-O2`, kernel microbench over the runtime's own `GemmNT*` entry points):

| shape (M×N×K) | threads | NT f32 | NT bf16 | NT q8 |
|---|---|---|---|---|
| 256×576×1536 (SmolLM dX) | 10 | 214 GF/s | 227 GF/s | **30 GF/s** |
| 256×576×1536 | 1 | 28.4 | 33.0 | **4.3** |
| 512×1536×576 | 10 | 164 | 217 | **28.5** |
| 128×49152×576 (lm-head dX) | 10 | 87 | 109 | **27.8** |

The 4.3 GF/s serial q8 rate is the pre-#66 serial chain's rate: for int8
B, #66 changed the reduction order and delivered none of the speedup.

## Why it was invisible

- The nightly fixtures are f32 (`mlp_*`, `dec_*`, `tok_*`); the only q8
  fixture, `tok_smollm135m_q8`, is opt-in and never runs in CI.
- The v1.3.0 field report measured SmolLM on the CPU backend at 65 tok/s
  against the v1.2.4 report's 80 tok/s (an M4) and could not separate
  host, geometry and kernel. A bisect on one host and one plan (S=64,
  r16, batch 256, 10 threads) decomposes it: v1.2.4 code 2.35 s/step,
  cbdc5b2 2.46, **7957aff (#99, carrying #66) 3.58**, 263be4a 3.66. Of
  the −15 tok/s: host +29 (the M5 is faster), **this regression −36**,
  S=64→128 −8. SmolLM's backward is 208 q8 NT GEMMs per step — the
  bwd/fwd ratio of 5.3× on the frontier fixture is this kernel.

## Fix (bit-identical)

Widen each eight-wide block of B into a local `float w[8]` in its own
lane loop, then run the multiply-accumulate lane loop over `w`. The lane
assignment (`k mod 8`), the per-lane accumulation order and the combine
order are untouched, and `static_cast<float>` of an int8 or a bf16 is
exact — so the products, the sums and every C bit are the same as
today's kernel. Two compiler-facing constraints, both measured:

- the widening loop must be written inline — through a helper that takes
  the block by pointer, clang keeps `w` in memory and the f32 form loses
  its vectorization as well;
- the f32 instantiation must keep its single-loop body (`if constexpr`
  on the element type): a no-op widening pass through a local array
  costs the f32 kernel 3× under the same compiler.

Expected: q8 NT ≥ the f32 NT rate at every shape (the int8 B stream is
4× narrower), the frontier fixture's CPU row up from 65 tok/s, and
`--backend cpu` bit-identical to main (assert with the kernel goldens and
a full SmolLM step: same committed bytes).

## Verification

- Kernel goldens: `GemmNTQ8` and `GemmNTBF16` over ragged shapes (K % 8
  ≠ 0, N % 4 ≠ 0) hash-identical before and after at 1 and 10 threads.
- `seeml_kernels_test`, `seeml_update_engine_test` (thread-count
  invariance), `seeml_metal_backend_test` (the GPU twin compares against
  this CPU kernel at tolerance) all green.
- `seeml-bench --backend cpu --fixtures tok_smollm135m_q8` row before and
  after, same host, cooled.
- G13 (#97) already asks for a `GemmNTQ8` golden in the suite; this issue
  is the reason it should also be a *rate* row in the nightly set — a q8
  fixture small enough for CI, so the next silent scalarization is a
  −10% gate failure instead of a field-report footnote.
