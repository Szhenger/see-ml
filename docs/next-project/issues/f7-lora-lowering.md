---
title: "SeeAI F7: LoRA lowering — fold the rank-r scale into the GEMM alpha, batch the A-projections, lift the small-N Metal kernels off the occupancy floor, then fold B(Ax) into the base GEMM"
number: 138
labels: enhancement,efficiency,core-plane,frontier-parity
plane: Core plane
milestone: SeeAI v1.0.0.A
priority: P0
---
## Root cause: the LoRA lowering's structure (2026-09-24 systems review)

`compiler/analysis/algebra/lora_grafter.cc` lowers every adapted
projection `C = X @ W` into four separate ops: `t = X @ A` ([N, r]), `t' =
(α/r) · t` (`sc_high.scale`), `u = t' @ B` ([N, M]), `C += u`. On the
SmolLM-135M package that is 211 adapters, and on Metal the rank-8 family is
a family of kernels at the occupancy floor, not a FLOP cost. From the
per-opcode profile (`out/frontier-2026-09-22/prof_q8.txt`, kernel spans
per training step, estimated):

| family | fwd ms | bwd ms | share of kernel time | share of FLOPs |
|---|---:|---:|---:|---:|
| frozen q8 GEMMs (block + head) | 51 | 53 | ≈50 % | ≈97 % |
| rank-8 LoRA GEMMs + scale + add | 32 | 43 | **≈36 %** | ≈3 % |

- `k_gemm_nn_small+k_add_ew 512x1536x8` runs 158 µs: 3 MB of f32 moved
  at ≈40 GB/s, a quarter of the M5's bandwidth — occupancy-bound, not
  bandwidth-bound. `k_gemm_nn_rows 512x8x576` (the A-projection) is 37 µs,
  5,430 dispatches over the profile.
- The head's LoRA (`512x49152x8` + add, 5.1 ms) costs 60 % of the head
  GEMM itself (8.5 ms).
- `k_scale` on 4,096 elements (512 × r) is dispatched 7,174 times over the
  profile at 2.7 µs each: a whole kernel for a scalar the GEMM's `alpha`
  argument already carries (`KArgs.f[0]`).
- The profile over-weights the forward (30 forward passes to 4 backward,
  validation on); the shares above are per-step estimates and the first
  step of the work is a train-only profile (F9). The finding survives
  either way.

After F2 halves the frozen-GEMM half of the step, this family is the
largest term left.

**What is already done, and what this issue integrates.** E10 (#93,
closed, landed in #123) moved `α/r` off the activation-sized `dC` copy
onto the `[N, r]` operand in the backward and merged `dX += dh·Aᵀ`
through `gemm_acc` (on Metal: `k_gemm_nt_small+k_add_ew 512x576x8`, 712
rows of the profile). The 7,174 `k_scale` dispatches on 4,096 = 512 × r
elements are exactly the scales E10 left as their own op — the forward
`t' = (α/r)·t` the grafter still emits (`lora_grafter.cc`, "s == 1 needs
no op") and the backward's `dh` scale — so step 1 below is E10's fold
carried the last inch, into the GEMM's own alpha. (The profile's
`k_gemm_acc_small 576x1536x8` / `576x576x8` rows are the merge program's
`W += α·A@B` at commit, not E10's dX path.) `docs/roadmap.md` Project 5
already lists "the LoRA residual chain (four dispatches per adapter)"
and "fusing x·A·B per adapter" as what is left on the GPU side; this
issue is that remainder, now priced.

## Design — four steps, each measured on the Metal profile

1. **Scale into alpha** (compiler, bit-neutral where `alpha == 1`,
   otherwise a rounding change the certificate prices): the grafter emits
   no `scale` op; the `t @ B` GEMM carries `α/r` as its alpha, the
   backward's `dC` scaling likewise. Removes every `k_scale` dispatch.
2. **Batch the A-projections** per block: `X @ [A_q | A_k | A_v]` is one
   512×24×576 GEMM instead of three 512×8×576; gate/up likewise. Same
   per-element reduction order (each column's K-sum is unchanged), so
   bit-identical.
3. **Small-N / small-K Metal kernels** rewritten for occupancy: the
   `nn_small`, `nn_rows`, `nt_small`, `tn_rows`, `tn_colsg` family at N or
   K ≤ 16 splits rows across simdgroups and vectorizes the add-epilogue;
   target ≥ 100 GB/s effective on `512x1536x8+add`. Bit-identical: the
   per-output K order does not change.
4. **Fold B(Ax) into the base GEMM** as a K-extension (`[X | t'] @ [W ;
   B]`, K = 576 + r) or as a second accumulate in the epilogue. This
   changes the reduction order of `C`, so it is a new opcode / flag the
   old path never emits, admitted under the same certificate as F2's
   relaxed family; the four-op lowering stays the exact reference.

## Acceptance

- Train-only Metal profile of SmolLM-135M: the LoRA family's share of
  kernel time reported before/after each step; steps 1–3 bit-identical
  committed models against main; step 4 certified.
- SmolLM-135M `metal` tok/s and the F2 relaxed package's tok/s reported
  against F1's MLX-LM rows.
- The head's LoRA cost reported as a fraction of the head GEMM.
