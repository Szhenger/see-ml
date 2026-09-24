---
title: "Core plane E10: two avoidable full-activation passes per adapted layer in the backward — fold α into the rank-r GEMMs, merge dX through gemm_acc"
labels: enhancement,efficiency,core-plane
plane: Core plane
origin: Algorithm review
priority: P1
---
## Algorithmic root cause

For every LoRA site the grafter emits `s = (α/r)·u` as a separate
`sc_high.scale` op (`compiler/analysis/algebra/lora_grafter.cc:142-150`).
Autodiff then materializes, per adapted matmul and per step:

1. a full **N×M** scaled copy of `dC` for the scale op's VJP
   (`compiler/analysis/calculus/autodiff.cc:280-293`);
2. a full **N×K** `sc_high.add` merging `dX_base = dC·Wᵀ` with
   `dX_lora = dh·Aᵀ` (fan-out accumulation, `autodiff.cc:44-51`).

Both are memory-bound passes over activation-sized tensors, in a runtime
that `elementwise.cc` itself declares bandwidth-bound. The adapter GEMMs
they feed are rank-r (tiny); the copies are full-size.

## Design — values unchanged, op order changed once

- Fold `α/r` into the two rank-r GEMMs (`dB = tᵀ·(α/r · dC)` becomes
  `dB = (α/r)·(tᵀ·dC)` on an `[r, M]` result; `dh = (α/r)·(dC·Bᵀ)` on an
  `[N, r]` result) — the scale moves to the small operand.
- Emit `dX += dh·Aᵀ` as the existing `sc_low.gemm_acc` (`C += α·A@B`,
  already used by the merge program) instead of `matmul_nt` + `add`.
- Per-element expression order changes for `dX` (one rounding difference
  in the accumulate), so this is **not** bitwise vs v1.2.4; it is
  deterministic across thread widths like every other kernel.
- Whole-program finite-difference checks must pass unchanged; the
  `TrainMergeCommit` system test re-baselines its golden bits.

## Acceptance

- Instruction-stream diff: per adapted matmul, one `kScale` and one
  `kAddEW` over activation-sized operands removed; arena high-water for
  the transient segment drops accordingly (report the delta).
- `seeml-bench` bwd/fwd ratio drops on decoder fixtures (measure; the
  Algorithm Review estimates the two passes at a low-single-digit share of
  a step on dec_mid, more on wide-D models).
- FD suite green; serial-vs-8-thread bit-identity holds.

## Goal alignment

Objective 3 / Frontier Bridge "fusion depth" row: Inductor and MLX never
materialize either copy (the scale folds into the small GEMM; the residual
add is an accumulate). Distinct from E4 (#83) chain fusion, which would
not remove the second GEMM's output pass.

Refs: SeeAI Algorithm Review (2026-09-14) §02-A; #83, #66.
