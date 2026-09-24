---
title: "SeeAI S5: analysis restructure — calculus, algebra, statistics, topology, optimization; the reduction-order ledger; the seed handed into autodiff; the analysis screen"
number: 151
labels: enhancement,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

`compiler/analysis/` is `algebra`, `calculus`, `reviewer`, `updater` — two
names that say what the mathematics is and two that do not. Optimizer
synthesis sits in calculus though it is minimization, not differentiation;
the schedule lives in the driver; the grad-accumulation `1/G` seed is a
coupling hidden inside autodiff; precision assignment for `kFlagRelaxed` is
decided in the driver; and nothing records, per compiled plan, which passes
changed reduction order and under which certificate (E10, F7's fourth step,
the relaxed family all do). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §4.

## Design

1. Five directories: `calculus/` (autodiff, seed as a parameter, the masked
   VJP from S3, remat machinery for F5), `algebra/` (LoRA graft, merge, the
   three fusers, RoPE table, the attention segment mask), `statistics/`
   (quantization review, attention tiling, precision assignment, adapter
   policy), `topology/` (DCE, verify-after-every-pass, the ledger),
   `optimization/` (SGD/AdamW, schedule and horizon, clipping, the seed,
   8-bit moments). Pass manager at the root; `updater/` and `reviewer/`
   gone; conv lowering deleted (LM scope).
2. The **reduction-order ledger**: one line per pass — bit-identical to its
   input or not, and why — stored in the plan header (S6) and printed.
3. The **analysis screen**: trainable parameters and fraction, adjoints
   emitted, activation bytes saved for the backward, FLOPs
   forward/backward/optimizer, bytes of arena traffic removed per fusion,
   dead ops removed, the ledger.
4. Best-practice defaults from §4, changed in the recipe (S4) under the
   three rules: clipping 1.0, learning rate 1e-4 with cosine and warmup,
   α/√r scaling — each with a frontier row (F1) before it flips.

The LoRA lowering itself is F7 (#138) and is not restated here; F7 lands in
`algebra/`.

## Acceptance

- Every fixture compiles to a bit-identical plan before and after the move
  (directory change only), then the defaults flip in their own PR with the
  F1 rows attached.
- The ledger names every pass that changes arithmetic on the F2 relaxed
  package and none on an exact package.
- The analysis screen's numbers match `seeml-plan-probe --profile` for
  FLOPs and the arena binder for bytes.
