---
title: "Python plane P3: numerics certification enabling an opt-in relaxed-reduction opcode family"
number: 77
labels: enhancement,correctness,design,python-plane,doctrine
plane: Python plane
origin: Frontier bridge
priority: P1
---
## Doctrinal root cause

The bitwise-determinism contract makes every reduction accumulate in
**f64, in fixed chunk order** — half the SIMD width plus a convert per
element, on every norm, loss, softmax, and attention score. The Frontier
Bridge prices this at **up to ≈2× on reductions**; the largest single
instance is attention's QK^T dot, an O(B·H·S²·d) loop
(`attention.cc:110-112`), which the Performance Audit flagged (finding #9)
as "plausibly ~2× on the hottest transformer loop — measure-first policy
change, not a fix". torch reassociates freely; SeeAI cannot — *by
doctrine, not by accident*. The audits' conclusion: spend this doctrine
**consciously** or not at all.

## Python-plane design

Relaxation becomes a *certified, opt-in* path, never a silent default:

- **Core plane:** a relaxed-reduction opcode family (vectorized f32
  accumulation, reassociation allowed *within* a backend's fixed schedule)
  that the default compilation path **never emits**. Emitted only under
  `--relaxed-reductions`, recorded in the plan header so the validator,
  report, and regression gate all know which contract they are checking.
  Per-backend reproducibility is retained (same G1a re-scoping that
  admitted the GPU); only cross-schedule bit-equality is traded.
- **Python plane:** `tool/certify_numerics.py` runs the bit-exact
  reference and the relaxed kernels over plan-specific operand
  distributions (drawn from the actual SMF weights + SDS corpus), computes
  error bounds, and writes a **tolerance certificate** into the package
  metadata. No certificate, no relaxed plan.
- Absorbs Performance Audit finding #9 (f32 attention-score accumulation)
  as the first certified candidate.

## Acceptance

- Default path byte-identical to today; `seeml_seeu_dump` shows zero
  relaxed opcodes in a default compile.
- A certified relaxed plan demonstrates the reduction speedup on Tier B/C
  fixtures and trains within its certificate's bounds.
- The commit gate still holds: loss must improve or the device is
  untouched, under either contract.

Refs: Frontier Bridge §02 (reduction row, conclusion); Performance Audit
§04 finding #9; depends on P4 (the frontier executor is the natural
certification oracle).
