---
title: "SeeAI F4: CPU matrix units — SME kernels (M4+) or a certified Accelerate exception for the frozen-weight GEMMs"
number: 132
labels: enhancement,efficiency,core-plane,doctrine,frontier-parity
plane: Core plane
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P1
---
## Why

SeeAI's own SIMD GEMMs peak at 332 GFLOP/s (8 threads, E1 #80) and the
SmolLM-135M CPU row is 332 tok/s on the real package (the 2026-09-22
frontier row in `docs/benchmarks.md`, 10 threads; the `tok_smollm135m_q8`
fixture under-reads it at 233–269, #105 / E1), 0.39× of `torch.compile`'s
847 on the same host. PyTorch reaches the AMX/SME
matrix units through Accelerate: even interpreted op by op it runs the
dec_wide plan 2.5× faster than SeeAI's CPU backend (5,142 vs 2,060 tok/s,
`frontier_exec.py price`, 2026-09-17). The speculated gap to
`torch.compile` on the CPU is 2–3×.

## Options (decide in this issue)

- **A. Native SME kernels** (Apple M4 and later, Armv9 SME/SME2): owned
  code, fixed accumulation order, so the result can stay bit-identical run
  to run; a new opcode family if the per-element order differs from the
  NEON kernels; runtime capability check with the NEON path as fallback.
- **B. Accelerate exception**: `cblas_sgemm` for frozen-weight GEMMs only,
  behind the P3 certificate (F2's relaxed family), because Accelerate's
  reduction order is not ours and may change across OS releases —
  determinism holds per OS build, not across them. Packages record the
  requirement.

Recommendation: A first — it keeps the determinism contract whole; B only
if A cannot reach ≥ 2× on M4/M5.

## Integrated from the 2026-09-24 systems review

Root cause 1, *arithmetic doctrine*, CPU side — measured on #136 (option
B, the Accelerate exception): the f32-base SmolLM-135M package on
Accelerate's SGEMM runs 681–708 ms/step (724–752 tok/s) against the exact
kernels' 1,830 ms (280 tok/s), 2.6×, i.e. **0.85–0.89× of
`torch.compile`'s 847**, validation equal to six decimals, two runs
byte-identical. But only an f32-weight relaxed GEMM takes the Accelerate
path; int8 and bf16 bases stay on the portable NEON kernels (#136), and
the int8 package is the one that ships (rodata 262 MB vs a 741 MB f32
plan). So the shipped package still trains at 332 tok/s, 0.39×.

Executable next step: give the int8 base the same exception — widen a
K×nc panel of the int8 weight into the arena-planned pack segment (E1's
buffer, already f32) and hand that panel to SGEMM, or SME kernels that
consume int8 directly (option A). Measure against the real-package row
(332 tok/s), never the `tok_smollm135m_q8` fixture, which under-reads it
(233–269; F9).

## Acceptance

- GEMM GFLOP/s at SmolLM shapes (q8 and f32, NN/NT/TN) vs the committed
  kernels, timed against the committed object.
- SmolLM-135M CPU tok/s vs F1's `torch.compile` CPU row; target ≥ 0.8×.
- Determinism: serial vs 8-thread committed models byte-identical on the
  new path.
