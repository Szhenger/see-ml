---
title: "SeeAI CR1: a non-finite policy — NaN and Inf refused at the boundary: ingest, int8 selection and packing, clip norms on both backends, KL temperature, q8 scales, and a test comparison NaN cannot pass"
number: 142
labels: bug,correctness,core-plane,doctrine
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

Ten findings of the 2026-09-24 review are one missing rule. A NaN weight is
selected for int8 because `std::max` ignores NaN (F12) and then cast to
int8, undefined behavior that makes plans differ across hosts (F34); the
standalone `kClipNorm` does not refuse a non-finite norm on the CPU (F09)
and the GPU fused clip writes NaN into p, m and v where the CPU refuses the
step (F11); `max_norm == 0` multiplies by stale scratch (F13); the KL
temperature is never validated, and a negative T inverts both softmaxes
(F17, F10); the standalone clip threshold is unvalidated, −1.0 is gradient
ascent (F18); a NaN q8 dequant scale passes and the test expects it (F33);
`--temperature` is unbounded (N26); `ExpectClose` accepts NaN because
`err > tol` is false (F29). Full text: the "SeeML Code Review" page, first
review detail. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §8.

## Design

One rule, applied at each layer: **non-finite is refused at the boundary
it crosses.** Ingressor: non-finite weights refused (S1). Verifier: every
immediate float (clip threshold, KL temperature, q8 scale) must be finite
and in range; the validator's `clip_word_ok` checks applied to every site.
Executor: `ClipNorm` returns a status, CPU and Metal both map a non-finite
norm to `kNonFiniteNorm` with a flag checked after `Flush`; the sum of
squares accumulates in f64. Tests: `!(err <= tol)` everywhere a tolerance
is compared. The threat model on the screens (S0) states which classes the
sanitizer covers.

## Acceptance

- Each of the ten reproductions is a refusal or a defined result, with a
  test; the twin test pins CPU and Metal to the same verdict.
- UBSan clean on a plan compiled from a model with a planted NaN (it is
  refused before packing).
