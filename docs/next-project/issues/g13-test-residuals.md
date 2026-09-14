---
title: "Gates G13: test residuals of closed #17 / #24 — eval tail weighting, GemmNTQ8, the LR floor default, tile-boundary GEMM shapes, program-level FD for KL / MSE / composite"
labels: testing,core-plane
plane: Gates & docs
origin: Algorithm review
priority: P2
---
## Root cause

Two closed testing issues left halves undone, and the Algorithm Review had
to fill the gaps by hand (a scratch six-variant GEMM check against a
double reference at M=37, N=517, K=131 — ragged K%4 and N%4, crossing both
tile boundaries — passed at 6.2e-6 max relative error, 1 and 10 threads):

- **#17 residual:** `Evaluate` (`runtime/engine/update_engine.cc:599-651`)
  now snapshots/restores the cursor (both gate passes see the same
  multiset), but the wrapped final batch's duplicates are still averaged
  into the reported loss (accuracy correctly excludes them). Val sets
  smaller than a batch report a loss dominated by duplicates.
- **#24 residual:** `GemmNTQ8` appears in no test file; the LR schedule
  test pins `min_lr_factor = 0.1`, never the default 0 past the horizon;
  no kernels test crosses `kTileK = 64` / `kTileN = 256` (largest shape is
  64×96×48) so the `k0 > 0` / `n0 > 0` paths are exercised only by the
  system FD checks.
- **New:** no program-level finite-difference check exists for `kMse`,
  `kKLDistill` or `kXEntPlusKL` (all system FD checks use the xent
  config); the KL kernel tests check sign and monotonicity, not
  `d loss / d logit` at T ≠ 1. `ClipNorm` has no test asserting a vector
  above the cap lands at exactly `max_norm`, nor the inf / NaN cases.

## Design

- Weight the final eval batch by real samples (or run it at the exact
  remainder shape when the plan permits); the accuracy path already knows
  the count.
- `kernels_test`: `GemmNTQ8` vs `dq(q8)@` f32 reference; a big-shape
  six-variant test against a double reference at a tile-crossing ragged
  shape; `ClipNorm` cap, inf and NaN cases.
- `update_engine_test`: `EffectiveLr` at `default_steps + 1` with the
  default floor (documents E8's contract once it lands).
- `update_system_test`: FD checks under `kMse`, `kKLDistill` (T = 2) and
  `kXEntPlusKL`.

## Acceptance

- Each test fails against a deliberately broken kernel (mutation check in
  the PR description) and passes on main.
- Reported val loss on a 5-sample val set with batch 32 equals the
  5-sample mean.

## Goal alignment

Objective 5 (trustworthy measurement): every frontier comparison quotes
`train_loss_first/last` and val loss from these paths, and every kernel
project (E1, E7, #66, E11) needs a tile-crossing reference test to land
safely.

Refs: SeeML Algorithm Review (2026-09-14) §02-C/D, §04; closed #17, #24.
