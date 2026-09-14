---
title: "Core plane E8: the cosine horizon is the compiled step budget, not the run — zero-LR tails, un-annealed short runs, resume overshoot (residual of #18 / #24)"
labels: correctness,core-plane,sev:medium
plane: Core plane
origin: Algorithm review
priority: P1
---
## Root cause

`EffectiveLr` (`runtime/engine/update_engine.cc:527-538`) anneals over
`header_.default_steps − warmup_steps`, the **compile-time** budget, not
the `steps` passed to `Train()` (`:675`). `min_lr_factor` defaults to
**0.0** (`source/plan/config.h:53`). Consequences, all silent:

- `model_update --steps 200` on a plan compiled with `--steps 1000` never
  decays below ~91 % of the base LR (no annealing).
- `--steps 5000` on the same plan sits at LR = 0 for 80 % of the run —
  steps that move nothing but still update the AdamW moments and burn
  time.
- `--resume` with no `--steps` requests `default_steps` **further** steps
  (`TrainImpl`: `if (steps == 0) steps = header_.default_steps;` runs
  before the checkpoint loads), i.e. a full extra budget entirely past the
  horizon at LR = 0. This is the unresolved half of closed **#18**, which
  asked to "train the remainder"; the doc-side decision
  (`docs/usage.md:118`: `--steps` after `--resume` means further steps)
  left the LR consequence unaddressed.
- The compiler accepts `--steps 0` and never checks `warmup < steps`
  (`tool/seeml_update_compile.cc:233-235, 297-305`); with `horizon == 0`
  every post-warmup step is at the floor. `--warmup` / `--min-lr-factor`
  under `const` are silently ignored, against the CLI's own rule. Closed
  **#24** named this latent bug; the schedule test that closed it pins
  `min_lr_factor = 0.1`, never the default 0.

## Design

1. The horizon is the **run's** step count: `start + steps` for a fresh
   run; on resume, the remainder to the original horizon (stored in the
   checkpoint, see E9) unless `--steps` overrides it explicitly.
2. `min_lr_factor` default → 0.1 (documented); a run that would execute
   any step at LR = 0 is refused with a diagnostic unless
   `--allow-zero-lr`.
3. Compile-time validation: `steps > 0`, `warmup < steps`, and
   `--warmup` / `--min-lr-factor` rejected under `--lr-schedule const`.
4. The header keeps `default_steps` for the generated main's default; no
   ABI change if the run horizon is a `TrainOptions` field.

## Acceptance

- `Train(data, 200)` on a 1000-step plan reaches `lr·min_lr_factor` at
  step 200; `Train(data, 5000)` reaches it at step 5000.
- A resume at step 500 of 1000 with no `--steps` trains exactly 500 more
  and anneals to the floor at step 1000; committed bits equal the
  uninterrupted run (the existing bit-identity resume test, extended).
- `EffectiveLr` past the horizon with the default floor is tested
  explicitly; `--steps 0` and `--warmup 1000 --steps 1000` are compile
  errors.

## Goal alignment

Outcome parity with the reference recipes: `torch` schedulers and
`mlx_lm.lora` anneal over the iterations actually run. The frontier
comparison in `docs/benchmarks.md` compares loss trajectories at equal
step counts; a SeeML run that silently trains at LR 0 or never anneals
loses that comparison for reasons unrelated to kernels. No throughput
effect except the wasted steps themselves.

Refs: SeeML Algorithm Review (2026-09-14) §02-A/D, §04; closed #18, #24;
`update_engine_test.cc:811-843` (pins current behaviour).
