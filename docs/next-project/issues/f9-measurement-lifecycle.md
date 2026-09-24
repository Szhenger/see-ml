---
title: "SeeAI F9: measurement and lifecycle — a train-only Metal profile, the gate's share of the update wall, the real-package CPU baseline, and the per-column int8 validation row"
labels: enhancement,testing,docs,frontier-parity
plane: Gates & docs
milestone: SeeAI v1.0.0.A
priority: P1
---
## Root cause: numbers that gate decisions were taken on the wrong object (2026-09-24 systems review)

- **The Metal profile weights the forward 7:1.** `prof_q8.txt` was taken
  with validation on: 30 forward passes (26 of them evaluation) against 4
  backward passes, so every per-family share read from it over-weights
  the forward. F7's and F8's per-step shares are estimates from it.
- **The CPU fixture under-reads the real package.** `tok_smollm135m_q8`
  measures 233–269 tok/s where the real SmolLM-135M package measures 332
  (F1, 2026-09-22); F4's baseline is set from the real row and the fixture
  cannot gate CPU decisions.
- **Tier D's gate share is defined and never measured.** `--eval-every
  auto` evaluates every `steps / 10` over the whole validation split, plus
  the pre- and post-training passes; the F2 CPU run's lifecycle intercept
  was 39.5 s on a 5-step run (`f2_measure/cpu.json`: wall 41.75 s), most of
  it loading a plan that embeds 262 MB of rodata (741 MB for the f32 plan)
  and two full evaluations. `torch.compile`'s 40–54 s warm-up is the
  counterweight the doc already names; SeeAI's own lifecycle has no
  number beside it.
- **The per-column int8 fix is unmeasured on the frontier row.** Per-tensor
  int8 costs +0.10–0.14 nats of validation loss at step 0 on every model;
  #128 (E12) landed per-column scales and `pkg_smollm135_q8col` is built in
  `out/frontier-2026-09-22/`, but the validation-loss row was never rerun
  on it.

## Design

1. `SEEML_METAL_PROFILE` gains a train-only mode (`--eval-every 0`,
   `--val-frac 0`) documented as *the* profile for kernel decisions; the
   profile header prints the forward/backward pass counts so a reader
   can see the weighting.
2. `seeml-bench` (or `frontier_run.py`) reports the Tier D lifecycle
   split for the real package — load+validate, each evaluation, merge,
   commit — and `docs/benchmarks.md` carries the gate's share of the
   update wall for SmolLM-135M on `cpu` and `metal`.
3. `docs/benchmarks.md` states the rule: CPU decisions are gated on the
   real-package row; the fixture is a regression canary only. The fixture's
   under-read is either explained (weights, corpus, thread pinning) or the
   fixture is re-seeded to reproduce the real per-step cost as it does on
   Metal.
4. Rerun the frontier validation-loss row on the per-column package and
   record the step-0 and step-300 deltas against f32 beside the
   per-tensor numbers.

## Acceptance

- A train-only Metal profile of SmolLM-135M committed to
  `docs/benchmarks.md` with per-family shares (frozen GEMMs, LoRA,
  elementwise, attention, CPU-resident, optimizer) and the outside-kernel
  remainder.
- The gate-share and lifecycle row for the real package on both backends.
- The per-column int8 validation delta reported; if it does not close the
  +0.10 nats, a follow-up issue names the residual.
