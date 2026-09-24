---
title: "SeeAI S0: no leaky abstractions — every subsystem prints its first-principles statistics screen, one struct rendered to console and report, tested on the fixtures"
number: 146
labels: enhancement,doctrine,design,docs
plane: Gates & docs
milestone: SeeAI v1.0.0.B
priority: P0
---
## Why

The compiler computes almost every number an engineer would want and prints
eight prose notes and one summary line. The compile report on disk carries
17 passes with op counts and milliseconds, the resource analyzer's byte
split, 211 adapter geometries with their scales, and the three programs'
instruction counts; the console shows none of it. The runtime knows the
per-phase step split, the per-opcode Metal profile and the arena, and prints
"resource contract" and "accepted". This issue heads the milestone: the
doctrine that each subsystem explains itself quantitatively. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §1–§2.

## Design

1. One statistics struct per subsystem (frontend, analysis, backend,
   runtime); the console screen and the compile report's JSON are two
   renderings of that struct — one source of truth, the P6 seam rule.
2. One screen per subsystem by default, per-tensor and per-op detail behind
   `--verbose`. Every number carries its unit and its derivation clause.
3. The backend screen ends with a predicted step time (plan FLOPs ÷ the
   host peak the user passes, the `--peak-gflops` convention of
   `seeml-bench`); the runtime's screen prints the measured step beside it,
   so the ratio is the utilization and the gap is where the next issue
   comes from. RSS beside arena; loss beside the accountant's floor.
4. Screens print aggregates, hashes and losses, never record contents.
5. Golden statistic blocks for the standard fixtures in the test suite, so a
   pass that silently changes what it does changes a number the suite sees.
6. `docs/usage.md` gains "What the compiler tells you": the real
   SmolLM-135M screens, annotated.

## Acceptance

- SmolLM-135M compile prints four screens totalling ≤ 60 lines by default;
  `--report` JSON carries the same structs; `seeml-abi` publishes their
  schema.
- Golden blocks for `dec_wide` and the MLP fixtures pass under all
  toolchains; a deliberate one-op change in a pass fails them.
- The runtime's measured-vs-predicted line appears in every `model_update`
  run and in `seeml-bench` rows.
