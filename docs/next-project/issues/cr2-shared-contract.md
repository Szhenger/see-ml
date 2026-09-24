---
title: "SeeAI CR2: one contract function in source/ — the compiler's topology, the loader and verifier, the plan probe and the Python oracle all call it, and a program × check coverage table proves nothing is skipped"
number: 143
labels: bug,correctness,core-plane,python-plane
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The contract checks exist in several copies and each copy skips something.
The step program is left out of the softmax class-width and vocabulary scans
(F07: a heap write from a plan that passed every contract); the provenance
proof assumes the program never writes the staged IO slots (N03); header
input and label refs may carry the source bit and bypass the feeder's
validated labels (N04); rodata alignment is never proven (N05); the plan
probe skips `VerifyPlanContract`, `VerifyExecutorContract` and the feeder
checks (F58); the Python oracle's `Corpus.check` skips most of the feeder
contract (F47) and its `diff` never checks bytes outside declared write
extents, so an out-of-bounds kernel write still agrees (N33). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3
(doctrine: shared functions in `source/`), §8.

## Design

1. Move the geometry contract out of `runtime/engine/contract.cc` into
   `source/contract/` as pure functions over the plan header and the
   corpus header; the compiler's topology (S1), the loader (S7), the probe
   and, through the ABI manifest, the Python oracle call the same code.
2. A **coverage table** in the test suite: every program (train, step, eval,
   merge) × every check (class width, vocab bound, IO-slot write
   disjointness, source-bit refs, rodata alignment, label kind, label width,
   token bound); a check missing from a program is a failing test, not a
   review finding.
3. The oracle's `diff` compares the whole arena, not declared extents.

## Acceptance

- F07, N03, N04, N05, F58, F47, N33 reproductions refused by both planes.
- The coverage table is complete and is what the review's "three-way check"
  would have found in CI.
