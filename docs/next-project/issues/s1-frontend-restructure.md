---
title: "SeeAI S1: frontend restructure — ingressor, accountant, topology, computation; shape checks lifted out of the build loop; the cross-input and non-finite checks the review found missing"
number: 147
labels: enhancement,correctness,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The frontend's stages are interleaved: whole-graph checks run before
construction, but the per-op shape checks run inside `BuildForward` one op
at a time, so "verify, then build" does not exist as a boundary and the
cross-input checks have no home. The 2026-09-24 review found the holes that
follow from that: const payload placement never validated (F04: alignment,
header overlap, payload overlap); student names starting `t::` not reserved
(F06); output row count never checked against the compiled batch (F68, a
heap write downstream); teacher `seq_len` never checked against the student
(F02, high — a wrong distillation target with no diagnostic); `--loss mse`
on a token model compiles and every dataset is rejected (N19); the "lower
bound" footprint over-estimates and refuses compiles that fit (N29) and
ignores `--bf16-base` (F03); a NaN weight passes ingest and is later cast to
int8 (F12/F34). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3.

## Design

1. Directories `frontend/ingressor`, `frontend/accountant`,
   `frontend/topology`, `frontend/computation`; `representation/` and
   `operator/` stay as shared infrastructure; `model_writer` moves out as
   egress.
2. Topology runs to completion over the decoded model before computation
   starts: order, names, output, per-op shapes (from the tensor table), and
   the cross-input checks — student↔teacher (seq_len, output rows vs batch),
   loss↔input kind, adapter↔weight, tokenizer vocab == embedding rows ==
   head width. Computation becomes a mechanical translation with no failure
   modes of its own.
3. The ingressor refuses non-finite weights and validates payload placement
   (64-byte alignment, after the metadata, non-overlapping sorted ranges).
4. The accountant's estimate is a **tested lower bound**: estimate ≤ the
   backend's exact arena on every fixture, bf16 counted at 2 bytes; the
   screen prints it as an estimate with the honesty ratio.
5. Each stage fills its statistics struct (S0).

## Acceptance

- Every F04/F06/F68/F02/N19/N29/F03/F12 reproduction from the review is a
  compile-time refusal with a diagnostic naming the stage.
- Bit-identical plans for every fixture against main (a restructure, not a
  semantic change) — except the refusals above.
- The frontend screen prints the parameter census by role, per-tensor
  ranges, the estimate and its ratio.
