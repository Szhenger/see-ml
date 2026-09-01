---
title: "Python plane P5: calibration-ratio gate in bench_compare.py — Nightly #20–#23 were runner-variance false positives"
labels: testing,correctness,python-plane
plane: Python plane
origin: Nightly eval
priority: P1
---
## Root cause

Nightly runs #20–#23 failed **only** the bench Tier A gate, at the *same
commit* (9768709) that #24 passed green, against the *same rolling
baseline* (from #15 — the cache saves only on green). The green nights'
bench jobs ran 230–235 s; the red nights ran 264–315 s on identical work:
**15–35% slower runners tripped the 10% throughput gate.** The mechanism
is a high-water trap: one fast runner sets the baseline, and every
ordinarily-provisioned runner after it fails until an equally fast one
lands — during which a *real* regression would be indistinguishable from
noise. The epoch baseline (`bench-epoch-v1`) was seeded from fast #24, so
recurrence is likely.

## Design

Every objective in this project is gated on Tier A/B numbers — the
measurement plane must stop lying first:

- **Calibration workload:** `seeml-bench` runs a small fixed
  single-threaded reference kernel first and records `calib_rows_per_s`
  plus a real host identifier (CPU model, core count) in `bench.json`.
- **Ratio normalization:** `tool/bench_compare.py` scales each Tier A
  delta by the calibration ratio between the two runs before applying the
  10%/15% thresholds — a uniformly slow runner no longer regresses
  every fixture at once.
- **Two-consecutive-red rule:** the nightly gate reports red-once as a
  warning artifact and fails the job only on a second consecutive
  regression of the same key (state carried in the baseline cache dir).
- Keep fail-closed behavior: zero key overlap and missing calibration in
  a schema-3 report remain hard errors; schema-2 baselines compare
  un-normalized with a note (same migration pattern the `rows_per_s`
  comment in `bench_compare.py` already uses).

## Acceptance

- Replaying the #20–#24 sequence (recorded timings, synthetic reports)
  yields five green gates and zero missed true regressions in the
  injected-regression test.
- `bench_compare.py` stays stdlib-only.

Refs: `.github/workflows/nightly.yml` bench job, `tool/bench_compare.py`,
the Nightly #20–#23 incident analysis (2026-09-01); `docs/benchmarks.md`
measurement discipline.
