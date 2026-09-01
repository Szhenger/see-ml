---
title: "Python plane P2: offline autotuner with host-keyed kernel-policy persistence (retire the dead in-tree UCB1)"
labels: enhancement,efficiency,python-plane
plane: Python plane
origin: Both audits
priority: P2
---
## Root cause

`torch.compile` closes schedule-selection by **benchmarking at compile
time** (`max-autotune`). SeeML derives tilings analytically
(`SuggestGemmTiling`, pure arithmetic against detected cache sizes,
`native_emitter.cc:462-463`) — and the in-tree UCB1 bandit
(`AutotuneGemmTiling`) is **dead code**: no production path calls it, it
has no caching or persistence, and `docs/compiler.md:287-297` narrates it
as a compile participant the code contradicts (Performance Audit §03).
Zero compile cost — but also zero measurement, exactly when E1's
microkernel and G1b-4's GPU tile arms will need measured go/no-go data.

## Python-plane design

`tool/autotune.py` — an *offline* subsystem, never in the compile path:

- Drives `seeml-bench` sweeps over tiling/kernel-policy arms on the target
  host; medians-of-medians per `docs/benchmarks.md` discipline.
- Persists a **kernel-policy table keyed on host identity** (CPU model,
  cache sizes, core count); the compiler consumes the table when present
  and falls back to the analytic `SuggestGemmTiling` otherwise —
  determinism is untouched because the table only picks among
  bitwise-equivalent schedules per backend.
- Retires (or demotes to a test utility) the C++ UCB1 bandit; corrects
  `docs/compiler.md`.

## Acceptance

- A tuned host beats the analytic default on Tier B GEMM fixtures or the
  table records "analytic is best" — either way the decision is measured.
- Compile wall time unchanged when no table exists.
- `docs/compiler.md:287-297` matches reality.

Refs: Performance Audit §03 (autotuner), doc-correction list; Frontier
Bridge §02 (max-autotune paragraph); feeds E1 and #63 (G1b-4 GPU arms).
