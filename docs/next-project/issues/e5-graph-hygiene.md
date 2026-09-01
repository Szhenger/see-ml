---
title: "Core plane E5: graph/infra hygiene batch — DCE mark-and-compact, per-pass timing, gated SIR dump, cached label validation, single-wakeup pool notify"
labels: enhancement,efficiency,core-plane,good first issue
plane: Core plane
origin: Performance audit
priority: P2
---
## Root causes

Five small, independent items (Performance Audit finding #7) — all cheap,
none regress anything, several are prerequisites for the bigger tracks:

1. **DCE removes ops one `removeOp` at a time** — O(removed × ops), a
   latent O(ops²) cliff the first time a pass (e.g. E4's chain fusion)
   dead-codes a real fraction of the graph (`dce.cc:41`). Fix
   preemptively: mark + one `erase_if` compaction.
2. **No per-pass timing exists anywhere** — `PassNote` records op counts
   only (`pass_manager.cc:16`). Add elapsed time so every compile-side
   claim in this project becomes measurable for free.
3. **The SIR dump streams the whole graph through `ostringstream`
   unconditionally and retains the string**
   (`update_compiler.cc:549-553`) — the largest pure-CPU graph cost in
   the driver, and it is debug output. Gate it behind the flag that
   prints it.
4. **Corpus label validation rescans the full corpus up to 4× per
   `Train()`** (`dataset.cc:205-228`). Validate once, cache the verdict.
5. **Every finishing pool worker takes the mutex and broadcasts
   `done_cv_`** — O(T²) wakeup checks per job (`parallel_for.cc:200-204`).
   Last-worker `notify_one` cuts it to one; the static chunk geometry and
   claiming protocol are untouched, so determinism is unaffected.

## Acceptance

- Each item lands independently; bit-identity everywhere (none touch
  numerics).
- Item 2's timing output appears in the compile report and is cited by
  E2/P1 acceptance measurements.

Refs: Performance Audit §02–§04 (finding #7).
