---
title: "SeeAI CR3: CI that can fail — the seam job's lost exit status first, then the fuzzer that never runs an accepted plan, the instrumentation that only nightly builds, and the tests whose assertions cannot fire"
number: 144
labels: bug,testing,correctness
plane: Gates & docs
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The review's one high finding in CI: the seam job pipes `unittest` into
`tee` without `pipefail`, so a failed Python suite merges green (F00). With
it: the `! grep` doctrine check is not enforced under errexit (F01); the
plan fuzzer never executes an accepted plan (F22) and the fuzz target
instruments only the harness file (N41); the HF-import compile check
hard-codes a build directory, skips silently and runs before any build
(F36); `seeml-bench` and `SEEML_STEP_TIMING` code compile only in the
nightly job (N44) and the NumPy- and tier-2-gated Python tests run in no job
(N45); the `SEEML_ACCELERATE` path is never compiled or tested (N43); the
overflow tests do not overflow (F30, F31), the addend rejection fires on the
wrong check (F32), the non-const-weight test tests a missing tensor (F26),
the concurrent-writer test passes when every write fails (F27), the accuracy
test cannot detect the leak it guards (F28); `build.sh` treats
`SEEML_ACCELERATE=off` as on (F23). F9 (#140) owns the measurement-code
items (N43, N44, N45, F36) and is referenced, not duplicated. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §8.

## Design

F00 ships first as its own one-line PR. Then, in one PR: `set -o pipefail`
and explicit `if grep` in every workflow step; the fuzzer runs one bounded
step under ASan per accepted plan with the parsers instrumented; each named
test rewritten so removing the guard it protects fails it; the falsy-value
check shared by every build-script switch.

## Acceptance

- A deliberately failing Python test turns the seam job red.
- Each listed test fails when its guard is removed (mutation check in the
  PR description).
- The fuzz corpus reports edges inside `runtime/` and `compiler/`, not only
  the harness.
