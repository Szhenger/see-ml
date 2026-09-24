---
title: "Python plane P6: seam contracts and gates — one source of truth per Python/C++ format, a tool/ package with dependency tiers, and CI parity for the build-host plane"
number: 86
labels: enhancement,testing,correctness,python-plane,core-plane
plane: Python plane
origin: Seam review
priority: P0
---
## Root cause

The Two-Plane Overhaul takes the Python surface from two scripts (~750
lines, `export_model.py` + `bench_compare.py`) to seven subsystems, and
every one of P1–P5 adds a new file contract between the planes. Today
each existing contract is held together by two hand-maintained copies and
a developer remembering to edit both (seam review, 2026-09-03, at
8993054):

| Contract | Writer → reader | Source of truth today | Checked by |
|---|---|---|---|
| SMF | Python → C++ | magic/version/op kinds duplicated in `model_format.h` and `export_model.py` | one `--demo` export in the CI e2e job (PyTorch required); no golden bytes in tree |
| SDS | Python → C++ | same duplication vs `runtime/feeder/dataset.h` | same single demo |
| `bench.json` (schema 2) | C++ → Python | hand-rolled `fprintf` JSON in `seeml_bench.cc`; keys live in comments | nightly only; `bench_compare.py` has no syntax check in CI |
| `--report` JSON | C++ → Python (P1) | no `schema` field; lacks the tiling, host arch, and vendored-file list a packer needs | nothing |
| SEEU v7 | C++ → Python (P4) | `PlanHeader` + 42 opcodes with packed dim words (`N<<32\|D`) in `source/plan/` | fuzzed and validated in C++; a hand-ported Python decoder would be a second, unchecked copy of the most-fuzzed format in the tree |
| kernel-policy table (P2), tolerance certificate (P3) | Python → C++ | undefined | undefined |

Meanwhile the C++ half is gated by three toolchains, ASan/UBSan/TSan,
libFuzzer, CodeQL, and the width-1/3/8 determinism matrix; the Python half
is gated by `py_compile` of one file. The in-tree UCB1 autotuner — dead
code that `docs/compiler.md` still narrates as live — is what an
unchecked seam looks like after a few releases.

One concern with P1 as written: the 16.5-minute `--build` is a
decimal-TU problem, and `.incbin` fixes it in ~20 lines of C++ emitter
change. A Python packer earns its place for the G1c extras (Metal
vendoring, page alignment, direct object emission) and must stay
**optional** on the default compile path, which today runs — and is
tested — without Python.

## Design

**1. One source of truth per contract, checked by machine.**

- An ABI manifest emitted from C++ (`seeml-seeu-dump --abi-json`, or a
  tiny `seeml-abi` tool): every format constant, enum value, and
  `sizeof`/`offsetof` of the packed headers (SMF, SDS, SEEU, checkpoint).
  Committed as `tool/seeml/abi.json`; a Python test asserts the Python
  constants and `struct` layouts agree; CI regenerates and fails on diff.
- Golden bytes in tree: a few KB of seeded SMF/SDS/SEEU fixtures. C++
  suites load them; the Python suite re-exports and compares bytes, which
  finally asserts the "classic demos are byte-for-byte" claim.
- P4 never hand-ports the plan decoder: `seeml-seeu-dump` gains `--json`
  and the frontier executor consumes the disassembly. The C++ decoder
  stays the only decoder; the Python interpreter's independence lives in
  opcode *semantics*, where the differential test earns its value.
- `--report` gains `"schema": 1` now, before P1 depends on it, plus the
  fields the packer needs: GEMM tiling, host arch, vendored source list,
  plan hash, `seeml_version`.

**2. `tool/` becomes a package with declared dependency tiers.**

- `tool/seeml/` with one `formats.py` owning every byte layout, shared by
  exporter, packer, and executor; the documented entry points stay as
  thin scripts.
- Tiers, enforced: **tier 0 stdlib-only** for anything on the compile or
  gate path (`pack_update.py`, `bench_compare.py`); **tier 1 NumPy**
  (token demos, `--corpus`); **tier 2 PyTorch/MLX** (MLP export,
  `frontier_exec.py`, `certify_numerics.py`). Declared as extras in
  `pyproject.toml`; tier 0 runs in CI under a bare interpreter so a stray
  import fails loudly.
- Mirror the C++ discipline: strict CLIs that reject inapplicable flags
  (already the exporter's rule), typed errors with a documented exit-code
  table, and atomic writes (temp + rename) for every artifact — the
  staging pattern `seeml_bench.cc` already uses.

**3. Python under the same gate as C++ (`ci.yml`).**

- A per-PR Python job: unit tests for formats, ABI-manifest agreement,
  golden-byte round trips, seed determinism; `ruff` + strict `mypy` over
  `tool/`.
- A torch-free seam job on every PR: `--demo-decoder` needs NumPy only,
  so export → compile → dump → validate runs in under a minute.
- Byte-compile every script (the bench gate is uncovered today).
- A doctrine assertion: after `--build`, grep the emitted package for any
  reference to Python and build it on a runner with Python off `PATH` —
  "nothing Python ships on device" becomes a test, not a sentence.

**4. Every contract versioned; both sides fail closed.**

- P2 table: JSON with schema version + host key. Absent → analytic
  fallback; present-but-invalid → hard error (the CLI's own rule: a
  silently ignored input is worse than a refusal). Entries validated
  through `ValidateGemmTiling`.
- P3 certificate: sidecar bound to the plan by hash plus a header flag;
  the validator refuses relaxed opcodes with no matching certificate.
- P5 schema 3: keep the migration pattern, and compare `seeml_version` as
  well as `schema` so a baseline from another release cannot compare
  silently.

**5. Strangler-style port; delete what is replaced.** Each Python
subsystem lands behind a flag with the C++ path default → proves
equivalence with its own acceptance test (byte-identical `.seeu` for P1,
tolerance agreement for P4, replayed nightlies for P5) → flips the default
in its own PR together with its E6 doc change → removes the superseded
C++ path in a following PR (for P2: the UCB1 bandit and its test, not
left beside the new tuner). Definition of done per subsystem: ABI fixture
updated, CI gate present, docs corrected, old path gone, ownership row
added to `SPECIFICATION.md` §1 saying which plane owns which artifact.

**Drift to fold into E6 now:** `docs/formats.md` heads SMF "v2" and marks
`seq_len` "(v3 only)" while code and exporter are at v5;
`SPECIFICATION.md` §1 says the Python surface is "one file"; CI's
byte-compile names one of two scripts.

## Acceptance

- `tool/seeml/abi.json` exists, is regenerated in CI, and a Python test
  fails when any SMF/SDS/SEEU constant or header layout drifts.
- Golden SMF/SDS/SEEU fixtures are loaded by the C++ suites and
  byte-compared by the Python suite; `--demo` / `--demo-decoder` with
  default flags reproduce them exactly.
- `ci.yml` runs the Python test job, `ruff`, `mypy --strict`, the
  torch-free seam job, and the no-Python-in-package assertion on every PR;
  tier-0 tools import cleanly under a bare interpreter.
- `--report` carries `"schema": 1` and the packer inputs; `seeml-seeu-dump
  --json` round-trips the full fixture suite.
- Lands before the first Python-plane subsystem (P1 or P5) flips its
  default; the P1 default compile path still needs no Python.

Refs: seam review of 2026-09-03 (this issue's source); P1–P5 (#75–#79),
E6 (#85); `tool/README.md`, `SPECIFICATION.md` §1–2, `.github/workflows/ci.yml`.
