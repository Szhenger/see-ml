---
title: "Python plane P1: package assembler in tool/ replaces the decimal C++ TU (implements G1c #64)"
number: 75
labels: enhancement,efficiency,design,python-plane,doctrine
plane: Python plane
origin: Both audits
priority: P0
---
## Doctrinal root cause

The zero-dependency doctrine says a package must build anywhere with a C++
compiler — so the emitter re-encodes the entire plan as a ~3.9× **decimal
byte-array translation unit** and `--build` hands it to the host compiler
at `-O2` (`native_emitter.cc:383`, `:423-435`). The consequences, from the
Performance Audit:

- `--build` **dominates total compile wall time**: 16.5 min at 135M params
  (v1.2.4 field report); at 1B it would not build on a 16 GB host.
- Emission peak RSS ≈ **11× plan size** — model + plan + 4× chunk strings +
  the 5×-reserved output string simultaneously live
  (`native_emitter.cc:46,65-69,87`).
- The plan lands on disk twice (raw `.seeu` + the array TU).

## Python-plane design

Move package *assembly* (byte-scaling work) out of the C++ compiler into a
build-host Python subsystem, `tool/pack_update.py`:

- Writes the `.seeu`, generates the **`.incbin` assembly stub** (or emits
  the object file directly), and drives the vendored-runtime build — the
  host C++ compiler never parses weight bytes again.
- Implements the G1c/#64 `.incbin` plan (the roadmap already measured the
  decimal-TU route as untenable) and doubles as the packaging point for
  vendoring the Metal path (G1c) and the page-alignment prerequisite of
  zero-copy residency (G1b-2, #61).
- The **emitted artifact is unchanged**: a dependency-free C++ package.
  Doctrine is spent on the build host only, where it never applied.

## Acceptance

- `--build` at 135M drops from minutes to seconds; a 1B-param package
  builds on a 16 GB host.
- Emission peak RSS ≤ ~2× plan size (coordinate with E2's streaming
  emission — P1 makes most of the chunk-string machinery unnecessary).
- Byte-identical `.seeu` vs the current emitter on the test fixtures.

Refs: Performance Audit §03; Frontier Bridge §03 (G1c row); #64, #61.
