---
title: "Docs E6: four claims the code contradicts, plus the Two-Plane doctrine update"
labels: documentation,core-plane
plane: Gates & docs
origin: Performance audit
priority: P2
---
## The corrections (Performance Audit, verified against source)

1. `update_kernels.h:18-20` claims AVX-512/NEON variants are "swapped in
   at link time" — none exist in the tree (zero intrinsics anywhere).
2. `docs/runtime.md:120` — "literally zero extra instructions" for int8
   dequant overstates it: the widening chain per element is real, and the
   bench harness was built to measure exactly that ratio. Say what's true.
3. `docs/compiler.md:287-297` narrates the autotuner as a compile
   participant; `AutotuneGemmTiling` is test-only dead code (P2 owns the
   real fix; correct the prose now).
4. `update_engine.h:19-21` — "mmap-equivalent ingest" is a buffered read
   into a heap vector (`durable_io.cc:135-150`). The distinction matters
   for the RSS accounting the docs elsewhere treat as exact.

## The doctrine update

When the Python plane lands (P1–P5), `tool/README.md`'s "the only Python
in the product, and everything downstream is dependency-free C++" stops
being the right sentence. Replace it with the two-plane statement
(`docs/next-project/README.md`): **the emitted package and runtime are
dependency-free C++ under the full doctrine; the build host carries a
Python frontier plane** (packaging, measurement, autotuning,
certification, reference execution). Update `docs/compiler.md` /
`docs/workflows.md` cross-references accordingly, and state explicitly
that nothing Python ships on device.

## Acceptance

- Every quoted claim either matches the code or is gone.
- The doctrine paragraph lands in the same PR as the first Python-plane
  subsystem it describes (P1 or P5, whichever merges first).

Refs: Performance Audit (documentation-corrections list); this project's
README.
