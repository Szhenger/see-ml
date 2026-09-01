---
title: "Python plane P4: frontier reference executor — run .seeu plans through PyTorch/MLX for differential testing and AMX/GPU ceiling pricing"
labels: enhancement,testing,design,python-plane,doctrine
plane: Python plane
origin: Frontier bridge
priority: P1
---
## Doctrinal root cause

torch reaches the Apple **AMX** matrix coprocessor only via Accelerate
(there is no public ISA); SeeML's runtime builds with a C++ compiler
alone, so its CPU ceiling is the NEON peak — **0.84 vs 1.49 TFLOP/s,
≈1.8× before a single line of kernel code is compared** (Frontier Bridge
§02). The audits concluded this is a *product decision* (a dependency
exception), but today it is unpriced: nobody has measured what
Accelerate/MLX would actually buy **on SeeML's own plans and shapes**.

## Python-plane design

`tool/frontier_exec.py` — a build-host interpreter for the 31-opcode plan
ISA, never shipped on device:

- **Executes a `.seeu` plan** step-for-step with a PyTorch backend (CPU →
  Accelerate/AMX) and an MLX backend (unified-memory GPU), consuming the
  same SMF/SDS inputs as the C++ runtime.
- **Differential-testing oracle:** compares against the C++ runtime at
  tolerance per backend — a second, independent implementation of every
  opcode's semantics, catching kernel bugs the bit-exact self-comparison
  structurally cannot (it would reproduce them faithfully).
- **Ceiling pricing:** reports per-plan tok/s and MFU under each frontier
  backend next to the C++ numbers, so the Accelerate-exception and
  GPU-priority decisions are data, not vibes — the Frontier Bridge's
  anchor table (~1,605 tok/s MLX vs 80 tok/s SeeML at 135M) regenerated
  on demand for *any* plan.
- Extends the existing Python seam (`export_model.py` already owns
  PyTorch on the on-ramp; this adds the off-ramp).

## Acceptance

- Round-trips the full v1.2.4 fixture suite: every opcode interpreted,
  tolerance-agreement with the C++ runtime on CPU.
- Emits a `frontier` section (tok/s, it/s, MFU, peak RSS per backend)
  comparable to `seeml-bench --out` schema-2 fields.
- One command answers "what would AMX buy on this plan?".

Refs: Frontier Bridge §01–§02 (anchors, AMX row) and §04 (the properties
the oracle must respect); enables P3's certification.
