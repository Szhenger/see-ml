---
title: "SeeAI S6: backend restructure — architecture, allocation, selection, assembly, packaging; assembly out of the driver; tuner and kernel emitter retired; the GPU half of the target description"
number: 152
labels: enhancement,core-plane,design
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

Plan assembly, the seeded persistent image and the integrity seal are
implemented inside the driver's `CompileImpl`, a 560-line function the
Style Audit named; the tuner directory holds a table reader that belongs to
the target description; `kernel_emitter.cc` generates Metal source nobody
ships since the runtime took ownership of the kernel library; and the host
description cannot say the device has a GPU, so the backend screen's
predicted step time has no GPU denominator. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §5.

## Design

1. Directories `architecture/` (host + GPU target description, analytic
   tiling, policy-table read), `allocation/`, `selection/`, `assembly/`
   (moved out of the driver: layout, persistent image, emit table, seal;
   the header gains the identity manifest (S4) and the ledger (S5)),
   `packaging/` (the emitter, now also vendoring `source/`'s contract
   check, sanitizer and tokenizer). `tuner/` folds into architecture;
   `kernel_emitter.cc` retires or its GPU tiling clamp moves under
   architecture as data.
2. Selection: ISA drops MSE and conv opcodes, adds the masked cross-entropy
   pair (S3) and the attention segment operand; SEEU version bump with the
   header fields above.
3. Scheduling stays a candidate gated on F8 (#139): if the measured step
   time outside kernels is real, it becomes a directory that runs before
   allocation; otherwise a function inside selection.
4. The backend screen: arena by segment with the five largest tensors, the
   honesty ratio against the accountant, the GEMM census (shape, FLOPs,
   share), dispatch count per program, plan and package bytes, predicted
   step time at the host peak.

## Acceptance

- Bit-identical `.seeu` and packages for every fixture before and after the
  move, except the header fields the bump adds.
- The driver contains orchestration only: no function over 150 lines.
- The backend screen's predicted step time for SmolLM-135M on the M5 GPU is
  printed and the runtime's measured line lands beside it.
