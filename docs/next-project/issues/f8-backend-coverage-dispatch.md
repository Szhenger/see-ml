---
title: "SeeAI F8: backend coverage and the dispatch model — softmax cross-entropy on the GPU (opcodes 11/12), no mid-step drains, and the dispatch overhead measured on Metal before the fusion verdict"
labels: enhancement,efficiency,core-plane,frontier-parity
plane: GPU backend
milestone: SeeAI v1.0.0.A
priority: P0
---
## Root cause: opcodes the Metal backend does not cover, and what a CPU-resident opcode costs (2026-09-24 systems review)

`MetalBackend::IsGpuOpcode` returns `false` for `kSoftmaxXEntFwd` (11),
`kSoftmaxXEntBwd` (12), `kKLDistillFwd/Bwd`, `kEmbedFwd`, `kRopeTable` and
the tiled attention family (47–50, F3). A CPU-resident instruction whose
operands the GPU still owns makes `Execute` flush the command buffer and
`waitUntilCompleted` (`runtime/executor/metal_backend.mm`,
`HazardWithPending` → `Flush`). On the SmolLM-135M package the loss is
therefore a drain every pass: the head GEMM finishes, the GPU idles, the
CPU runs `cpu opcode 11` for 7.5 ms over the 100 MB of f32 logits
(512 × 49,152), then opcode 12 for 1.9 ms, then the backward is encoded
from a cold queue (`out/frontier-2026-09-22/prof_q8.txt`: 30 × 7,533 µs
and 4 × 1,870 µs). That is ≈5 % of kernel time plus the drain itself.

The second half of the root cause is unmeasured: summing the profile's
isolated kernel spans per training step gives ≈100 ms forward and ≈108 ms
backward against the measured 127 / 158 ms (`seeml-bench` split, the
2026-09-15 baseline row), so **≈27 % of the step is outside any kernel** —
encode time, inter-dispatch gaps in a serial encoder, the CPU-resident
opcodes and the flush. E4 (#83) concluded elementwise fusion is worth
2–6 % of the step, but that was measured on the CPU backend, where a
dispatch costs nothing; on Metal the train program is 3,309 instructions
and the profile counts ≈6,000 dispatches per step. The verdict on fusion
for the GPU has not been measured.

`docs/roadmap.md` Project 5 (G1b, #59–#65) lists "the loss families on
the GPU (the last CPU round-trips)" as a remainder after the v1.3.0 gate;
this issue is that remainder with its cost measured, plus the dispatch
question the roadmap does not ask.

## Design

1. **Port 11 and 12 to Metal** with a fixed-order row reduction (one
   threadgroup per row, a fixed tree over the 49,152 columns, f64
   partials as the CPU kernel keeps them, no atomics) so the CPU and Metal
   results stay within the existing backend-agreement bound (1e-5) and
   two Metal runs stay byte-identical. `kEmbedFwd` stays on the CPU (it
   runs before any GPU work and forces no drain); `kRopeTable` likewise.
2. **Measure the dispatch overhead directly**: with `SEEML_METAL_PROFILE`
   off, record one `ExecuteRange`'s command-buffer `GPUStartTime → GPUEndTime`
   span and the CPU wall of the same range, and set them beside the
   summed per-kernel spans from the profiled run. The difference is the
   dispatch and serialization cost, per program.
3. **Then decide fusion for the GPU**: if step 2 attributes more than
   ~10 % of the step to gaps, extend `kFusedMap` (E4) to the Metal
   backend and fuse the elementwise chains it already describes (the
   RMSNorm → GEMM prologue, SiLU × up, residual add), keyed on the
   per-dispatch cost measured rather than the CPU's 2–6 %.
4. A concurrent dispatch type is *not* the answer under the doctrine
   (the serial encoder is what keeps two runs byte-identical without
   barriers); the hazard tracker exists so CPU-resident ops flush late,
   and step 1 removes the only mid-step flush of a SmolLM-shaped step.

## Acceptance

- No CPU-resident opcode in the train program of a SmolLM-shaped plan on
  Metal (`seeml-plan-probe` lists none; the profile shows no `cpu opcode`
  row); CPU vs Metal loss within 1e-5; two Metal runs byte-identical.
- The dispatch-overhead number reported per program (train / eval /
  merge) for SmolLM-135M and `dec_wide` in `docs/benchmarks.md`.
- The fusion decision for the GPU recorded from that number, with the
  E4 CPU number beside it.
