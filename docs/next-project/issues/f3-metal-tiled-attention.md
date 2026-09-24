---
title: "SeeAI F3: tiled attention on Metal — port opcodes 47-50 to the GPU, then fuse forward and backward flash-style"
labels: enhancement,efficiency,core-plane,frontier-parity
plane: GPU backend
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P1
---
## Why

E11 (#94, plan v15) added the tiled attention family (`kAttnFwdTiled`,
`kAttnDQTiled`, `kAttnDKTiled`, `kAttnDVTiled`) so long sequences need no
O(S²) probability cache (S=2048 peak 1,039 → 324 MiB). On Metal those four
opcodes still return `false` from the backend's coverage switch
(`runtime/executor/metal_backend.mm`, "runs on the CPU until #63 ports it"),
so a long-context plan either keeps the cached family or leaves the GPU for
every attention op. Attention is ~24% of a decoder step; MLX and PyTorch
both ship fused SDPA on Apple GPUs.

## Design

1. Port: Metal kernels for the four tiled opcodes with the CPU form as the
   bit-for-bit reference in `seeml_metal_backend_test` (stats row of 4
   floats per query row, same tile order); `HazardWithPending` extents from
   `DescribeInstruction` as for every other kernel.
2. Fuse: one flash-style forward kernel (QKᵀ, online softmax, PV in
   threadgroup memory) and a fused backward (dQ, dK, dV from the saved row
   stats), keyed on head dim ≤ 128. New opcodes if the reduction order
   changes; the unfused port stays the reference.
3. `--attention auto` extends its decision to the Metal backend: tiled
   whenever the cached family's `probs_cache_bytes` exceeds the device
   budget.

## Integrated from the 2026-09-24 systems review

Root cause 3, *backend coverage and the dispatch model*: a CPU-resident
opcode is not merely slower — `MetalBackend::Execute` flushes the command
buffer and waits (`HazardWithPending` → `Flush` → `waitUntilCompleted`)
before every CPU-resident instruction whose operands the GPU still owns.
On a SmolLM-shaped long-sequence plan the four tiled opcodes drain the GPU
four times per block per step, ≈120 mid-step drains, the same mechanism
that costs the softmax cross-entropy its drain every pass in F8. The port
is therefore a pipeline fix before it is a kernel fix: the acceptance adds
that a long-S train program on Metal contains no CPU-resident opcode.

## Acceptance

- GPU and CPU tiled attention agree bit-for-bit (unfused port) or within
  the certified bound (fused), at model shapes including ragged S.
- S=2048 SmolLM-shaped step runs entirely on the GPU; peak memory and
  tok/s reported against the cached family and against F1's MLX row.
