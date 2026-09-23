---
title: "SeeRL F5: scale ladder — SmolLM-360M, Qwen2.5-0.5B, then 1.5B on a 16 GB device with rematerialization and 8-bit optimizer states"
labels: enhancement,core-plane,frontier-parity
plane: Core plane
origin: Frontier Outlook
milestone: SeeRL v1.0.0.A
priority: P1
---
## Why

Only SmolLM-135M has been validated end to end (import #69, Metal #103,
CPU #105). The arena doctrine — one allocation, refused compiles instead
of OOM — is the property that should beat frontier frameworks as models
grow on a fixed-memory device, and it has never been exercised past 135M.
At 135M on the M5 (2026-09-22) MLX-LM peaks at 1.34 GB in the Metal
allocator with 1.01 GB resident, while SeeML's whole footprint is 1.89 GB
resident (arena, plan and Metal buffers are one shared allocation) — two
quantities, to be compared like against like in this ladder; nothing is
known about SeeML's arena at 0.5B–1.5B.

## Design

1. Import and field-run SmolLM-360M and Qwen2.5-0.5B through `--hf`
   (Qwen2 is a supported walker family); record arena bytes / model bytes,
   step-0 estimate honesty, tok/s cpu/metal, val-loss parity vs torch.
2. **Rematerialization (roadmap 2b):** a compile-time policy that drops
   selected activations from the primal snapshot and recomputes them in
   the backward; chosen by the memory gate when the arena would exceed the
   budget. Bit-identical to the non-remat plan.
3. **8-bit optimizer states:** block-quantized AdamW moments (new opcode;
   SEKP change), opt-in, measured on loss curves.
4. Qwen2.5-1.5B on a 16 GB M-series machine with q8 base + remat + 8-bit
   states + grad accumulation (#70): the compile must either fit or refuse
   with the exact shortfall.

## Acceptance

- A field-report table per model: arena, peak RSS / arena, tok/s,
  val-loss parity, compile time.
- 1.5B either trains on 16 GB or is refused at compile time with a correct
  estimate — never an OOM at run time.
