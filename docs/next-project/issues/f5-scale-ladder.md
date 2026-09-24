---
title: "SeeAI F5: scale ladder — SmolLM-360M, Qwen2.5-0.5B, then 1.5B on a 16 GB device with rematerialization and 8-bit optimizer states"
labels: enhancement,core-plane,frontier-parity
plane: Core plane
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P1
---
## Why

Only SmolLM-135M has been validated end to end (import #69, Metal #103,
CPU #105). The arena doctrine — one allocation, refused compiles instead
of OOM — is the property that should beat frontier frameworks as models
grow on a fixed-memory device, and it has never been exercised past 135M.
At 135M on the M5 (2026-09-22) MLX-LM peaks at 1.34 GB in the Metal
allocator with 1.01 GB resident, while SeeAI's whole footprint is 1.89 GB
resident (arena, plan and Metal buffers are one shared allocation) — two
quantities, to be compared like against like in this ladder; nothing is
known about SeeAI's arena at 0.5B–1.5B.

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

## Integrated from the 2026-09-24 systems review

Root cause 4, *memory planning* — the arena at each rung, measured
2026-09-22 (`out/frontier-2026-09-22/summary.json`, the q8 packages):

| model | arena | persistent | cached probs | rodata |
|---|---:|---:|---:|---:|
| SmolLM-135M | 1.56 GB | 36.3 MB | 70.8 MB | 262 MB |
| SmolLM2-360M | 3.17 GB | | | |
| Qwen2.5-0.5B | 4.20 GB | | | |

At 135M ≈1.2–1.5 GB of the arena is activations and adjoints for 512
tokens; a linear extrapolation puts 1.5B past 12 GB, a refused compile on
a 16 GB device. Like against like at 135M SeeAI is *below* the frameworks
in device memory (MLX-LM f32 allocator peak 2.11 GB, PyTorch MPS device
peak 2.4–2.6 GB), so this is a scaling problem, not a parity problem.
Executable order: first the compile report's persistent / IO / transient
split at each rung and adjoint liveness in the arena binder (bit-identical,
no new opcode), then rematerialization (2b) for what liveness cannot
recover, then the 8-bit states.

## Acceptance

- A field-report table per model: arena, peak RSS / arena, tok/s,
  val-loss parity, compile time.
- 1.5B either trains on 16 GB or is refused at compile time with a correct
  estimate — never an OOM at run time.
