---
title: "SeeRL F2: certified half-precision GPU GEMMs — bf16/f16 simdgroup-MMA and M5 neural-accelerator kernels as a relaxed opcode family behind a P3 certificate"
labels: enhancement,efficiency,core-plane,doctrine,frontier-parity
plane: GPU backend
priority: P0
---
## Why

SeeML's Metal backend computes every GEMM in f32 (bf16/int8 are
*storage* formats widened into f32 panels, #71/#102). MLX runs bf16 or
TF32 by default and, on M5-class GPUs, can reach the per-core neural
accelerators. That precision gap — not compiler design — is the largest
single cause of the speculated 0.55–0.8× SmolLM-135M GPU ratio versus
MLX-LM (Frontier Outlook, 2026-09-18). GEMMs are ~65% of a decoder step
(`seeml-plan-probe --profile`, E1 #80); fusion work moved step time <6%.

## Doctrine

Bitwise determinism stays the default. `docs/roadmap.md` already states
the rule: a relaxed computation is an **opt-in opcode family** that the
old path never emits, admitted only for a plan that
`tool/certify_numerics.py` has certified on its own operands. The tool
and the contract exist (P3, #77); the opcode family does not. This issue
builds the first member of it. Relaxed kernels must still be
run-to-run deterministic on a given device (fixed reduction order, no
atomics), so two runs of the same relaxed plan stay byte-identical.

## Design

1. Plan: relaxed GEMM opcodes (or a `kFlagRelaxedPrecision` bit on
   kGemmNN/NT/TN) carrying the compute type (bf16 or f16 inputs, f32
   accumulate); a header field or section recording the certificate hash;
   the validator refuses relaxed instructions without it. SEEU version bump.
2. Compiler: `--precision f32|certified-bf16|certified-f16`; lowering picks
   the relaxed opcode only for the frozen-weight GEMMs and the LoRA
   projections the certificate covers; optimizer, norms, softmax and loss
   stay f32.
3. Metal: bf16/f16 `simdgroup_matrix` GEMM templates (same tile family as
   the f32 kernels, #103); a Metal 4 tensor-op path for M5 neural
   accelerators behind a device-capability check, falling back to simdgroup.
4. `certify_numerics.py`: price the relaxed GEMM sites against the f64
   oracle, emit the certificate; `pack_update.py` refuses an uncertified
   relaxed plan (it already checks certificates).
5. CPU backend: executes relaxed opcodes by widening to f32 (reference
   semantics), so every plan still runs everywhere.

## Acceptance

- SmolLM-135M GPU tok/s measured against F1's MLX-LM row; target ≥ 0.9×
  MLX-LM default mode on the same M5.
- Certificate: per-site error inside the stated bound; 300-step val loss
  within 1e-3 relative of the f32 run; gate decision unchanged.
- Two runs of a relaxed plan byte-identical; f32 plans bit-identical to
  today.
