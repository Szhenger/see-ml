---
title: "SeeAI F2: certified half-precision GPU GEMMs — bf16/f16 simdgroup-MMA and M5 neural-accelerator kernels as a relaxed opcode family behind a P3 certificate"
number: 130
labels: enhancement,efficiency,core-plane,doctrine,frontier-parity
plane: GPU backend
origin: Frontier Outlook
milestone: SeeAI v1.0.0.A
priority: P0
---
## Why

SeeAI's Metal backend computes every GEMM in f32 (bf16/int8 are
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

## Integrated from the 2026-09-24 systems review

Root cause 1, *arithmetic doctrine*: every GEMM is f32 math. The Metal
per-opcode profile of the SmolLM-135M q8 package
(`out/frontier-2026-09-22/prof_q8.txt`, `SEEML_METAL_PROFILE`) shows the
exact kernels are already at the M5's f32 ceiling — the head
512×49152×576 in 8.46 ms (≈3.4 TFLOP/s), the block GEMMs 512×1536×576 in
299 µs and 512×576×576 in 122 µs (2.8–3.0 TFLOP/s) — so no further f32
kernel work can close the 0.36–0.48× bf16 gap; only this issue can. Base
storage does not move the step (int8 1,802 / bf16 1,756 / f32 1,776
tok/s), so dequantization is not the bottleneck either. Per training step
the frozen GEMMs are ≈50 % of kernel time for ≈97 % of the FLOPs: this
issue's ceiling on the step is ≈2× on that half, after which the LoRA
family (F7) and the dispatch model (F8) are the majority of what remains.

Status 2026-09-24: plan v18 `kFlagRelaxed`, the Metal 4 tensor-op GEMMs,
the Accelerate CPU path and the certificate gate at packaging landed in
#136; the SmolLM-135M certificates priced 419 relaxed GEMMs (gemm.relaxed
6.2e-3 / 5.5e-3 against 2^-7, validation within 3e-6). **The Metal rows
are unmeasured**: every row of `out/frontier-2026-09-22/f2_measure/metal.json`
is exit 1 and marked suspect (the session had no Metal access). The CPU
row is the only real F2 number so far (F4 carries it). The first
executable step is therefore the measurement itself, on a host with Metal
access, with validation off so the profile weights forward and backward
as a training step does (F9).

## Integrated from the 2026-09-24 code review

The certified family has gate holes the review reached, and they are this
issue's to close: a relaxed (`--precision certified-bf16`) plan is built into
a runnable binary by the C++ `--build` path with **no numerics certificate**
(N17) — only `pack_update.py` refuses; the certifier's `worst_ratio * rtol`
is a lower bound on relative error, not an upper bound, and with `rtol = 0`
every site measures 0 and NaN passes (F41); `--observed` is not bound to the
plan being verified and its own failures are ignored (F42); `check_observed`
looks for `gemm.relaxed` while reports use opcodes (F72); `run`/`certify`
accept a `--source` whose hash differs from the plan's (N35). Acceptance
gains: every build path refuses an uncertified relaxed plan; the certificate
carries the per-opcode max relative error and refuses `rtol <= 0` and
non-finite values; the certificate binds to the plan hash and the source
hash.

## Acceptance

- SmolLM-135M GPU tok/s measured against F1's MLX-LM **all-bf16** row
  (4,605 tok/s on the M5, the 2026-09-22 frontier row in
  `docs/benchmarks.md`; not the trainer's default mode, which on this
  F32-stored checkpoint is f32 weights under TF32-class matmul); target
  ≥ 0.9× on the same M5.
- Certificate: per-site error inside the stated bound; 300-step val loss
  within 1e-3 relative of the f32 run; gate decision unchanged.
- Two runs of a relaxed plan byte-identical; f32 plans bit-identical to
  today.
