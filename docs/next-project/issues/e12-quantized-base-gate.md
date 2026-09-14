---
title: "Core plane E12: under --quantize-base the gate scores W_q + Δ but commit ships W_f32 + Δ — score what ships; per-column int8 scales"
labels: correctness,core-plane,sev:medium
plane: Core plane
origin: Algorithm review
priority: P2
---
## Root cause

With `--quantize-base`, both the train and the eval programs lower
against `quant_scales` (`compiler/driver/update_compiler.cc:382-385`,
`instruction_lowering.cc:58-74` → `kGemmNNQ8` / `kGemmNTQ8`). The adapter
therefore learns Δ conditioned on **W_q** and partially compensates
`W_q − W`; the improvement gate validates `W_q + Δ`. Commit then writes
`W_f32 + Δ` onto the pristine source (`runtime/engine/update_engine.cc:864-895`).
The docs' "quantization error is never baked in" is byte-true, but the
function that ships is not the function the gate scored — the QLoRA merge
mismatch. No test compares committed-model loss with gated loss.

The scheme is per-tensor symmetric max-abs (`compiler/analysis/reviewer/quantization.cc:50`);
LLM projections carry |w| outliers 10–50× the bulk, which collapses most
weights to a few levels and widens exactly the `W_q − W` gap the adapter
absorbs. Teacher weights are quantized too (no exclusion), perturbing the
distillation target.

## Design

1. **Score what ships:** after `RunMerge`, evaluate the validation set
   once through an f32 program with the merged deltas applied
   (the eval program lowered against un-quantized rodata is one more
   instruction stream; or evaluate the committed file through the same
   package before the rename). The gate's recorded `val_final` is that
   number; report both.
2. **Per-output-column scales:** one `float[M]` per weight, applied in the
   GEMM's column epilogue; no loop restructuring. Version-gated (rodata
   layout changes). Teacher weights excluded from quantization by default.
3. Test: committed-model loss vs gated loss within a stated tolerance on
   the quantized-base system fixture; a synthetic outlier weight shows the
   per-column error reduction.

## Acceptance

- `report.json` carries `val_final_shipped`; the gate uses it under
  `--quantize-base`.
- Per-column scales reduce max-abs dequant error on the outlier fixture
  by ≥ 10× with < 3 % GEMM-time overhead (Tier B q8 ratio).

## Goal alignment

MLX-LM's QLoRA path fuses dequant into the matmul with group-wise scales
and merges onto the quantized weights it trained against; SeeML's frontier
config is a q8 base (#63 item 3). The comparison is only fair if SeeML's
gate proves the function it ships and its quantization error is in the
same class as MLX's group-wise scheme.

Refs: SeeML Algorithm Review (2026-09-14) §02-B; #63; `docs/compiler.md`
"Quantization review".
