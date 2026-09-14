---
title: "Python plane P7: on-ramp fidelity guards — refuse erf-GELU, carry normalization epsilon in SMF, warn on silently frozen tied heads (ahead of #69)"
labels: correctness,python-plane,core-plane
plane: Python plane
origin: Algorithm review
priority: P1
---
## Root cause

Three places where the exporter accepts a model the device then computes
differently, without a word:

1. **GELU.** `nn.GELU` exports to `OP_GELU` unconditionally
   (`tool/export_model.py:249-253`); the runtime implements only the tanh
   approximation (`runtime/executor/kernel_policy.h:55-63`). torch's
   default is `approximate='none'` (erf). A model pretrained with erf-GELU
   starts from a forward that differs in every MLP layer and the adapter
   must absorb it.
2. **Normalization epsilon.** LayerNorm and RmsNorm use a hard-coded
   `1e-5` (`runtime/executor/normalization.cc:20,77`); the SMF container
   has no epsilon field and the exporter ignores `nn.LayerNorm.eps`
   (`export_model.py:255-268`). Qwen2-class models (`rms_norm_eps=1e-6`)
   fine-tune against a forward that is not their own. **#69 lists "RMSNorm
   eps" in its walker scope but cannot deliver it** without this format
   change.
3. **Tied embedding / LM head.** A weight with any non-matmul user is
   ineligible for LoRA (`compiler/analysis/algebra/lora_grafter.cc:43-51`),
   so a tied `emb` / `lm_head` stays frozen with only a Note; a
   `--targets` filter naming it is silently unmet. GPT-2, Gemma,
   Llama-3.2-1B and Qwen-0.5B are tied. The decoder fixture is untied, so
   the path is untested.

The demos and seeded fixtures never hit any of these; a real checkpoint
hits several at once, and "step 0 equals the source model"
(`lora_grafter.h:44`) becomes approximate without anyone being told.

## Design

- Exporter: `nn.GELU(approximate='none')` is a hard error naming the
  fix (`approximate='tanh'` on the module, or `--gelu-tanh-ok` to accept
  the drift with a printed max-|Δ| estimate). Same rule the CLI already
  applies to inapplicable flags.
- SMF v6: per-op `attr2` = f32 bits of epsilon on `kLayerNorm` /
  `kRmsNorm` (0 = 1e-5, keeping v5 files valid — the v5 `attr1` RoPE
  pattern). Reader/writer/parser/sema/validator/lowering carry it; the
  kernels take `eps` as a parameter (bitwise for `eps = 1e-5`).
- Compiler: a `--targets` substring that matches only ineligible weights
  is an error; an eligible-but-tied head gets a `Note` that names the
  reason. Add a tied-head decoder fixture.
- Golden fixtures for an erf-GELU / eps-1e-6 model land in P6's (#86)
  golden-bytes set so the guards are tested at the seam.

## Acceptance

- Exporting an unmodified `nn.GELU()` model fails with the documented
  message; with `approximate='tanh'` it round-trips byte-identically to
  today.
- A Qwen2-style `eps=1e-6` RmsNorm exports, compiles, and the plan dump
  shows the epsilon on every norm op; step-0 logits match the NumPy
  reference to the parity tolerance #69 uses (6.3e-05).
- `--targets emb` on the tied fixture is a compile error.

## Goal alignment

The torch.compile and MLX comparisons in `docs/benchmarks.md` are run on
the same pretrained checkpoint; if SeeML's step-0 function is not that
checkpoint, its loss trajectory starts from a handicap the kernels cannot
recover, and #69's HF import will reproduce all three drifts at scale.
Prerequisite for an honest frontier comparison, not a throughput item.

Refs: SeeML Algorithm Review (2026-09-14) §02-B/C/E, §04; #69, #86; closed
#68 (the SMF v5 RoPE-θ precedent).
