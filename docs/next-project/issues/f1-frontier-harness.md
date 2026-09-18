---
title: "SeeRL F1: measured frontier harness — real mlx_lm.lora and torch.compile runs on the same host, model, corpus and tokens/step, plus a nightly frontier row"
labels: enhancement,python-plane,frontier-parity
plane: Python plane
priority: P0
---
## Why

Every frontier comparison SeeML has today is either from another host
(`mlx_lm.lora` ~1,600 tok/s on an M4, `torch.compile` ~40% CPU MFU, the
v1.2.4 field report) or a **floor**: `tool/frontier_exec.py price`
interprets a plan op by op with no `mx.compile`, no `torch.compile` and no
fusion. The 2026-09-18 Frontier Outlook therefore had to *speculate* every
frontier cell (SmolLM-135M GPU: SeeML 1,749 tok/s measured vs MLX-LM
2,200–3,200 guessed). The rest of this milestone (F2–F6) is priced against
those guesses; it must be priced against measurements instead.

## Design

1. `tool/frontier_run.py` (python plane, optional tier — needs mlx-lm /
   torch, never imported by the core tools): given an SMF model or an HF
   directory, a corpus and `--seq-len/--batch/--rank`, runs
   - `mlx_lm.lora` with the matching LoRA targets, rank, alpha, batch,
     sequence length and step count, in both `MLX_ENABLE_TF32=0` (f32) and
     the default mode (one process each — MLX latches the switch);
   - a hand-matched PyTorch LoRA loop under `torch.compile` on `cpu` and
     `mps` (inductor on MPS where it compiles; eager MPS otherwise, labelled);
   - SeeML `model_update` on `cpu` and `metal` from the same package.
2. One report per run in `seeml-bench`'s units and definitions (`tokens_per_s`,
   `it_per_s`, `step_ms` by steps-regression, `peak_rss_bytes`, train loss
   first/last, precision mode named on every row), schema-versioned; the
   loss trajectories must agree within a stated tolerance or the row is
   refused (a faster run that trains a different function is not a number).
3. `nightly.yml`: an opt-in **frontier row** on a self-hosted Apple-silicon
   runner (the Linux runner cannot run MLX / MPS / Metal) for
   `tok_smollm135m_q8` and `dec_wide`; trend only, never a merge gate.
4. `docs/benchmarks.md`: replace the carried M4 anchors with the measured
   same-host table; the Frontier Outlook's asterisks become numbers.

## Acceptance

- Same-host table for SmolLM-135M (r8, S=128, 512 tok/step) and dec_wide:
  SeeML cpu/metal, MLX-LM f32/default, torch.compile cpu, torch mps.
- Every row names device, precision mode, library versions and host key.
- Loss-agreement check fails loudly on a mismatched configuration (tested
  with a deliberately wrong rank).
