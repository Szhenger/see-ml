---
title: "Core plane E11: attention materializes three S² matrices per layer in double-accumulated f32 — decide and land the tiled (flash-style) backward before G1b-4 freezes the cached-P design"
labels: enhancement,efficiency,design,core-plane
plane: Core plane
origin: Algorithm review
priority: P1
---
## Algorithmic root cause

`AttnFwd` caches `P` as `[B·H·S, S]` f32 (`runtime/executor/attention.cc:94-136`)
and the attention VJP (`compiler/analysis/calculus/autodiff.cc:242-278`)
materializes `dP` and `dS` at the same size; `P` stays live from forward
to backward for every layer. At S = 2048, H = 8, B = 1 that is **134 MB
per matrix per layer**; a 12-layer model needs ~1.6 GB for `P` alone —
this, not weights, is the sequence-length ceiling on a device. `AttnDV`
(`:176-177`) and `AttnDK` (`:235-236`) also walk `P` / `dS` column-strided
(one useful element per cache line for S ≥ 16).

`docs/benchmarks.md` names "the flash-attention decision" as a Tier B/C
question and stops there; G1b-4 (#63) ports the cached-P decomposition to
Metal as-is, which would freeze the O(S²) design into a second backend.

## Design

1. **Tier B/C sweep first** (the harness the bench doc specifies:
   `AttnFwd` + backward chain at S ∈ {64, 256, 1024, 2048}; probs-cache
   share of the arena) — the go/no-go the roadmap asked for, run and
   recorded.
2. Tiled forward that stores only the per-row `(max, logsumexp)` plus `O`;
   tiled backward that recomputes `P` per (q-block, k-block) and
   accumulates `dQ`, `dK`, `dV` in a fixed block order — deterministic per
   backend by construction (block geometry is a pure function of shape).
   ~2× attention FLOPs for O(B·H·S) memory.
3. New opcodes (`kAttnFwdTiled`, `kAttnBwdTiled`) behind a version bump;
   the old family stays for plans that fit (the compiler picks by
   `probs_cache_bytes` vs a threshold in the memory gate).
4. Fold the Algorithm Review's medium item — scores/`dP`/rowsum
   accumulated in `double` — into P3's (#77) first certificate; the tiled
   kernel's f32-vs-f64 accumulation choice is the certified knob.

## Acceptance

- Arena high-water at S = 2048 on the token decoder fixture drops from
  O(S²) to O(S) per layer (report the ratio); step time within 2× of the
  cached-P path at S = 256 and faster beyond the crossover the sweep finds.
- FD gradient checks pass on the tiled family; serial-vs-8-thread bitwise.
- #63 targets the tiled family for its attention kernels.

## Goal alignment

MLX-LM fine-tunes SmolLM at 2048 tokens with a fused SDPA and 1.28 GB
peak; SeeML's O(S²) cache makes the same sequence length a memory-gate
refusal on the devices it targets. This is a prerequisite for the
v1.3.0 gate (#65) to compare at MLX's sequence lengths, and for
objective 3's Tier B attention row to mean anything at S > 256.

Refs: SeeML Algorithm Review (2026-09-14) §02-C; Performance Audit §04
finding #9; #63, #65, #77.
