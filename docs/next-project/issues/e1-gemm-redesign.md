---
title: "Core plane E1: GEMM redesign — m-hoist (bitwise), MR×NR microkernel with arena-planned pack buffers, 2D partition for small-M"
labels: enhancement,efficiency,core-plane
plane: Core plane
origin: Both audits
priority: P0
---
## Algorithmic root cause

100% of training FLOPs go through two autovectorized blocked loop nests —
zero intrinsics in the repo, no panel packing, no M register blocking.
Vendor BLAS (what `torch.compile` calls) packs panels and holds C in an
MR×NR register block at 60–90% of peak; SeeAI sits at **4–13% MFU**. The
Frontier Bridge prices this single factor at **≈3–5×** — the largest
*engineering* (non-doctrinal) term in the CPU gap.

## Design — three stages, each gated on Tier B measurements

1. **m-hoist in `BlockedNN`** (`gemm.cc:62-83`). The current
   `k0→n0→m→k→n` order streams C from memory ⌈K/64⌉ times; hoisting `m`
   outside the k-tile loop keeps the C row L1-resident across the K sweep.
   **Bitwise-identical** (the k order per (m,n) is unchanged) and
   near-free — Performance Audit finding #1, land first.
2. **MR×NR microkernel + packing inside the one-allocation contract.**
   Packing needs scratch, and the doctrine forbids runtime allocation —
   so the *compiler* plans the pack buffers as arena segments (it already
   computes `mc` and discards it for CPU, `native_emitter.cc:373-375`) and
   the step-0 memory gate prices them honestly. Today's ~4 FMAs per C
   round-trip becomes 4–8× more per the audit's arithmetic. Also fix the
   64 KiB default B panel (2× a typical L1, `gemm.cc:33-40`) so
   hand-built packages stop thrashing.
3. **2D partition of C.** Threading currently splits C rows only, so a
   small-M GEMM — a LoRA down-projection with M = rank — runs on one core
   (`gemm.cc:175-225`). Chunk over an (M,N) grid with the same static
   geometry + dynamic claiming so determinism is preserved.

`BlockedNT` (the dX backward, measured 3–5× slower than NN at equal
FLOPs) stays owned by **#66** — its split-accumulator fix should land on
the redesigned nest, not the old one.

## Acceptance

- Stage 1: no bit change on any fixture; measurable Tier B win on
  large-K shapes.
- Stages 2–3: serial-vs-8-thread bit-identity holds; Tier B MFU on
  `dec_wide`-class GEMMs reaches the **20–30%-of-NEON-peak** band the
  Frontier Bridge names as the in-doctrine target.
- `seeml-bench` bwd/fwd ratio (with #66) drops from 3.4–4.5× toward ~2×.

Refs: Performance Audit §04 (GEMM table, findings #1); Frontier Bridge
§02 (kernel-quality row); roadmap microkernel go/no-go; #66, P2 (tuned
tile arms).
