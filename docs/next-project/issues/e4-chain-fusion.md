---
title: "Core plane E4: elementwise chain fusion — kFusedMap (roadmap Phase 1b), closing the fusion-depth gap"
labels: enhancement,efficiency,core-plane,design
plane: Core plane
origin: Both audits
priority: P2
---
## Algorithmic root cause

Inductor fuses arbitrary pointwise/reduction chains into single loops;
SeeAI ships exactly **one** fusion pattern — GEMM bias+activation
epilogues, forward-only (1a, shipped) — while `elementwise.cc` itself
declares the runtime CPU-bandwidth-bound. Every unfused elementwise op is
a full arena round trip. The Frontier Bridge names fusion depth as the
third factor in the CPU gap; it is pure engineering, no doctrine involved.

## Design

This is roadmap **Phase 1b** (`docs/roadmap.md:84-97`), owned here so the
project board carries it: a general `kFusedMap` opcode — a short
micro-program (≤4 unary/binary stages) encoded in the instruction,
executed per element in one pass. The compiler pass builds chains out of
adjacent elementwise ops with single-use intermediates; the validator
checks the micro-program the same way it checks everything else.

Honor the roadmap's own gate: **contingent on post-1a profiling** and
sequenced after 2b — land it when `seeml-bench` Tier B shows elementwise
round trips as the surviving bottleneck once E1/E3 are in (the audits
predict it will be, but the discipline is measure-first).

## Acceptance

- Bitwise-identical to the unfused sequence (same per-element expression
  order — the 1a epilogue precedent: fused write-back calls the same
  inline expressions as the standalone kernels, `kernel_policy.h:47-67`).
- Tier B elementwise-chain fixtures show the removed arena round trips;
  no regression on non-fusable graphs.
- DCE interaction covered (chains dead-code their intermediates — see E5's
  mark-and-compact prerequisite).

Refs: Frontier Bridge §02 (fusion row); Performance Audit §04; roadmap
Project 1 / Phase 1b.
