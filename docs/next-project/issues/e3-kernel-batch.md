---
title: "Core plane E3: bitwise-safe kernel batch — RoPE sin/cos table, ReduceRows false sharing, clip-fused optimizer step, parallel eval argmax"
number: 82
labels: enhancement,efficiency,core-plane
plane: Core plane
origin: Performance audit
priority: P1
---
## Algorithmic root causes

Four independent kernel inefficiencies, every fix **provably
bit-identical** under the determinism contract (Performance Audit findings
#3–#6):

1. **RoPE recomputes sin/cos per (b, h)** of values that depend only on
   (s, c) (`attention.cc:56-89`). A [S, d/2] table built with the
   *identical* recurrence eliminates ~99% of the transcendental work; it
   needs one compiler-allocated arena slot (same pattern E1 stage 2 adds
   for pack buffers).
2. **`ReduceRows` false sharing** — the bias-gradient kernel's adjacent
   chunks read-modify-write shared boundary cache lines once per row
   (`elementwise.cc:59-66`). Chunk-local accumulation, then a fixed-order
   combine: several× at high thread counts.
3. **`ClipNorm` + AdamW stream every gradient tensor 3× per step**
   (`optimizer.cc:17-68`). Folding the clip scale into the step
   instruction (one ABI operand) removes a full read+write pass —
   verify with `opt_seconds`; ~⅓ of optimizer-phase bandwidth.
4. **The eval accuracy argmax is a serial, scalar O(rows × classes) loop**
   (`update_engine.cc:637-644`) — invisible at 10 classes, plausibly the
   dominant eval cost at a 32k vocab while the GEMM that fed it used all
   cores. Parallelize with the standard chunk geometry.

## Acceptance

- Bit-identity on every fixture, serial vs 8-thread, before/after each
  item independently (they must be landable one at a time).
- Tier B/C: RoPE and bias-grad kernel timings drop as predicted;
  `opt_seconds` shows the removed pass; eval time at large-vocab fixtures
  becomes negligible.
- Items 1 and 3 bump the plan version (arena slot / ABI operand) with
  validator coverage for the new operand.

Refs: Performance Audit §04 (kernels + engine); pairs with #66 and E1.
