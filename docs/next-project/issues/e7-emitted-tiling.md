---
title: "Core plane E7: the emitted GEMM tiling is 1.3–3.3× slower than the kernel defaults — stop emitting it, bench what ships, re-premise E1 stage 2"
labels: correctness,efficiency,core-plane,sev:high
plane: Core plane
origin: Algorithm review
priority: P0
---
## Algorithmic root cause

`SuggestGemmTiling` (`compiler/backend/architecture/host_arch.cc:172-178`)
defines `nc` as a **register-block width** — "4 vectors of C columns" —
which is the right quantity for a packed BLIS microkernel. The runtime has
no such kernel: `BlockedNN` (`runtime/executor/gemm.cc:62-83`) consumes
`SEEML_GEMM_TILE_N` as the **N cache tile** of an unpacked `k0→n0→m→k→n`
nest. With `nc = 16` every A row is re-streamed N/16 times and the
vectorized inner loop is 16 wide.

`native_emitter.cc:373-375` bakes the pair into every emitted `build.sh`.
On an Apple M-series host the line is

```
TILE_FLAGS="${SEEML_TILE_FLAGS--DSEEML_GEMM_TILE_K=512 -DSEEML_GEMM_TILE_N=16}"
```

Measured 2026-09-14 (`GemmNN`, same binary built both ways, medians):

| shape (M×K×N) | threads | kernel defaults 64/256 | emitted 512/16 | slowdown |
|---|---|---|---|---|
| 1024×576×8192 (lm-head) | 1 | 18.9 GF/s | 5.7 GF/s | 3.3× |
| 1024×576×8192 | 8 | 84.7 GF/s | 40.3 GF/s | 2.1× |
| 128×192×768 | 8 | 129.7 GF/s | 98.2 GF/s | 1.3× |
| 32×128×1024 | 8 | 95.7 GF/s | 70.6 GF/s | 1.4× |

`GemmNT` is unchanged (4.8–4.9 GF/s serial either way — that is #66).

The nightly bench cannot see this: `seeml-bench` is built through CMake
with no tile flags (`nightly.yml`), so the gate measures a tiling **no
package ships**. `native_emitter_test.cc:106-109` only asserts the defines
exist.

**This inverts the premise of E1 (#80) stage 2**, which calls the kernel's
64 KiB default B panel "2× a typical L1" and plans to bring hand-built
packages onto the compiler-tuned half-L1 contract. On the kernel that
exists, the default panel is the fast configuration and the tuned one
thrashes. P2 (#76) inherits the same analytic fallback.

## Design — bitwise-safe

1. **Now:** `native_emitter.cc` stops emitting `TILE_FLAGS` from
   `SuggestGemmTiling` (emit the empty default; keep the `SEEML_TILE_FLAGS`
   override hook). Tile choice never changes bits (the K tile must stay a
   multiple of 4, enforced by `static_assert`).
2. **Bench parity:** `seeml-bench` gains `--tile-flags` (or the CMake bench
   target reads the compiler's `--report` tiling), so Tier A/B numbers are
   taken on the configuration packages actually ship. Record the tiling in
   `bench.json`.
3. **Contract rewrite:** `ValidateGemmTiling` / `SuggestGemmTiling` are
   re-derived for the unpacked nest (N tile = the vectorized sweep width
   the core wants, K tile = the A-row reuse window), or retired until E1
   stage 2 lands the packed microkernel the current formula assumes.
   `docs/compiler.md` "Host architecture and GEMM tiling" corrected.
4. E1 (#80) stage 2 and P2 (#76) re-premised: the analytic default is not
   the baseline to beat; the kernel defaults are.

## Acceptance

- A package emitted by `--build` and one built with `SEEML_TILE_FLAGS=""`
  train at the same rows/s on every fixture (±5 %), and both match
  `seeml-bench` for the same fixture.
- Bit-identical committed models before/after on every system test.
- `seeml-bench --out` records the tiling in use; the nightly gate's keys
  include it.

## Goal alignment

Objective 3 (CPU MFU 4–13 % → 20–30 % of NEON peak, the torch.compile
bridge): this is the largest CPU-side win with zero risk, and every later
CPU measurement (E1 go/no-go, P2 tuning, P4 ceiling pricing) is
mis-baselined until the bench and the package run the same kernel
configuration. Objective 5 (trustworthy measurement): the gate must
measure what ships.

Refs: SeeML Algorithm Review (2026-09-14) §02-E, §04; Performance Audit
§04 (GEMM table row "default tiles give a 64 KiB B panel" — now known to
be the fast case); #80, #76, #66, #79.
