# Benchmarks — The Metrics That Gate Frontier Development

As the frontier program lands (token-native training now; QLoRA-class
quantization, GPU dispatch, chunked cross-entropy, and scale passes next),
"is it faster / does it still fit" stops being something you can eyeball.
This document defines the metric program: what to measure, why each metric
gates a specific frontier decision, and the measurement discipline that
keeps numbers comparable across commits.

## Measurement discipline (non-negotiable)

- **Fixed seeds, fixed inputs, pinned `SEEML_THREADS`.** Every benchmark
  is a compiled plan + seeded corpus; determinism makes medians of 5 runs
  tight enough that a >3% delta is signal, not noise.
- **Medians, not means; report the machine.** Apple M-series vs x86 AVX2
  are different regimes — a number without its host string is not a
  benchmark.
- **Per-commit trend, not absolutes.** The gate is regression against the
  previous commit's number on the same host (CI can carry the baseline as
  an artifact the way the fuzz corpus is cached).

## Tier A — Training throughput (the headline numbers)

| metric | definition | what it gates |
|---|---|---|
| **tokens/sec** (token plans) | `batch × steps / wall` on the token tiny-GPT and a mid decoder (e.g. D=512, 8 heads, S=256) | every kernel/GPU/fusion decision rolls up here; THE frontier comparison number against MLX / llama.cpp-finetune |
| **step latency breakdown** | wall time of one `ExecuteTrainOnce`, split fwd / bwd / optimizer (instrument at the instruction-stream level: the three programs' boundaries are known offsets) | where the next kernel dollar goes — if bwd ≫ 2×fwd, attention backward or GEMM-TN is the target |
| **samples/sec** (feature plans) | the same for MLP-class plans | regression canary for the classic path while transformer work churns shared kernels |
| **scaling efficiency** | tokens/sec at `SEEML_THREADS=1,2,4,8` vs ideal | the parallel_for chunking + feeder overlap health; a flat curve says grain constants (`kGrainCheap/kGrainMath`, `RowGrain`) need retuning |

## Tier B — Kernel-level (why Tier A moved)

| metric | definition | what it gates |
|---|---|---|
| **GEMM GFLOP/s vs peak** | blocked cores at the shapes the decoders actually emit (projections, SwiGLU, lm-head) | the explicit-SIMD kernel project (frontier pillar C2): if autovectorized cores sit at <30% of peak, hand SIMD is worth it; if 60%+, it is not |
| **attention μs and bytes** | `AttnFwd` + backward chain per (B,H,S,d) sweep of S ∈ {64, 256, 1024} | the flash-attention decision: the S² probs cache's cost curve tells you exactly when the tiled rewrite pays |
| **EmbedFwd GB/s** | gather bandwidth vs `memcpy` bandwidth | it should be memory-bound; if not, the row-grain is wrong |
| **softmax-xent μs at vocab ∈ {1k, 32k, 128k}** | fwd+bwd per row | the chunked-CE project (pillar A3): this curve IS the justification |
| **q8 dequant overhead** | q8 GEMM / f32 GEMM time ratio | NF4/int4 design: if int8 dequant already costs >15%, block-wise 4-bit needs a fused design, not a naive port |
| **Metal vs CPU GEMM** | the G1a harness at real shapes | the G1b engine-integration go/no-go: dispatch overhead amortization point (at which M×N×K does GPU win?) |

## Tier C — Memory (the gate that refuses compiles)

| metric | definition | what it gates |
|---|---|---|
| **arena bytes / model bytes** | header `arena_size` vs source `.smf` size across model scales | the honest cost of training; grad-accumulation (2a) and remat (2b) success = this ratio falling |
| **arena breakdown** | persistent / IO / transient split (compile report already knows it) | which memory project fires first: transients ⇒ remat; persistent ⇒ 8-bit optimizer states |
| **probs-cache share** | attention P cache bytes / arena bytes as S grows | same flash-attention decision from the memory side |
| **estimator honesty ratio** | step-0 estimate / exact final gate value | must stay ≥ 1 and near 1 — the review found it 16× off for teachers once; this metric keeps the lower-bound claim measured, not asserted |
| **peak RSS vs arena** | OS-observed peak / (arena + plan) | the "one allocation" doctrine, verified: a drifting ratio means hidden allocation crept in |

## Tier D — Lifecycle latencies (the product feel)

| metric | definition | what it gates |
|---|---|---|
| **compile wall time** | `seeml-update-compile` end to end, per model scale | developer loop; autodiff/fusion passes are O(ops²) risks as models deepen |
| **load+validate ms** | `LoadFromMemory` (hash + contracts + validator over every instruction) | on-device startup; the validator is O(instructions) and must stay trivial vs training |
| **plan size / delta size** | `.seeu` bytes; later: adapter-only package bytes | the delta-only update-package project (pillar D2) — the headline there is this number collapsing from model-scale to adapter-scale |
| **gate (eval) time share** | 2×eval-pass wall / total update wall | if the gate costs >20% of the update, eval batching or fused-eval work is justified |
| **merge+commit ms/MB** | `RunMerge`+`CommitToModel` per model MB | the streamed COW commit's O(chunk) claim under load; the new delta-finiteness scan rides this path and must stay invisible |
| **checkpoint save/load ms** | per persistent-segment MB | resume UX; 8-bit optimizer states will change this |

## Tier E — Development-velocity metrics (speeding up the project itself)

| metric | definition | what it gates |
|---|---|---|
| **CI wall time per job** | each ci.yml job's duration trend | the per-diff feedback loop; when build-and-test crosses ~10 min, precompiled-header or unity-build work pays |
| **full local build time** | `build/build.sh` clean | same loop locally; the single biggest dev-speed lever in a -O2 -Werror tree |
| **suite runtime top-10** | slowest tests trend | keeps the 251-test suite honest — one 60 s test taxes every diff forever |
| **fuzz corpus coverage** | edges covered nightly (libFuzzer `-print_final_stats`) | whether the fourth arm (compile-of-SMF) is still finding new ground or needs structure-aware mutators |

## Reference points from this branch (not benchmarks — sanity anchors)

Measured once on the dev host (Apple Silicon, default threads) during
validation, to seed expectations until the harness lands: the demo MLP
e2e package compiles + builds in seconds; 300 training steps of the
16-wide MLP run in ~1 s; the token tiny-GPT (V=16, D=8, S=4) trains 300
steps to 0.7× loss in ~2 s; serial vs 8-thread committed models are
bitwise identical (the determinism overhead is therefore *measurable as
pure speedup*, no correctness tax).

## What to build next for this program

1. A `tool/seeml_bench.cc` harness: compiles the standard fixture set,
   runs Tier A/B sweeps with pinned seeds, and emits one JSON per run —
   the same strict-CLI discipline as the other tools.
2. A CI `bench` job (nightly, not per-diff): runs the harness, diffs
   against the cached baseline artifact, and fails on >10% regression in
   any Tier A metric — the same trust model as the fuzz corpus cache.
3. The step-latency instrumentation: three timestamps around the
   train/eval/merge program boundaries in `ExecuteTrainOnce`, behind a
   compile-time flag so the shipped runtime stays untouched.
