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
- **External definitions, not house units.** Every headline number is
  reported under the definition an outside harness already prints for it
  (next section). A metric only this repository defines cannot be set
  beside an MLX-LM or llama.cpp log, so it cannot answer "how far behind
  the frontier are we" — the question the whole program exists to answer.

## External standards the harness reports in

`seeml-bench` emits each Tier A/C cell twice: once under the internal key
the regression gate has always compared (`rows_per_s`), and once under the
key and definition the external tool uses. The numbers are identical by
construction where the definitions coincide; the point is that a reader
of an `mlx_lm.lora` log or a `llama-bench` table needs no unit translation.

| field | external standard | definition here | note |
|---|---|---|---|
| `tokens_per_s` (token/decoder fixtures), `samples_per_s` (feature fixtures) | MLX-LM `lora` **Tokens/sec** | loss-target rows per wall second of training | MLX counts loss-masked target tokens only; every row of a SeeML batch is a target (next-token label per position), so this is exactly `rows_per_s` — the gate key is unchanged, the unit is now named |
| `it_per_s` | MLX-LM **It/sec** | optimizer steps per second = `1000 / step_ms` | per-step, from the steps-regression slope, so lifecycle cost is excluded as MLX's warm iterations exclude it |
| `step_ms_min` / `step_ms_max` | `llama-bench` **t/s ± σ** | spread of the per-step slope across `--repeats` | medians stay the headline; the spread says whether a delta is signal |
| `train_loss_first` / `train_loss_last`, `trained_tokens` | MLX-LM **Train loss**, **Trained Tokens** | first/last windowed mean training loss over the whole sweep; rows trained | a sanity anchor that the timed work was real training, not a stalled loop |
| `peak_rss_bytes`, `rss_over_planned` | MLX-LM **Peak mem**; Tier C "peak RSS vs arena" | OS-observed peak resident set (`getrusage`); ratio to `arena + plan` | process-cumulative, so it rises monotonically across fixtures; on the MB-scale fixtures the ~20 MB process floor dominates the ratio, so read it at model scale (where the field audits found 1.00–1.13×) |
| `mfu` (only with `--peak-gflops F`) | PaLM / Megatron **model-FLOPs utilization** | achieved GEMM FLOP/s ÷ the host peak you pass | the harness never guesses a peak — a guessed denominator is not a standard. Published anchors (arXiv 2502.05317): Apple M4 GPU ≈ 2.9 TFLOP/s measured, M-series CPU AMX ≈ 1.49, NEON-only ≈ 0.84 |
| `gemm_gflops` | (internal) | `gemm_flops_per_step / step_ms`, FLOPs summed as 2·M·N·K over the plan's own GEMM instructions | the numerator of `mfu`; kept because it needs no external input |
| `host` | every standard's "report the machine" rule | `uname` sysname + machine | the JSON is self-describing; a number without its host string is not a benchmark |

What is deliberately *not* reported: a validation perplexity. The fixtures
train on seeded synthetic corpora, so a val loss here would be an internal
number dressed as lm-eval output. Quality metrics against a real model
(SmolLM-135M val loss/accuracy vs the torch reference) live in the
validation field reports, where the corpus is real text.

Reference frontier numbers for the same host class, for scale: on Apple
M4, `mlx_lm.lora` on SmolLM-135M (bf16, r8, GPU) reports ~6.4 It/sec and
~1,600 Tokens/sec at 1.28 GB peak; `torch.compile` on CPU reaches ~40% MFU
against the AMX peak. SeeML's CPU path sits at 4–13% MFU on the same
shapes — the gap the C2 SIMD and G1b Metal projects are priced against.

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

## The harness

All three pieces of this program exist:

1. **`tool/seeml_bench.cc`** compiles the standard fixture set in-process
   (the seeded builders the test suites share — two MLP stacks, three
   4-block decoders, one token-native decoder) and emits one JSON per run:

   ```bash
   cmake -B build -DSEEML_BENCH=ON && cmake --build build --target seeml-bench
   # or: SEEML_BENCH=1 sh build/build.sh
   ./build/seeml-bench --out bench.json --threads 1,2,4,8
   ```

   Per fixture, per thread width: per-step latency and fixed lifecycle
   cost by **steps-regression** (train `--steps-lo` then `--steps-hi`;
   slope = per-step, intercept = lifecycle — startup never pollutes the
   step number), Tier A `rows_per_s`, effective GEMM GFLOP/s (the FLOPs
   are summed from the plan's own instruction stream, 2·M·N·K over every
   GEMM), the fwd/bwd/optimizer split, and the Tier D lifecycle numbers
   (compile, load+validate, merge, checkpoint save), plus the
   external-standard fields of the table above (`tokens_per_s`,
   `it_per_s`, spread, train loss, peak RSS, and `mfu` when
   `--peak-gflops` names the host peak). Medians of `--repeats`
   (default 3); seeds and thread widths pinned. The CLI is strict, like
   the other tools: an unknown flag or a flag that cannot parse is exit 2,
   never a default. Report schema is 2; every schema-1 key is unchanged,
   so stored baselines stay comparable.

2. **The nightly `bench` CI job** restores the previous night's numbers
   from the actions cache (the fuzz-corpus trust model), runs the harness,
   and **fails on a >10% regression in any Tier A `rows_per_s`** via
   `tool/bench_compare.py` — which also seeds the baseline on first run
   and skips (never fails) fixtures that exist on only one side, so adding
   a fixture can't fail the night it lands.

3. **The step-latency instrumentation** lives in the engine behind
   `-DSEEML_STEP_TIMING` (set by `-DSEEML_BENCH=ON` / `SEEML_BENCH=1`):
   the training stream's phase boundaries — after the last loss-forward
   instruction, and at the first clip/optimizer instruction — are scanned
   once at load, and each step's wall clock is split across them into
   `UpdateEngine::step_timings()`. Default builds, and every emitted
   package, compile the exact untouched dispatch loop; the split answers
   Tier A's "where does the next kernel dollar go" (if bwd ≫ 2×fwd,
   attention backward or GEMM-TN is the target).
