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
- **Measure what ships, and say what ran.** The CPU GEMM tile geometry
  travels in the plan (v11), so the harness compiles every fixture with the
  policy a package would carry — the runtime defaults, or `--gemm-tiles` /
  `--kernel-policy` exactly as `seeml-update-compile` takes them — and the
  report (schema 3) records the policy, the host key a table is keyed on,
  and the host description. `bench_compare.py` refuses to compare two runs
  whose policies differ: that delta is a tuning result, not a regression.
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
| **backend split** | `seeml-bench --backend metal` vs `cpu` on the same fixtures; `bench.json` carries `"backend"` and the gate keys rows by it | where the GPU pays: on this suite's tiny fixtures Metal runs at the dispatch floor (0.64–3.50× the CPU, Apple M5); on the SmolLM-135M-shaped `tok_smollm135m_q8` fixture 7.7× (1,786 vs 233 tok/s; 27.5× before #104 fixed the CPU's q8 NT kernel) — see "The first GPU baseline row" below |

### The first GPU baseline row (v1.3.0 gate, 2026-09-15)

`tok_smollm135m_q8` on an idle Apple M5 (10 GPU cores), `--threads 10`,
per-step medians by steps-regression; the Metal row at the harness
defaults (20→80 steps, 3 repeats), the CPU row at `--steps-lo 5
--steps-hi 15 --repeats 1` because a default sweep of it takes the better
part of an hour. The step trains 512 tokens; `gemm_gflops` sums 2·M·N·K
over the plan's own GEMM instructions.

| backend | step ms | tok/s | it/s | GEMM GFLOP/s | fwd / bwd / opt ms |
|---|---|---|---|---|---|
| metal | 286.6 | 1,786 | 3.49 | 1,084 | 127 / 158 / 1.9 |
| cpu | 2,194 | 233 | 0.456 | 141.7 | 1071 / 1134 / 13 |
| cpu, before #104 | 7,841 | 65 | 0.128 | 39.6 | 1253 / 6591 / 14 |

The same binary on the real SmolLM-135M package (the T2 import, the docs
corpus, `model_update --backend metal`) measures 1,749 tok/s by the
same regression; the fixture's seeded weights and corpus reproduce the
model's per-step cost, not its loss. The CPU row was re-measured on
2026-09-15 after #104: `GemmNTQ8` — the dX backward through every frozen
projection, 208 of them per step at this geometry — had run scalar since
#66 because the int8 cast sat inside the vectorized lane loop; widening
each block first (bit-identical, see `gemm.cc`) took the backward from
6.6 s to 1.1 s and the row from 65 to 233 tok/s.

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
| **suite runtime top-10** | slowest tests trend | keeps the 367-test suite honest — one 60 s test taxes every diff forever |
| **fuzz corpus coverage** | edges covered nightly (libFuzzer `-print_final_stats`) | whether the fourth arm (compile-of-SMF) is still finding new ground or needs structure-aware mutators |

## Reference points from this branch (not benchmarks — sanity anchors)

Measured once on the dev host (Apple Silicon, default threads) during
validation, to seed expectations until the harness lands: the demo MLP
e2e package compiles + builds in seconds; 300 training steps of the
16-wide MLP run in ~1 s; the token tiny-GPT (V=16, D=8, S=4) trains 300
steps to 0.7× loss in ~2 s; serial vs 8-thread committed models are
bitwise identical (the determinism overhead is therefore *measurable as
pure speedup*, no correctness tax).

## Tuning the kernel policy (`tool/autotune.py`)

The one knob the CPU kernels expose — the blocked GEMM's K and N tiles —
is measured offline, per host, by the Python plane, never inside the
compiler (Two-Plane Overhaul P2, #76; the design is in
[compiler.md](compiler.md)). One command tunes a host:

```bash
python3 tool/autotune.py tune --bench build/seeml-bench --out kernel_policy.json \
    --fixtures mlp_128x1024x2,dec_v256_d128_s128,dec_v512_d192_s32 --threads 10
python3 tool/autotune.py show kernel_policy.json
seeml-update-compile ... --kernel-policy kernel_policy.json   # consumes it
```

Every arm is one `seeml-bench --gemm-tiles K,N` run under this document's
discipline; arms are visited round-robin for `--rounds` (3) rounds and
each arm's per-key result is the median over rounds — medians of medians.
The score is the geometric mean over (fixture, threads) keys of rows/s
relative to the kernel-default arm, which is always swept, as is the
compiler's analytic tiling (`analytic_gemm_tiles` in the report — the
hypothesis #90 found 1.3–3.3× slow, now measured instead of emitted). The
winner is recorded only if it beats the defaults by `--min-gain` (3 %);
otherwise the table records that the defaults are best. Either way every
arm's numbers are kept beside the decision (`tuned.arms`), so the table is
an audit trail, not just an answer. The table is keyed on the host key the
bench printed; the compiler recomputes the same key and writes the tiles
into the plan header.

**Measured on the development host (2026-09-15, Apple M5, 10 threads,
fixtures `mlp_128x1024x2`, `dec_v256_d128_s128`, `dec_v512_d192_s32`, 3
rounds x 3 repeats, 18 arms, 10 minutes):** the kernel defaults are the
policy. Scores are geometric-mean rows/s relative to the default arm.

| arm | label | score |
|---|---|---|
| 256x512 | grid (best) | 1.005 |
| 32x256 | grid | 1.004 |
| 64x512 | grid | 1.004 |
| **64x256** | **default (recorded)** | **1.000** |
| 128x128 | grid | 0.989 |
| 64x64 | grid | 0.976 |
| 256x64 | grid | 0.960 |
| 512x16 | analytic (`SuggestGemmTiling`) | 0.903 |

Decision: `default-within-margin` — the best arm's 0.5 % is inside the
3 % noise line, so the table records 64x256 for this host key
(`arm64;Apple M5;cores=10;l1d=65536;l2=6291456;simd=4`). The analytic
tiling that packages used to ship with is 9.7 % slower here, in line with
#90's finding at larger shapes. A tuned entry is therefore not what this
host gets from P2; what it gets is the measurement that says so, and a
gate that cannot be fooled by the tiling again.

## The harness

All three pieces of this program exist:

1. **`tool/seeml_bench.cc`** compiles the standard fixture set in-process
   (the seeded builders the test suites share — two MLP stacks, three
   4-block decoders, one token-native decoder, plus an opt-in
   SmolLM-135M-shaped token decoder, `tok_smollm135m_q8`, that runs only
   when `--fixtures` names it: 30 blocks of D=576 / 9 heads / ffn 1536
   over a 49,152 vocabulary, q8 base, 512 tokens per step — the frontier
   row below, kept out of the default set and the nightly keys because a
   CPU sweep of it takes the better part of an hour) and emits one JSON
   per run:

   ```bash
   cmake -B build -DSEEML_BENCH=ON && cmake --build build --target seeml-bench
   # or: SEEML_BENCH=1 sh build/build.sh
   ./build/seeml-bench --out bench.json --threads 1,2,4,8
   # the geometry a package ships with: --gemm-tiles K,N or
   # --kernel-policy table.json [--target-host KEY], as the compiler takes them
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
   a fixture can't fail the night it lands. Two fail-closed rules keep the
   gate honest: a comparison with **zero overlapping keys is an error**
   (a renamed metric must re-seed deliberately, not pass silently), and a
   **pinned epoch baseline** (`--epoch-baseline`, cache key `bench-epoch-v1`,
   saved once and never rolled forward) is diffed at a 15% threshold so a
   sub-10%/night drift cannot compound unnoticed.

3. **The step-latency instrumentation** lives in the engine behind
   `-DSEEML_STEP_TIMING` (set by `-DSEEML_BENCH=ON` / `SEEML_BENCH=1`):
   the training stream's phase boundaries — after the last loss-forward
   instruction, and at the first clip/optimizer instruction — are scanned
   once at load, and each step's wall clock is split across them into
   `UpdateEngine::step_timings()`. Default builds, and every emitted
   package, compile the exact untouched dispatch loop; the split answers
   Tier A's "where does the next kernel dollar go" (if bwd ≫ 2×fwd,
   attention backward or GEMM-TN is the target).
