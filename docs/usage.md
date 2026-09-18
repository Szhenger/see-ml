# Using SeeML

## The Shape of the Workflow

SeeML compiles an on-device model update *ahead of time*: LoRA adapters are grafted onto a frozen model, the backward pass and optimizer are synthesized as a fixed instruction stream bound to a pre-planned arena, and the result is executed on-device by a zero-dependency VM. One `.seeu` plan = one complete, gated, resumable, atomically-committed update.

The workflow has three steps on two machines:

```
build host:  1. export   PyTorch model  ──▶  model.smf (+ corpus.sds)
             2. compile  model.smf      ──▶  pkg/  (plan + vendored runtime + build.sh)
device:      3. update   model.smf + corpus.sds  ──▶  updated.smf, or no change at all
```

Notice what does *not* travel to the device: PyTorch, Python, this repository. The device receives a folder that builds with any C++23 compiler. If you want to understand what each step does internally, [compiler.md](compiler.md) and [runtime.md](runtime.md) go deep; the binary files exchanged between the steps are specified in [formats.md](formats.md). This document just gets you running.

## Step 1: Export the Model (build host, PyTorch)

The exporter needs Python 3 with PyTorch and NumPy on the build host. The stack it is developed and tested on is CPython 3.14.7 with PyTorch 2.14 and NumPy 2.5 (`pip install -r tool/requirements-pinned.txt`); it also runs on anything down to Python 3.9, NumPy 1.17 and torch 1.7 (`tool/requirements.txt` states those floors), and the compiler reads every SMF/SDS file any version has written. Export streams the model straight from the framework's arrays to disk, so a 500 MB model needs about 500 MB, not several times that. Nothing on the device side touches Python.

The quickest start — a demo model, teacher, and synthetic corpus in one command:

```bash
python3 tool/export_model.py --demo out/
```

This writes `model.smf` (a small `Linear(16,32) → ReLU → Linear(32,4)` classifier), `teacher.smf` (a wider sibling, for distillation experiments), and `corpus.sds` (2,048 labeled samples). Everything downstream can be tried against these three files.

Every demo dimension is a flag, so randomized experiments need no bespoke script: `--width`, `--depth`, `--samples`, `--seed`, and `--corpus-kind class|dense|none` (`none` writes the unlabeled corpus that `--loss kl` distillation wants) for `--demo`; add `--vocab`, `--heads`, `--seq-len`, `--blocks`, `--ffn`, and `--rope-base` (the rotary θ — 10000 by default, 500000 for Llama 3, 1000000 for Qwen; written per Rope op as SMF v5 `attr1` and lowered into the plan verbatim) for `--demo-decoder`. Defaults reproduce the classic demos byte-for-byte, and — as with `seeml-update-compile` — a flag that cannot apply to the requested mode is a hard error (exit 2), never silently ignored. There is also `--corpus data.npz out.sds`, which converts saved NumPy arrays (`records` for token corpora; `inputs` + optional `labels` for feature corpora) into an SDS file without writing any Python.

For your own model, use the two functions the script exports:

```python
from export_model import export_smf, export_sds
export_smf(sequential_model, "model.smf")   # Linear/ReLU/GELU/SiLU/LayerNorm
export_sds(inputs, labels, "corpus.sds")    # labels: int32 classes, dense f32, or None
```

The exporter accepts an `nn.Sequential` of `Linear`, `ReLU`, `GELU`, `SiLU`, and `LayerNorm` modules — anything else is a loud `ValueError`, not a silent skip. One detail worth knowing so the format makes sense later: PyTorch stores a `Linear`'s weight as `[out, in]`, but SMF's `MatMul(x, W)` wants `[in, out]`, so the exporter transposes on the way out. Labels: pass int32 class indices for cross-entropy, dense float vectors for MSE, or `None` for a distillation corpus (the teacher provides the targets).

**Token-native decoders.** For a decoder transformer that consumes raw token ids (SMF v4+; the writer emits v5, which adds the per-op RoPE base), export the frozen embedding table alongside the blocks, and a corpus of plain ids — no labels, no embedded vectors:

```python
from export_model import export_token_decoder_smf, export_token_sds
export_token_decoder_smf(embedding, blocks, head, "decoder.smf",
                         seq_len=S, num_heads=H)   # embedding: [V, D] f32
export_token_sds(records, "decoder_corpus.sds")    # records: [N, S+1] i32
```

**Importing a Hugging Face decoder.** A local Llama-class checkpoint directory (`llama`, `qwen2`, SmolLM2 — `config.json` plus `model.safetensors` or its shard index) becomes a token-native SMF with NumPy alone:

```bash
huggingface-cli download HuggingFaceTB/SmolLM-135M            # once; any local dir works
python3 tool/export_model.py --hf <model_dir> smollm.smf --seq-len 128 \
  --text-corpus docs.txt docs.sds --hf-parity
```

The walk transposes every Linear to `MatMul(x, W)` layout, repeats grouped-query k/v heads to one per query head (the format carries no KV heads yet), permutes q/k features within each head from Hugging Face's rotate-half RoPE pairs `(c, c + d/2)` to SeeML's interleaved `(2c, 2c+1)` (the scores are a dot product over `d`, so a permutation applied to both sides is exact), carries `rope_theta` per Rope op, adds Qwen2's q/k/v biases as `AddBias` ops, and ties `w_head` to the embedding when the checkpoint does. What it refuses, loudly: a non-SwiGLU block, `rope_scaling`, MLP biases, a `--seq-len` past `max_position_embeddings`, and an `rms_norm_eps` other than the runtime's fixed `1e-5` (Qwen2's `1e-6` is a drift; `--allow-eps-drift` accepts it knowing step 0 will not equal the source model — the attribute is P7, #96). Two tier-2 extras: `--text-corpus` tokenizes a UTF-8 file with the checkpoint's `tokenizer.json` into `S + 1`-token records (needs `tokenizers`), and `--hf-parity` runs a NumPy forward with SeeML's exact semantics against `transformers` on seeded random tokens and prints the max logit delta (SmolLM-135M: `6.4e-05` at `S = 128`; needs torch + transformers). Neither is needed to import.

Each corpus record is `S + 1` ids: the runtime feeds the first `S` and derives next-token labels from the shifted view, and the embedding gathers on-device. `python3 tool/export_model.py --demo-decoder out/` writes a working example of both files — `decoder.smf` and `decoder_corpus.sds` (NumPy only — no PyTorch needed); the corpus is deliberately *not* named `corpus.sds`, so both demos can share one output directory. Pass those names to the compile and update steps below in place of `model.smf` / `corpus.sds`. Remember that `--data-batch` counts *rows* (tokens), so it must be a multiple of `seq_len`.

## Step 2: Compile the Update Plan (build host)

Here's a full-featured invocation; we'll unpack it flag by flag:

```bash
seeml-update-compile \
  --source model.smf --out pkg/ \
  --data-batch 32 --loss xent \
  --lora-rank 8 --lora-alpha 16 \
  --optimizer adamw --lr 1e-3 --clip-norm 1.0 \
  --lr-schedule cosine --warmup 100 --min-lr-factor 0.1 \
  --quantize-base \
  --steps 1000 --report pkg/report.json --build
```

**What to train on.** `--data-batch` (default 32) fixes the batch size — how many samples are processed together in each training step — *into the plan*; shapes are compile-time facts in SeeML, so this isn't a runtime knob. `--loss` picks the objective, the measure of wrongness that training drives down: `xent` (softmax cross-entropy, needs class labels), `mse` (mean squared error, dense labels), `kl` (**distillation**: the model learns to imitate a teacher model's output probabilities rather than hard labels — add `--teacher teacher.smf` and use an unlabeled corpus), or `xent+kl` (both, blended by `--distill-weight`, default 0.5; `--temperature`, default 2.0, softens both distributions, and the KL term is scaled by `T²` so that `--distill-weight` means the same thing at every temperature — see [runtime.md](runtime.md) for why).

**What to adapt.** `--lora-rank` (default 8) and `--lora-alpha` (default 16) set the adapter geometry — the update lives in `r·(K+M)` parameters per adapted matmul instead of `K·M` ([compiler.md](compiler.md) does the math). `--targets substr1,substr2` restricts grafting to weights whose names match a substring; by default every eligible frozen matmul is adapted. `--lora-seed` (default 42) makes the adapter initialization reproducible.

**How to optimize.** `--optimizer adamw|sgd` (default adamw), `--lr` (default 1e-3), `--weight-decay` (default 0.01), and `--clip-norm` (default 0 = off; a positive value bakes per-tensor gradient clipping instructions into the stream). The schedule — `--lr-schedule const|cosine`, `--warmup N`, `--min-lr-factor F` (default 0.1) — travels in the plan header and is evaluated per step on the device. The cosine anneals over **the steps the update actually runs** — `model_update --steps 200` on a plan compiled with `--steps 1000` reaches the floor at step 200, not never — and the compiler refuses a schedule that would misbehave silently: `--steps 0`, a warmup as long as the budget, `--warmup` or `--min-lr-factor` under `const`, and a floor of 0 (the last step would train at a learning rate of zero) unless `--allow-zero-lr` asks for it.

**How big.** `--quantize-base` stores eligible frozen weights as int8 in the plan (4× smaller, dequantization fused into the GEMM for free); `--bf16-base` stores them as bfloat16 instead (2× smaller, float32's exponent range with 8 bits of mantissa, widened exactly inside the GEMM — the gentler choice for outlier-heavy weights; the two flags are mutually exclusive). Either way, because commit patches the *original file's* floats, storage rounding never reaches the committed model. `--steps` (default 1000) is the default optimizer-step count baked into the plan (the device may override it). `--grad-accum G` (default 1) compiles at the micro-batch `--data-batch` and accumulates `G` micro-batch gradients per optimizer step: the effective batch is `data-batch × G` while activation memory stays that of one micro-batch, at the cost of one gradient-sized accumulator per adapter in the persistent segment (the compile report lists `effective_batch` and `step_instructions`).

**How it's optimized.** By default the compiler fuses each frozen `X@W → +bias → activation` chain into a single matmul instruction with a fused write-back epilogue, so an MLP layer's three arena round-trips become one. Fusion is bitwise-neutral by construction: it only matches chains no backward instruction reads (the frozen teacher subgraph, the bias step of unadapted layers), and the runtime applies the epilogue with the same per-element expressions as the standalone kernels. `--report out.json` carries, beside the plan's geometry, a `"passes"` array — every compiler pass and driver phase with its op count and wall time — and `--dump-sir out.txt` writes the final SIR (debug output; nothing is rendered unless asked). `--no-fuse-epilogue` disables the pass — the plan gets more instructions and more transient arena, never different bits; useful when diffing `seeml-seeu-dump` output across compiler versions or isolating a kernel while debugging. Same-shape elementwise chains whose intermediates have a single reader — `α·u` then `c + s` at every LoRA site — fold into one `kFusedMap` instruction (plan v13; `--no-fuse-elementwise`). Two more passes of the same kind ship on by default (plan v12), each with its switch for bit-for-bit comparison: the RoPE angles are hoisted into one device-built table per sequence geometry instead of being recomputed by every rotation (`--no-rope-table`), and `--clip-norm` is folded into the optimizer step, which removes a full read+write pass over every gradient and refuses to step a tensor whose gradient norm is not finite (`--no-fuse-clip` restores the standalone clip instruction). At every LoRA site the two activation-sized sums — `C + (α/r·t)@B` forward, `dC@Wᵀ + dt@Aᵀ` backward — are written by the rank-r GEMM itself rather than by a separate add over the whole tensor (plan v14; bit-identical, `--no-fuse-addend` restores the separate instructions).

**How fast (the kernel policy).** The CPU GEMM tile geometry — a throughput knob only, never a bit — travels *in the plan* (header v11), decided here. By default it is the runtime's own defaults. `--kernel-policy table.json` consults the table `tool/autotune.py` measured on this host (keyed on the host's identity; the compile note prints the key, and a table with no entry for it is a note and the defaults), `--target-host KEY` looks another host up when cross-compiling, and `--gemm-tiles K,N` sets the geometry outright (K a multiple of 4). Nothing is read when no table is given, so compile time is unchanged; a table that does not parse or names a geometry the kernels reject is a hard error. See [benchmarks.md](benchmarks.md) for tuning a host.

**What comes out.** The emitted `pkg/` is **self-contained**: the plan (`update_plan.seeu`), the same plan embedded as a C array, a generated driver `main`, the vendored runtime sources, and a `build.sh`. `--build` runs that script immediately; on any machine, `sh pkg/build.sh` builds the `model_update` binary with nothing but a C++23 compiler — set `CXX` to cross-compile for the device. `--report pkg/report.json` writes a machine-readable summary (arena bytes, instruction counts, per-adapter shapes and scales, the kernel policy — source, host key, tiles — and which embedded TU the package carries) worth archiving with each release.

**Big plans: embed with `.incbin` instead.** The C-array embedding renders every plan byte as ~4 characters of C++ that the host compiler must parse at `-O2` — fine at kilobytes, but a 135M-parameter plan becomes a ~950 MB translation unit and a 16-minute `--build`. For anything past a few tens of megabytes, pass `--no-embed` (no C array is written; `--build` is refused, since there is nothing yet to link) and let the build-host packer embed the plan as an assembly stub that splices the `.seeu` in verbatim:

```bash
seeml-update-compile --source model.smf --out pkg/ ... --no-embed
python3 tool/pack_update.py pkg/ --build          # writes update_plan_embedded.S, runs build.sh
```

The packer is standard-library Python on the build host only; the package it leaves behind is the same dependency-free C++ (the stub needs only the preprocessor and assembler every C++ toolchain carries), the `.seeu` is untouched, and the binary trains the very same bits as the C-array build — CI asserts all three on every change. The embedded plan is page-aligned (`--page-align`, default 16 KiB) and, on Apple, placed in writable data — exactly what the Metal backend's zero-copy residency wants: a packed package trains from the plan's own pages, a heap-loaded plan is copied once. `build.sh` prefers the stub whenever one is present, so `pack_update.py` also upgrades an unmodified package emitted before `--no-embed` existed (it rewrites the one compile line and drops the old C array); a hand-edited `build.sh` is refused rather than guessed at.

Compilation is also your first line of defense: infeasible memory footprints, shape mismatches, a loss that can't see any trainable parameter — all fail *here*, on the build host, with a one-line `"<unit>: <message>"` diagnostic.

Curious what was actually generated? Disassemble any plan:

```bash
seeml-seeu-dump pkg/update_plan.seeu            # header, integrity check, emit table
seeml-seeu-dump pkg/update_plan.seeu --instrs   # all three instruction streams
```

Reading a training loop as thirty-ish opcodes of straight-line code is a genuinely instructive way to see what the compiler did — and a good sanity check that, say, your clip instructions exist.

## Step 3: Run the Update (device)

```bash
model_update --model model.smf --data corpus.sds --out updated.smf \
  --val-frac 0.1 --seed 7 \
  --checkpoint state.ckpt --checkpoint-every 100 --resume \
  --loss-log curve.csv
```

| flag | meaning (default) |
|---|---|
| `--model source.smf` | the model the plan was compiled from (required; must hash-match the plan) |
| `--data corpus.sds` | the training corpus (required) |
| `--out updated.smf` | where the committed model is written (`updated_model.smf`) |
| `--steps N` | optimizer steps (0 = the plan's compiled default); under `--grad-accum G` each consumes `G` micro-batches |
| `--seed S` | shuffle-permutation seed (0) |
| `--val-frac F` | held-out fraction for the regression gate (0.1); 0 gates on the training-loss trend instead |
| `--checkpoint path` | checkpoint file, hash-bound to the plan |
| `--checkpoint-every N` | steps between checkpoints (0 = off) |
| `--resume` | resume from `--checkpoint` if present. With no `--steps` it trains exactly the **remainder** of the interrupted run, on that run's learning-rate schedule — the checkpoint records the horizon — so the result is bit-identical to the run left uninterrupted; with `--steps N` it trains N *further* steps and the schedule is stretched to cover them |
| `--loss-log curve.csv` | write the per-step loss curve |
| `--backend cpu\|metal\|auto` | the executor (cpu; or `$SEEML_BACKEND`). `cpu` is the bitwise-deterministic reference and builds anywhere; `metal` runs the update on an Apple GPU and is an error where none exists; `auto` takes the GPU when present and says so, else cpu |
| `--min-improvement F` | commit only if the gated loss fell by at least the fraction F of its initial value (0 = any strict fall) |
| `--require-accuracy` | additionally require held-out accuracy not to drop; a usage error on plans without class labels or without a validation split |
| `--attention auto\|cached\|tiled` (compiler) | attention memory: `cached` keeps the `[B·H·S, S]` probability cache from forward to backward, `tiled` keeps four floats per query row and recomputes (about 3× the attention backward, 1.2–1.7× a whole step in the measurements of `docs/benchmarks.md`); `auto` (default) tiles when the caches of all layers exceed `--attention-cache-budget-mib` (256) or the footprint would not fit local memory with them. Identical bits either way |
| `--eval-every N` | with a validation split, evaluate every N steps and commit the **best** evaluated state rather than the last; default a tenth of the run (at least 1); `0` evaluates only before and after and commits the last state, bit for bit as before |
| `--patience K` | stop after K consecutive evaluations without a new best (0 = never); needs `--eval-every > 0` and a split |
| `--report report.json` | write the backend, the gate parameters, the loss and accuracy pairs (`validation_loss[1]` is the committed state's; `best_state` carries its step, the endpoint's loss and the evaluation settings — schema 2), the verdict and whether the commit landed as JSON (written last; an unwritable path is a warning, the exit code still carries the verdict) |
| `--force` | commit even if the gate shows no improvement |

What happens, in order:

1. **Load + verify** — the plan's hash is checked, then every instruction operand is bounds-validated *before anything executes*. One arena allocation, sized at compile time. A corrupt or foreign plan is refused at this door.
2. **Split + shuffle** — the last `--val-frac` (here 10%) of the corpus is held out for validation; training batches are then served through a seeded per-epoch permutation. Same `--seed`, same batches, same bits — on any machine.
3. **Train** — N steps of forward + backward + clip + optimizer. Interruptible: checkpoints (`--checkpoint`, `--checkpoint-every`) are hash-bound to the plan and fsync-durable, and `--resume` picks up from one — optimizer momentum, step counter, *and* the data-shuffle position, which is replayed so a resumed run produces the same bits as one that was never interrupted. `--steps` after `--resume` counts further steps, not a total. A non-finite loss aborts immediately rather than continuing on garbage.
4. **Gate** — validation loss is evaluated before training, every `--eval-every` steps, and after, with the plan's eval program; the state that scored best is the one gated and committed (the log line names its step and the endpoint's loss; the report's `best_state` records both). A resume keeps the *source model's* "before" from the checkpoint rather than re-scoring the resumed adapter, and refuses a `--seed` or `--val-frac` other than the first run's (exit 2). No improvement — or less than `--min-improvement`, or a dropped accuracy under `--require-accuracy` — → exit code 3 and the device is left *untouched* (`--force` overrides, if you must). The gate line names the failing gate and the backend that produced the numbers.
5. **Merge + commit** — deltas `Δ = (α/r)·A@B` are materialized and added to the pristine f32 weights of the source file (which must hash-match the plan), written durably, renamed atomically. A power cut at any moment leaves the old model or the new one — never a torn file.

Exit codes are the API for your update orchestrator: `0` committed, `1` runtime error, `2` bad arguments, `3` regression-gate rejection. `--loss-log` writes the full per-step CSV loss curve once training completes (it is not appended live, so tail it after the run, not during).

## Threading and Determinism

Both halves parallelize: the compiler's byte-heavy passes (int8 quantization, adapter initialization, plan embedding) and the runtime's kernels, plus a feeder thread that stages the next batch while the current step computes. Control it with one variable:

```bash
SEEML_THREADS=1 model_update ...   # fully serial: no thread is ever created
SEEML_THREADS=4 model_update ...   # pin the pool width; default = all cores
SEEML_METAL_PROFILE=1 model_update --backend metal ...   # per-kernel GPU time at exit (serializes; diagnostic only)
```

Here's the property that makes this knob safe to turn: parallel execution is **bitwise-deterministic**. Work is split into chunks whose boundaries depend only on the problem shape (never the thread count), and reductions combine per-chunk partials in a fixed order — so the same plan, data, and seed produce the *same bits* at any thread count. Thread count is a throughput knob, not a numerics knob. Practically: a loss curve from an 8-core dev board reproduces exactly on a single-core target, and any bug you find is reproducible by construction. ([runtime.md](runtime.md) explains the mechanism.)

This is a tested contract, not an aspiration:
`ParallelFor.OrderedPartialReductionIsBitwiseThreadCountInvariant`
(`test/source/parallel/parallel_for_test.cc`) proves the substrate, and
the `ParallelDeterminism` suite (`test/runtime/executor/kernels_test.cc`)
re-proves every kernel family bit-for-bit across thread counts.

## Development

Working on SeeML itself:

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
# without CMake (builds every per-module SeeTest suite):
sh build/build.sh && for t in build/seeml_*_test; do "$t"; done
# one suite, one test:
./build/seeml_update_engine_test --filter=UpdateEngineCheckpoint
# sanitizers / fuzzing:
cmake -B build -DSEEML_SANITIZE="address;undefined"
cmake -B build -DSEEML_SANITIZE="thread"   # proves the pool + feeder sync
cmake -B build -DSEEML_FUZZ=ON && ./build/seeml_fuzz_formats
```

The test tree mirrors the source tree one-to-one — a suite lives where its subject lives — and [test/README.md](../test/README.md) explains the layout and the in-tree harness. The thread sanitizer build is not decoration: it's the mechanical proof that the worker pool and the feeder handoff are race-free, and the fuzzer hammers the binary-format parsers with hostile bytes — the same parsers whose paranoia [formats.md](formats.md) describes.
