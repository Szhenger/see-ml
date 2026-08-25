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

The exporter needs Python 3 with PyTorch and NumPy on the build host — `pip install -r tool/requirements.txt` installs both. Nothing on the device side touches Python.

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

**What to train on.** `--data-batch` (default 32) fixes the batch size — how many samples are processed together in each training step — *into the plan*; shapes are compile-time facts in SeeML, so this isn't a runtime knob. `--loss` picks the objective, the measure of wrongness that training drives down: `xent` (softmax cross-entropy, needs class labels), `mse` (mean squared error, dense labels), `kl` (**distillation**: the model learns to imitate a teacher model's output probabilities rather than hard labels — add `--teacher teacher.smf` and use an unlabeled corpus), or `xent+kl` (both, blended by `--distill-weight`, default 0.5; `--temperature`, default 2.0, softens both distributions — see [runtime.md](runtime.md) for why).

**What to adapt.** `--lora-rank` (default 8) and `--lora-alpha` (default 16) set the adapter geometry — the update lives in `r·(K+M)` parameters per adapted matmul instead of `K·M` ([compiler.md](compiler.md) does the math). `--targets substr1,substr2` restricts grafting to weights whose names match a substring; by default every eligible frozen matmul is adapted. `--lora-seed` (default 42) makes the adapter initialization reproducible.

**How to optimize.** `--optimizer adamw|sgd` (default adamw), `--lr` (default 1e-3), `--weight-decay` (default 0.01), and `--clip-norm` (default 0 = off; a positive value bakes per-tensor gradient clipping instructions into the stream). The schedule — `--lr-schedule const|cosine`, `--warmup N`, `--min-lr-factor F` — travels in the plan header and is evaluated per step on the device.

**How big.** `--quantize-base` stores eligible frozen weights as int8 in the plan (4× smaller, dequantization fused into the GEMM for free) — and, because commit patches the *original file's* floats, quantization error never reaches the committed model. `--steps` (default 1000) is the default step count baked into the plan (the device may override it).

**How it's optimized.** By default the compiler fuses each frozen `X@W → +bias → activation` chain into a single matmul instruction with a fused write-back epilogue, so an MLP layer's three arena round-trips become one. Fusion is bitwise-neutral by construction: it only matches chains no backward instruction reads (the frozen teacher subgraph, the bias step of unadapted layers), and the runtime applies the epilogue with the same per-element expressions as the standalone kernels. `--no-fuse-epilogue` disables the pass — the plan gets more instructions and more transient arena, never different bits; useful when diffing `seeml-seeu-dump` output across compiler versions or isolating a kernel while debugging.

**What comes out.** The emitted `pkg/` is **self-contained**: the plan (`update_plan.seeu`), the same plan embedded as a C array, a generated driver `main`, the vendored runtime sources, and a `build.sh`. `--build` runs that script immediately; on any machine, `sh pkg/build.sh` builds the `model_update` binary with nothing but a C++23 compiler — set `CXX` to cross-compile for the device. `--report pkg/report.json` writes a machine-readable summary (arena bytes, instruction counts, per-adapter shapes and scales) worth archiving with each release.

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
| `--steps N` | training steps (0 = the plan's compiled default) |
| `--seed S` | shuffle-permutation seed (0) |
| `--val-frac F` | held-out fraction for the regression gate (0.1); 0 gates on the training-loss trend instead |
| `--checkpoint path` | checkpoint file, hash-bound to the plan |
| `--checkpoint-every N` | steps between checkpoints (0 = off) |
| `--resume` | resume from `--checkpoint` if present; `--steps N` then means N *further* steps |
| `--loss-log curve.csv` | write the per-step loss curve |
| `--force` | commit even if the gate shows no improvement |

What happens, in order:

1. **Load + verify** — the plan's hash is checked, then every instruction operand is bounds-validated *before anything executes*. One arena allocation, sized at compile time. A corrupt or foreign plan is refused at this door.
2. **Split + shuffle** — the last `--val-frac` (here 10%) of the corpus is held out for validation; training batches are then served through a seeded per-epoch permutation. Same `--seed`, same batches, same bits — on any machine.
3. **Train** — N steps of forward + backward + clip + optimizer. Interruptible: checkpoints (`--checkpoint`, `--checkpoint-every`) are hash-bound to the plan and fsync-durable, and `--resume` picks up from one — optimizer momentum, step counter, *and* the data-shuffle position, which is replayed so a resumed run produces the same bits as one that was never interrupted. `--steps` after `--resume` counts further steps, not a total. A non-finite loss aborts immediately rather than continuing on garbage.
4. **Gate** — validation loss is evaluated before and after training with the plan's eval program. No improvement → exit code 3 and the device is left *untouched* (`--force` overrides, if you must).
5. **Merge + commit** — deltas `Δ = (α/r)·A@B` are materialized and added to the pristine f32 weights of the source file (which must hash-match the plan), written durably, renamed atomically. A power cut at any moment leaves the old model or the new one — never a torn file.

Exit codes are the API for your update orchestrator: `0` committed, `1` runtime error, `2` bad arguments, `3` regression-gate rejection. `--loss-log` writes the full per-step CSV loss curve once training completes (it is not appended live, so tail it after the run, not during).

## Threading and Determinism

Both halves parallelize: the compiler's byte-heavy passes (int8 quantization, adapter initialization, plan embedding) and the runtime's kernels, plus a feeder thread that stages the next batch while the current step computes. Control it with one variable:

```bash
SEEML_THREADS=1 model_update ...   # fully serial: no thread is ever created
SEEML_THREADS=4 model_update ...   # pin the pool width; default = all cores
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
