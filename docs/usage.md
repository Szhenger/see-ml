# SeeML Update Compiler — Usage

SeeML compiles an on-device model update ahead of time: LoRA adapters
(Low-Rank Adaptation: a pair of small trainable matrices per frozen weight)
are grafted onto a frozen model, the backward pass and optimizer are
synthesized as a fixed instruction stream bound to a pre-planned memory
arena, and the result is executed on-device by a zero-dependency virtual
machine (VM). One `.seeu` plan = one complete, gated, resumable,
atomically-committed update.

How the compiler is organized internally: [compiler.md](compiler.md);
the on-device runtime: [runtime.md](runtime.md).
The binary formats they read and write: [formats.md](formats.md).
Terms and abbreviations: the [README glossary](../README.md#glossary).

## 1. Export the model (build host, PyTorch)

```bash
python3 tool/export_model.py --demo out/        # demo model + teacher + corpus
```

or from your own code:

```python
from export_model import export_smf, export_sds
export_smf(sequential_model, "model.smf")   # Linear/ReLU/GELU/SiLU/LayerNorm
export_sds(inputs, labels, "corpus.sds")    # labels: int32 classes, dense f32, or None
```

`export_smf` accepts an `nn.Sequential` of `Linear`, `ReLU`, `GELU`,
`SiLU`, and `LayerNorm` modules (`Linear` weights are stored transposed
from PyTorch's layout); `export_sds` takes fixed-shape inputs with class,
dense, or no labels.

## 2. Compile the update plan (build host)

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

Every flag (parsing is strict — an unknown flag, a flag missing its value,
or a numeric with trailing garbage is a hard error, never a silent
default):

| flag | meaning (default) |
|---|---|
| `--source model.smf` | the on-device model to update (required) |
| `--out dir/` | emission directory (required) |
| `--data-batch N` | compiled batch size (32) |
| `--loss xent\|mse\|kl\|xent+kl` | training objective (`xent`) |
| `--teacher t.smf` | open-weights teacher model, for `kl` / `xent+kl` |
| `--distill-weight W` | KL weight in the composite loss (0.5) |
| `--temperature T` | distillation temperature (2.0) |
| `--lora-rank R` / `--lora-alpha A` / `--lora-seed S` | adapter rank / scale / init seed (8 / 16 / 42) |
| `--targets substr,substr` | restrict adapters to weights whose names match |
| `--optimizer adamw\|sgd` | optimizer (`adamw`) |
| `--lr LR` / `--weight-decay WD` | learning rate / decay (1e-3 / 0.01) |
| `--clip-norm C` | per-tensor L2 gradient clip (0 = off) |
| `--lr-schedule const\|cosine` | runtime LR schedule (`const`) |
| `--warmup N` / `--min-lr-factor F` | warmup steps / cosine floor as a fraction of `--lr` (0 / 0) |
| `--quantize-base` | int8-quantize frozen weights in rodata |
| `--steps N` | default step count baked into the plan (1000) |
| `--no-fuse-epilogue` | compile the unfused reference form |
| `--report out.json` | machine-readable compile report |
| `--build` | run the emitted `build.sh` after emission |

Distillation from an open-weights teacher: `--loss kl --teacher teacher.smf`
(unlabeled corpus), or `--loss xent+kl --distill-weight 0.5`.

GEMM (general matrix–matrix multiply) → AddBias → activation chains with
no backward reader (the frozen teacher,
unadapted layers) are fused into single-instruction epilogues by default —
bitwise-identical results, fewer arena round-trips. `--no-fuse-epilogue`
compiles the unfused reference form.

The emitted `pkg/` is **self-contained**: the plan, the generated driver, and
vendored runtime sources. `sh pkg/build.sh` builds `model_update` on any
machine with a C++23 compiler — set `CXX` to cross-compile for the device.

Inspect any plan with the disassembler:

```bash
seeml-seeu-dump pkg/update_plan.seeu --instrs
```

## 3. Run the update (device)

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
| `--resume` | resume from `--checkpoint` if present |
| `--loss-log curve.csv` | write the per-step loss curve |
| `--force` | commit even if the gate shows no improvement |

What happens, in order:

1. **Load + verify** — plan hash checked; every instruction operand
   bounds-validated before anything executes. One arena allocation, sized at
   compile time.
2. **Split + shuffle** — the last `--val-frac` of the corpus is held out;
   training batches are served through a seeded per-epoch permutation.
3. **Train** — N steps of fwd+bwd+clip+optimizer. Interruptible
   (checkpoints are hash-bound to the plan and fsync-durable); aborts on a
   non-finite loss.
4. **Gate** — validation loss is evaluated before and after with the plan's
   eval program; both passes rewind the held-out set so they average the
   identical sample multiset. No improvement → exit 3, device untouched
   (`--force` overrides).
5. **Merge + commit** — deltas `Δ = (α/r)·A@B` are materialized and added to
   the pristine f32 weights of the source file (which must hash-match the
   plan), written durably, renamed atomically.

Exit codes: `0` committed, `1` runtime error, `2` bad arguments,
`3` regression-gate rejection.

## Threading

Both sides of the product parallelize: the compiler's byte-heavy passes
(int8 quantization, adapter initialization, plan embedding) and the
runtime's kernels, plus a feeder thread that stages the next batch while the
current step computes. Control it with `SEEML_THREADS`:

```bash
SEEML_THREADS=1 model_update ...   # fully serial: no thread is ever created
SEEML_THREADS=4 model_update ...   # pin the pool width; default = all cores
```

Negative, zero, or malformed values (including `-1`, some tools' "all
cores" convention) fall back to hardware concurrency rather than being
taken literally.

Parallel execution is **bitwise-deterministic**: work is split into chunks
whose boundaries depend only on the problem shape (never the thread count),
and reductions combine per-chunk partials in a fixed order. The same plan,
data, and seed produce the same bits at any thread count — thread count is a
throughput knob, not a numerics knob — so a loss curve from an 8-core dev
board reproduces exactly on a single-core target.

This is a tested contract, not an aspiration:
`ParallelFor.OrderedPartialReductionIsBitwiseThreadCountInvariant`
(`test/source/parallel/parallel_for_test.cc`) proves the substrate, and
the `ParallelDeterminism` suite (`test/runtime/executor/kernels_test.cc`)
re-proves every kernel family bit-for-bit across thread counts.

## Development

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
