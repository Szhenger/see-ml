# SeeML Update Compiler — Usage

SeeML compiles an on-device model update ahead of time: LoRA adapters are
grafted onto a frozen model, the backward pass and optimizer are synthesized
as a fixed instruction stream bound to a pre-planned arena, and the result is
executed on-device by a zero-dependency VM. One `.seeu` plan = one complete,
gated, resumable, atomically-committed update.

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

Distillation from an open-weights teacher: `--loss kl --teacher teacher.smf`
(unlabeled corpus), or `--loss xent+kl --distill-weight 0.5`.

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
   eval program. No improvement → exit 3, device untouched (`--force`
   overrides).
5. **Merge + commit** — deltas `Δ = (α/r)·A@B` are materialized and added to
   the pristine f32 weights of the source file (which must hash-match the
   plan), written durably, renamed atomically.

Exit codes: `0` committed, `1` runtime error, `2` bad arguments,
`3` regression-gate rejection.

## Development

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
# without CMake (builds every per-module SeeTest suite):
sh build/build.sh && for t in build/seeml_*_test; do "$t"; done
# one suite, one test:
./build/seeml_update_engine_test --filter=UpdateEngineCheckpoint
# sanitizers / fuzzing:
cmake -B build -DSEEML_SANITIZE="address;undefined"
cmake -B build -DSEEML_FUZZ=ON && ./build/seeml_fuzz_formats
```
