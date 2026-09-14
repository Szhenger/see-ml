---
title: "Gates E9: commit the best evaluated state, not the last — periodic eval, patience, and a checkpoint that binds val_initial, seed and split (extends #67)"
labels: enhancement,correctness,core-plane
plane: Gates & docs
origin: Algorithm review
priority: P1
---
## Root cause

The step loop (`runtime/engine/update_engine.cc:718-754`) evaluates once
before training and once after; `TrainReport::improved()`
(`update_engine.h:113-116`) asks only `val_final < val_initial`, and the
**last** persistent segment is what `RunMerge` + `CommitToModel` ship. On
small on-device corpora (the demo runs ~15 epochs) val loss bottoms out
early; an overfit endpoint that is still below step 0 is committed while a
better intermediate state was discarded. #67 hardens the gate's
*threshold* (`--min-improvement`, `--require-accuracy`); it does not
change *which state* is gated.

On resume (`update_engine.cc:671-701`) `LoadCheckpoint` runs before the
initial `EvaluateMetrics`, so `val_initial` is the resumed adapter's loss,
not the source model's: a resumed segment that merely holds steady is
rejected, and the source model is never scored. The checkpoint
(`runtime/custodian/checkpoint.cc:27-34`) binds neither the shuffle seed
nor the val fraction, so a resume with a different `--seed` / `--val-frac`
silently moves the train/val boundary and can train on the first run's
validation rows.

## Design

1. `--eval-every N` (default: `steps/10`, min 1) runs the eval program on
   the validation set every N steps; `--patience K` stops after K
   non-improving evals. Both recorded in `report.json`.
2. A second persistent-segment buffer (already the checkpoint payload size)
   holds the **best** state; merge and commit operate on it. One extra
   `persistent_size` allocation at load, priced by the memory gate.
3. Checkpoint header gains `val_initial`, `shuffle_seed`, `val_frac`, the
   run horizon (E8) and the best-so-far loss; resume refuses a mismatched
   seed or split and reuses the stored `val_initial` so the gate measures
   the whole update against the source model.
4. `--min-improvement` / `--require-accuracy` (#67) apply to the best
   state.

## Acceptance

- A corpus engineered to overfit after step k commits the step-k state and
  reports it; the last-state loss is also reported.
- Resume at step 500 with the original seed: gate compares the final best
  against the stored step-0 loss; resume with a different seed → usage
  error.
- Bitwise: with `--eval-every 0 --patience 0` the committed bytes equal
  today's.

## Goal alignment

Outcome parity: HF `Trainer(load_best_model_at_end=True)` and
`mlx_lm.lora` (`--steps-per-eval`, adapter saves per interval) both expose
periodic evaluation; the frontier comparison quotes the best val loss each
stack reaches. Also the gate's own doctrine ("no improvement, no change")
is only as good as the state it scores. Eval cost is bounded by
`eval-every`; Tier D "gate time share" already tracks it.

Refs: SeeML Algorithm Review (2026-09-14) §02-D, §04; #67; closed #18.
