---
title: "SeeAI CR4: storage and egress durability — checkpoint metadata dropped on save, partial temp files left on failure, a writer that mutates offsets before a failable write, runners that drop report write failures"
number: 145
labels: bug,correctness,core-plane
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

The durability story has edges the review reached: `has_binding` is dropped
on checkpoint save so v5 loads report true (F71, previously refuted, now
real); `WriteFileDurable` leaves the partial temp file on every failure
path (F08); `SaveSmf` changes the model's offsets before a write that can
fail, so a failed save leaves new offsets paired with the old hash (F05);
`export_smf` leaves a truncated file when a non-CPU tensor write fails
(F55); the generated runner drops `--loss-log` open failures and `--report`
write errors, so a full disk leaves truncated JSON with exit 0 (F66);
`--attention-sweep` writes unstaged and ignores write errors (F61). Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §6
(storage), §8.

## Design

One rule for every writer on both planes: stage to a temp file, check every
write and close, replace atomically, unlink the temp on every error path,
and commit in-memory state only after the file is durable. Checkpoint
header round-trip tested field by field.

## Acceptance

- Each reproduction fixed with a test that injects the failure (full disk,
  failed write, kill mid-save).
- The storage role's screen line reports bytes written and fsyncs per
  checkpoint.
