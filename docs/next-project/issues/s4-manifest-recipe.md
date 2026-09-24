---
title: "SeeAI S4: the identity manifest and the recipe file — every input hashed into the plan, a declarative recipe with a schema and bounds, the exporter's output verified rather than trusted"
number: 150
labels: enhancement,correctness,python-plane,doctrine
plane: Python plane
milestone: SeeAI v1.0.0.B
priority: P1
---
## Root cause

A fine-tune is four inputs — model, tokenizer, corpus, recipe — and the plan
records the hash of one. The recipe is about forty flags with no schema, and
the review found what that costs: `ParseU64` accepts `" -1"` in the CLI and
the generated runner (F56, F63); `--teacher` / `--distill-weight` /
`--temperature` silently ignored unless the loss uses them (N15);
`--lora-rank` unbounded (N18, overflow emits an invalid plan);
`--temperature` unbounded (N26); an empty `--targets` adapts every weight
(N27); a runner `--steps` inside the compiled warmup (N20); the runner
ignoring `--checkpoint-every` without `--checkpoint` and `--eval-every`
without a split (N24). The exporter's output is trusted as-is. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3
(doctrine lines: recipe, exporter verification).

## Design

1. **Recipe file** with a JSON schema in `source/` (published by
   `seeml-abi`), every numeric field bounded, cross-field rules (warmup <
   steps, teacher fields require a distillation loss, targets non-empty);
   flags override fields; the resolved recipe is hashed. The generated
   runner validates its flags with the same rules.
2. **Identity manifest** in the plan header: model hash, tokenizer hash,
   corpus hashes (train, held-out), resolved recipe hash, split seed, mask
   policy, the exporter's parity figures. The loader (S7) verifies it; the
   accountant prints it.
3. **Exporter verification**: the frontend checks the exporter's output —
   file hash, tokenizer parity report, logits-parity figure — and records
   them; a missing report is a refusal, not a warning.

## Acceptance

- Every N15/N18/N20/N24/N26/N27/F56/F63 input is refused with a diagnostic
  naming the field and its bound, on both the compiler and the runner.
- A package states its four input hashes in `--report` and in the runtime
  screen; the loader refuses a source model or corpus whose hash disagrees.
- One recipe file reproduces the 2026-09-22 frontier row byte-for-byte.
