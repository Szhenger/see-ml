---
title: "SeeAI S3: the validation subsystem — the held-out split decided at compile time, and the loss mask for instruction data with a masked cross-entropy opcode"
number: 149
labels: enhancement,core-plane,doctrine
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

Two decisions define the objective the gate scores and neither is compiled.
The held-out set is a runtime `--val-frac` tail slice of whatever corpus the
device has, which the same data source controls — the last defense against
poisoned data is chosen by the data. And every position in a batch is a loss
target: there is no mask in the ISA, so instruction data (prompt + response)
trains the model to predict the prompt. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3 (validation row).

## Design

1. **Split.** With `--val-data`, validation tokenizes it with the same
   tokenizer (S2) and records both hashes. Without one, a seeded partition
   of the training corpus after deduplication (no held-out record has a twin
   in training); seed and held-out indices go into the plan so the device
   scores the same set. The feeder checks all of it at load from the shared
   contract function.
2. **Mask.** For instruction data the chat template from the SMF tokenizer
   section marks response positions as targets. The SDS record gains a
   per-position mask (part of S2's version bump); the ISA gains
   `kSoftmaxXEntMaskedFwd/Bwd` with the mean taken over target tokens; the
   VJP rule lands in `analysis/calculus`. Plain text records carry an
   all-ones mask and the existing opcode, bit-identical to today.
3. The accountant reports target tokens vs total tokens per record.
4. Attention across packed documents: the attention op gains a segment
   operand (block-diagonal within a window) or the screen prints how many
   windows straddle a document boundary — never silence (§4 algebra).

## Acceptance

- Two compiles of the same corpus and seed produce byte-identical split
  indices; the device refuses a plan whose split hashes disagree with its
  corpus.
- Masked loss: on a prompt/response fixture the response-only loss matches a
  PyTorch reference to 1e-5; plain-text plans are bit-identical to main.
- The gate's validation set is named in the compile report with its hash.
