---
title: "SeeAI S2: the tokenizer subsystem — raw text into the frontend, an SMF tokenizer section proven bit-exact against the reference, sanitize / deduplicate / encode / pack, vendored to the device"
number: 148
labels: enhancement,core-plane,python-plane,doctrine
plane: Core plane
milestone: SeeAI v1.0.0.B
priority: P0
---
## Root cause

The compiler never sees the user's data. The Python exporter tokenizes with
the Hugging Face tokenizer and the C++ side only ever sees ids, so
sanitization, deduplication and the contract check happen nowhere on the
build host and only the geometry check happens on the device. The review's
N01 (high: the feeder shuffles and splits sequences row by row), F19 (the
SDS loader accepts trailing bytes) and F52 (the Python split accepts what the
runtime refuses) are all symptoms of a data path with no owner. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §3
(tokenizer row and the doctrine lines).

## Design

1. SMF gains a **tokenizer section** (vocabulary, merges, pre-tokenizer
   pattern, normalization, special tokens, chat template) written by the
   exporter from the checkpoint's own files; the frontend's tokenizer is
   instantiated from it only — no `--tokenizer` flag exists. Hash of the
   whole definition = the tokenizer identity.
2. Exporter parity test: the C++ byte-level BPE against the reference on a
   golden adversarial corpus (mixed scripts, digits, whitespace runs, control
   characters, invalid UTF-8, every special token in and out of context);
   the section is refused when any string disagrees. Byte-level BPE first;
   SentencePiece refused with a diagnostic.
3. Stages: ingress (bounded) → normalize → sanitize and filter (invalid
   UTF-8, control characters, bidi overrides, empty/degenerate records) →
   exact deduplicate by hash → encode → pack (concatenate with EOS, cut
   fixed S+1 windows; never pad, since every position is a loss target) →
   canonical SDS + a manifest of dropped records by index and reason.
4. SDS version bump: header carries the tokenizer hash; the plan header
   carries it too; compiler and device refuse a corpus whose hash differs.
5. Sanitizer, contract check and tokenizer live in `source/` as pure
   deterministic functions and are vendored into the package (S6), so the
   pipeline re-runs them on device data. Near-duplicate detection stays in
   the Python plane (§7).
6. The accountant reports records in / kept / dropped per class, tokens,
   batches per epoch, epochs the step budget covers, distinct ids vs
   vocabulary, label entropy (the unigram loss floor), target vs total tokens.

## Acceptance

- `seeml-update-compile --data corpus.txt` on SmolLM-135M produces the same
  ids as the exporter's path for the 2026-09-22 frontier corpus (bit-exact).
- N01, F19, F52 reproductions refused on the build host; the device refuses
  a corpus with a foreign tokenizer hash.
- The frontier row (F1) re-run from raw text matches the ids-path loss curve
  to 1e-5.
