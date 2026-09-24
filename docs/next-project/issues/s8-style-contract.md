---
title: "SeeAI S8: the style contract — write the house style down and enforce it: the two-dialect decision, .clang-format, the packed-word helpers, the four oversized functions, tool/ idiom parity, the test-framework tightening, and the doc-drift list"
number: 154
labels: enhancement,docs,design,testing
plane: Gates & docs
milestone: SeeAI v1.0.0.B
priority: P2
---
## Root cause

The 2026-09-01 Style Audit graded the tree A− and found the discipline lives
entirely in people's heads: two undocumented dialects (Google-style in
`seeml::update`, MLIR-flavored camelCase and 4-space indent in
`seeml::sir`, ~24 files on the second side, a few straddling); no
`.clang-format`, no STYLE.md, no CONTRIBUTING.md; one bit-unpacking
expression hand-inlined ~47 times across engine, validator and the dumper
while `UnitOf` already exists; four functions of 277–560 lines
(`CompileImpl`, `ValidateInstruction`, `LowerOps`, `seeml-bench`'s
`main`); `source/` and `tool/` at 0 `[[nodiscard]]` against 93 elsewhere;
three tools with three argument parsers; `seeml-update-compile` exiting 1
where the docs promise 2 (also found by the v1.2.4 Field Report's ledger and
the STAR document); `using namespace` in 20 of 25 test suites; no
`SKIP()`, so hardware-gated tests report green for zero work. None of the
seven recommendations was ever filed. The Two-Plane Port Review's four
doc-drift items are folded in here. Design: [`docs/next-project/seeai.md`](https://github.com/Szhenger/see-ml/blob/main/docs/next-project/seeai.md) §8.

## Design

1. Decide the dialect question (document per-subsystem, or normalize), then
   commit `.clang-format` (Google base, 80 columns) and a short STYLE.md:
   the diagnostic-unit contract, the assert-vs-expected boundary, the
   exit-code table, ownership conventions. Add `-Wold-style-cast` (zero
   violations today).
2. `HiWord`/`LoWord` helpers beside `MakeArenaRef` in
   `source/plan/instruction.h`; delete the ~47 copies.
3. Decompose the four functions (S6 takes `CompileImpl`; this issue takes
   the other three).
4. `source/` and `tool/` to the tree's idiom: `[[nodiscard]]`, `std::span`
   for (ptr, size) pairs, one expected-based argument parser shared by the
   CLIs, exit codes per the documented contract, the dumper's bounds-checked
   wire-read rule in bench.
5. Test framework: per-symbol `using`, four duplicated helpers into
   `test/support/`, the two gtest-colliding macros renamed, `SKIP()`,
   operands printed by `EXPECT_TRUE` and haystacks by
   `EXPECT_STR_CONTAINS`.
6. Doc drift: `docs/formats.md` SMF version heading, `SPECIFICATION.md`'s
   Python-surface sentence, `tool/README.md`'s doctrine line, and the stale
   test count.

## Acceptance

- `clang-format --dry-run -Werror` clean in CI; STYLE.md linked from the
  README.
- `git grep -c '>> 32'` in engine/validator/dumper drops to the helpers'
  definitions.
- No function over 200 lines in `compiler/`, `runtime/`, `tool/`.
- All three CLIs exit 2 on a bad command line; a test pins it.
