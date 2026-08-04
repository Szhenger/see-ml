# The SeeML Journey: From SeeC to SeeML

This document traces the evolution of the project across its full git history —
635 commits from the initial commit on April 30, 2025 to the SeeML update
compiler of July 2026. The work arrived in three big bursts (June 2025,
May 2026, July 2026) separated by long quiet stretches, and the project
reinvented itself twice along the way.

## Phase 1 — SeeC: a C compiler in C (April – July 2025)

The project began as "SeeCompiler": a from-scratch compiler for the C language,
written in C, with a flat file layout (`lexer.c`, `token.c`, `parser.c`,
`ast.c`, `semantic.c`, `ir.c`, `codegen.c`), a Makefile, and small Python
scripts driving the tests.

- **May 2025** built the lexer and token machinery.
- **Late May – early June** added the parser and AST.
- **June 10** introduced the IR and code generator, emitting assembly
  (the demo pairs `hello.c` with `output.s`).

June 2025 was the most intense month in the repo's history at 259 commits,
mostly small iterative refinements. The phase culminated in
"SeeCompiler Complete" (July 7) and "Release Binary Done!" (July 11, 2025) —
the commit tagged `v1.0.0`, the only release tag in the repo.

## Phase 2 — SeeC++: the rewrite and the quiet pivot (October 2025 – March 2026)

After a quiet late summer, "SeeC++ Day 1!" (October 27, 2025) kicked off a
ground-up rewrite in C++, with day-numbered diary-style commits and a
`sourceCode/frontEnd` layout split into include, source, and utility
directories. Progress was sporadic — 29 commits in October, 8 in December,
then a gap.

The pivotal identity change happened here, quietly: on **February 17, 2026**
all the classic language-compiler files (lexer, parser, token, AST) were
deleted, and on **February 21** came "feat(frontend): implement MLIR-style
ingress, shape inference, and validation." The project stopped being a
C-language compiler and became an **ML graph compiler** — while still carrying
the SeeC++ name. March 2026 initialized the middle-end and backend for this
new direction.

## Phase 3 — The ML compiler buildout (May – June 2026)

May 2026 was the second great burst (160 commits), building out a full
ONNX-to-CPU inference compiler:

- **Frontend:** a ProtobufReader for ONNX ingestion, a TypeBridge, "SIR" as
  the internal IR, shape inference, canonicalizer, constant folder, validator,
  and diagnostics engine.
- **Middle-end:** a PassManager with conv lowering, operator fusion, dead-code
  elimination, and weight folding.
- **Backend:** three CPU codegen targets (generic C, AVX-512, ARM NEON), plus
  arena mapper, offset binder, weight packer, serializer, and a runtime engine
  with Python bindings.

Engineering discipline arrived alongside: CMake, `.clang-format` and
`.clang-tidy` (Google style), and in early June, integration/unit tests and
architecture documentation.

## Phase 4 — SeeML: the update compiler (July 2026)

On **July 12, 2026** the project was formally renamed SeeC++ → **SeeML**, and
days later it was re-scoped once more: "Engineer the SeeML Model Update
Compiler" (July 16) removed the legacy inference-pipeline sources and
reoriented the product around model updates and training — captured by the
README title, "SeeML: An ML Update Compiler (How to Train Your Local Model)".

The rest of July was rapid maturation:

- Reorganization into the current `{source, compiler, runtime, build, tool,
  test}` top level.
- A homegrown SeeTest framework with per-module suites.
- Industry-grade hardening: integrity checks, gating, ops, int8.
- Hot-path optimization and parallelism across the product.
- The late-July partitioning sweep: `compiler/frontend` into
  ingressor/parser/operator/representation, `compiler/analysis` into
  updater/algebra/calculus/reviewer, `compiler/backend` into
  trainer/architecture/tuner — with `runtime/` and `test/` mirroring that
  structure.

## Phase 5 — The adversarial audit and the great hardening (July 31, 2026)

Days after the partitioning sweep, the codebase went under a multi-agent
adversarial review: fifteen parallel reviewers (one per subsystem slice,
plus a dedicated concurrency specialist sweeping the whole tree), findings
deduplicated, then every finding handed to an independent skeptic
instructed to refute it by reading — and in one case compiling and
reproducing — the actual code. **34 bugs were confirmed, none refuted**,
spanning eight defect categories: memory safety, logic, concurrency,
error handling, I/O & persistence, API contracts, integer arithmetic, and
one defect in the test framework itself.

All 34 were fixed in a single hardening pass. The headline repairs:

- **Tied-weight merge corruption** (the one high-severity compiler bug):
  a frozen weight consumed by two MatMuls got per-site adapter pairs whose
  deltas were *both* committed to the single file range — `W + Δ₁ + Δ₂`,
  a model the training graph never computed. Tied weights now share one
  adapter pair, making the merge algebra exact; `MergeBuilder` rejects
  duplicates defensively.
- **Transactional plan loading**: a rejected re-`Load` used to leave the
  engine half-mutated (dangling rodata, programs from one plan validated
  against another's arena). `Initialize` now validates into locals and
  commits only after every contract passes.
- **Windows durability**: `WriteFileDurable` gained a real Win32 branch —
  checked writes, `FlushFileBuffers`, `MoveFileEx` replace — where CRT
  `rename` had failed every checkpoint overwrite after the first, unfsynced.
- **Exception-safe `ParallelFor`**: a throwing chunk body had been a
  cross-thread use-after-free of the stack-allocated job (or a straight
  `std::terminate` on a worker); exceptions are now captured and rethrown
  on the submitting thread after the loop retires.
- **Contract closure**: batch proven nonzero, every I/O slot and operand
  ref element-aligned, label widths checked against the narrowest softmax
  in *any* program, `merged_` invalidated by training and checkpoint
  loads, overflow-proof bounds math in the dump tool, and a test-framework
  fix making `EXPECT_LE/GE` fail on NaN — the exact value an ML suite most
  needs to catch.

The same pass finished wiring the parallel substrate through the whole
update path, closing the gaps the concurrency specialist flagged: commit's
delta-apply fans over `ParallelFor`; evaluation pipelines batches through
`BatchPipeline` like training (and rewinds, so the regression gate compares
identical sample multisets); checkpoint payloads hash with the parallel
`ContentHash64` (SEKP v3); the plan seal became the chunked-parallel
`PlanSelfHash`, one canonical function across compiler, engine, and dump
tool (`.seeu` v4); and randn adapter initialization moved to a
counter-based splitmix64 + Box–Muller stream — parallel *within* a tensor,
bit-identical at any thread count, and freed from
`std::normal_distribution`'s implementation-defined algorithm. The audit
closed with a clean `-Wall -Wextra -Werror` rebuild and all 24 suites —
245 tests — green.

## In one sentence

The repo evolved from a hand-rolled C compiler (SeeC, completed July 2025),
through a C++ rewrite that pivoted mid-stream into an ONNX inference compiler
(SeeC++, October 2025 – June 2026), into today's SeeML — a training and
model-update compiler with a fully partitioned compiler/runtime/test
architecture, adversarially audited and hardened end to end in July 2026.
