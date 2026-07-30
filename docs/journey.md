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

## In one sentence

The repo evolved from a hand-rolled C compiler (SeeC, completed July 2025),
through a C++ rewrite that pivoted mid-stream into an ONNX inference compiler
(SeeC++, October 2025 – June 2026), into today's SeeML — a training and
model-update compiler with a fully partitioned compiler/runtime/test
architecture.
