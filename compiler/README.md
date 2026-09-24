# The SeeAI Compiler

## What Does it Mean to *Compile* Training?

Ordinarily, training a model is something you *run* — a framework interprets
your graph, allocates memory as it goes, and dispatches kernels on the fly.
SeeAI's compiler moves all of that to the build host and decides it *once*:
it takes a frozen model plus a configuration ("adapt this with rank-8 LoRA,
cross-entropy, AdamW, 1,000 steps") and emits a `.seeu` **update plan** —
three flat instruction streams and every byte of memory layout the device
will ever need. Everything a compiler settles ahead of time is something
that *cannot go wrong on the device*.

The tree is partitioned into five subsystems, each named for its role in
the compilation, and each subsystem into folders named for the discipline
of the work inside. Where a folder splits across files, a single façade
header is the only include consumers need.

```
compiler/
  frontend/               SMF bytes -> verified forward SIR
    ingressor/            bounds-checked file load, content hash
    accountant/           the footprint lower bound and the memory gate
    topology/             whole-graph and per-op semantic checks
    computation/          op list -> the SIR computation graph
    egressor/             the SMF writer (the ingressor's inverse; tools and tests)
    tokenization/         the training-data ingress: raw text -> canonical SDS (S2 #148 lands the code)
    operator/             typed constructors for compound ops (shared)
    representation/       SIR itself (façade: sir.h; shared)
  analysis/               forward SIR -> complete training program
    pass_manager.{h,cc}   runs the passes, re-verifies after each
    algebra/              LoRA grafting, merge program, the fusers, RoPE table, conv lowering
    calculus/             autodiff
    statistics/           int8 quantization selection, attention-family decision
    topology/             dead-code elimination
    optimization/         optimizer synthesis
  backend/                training program -> .seeu plan + native package
    architecture/         host cache/ISA detection, GEMM tiling, kernel-policy table, Metal kernel source
    allocation/           arena binding
    selection/            instruction lowering
    packaging/            the self-contained native package
  driver/                 orchestrates the process, verifies every boundary (plan assembly still lives here — S6 moves it)
  diagnostics/            errors, partitioned by process
    tokenizing/ parsing/ passing/ updating/ architecting/ generating/
```

The sub-folder names follow one convention per subsystem: the frontend's
are roles, the analysis phase's are the fields of mathematics that study
what each does, the backend's are the classic compiler phases
(`docs/next-project/seeai.md`).

## The Shape of the Pipeline

Read the folders top to bottom and you have the pipeline. **`frontend/`**
turns untrusted SMF (SeeAI Model Format) bytes into SIR — the SSA
intermediate representation every later stage agrees on — refusing
malformed files and models that provably can't fit in memory.
**`analysis/`** is where the training program is *built*: LoRA adapters are
grafted onto the frozen weights, the backward pass and optimizer are
synthesized as more SIR (a program differentiating a program), and the
merge program is written. **`backend/`** is code generation: it lays out
one arena with no lifetime collisions, lowers SIR to the fixed 64-byte
instruction stream, and packages the result. **`driver/`** owns the
*process* — it sequences the stages and, at every seam, re-proves that each
subsystem produced what the next one assumes. **`diagnostics/`** gives every
error a single-line, unit-attributed home.

Two things worth internalizing: SIR ops carry dialect prefixes (`sc_mem.*`
storage, `sc_high.*` differentiable forward, `sc_low.*` synthesized
adjoints/steps) so you always know which layer of the story you're in; and
`driver/contract.h` is where the compiler distrusts *itself*, catching a
misused subsystem as a compiler bug rather than letting it reach the device.

## Where to Go Next

This is the orientation; the teaching is in
**[docs/compiler.md](../docs/compiler.md)**, which walks every stage from
first principles — what an IR is, the linear algebra of LoRA, how autodiff
works, memory planning as register allocation, cache-aware tiling, and the
offline tuner whose table decides it. The binary formats it reads and writes are in
[docs/formats.md](../docs/formats.md); the runtime that executes its output
is in [docs/runtime.md](../docs/runtime.md). The suites that verify each
subsystem live under [test/compiler/](../test/README.md), one folder per
subsystem here.
