# The SeeML Update Compiler — Architecture

`compiler/` is partitioned into five subsystems, each named for its role in
the compilation, and each subsystem into folders named for the discipline of
the work inside it. A folder's façade header (where one exists) is the only
include consumers need; the units behind it can be reorganized without
churning the rest of the tree.

```
source/                     the source language (not a compiler subsystem)
compiler/
  driver/                   orchestrates the process, verifies every boundary
  frontend/                 SMF bytes -> forward SIR
    ingressor/  parser/  operator/  representation/
  analysis/                 forward SIR -> complete training program
    updater/  algebra/  calculus/  reviewer/
  backend/                  training program -> .seeu plan + native package
    trainer/  architecture/  tuner/
  diagnostics/              error handling, partitioned by process
    tokenizing/  parsing/  passing/  updating/  architecting/  generating/
```

## The pipeline

`UpdateCompiler::Compile` (`compiler/driver/update_compiler.h`) runs the
whole ahead-of-time compilation; every byte the runtime will touch is bound
here:

```
SMF ingest ──▶ feasibility gate ──▶ forward SIR (+ frozen teacher)
           ──▶ loss grafting
           ──▶ pass phase A: conv-lowering, lora-graft     (PassManager)
           ──▶ primal snapshot (becomes the eval program)
           ──▶ pass phase B: autodiff, optimizer synthesis (PassManager)
           ──▶ merge program (Δ = (α/r)·A@B)
           ──▶ int8 quantization review
           ──▶ segmented arena binding: RODATA | PERSISTENT | IO | TRANSIENT
           ──▶ persistent image init (counter-based randn: bit-identical
               at any thread count, parallel within a tensor)
           ──▶ instruction lowering (train / eval / merge)
           ──▶ .seeu plan assembly, sealed with PlanSelfHash
           ──▶ (optional) native package emission
```

## source/ — the source language

Not under `compiler/` on purpose: these are the abstractions of the *source
language* and the substrate both the compiler and runtime share, not any
stage of compilation. Partitioned by functionality in the same fashion:

- **`language/`** — `model_format.{h,cc}`, the SMF container structs
  (`SmfModel`, `SmfTensor`, `SmfOp`); the byte format is specified in
  [formats.md](formats.md).
- **`plan/`** — the update-plan ABI behind the `update_types.h` façade,
  split per discipline: `config.h` (the compilation request —
  `UpdateConfig` and its specs), `instruction.h` (tensor refs, the opcode
  vocabulary, the 64-byte `UpdateInstruction`), `schema.h` (magic/version,
  `PlanHeader`, `EmitEntry`).
- **`parallel/`** — `parallel_for.{h,cc}`, deterministic chunked
  data-parallelism; chunk geometry depends only on the problem shape,
  never the thread count. Exception-safe: a throwing chunk body stops new
  chunks, the loop retires fully, and the first exception is rethrown on
  the calling thread — no worker ever unwinds or touches a dead job.
  `SEEML_THREADS` is parsed sign- and overflow-checked; invalid values
  fall back to hardware concurrency.
- **`identity/`** — `hash.h`, the integrity identity of every artifact:
  incremental 64-bit FNV-1a, the deterministic parallel `ContentHash64`
  (model and checkpoint identity), and `PlanSelfHash` (the plan seal —
  same chunked fold with the in-blob hash field treated as zero), shared
  verbatim by the compiler, the runtime, and the dump tool.

## compiler/frontend/ — SMF bytes to forward SIR

- **`ingressor/`** — container I/O and the ingest gate. `model_reader`
  (`LoadSmf` / `LoadSmfMany`) is fully bounds-checked — every blob is
  validated before the first byte moves, then large payload copies fan out
  over `ParallelFor`. `model_writer` serializes with 64-byte-aligned data
  offsets. `resource_analyzer` estimates the training footprint and refuses
  models that provably cannot train in local memory.
- **`parser/`** — `BuildForward` turns the decoded op list into SIR:
  whole-graph checks first (`sema`: topological order, unique outputs,
  producible model output), then per-op shape semantics, with
  `value_resolver` materializing frozen weights on first use and recording
  them in `GraphBuild` for rodata packing and the emit table.
- **`representation/`** — SIR, the SSA intermediate representation, split
  per core definition (`type` / `value` / `operation` / `block`) behind the
  `sir.h` façade. `Block::verify()` is the invariant gate the rest of the
  compiler leans on. Threading model: single writer — use-lists are written
  during construction, so a block is built by one thread, then freely read
  by many.
- **`operator/`** — `OpBuilder`, the typed constructors for compound ops,
  split per op family (`convolution` / `linear` / `normalization` /
  `activation`) behind the `op_builder.h` façade.

SIR ops live in three dialects — `sc_mem.*` (storage declarations),
`sc_high.*` (differentiable forward ops), `sc_low.*` (adjoints, optimizer
steps, merge kernels) — with the `sc_ctrl.*` prefix reserved for control
flow.

## compiler/analysis/ — the training program

All seven units re-exported by the `update_passes.h` façade.

- **`updater/`** — orchestration and structural lowering. `PassManager`
  runs named passes and re-verifies the block after each one, so a
  corrupting rewrite is attributed to its author; a pass's own error is
  propagated verbatim. `ConvLowering` rewrites `sc_high.conv2d` into
  im2col + filter-matrix + GEMM (+ bias) + col2im, rejecting group/dilated
  forms it cannot model.
- **`algebra/`** — kernel-fusion algebra. `LoraGrafter` grafts rank-r
  adapters onto every eligible frozen MatMul (A randn-initialized, B zeros,
  so step 0 is exactly the base model). A tied weight — one frozen tensor
  consumed by several MatMuls — shares a **single** adapter pair across its
  sites: every site computes `x_i @ (W + Δ)` during training, autodiff sums
  the pair's gradients across sites, and commit writes exactly that `W + Δ`
  (per-site pairs would commit `W + ΣΔ_i`, polluting every site with every
  other's delta). `MergeBuilder` builds the separate merge program — one
  fused `sc_low.gemm_acc` per adapter materializing Δ = (α/r)·A@B — and
  rejects duplicate frozen weights outright.
- **`calculus/`** — the mathematics of the update. `TrainableAutodiff` is
  reverse-mode autodiff pruned to the trainable set (needs-grad marking,
  a VJP rule registry; frozen and teacher subgraphs get no backward
  compute). `OptimizerSynthesizer` appends SGD or AdamW as SIR, so one
  program execution is one full training step.
- **`reviewer/`** — preprocessing that configures backend behavior:
  `SelectQuantizedWeights` scans frozen weights (parallel max-abs sweep,
  `kWeightSweepGrain` chunks) and selects per-tensor symmetric int8 scales
  for rodata packing.

## compiler/backend/ — code generation

- **`trainer/`** — generates the training program's code. `arena_binder`
  packs rodata and lays out the mutable arena (persistent parameters and
  moments, I/O, then liveness-scanned transient reuse). `instruction_lowering`
  lowers SIR ops to the fixed 64-byte `UpdateInstruction` stream.
  `native_emitter` writes the self-contained native package (plan, embedded
  TU, driver `main`, vendored runtime, build script). `kernel_emitter`
  generates threadgroup-tiled Metal GEMM kernel source (forward, both
  backward transposes, and the merge's scaled accumulate) from a `GpuTiling`
  clamped out of the host tiling.
- **`architecture/`** — local device analysis informing the trainer.
  `DetectHostArch` reads ISA/SIMD/FMA from the compilation target (host =
  target for this compiler) and cache/core geometry from sysctl on macOS
  and the sysfs CPU topology (physical cores, not hardware threads) plus
  sysconf on Linux, warning and falling back when a probe comes back
  empty. `SuggestGemmTiling` derives BLIS-style blocking (kc×nc panel in
  half of L1, mc×kc in half of L2) as a pure function of the host
  description, and `ValidateGemmTiling` enforces that contract —
  overflow-safely, since it is the trust boundary for deserialized
  tilings — on any tiling handed in from outside.
- **`tuner/`** — reinforcement tuning that refines the architecture hint on
  the real machine. `Ucb1Bandit` is a deterministic UCB1 multi-armed bandit
  (no RNG; ties break to the lowest index). `TilingCandidates` spans the
  hint and its halved/doubled neighbors; `AutotuneGemmTiling` spends an
  injected measurement budget (the benchmark is a parameter, so tuning is
  testable without a GPU or wall clock) and reports the winning tiling with
  full per-arm statistics.

## compiler/diagnostics/ — errors by process

Every compiler diagnostic is one line, `"<unit>: <message>"`, formed by the
process module it belongs to; errors travel as
`std::expected<T, std::string>`. The core (`diagnostic.h`) provides
`Fail` / `Note` / `Fallback` over the thread-safe `Logger`; each process
module is header-only and owns its unit registry and message shapes:

| module | process delimited | units |
|---|---|---|
| `tokenizing/` | SMF byte-stream decode/encode | `SMF`, `Ingressor` |
| `parsing/` | SMF graph → forward SIR | `Parser` |
| `passing/` | pass orchestration + lowering legality | `PassManager`, `ConvLowering` |
| `updating/` | the analytic methods | `TrainableAutodiff`, `LoraGrafter`, `MergeBuilder`, `OptimizerSynthesizer` |
| `architecting/` | local device analysis | `HostArch`, `Autotuner` |
| `generating/` | code generation + the driver | `UpdateCompiler`, `ArenaBinder`, `InstructionLowering`, `NativeEmitter` |

`architecting/` has a two-tier discipline: detection can never hard-fail
(warn and fall back to conservative defaults), while tiling-contract
violations are hard errors.

## compiler/driver/ — orchestration and verification

The driver owns the compilation *process*, nothing else. Beyond sequencing,
it verifies at every boundary that each subsystem was used correctly
(`contract.h`):

- `VerifyFrontendContract` — after the forward builds: graph build carries
  input/output, the SIR verifies, every weight source is a constant SMF
  tensor declared `sc_mem.weight`.
- `VerifyAnalysisContract` — after the merge program: adapters exist, every
  A/B is an `sc_mem.param` with a gradient reaching it (exactly two per
  adapter), the merge program verifies and covers every adapter.
- `VerifyGeneratedPlan` — before returning: plan non-empty, all three
  programs lowered (eval ≤ train; ≥ one merge instruction per adapter),
  the arena contains its persistent segment, frozen weights reached rodata,
  debug hooks cover the trainable set.
- `WellFormedDiagnostic` — at the outer error boundary: any error escaping
  `Compile` must be attributable to a unit registered in `diagnostics/`.

Contract violations are driver bugs, not user errors, and report under the
driver's own unit.

## Testing

One SeeTest suite per module, organized to mirror this partition
(`test/compiler/<subsystem>/*_test.cc` — see `test/README.md`), run via
`ctest` or directly from `build/` (see [usage.md](usage.md)). The
compiler-side suites: `frontend/` (`model_io`, `resource_analyzer`, `sir`,
`operator`, `parser`), `analysis/` (`update_passes`, `updater`,
`reviewer`), `backend/` (`tuner`, `trainer`, `native_emitter`), `driver/`
(`update_compiler`, `driver`), and `diagnostics/`, plus `source/` suites
(`hash`, `parallel_for`) and the end-to-end `system/update_system_test`.
