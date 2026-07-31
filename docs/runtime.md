# The SeeML Update Runtime — Architecture

`runtime/` is the zero-dependency half of the product: the code vendored
into every emitted package and executed on the device. It is partitioned in
the same fashion as [the compiler](compiler.md) — subsystems named for
their role in the update, folders for the discipline of the work inside,
façade headers where a subsystem's units are split — with the engine
playing the driver's role: it owns the *process* and verifies every
boundary it crosses.

```
runtime/
  engine/       orchestrates the update process, verifies every boundary
    update_engine  contract
  feeder/       corpus decode and pipelined batch staging
    dataset  batch_pipeline
  executor/     the kernel library the dispatcher executes
    update_kernels.h (façade)  kernel_policy.h
    gemm  elementwise  activation  normalization  loss  optimizer
  validator/    load-time bounds proof of every instruction
    plan_validator
  custodian/    durable state — checkpoints, atomic commits
    durable_io  checkpoint
  diagnostics/  error handling, partitioned by process
    feeding/  validating/  executing/  persisting/
```

## The update process

`UpdateEngine` (`runtime/engine/update_engine.h`) runs the ML analog of an
OS software update:

```
Load    ──▶ header identity + integrity (magic, version, whole-plan hash)
        ──▶ plan contract ──▶ decode ──▶ executor contract ──▶ one arena
Train   ──▶ feeder contract ──▶ N steps of {stage batch → execute stream}
Gate    ──▶ eval program before/after; no improvement → device untouched
Merge   ──▶ merge program materializes each Δ = (α/r)·A@B
Commit  ──▶ hash-checked source copy + deltas, fsync'd, atomically renamed
```

One allocation (the pre-planned arena), one thread pool, and a fixed
instruction stream — everything else was decided by the compiler.

## engine/ — orchestration and verification

The engine only abstracts the update process. Beyond sequencing, it
verifies at every boundary that each subsystem was used correctly
(`contract.h`, mirroring `compiler/driver/contract.h`):

- `VerifyPlanContract` — at load: the header is self-consistent with the
  blob (no section-size overflow, every section in bounds, the arena
  dominates its persistent segment and image, the input/label/loss I/O
  slots lie inside the mutable arena). This re-proves at load time what the
  compiler's driver promised at emit time.
- `VerifyExecutorContract` — after decode: every operand ref of every
  instruction passes the validator against its address space, and every
  emit entry targets the arena. After this gate, `Execute()` dispatches
  the programs blindly.
- `VerifyFeederContract` — before training: the corpus matches the
  compiled plan's geometry (batch × input width, label kind and width,
  class labels inside the softmax width).
- `WellFormedDiagnostic` — at the `Train` boundary (a `Train`/`TrainImpl`
  split, exactly like the driver's `Compile`/`CompileImpl`): any escaping
  error must be attributable to a unit registered in `diagnostics/`.

Contract violations mean a corrupt or foreign artifact, or a misused
subsystem, and report under the engine's own unit.

## feeder/ — the corpus

- **`dataset`** — the SDS container ([formats.md](formats.md)): fully
  validated before the first sample is served; batches served sequentially
  with wraparound or through a seeded per-epoch permutation, deterministic
  and allocation-free per batch.
- **`batch_pipeline`** — overlaps feeding with compute: a feeder thread
  stages batch s+1 while step s executes. The batch sequence is exactly the
  serial sequence — pipelining changes *when* batches are materialized,
  never *which* — and the destructor joins on every exit path. With
  `SEEML_THREADS=1` no thread is created at all.

## executor/ — the kernel library

`update_kernels.h` is the façade; the implementations are partitioned per
kernel family, all sharing `kernel_policy.h` — the decomposition policy
that makes parallel execution bitwise-deterministic (chunk boundaries are a
pure function of the problem shape; reductions combine per-chunk partials
in chunk order) and the no-overlap aliasing contract the arena allocator
guarantees.

- **`gemm`** — the four f32 variants and two dequantizing int8 variants,
  reduced to two cache-blocked cores.
- **`elementwise`** — pointwise arithmetic, bias broadcast, row reduction,
  serial fills/copies.
- **`activation`** — ReLU / GELU / SiLU forward-backward pairs; each
  backward differentiates exactly its forward's expression.
- **`normalization`** — LayerNorm with cached per-row statistics.
- **`loss`** — softmax cross-entropy, MSE, KL distillation; NaN losses
  propagate so the engine's finite-loss guard trips.
- **`optimizer`** — norm clipping, SGD, AdamW (in place).

Every kernel is allocation-free over caller-provided arena/rodata pointers.

## validator/ — the load-time proof

`ValidateInstruction` checks every operand ref of an instruction against
the address space it targets, with byte extents derived exactly as the
kernels derive their loop bounds; writes may only target the mutable
arena, and unknown opcodes are load errors, never silent skips. Its
overflow-safe helpers (`MulOk`, `RangeOk`) are the single way plan bounds
math is written.

## custodian/ — durable state

- **`durable_io`** — the runtime's only path for bytes that must survive a
  power cut: fsync'd sidecar, atomic rename, best-effort directory fsync.
  Also the commit path's scaling and concurrency primitives:
  `HashFileContent` streams a file's ContentHash64 one deterministic chunk
  at a time, `DurableFileEdit` is a copy-on-write editor that patches byte
  ranges of a sidecar and durably renames it into place, and `CommitLock`
  (an OS-level flock, released even on crash) refuses a second update
  committing to the same target instead of letting the rename race decide.
  Commit memory is O(chunk), never O(model), and the hash is verified on
  the exact copy being patched — no check-to-patch window.
- **`checkpoint`** — the SEKP container: the arena's persistent segment,
  hash-bound to the exact plan that laid it out; fully verified before a
  byte reaches the arena.

## diagnostics/ — errors by process

Mirrors `compiler/diagnostics/`: one-line `"<unit>: <message>"`
diagnostics formed by header-only process modules over a shared core
(`diagnostic.h` — message formation only; the zero-dependency runtime has
no logger).

| module | process delimited | units |
|---|---|---|
| `feeding/` | SDS decode + batch staging | `Dataset`, `BatchPipeline` |
| `validating/` | load-time plan verification | `PlanValidator` |
| `executing/` | engine lifecycle + dispatch | `UpdateEngine` |
| `persisting/` | checkpoints + atomic commits | `DurableIO`, `Checkpoint` |

## Vendoring

Every unit above (plus the `source/plan/` ABI headers,
`source/identity/hash.h`, and the `source/parallel/` substrate) is vendored into the emitted package by the
compiler's `native_emitter`, whose generated `build.sh` compiles the
thirteen runtime translation units and links `model_update` — the package
builds with no access to this repository.

## Testing

Runtime suites, organized to mirror this partition
(`test/runtime/<subsystem>/*_test.cc` — see `test/README.md`):
`executor/kernels` (per-family numeric checks against references),
`feeder/dataset` and `feeder/batch_pipeline` (the staged sequence is
exactly the serial one), `validator/validator` (per-opcode bounds proofs,
plus the regression that every compiled instruction validates),
`custodian/custodian` (durable writes, checkpoint binding and corruption
rejection), `engine/engine` (the boundary contracts and the unit
registry), and `engine/update_engine` (the VM lifecycle end to end,
gradient checks, checkpoint resume), plus the cross-half
`system/update_system_test` and the emitter suite that verifies the
vendored package layout.
