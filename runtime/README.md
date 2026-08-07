# The SeeML Runtime

## A virtual machine whose instruction set is *training*

A CPU fetches an instruction, decodes it, executes it, repeats — and nothing
says the machine doing that must be hardware. SeeML's runtime is a virtual
machine whose opcodes happen to be forward passes, gradients, and optimizer
steps. It is the **zero-dependency** half of the product: the code vendored
into every emitted package and run on the device. No framework, no allocator
churn, no graph library — because the [compiler](../compiler/README.md)
already decided everything. What's left on the device is deliberately
boring: one allocation, one thread pool, three fixed instruction streams,
and a dispatch loop.

But "boring" is *earned*. The runtime's real job, beyond executing, is
refusing to execute anything it cannot first prove safe. It is partitioned
in the same fashion as the compiler — subsystems by role, folders by
discipline, façade headers where units split — with `engine/` playing the
driver's role.

```
runtime/
  engine/                 orchestrates the update, verifies every boundary
    update_engine         load -> train -> gate -> merge -> commit
    contract              the load-time boundary proofs
  feeder/                 corpus decode and pipelined batch staging
    dataset               SDS load, shuffle/split, token records
    batch_pipeline        stage the next batch while this step computes
  executor/               the kernel library the dispatcher runs
    update_kernels.h      façade over the families below
    kernel_policy.h       the determinism + aliasing contract
    gemm elementwise activation normalization loss optimizer attention
    metal_gemm            (Apple-only) GPU GEMM dispatch harness
  validator/              load-time bounds proof of every instruction
  custodian/              durable state — checkpoints, atomic commits
  diagnostics/            errors, partitioned by process
    feeding/ validating/ executing/ persisting/
```

## The update, phase by phase

Read `engine/` as the spine. **Load** checks the plan's identity and
integrity, then re-proves — through **`validator/`** — that every operand of
every instruction is in bounds, so the hot loop can later dispatch *blindly*.
**Train** pulls batches from **`feeder/`** (which stages batch *s+1* while
step *s* runs) and executes the train stream through **`executor/`**. **Gate**
runs the eval program before and after; no improvement means the device is
left untouched. **Merge** materializes each adapter's delta, and **Commit**
adds those deltas onto the source file's pristine weights and writes the
result through **`custodian/`** — fsync then atomic rename, so a power cut
leaves the old file or the new one, never a torn one.

Two properties hold across all of it, and both are contracts the tests pin:
execution is **bitwise-deterministic** at any thread count (the executor
chunks work by problem shape, never by worker count), and every data-
dependent index — a class label, a token id — is proven inside its bound at
load, so a validated program can gather and index without a single runtime
check.

## Where to go next

This is the map; the walkthrough is in
**[docs/runtime.md](../docs/runtime.md)** — load-time validation as a safety
proof, why naive kernel formulas explode and how these stay numerically
stable, the deterministic parallelism, the producer-consumer pipeline, and
storage that survives a power cut. The plan format it consumes is in
[docs/formats.md](../docs/formats.md); the compiler that produces it is in
[docs/compiler.md](../docs/compiler.md). Every subsystem here has a matching
suite under [test/runtime/](../test/README.md).
