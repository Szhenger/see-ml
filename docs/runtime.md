# The SeeML Update Runtime

## A virtual machine, of all things

Think back to how a CPU works: fetch an instruction, decode it, execute it, repeat. Now recall that nothing says the "machine" doing this must be hardware. A program that loops over instructions and dispatches on an opcode is a **virtual machine**, and you've used dozens without noticing — the Python interpreter is one, the JVM is another.

SeeML's runtime (`runtime/`) is a virtual machine whose instruction set happens to be *training*. It is the zero-dependency half of the product: the code vendored into every emitted package and executed on the device. No framework, no allocator churn, no graph library — because the compiler ([compiler.md](compiler.md)) already decided everything. What remains on the device is deliberately boring: one allocation, one thread pool, three fixed instruction streams, and a dispatch loop.

But "boring" has to be *earned*. The runtime's real job, beyond executing, is refusing to execute anything it cannot first prove safe. Keep that lens as we go.

```
runtime/
  engine/       orchestrates the update process, verifies every boundary
  feeder/       corpus decode and pipelined batch staging
  executor/     the kernel library the dispatcher executes
  validator/    load-time bounds proof of every instruction
  custodian/    durable state — checkpoints, atomic commits
  diagnostics/  error handling, partitioned by process
```

The partition mirrors the compiler's — subsystems named for their role, façade headers where units split — with `engine/` playing the driver's role: it owns the *process* and verifies every boundary it crosses.

## The update process, end to end

`UpdateEngine` (`runtime/engine/update_engine.h`) runs the ML analog of an OS software update:

```
Load    ──▶ header identity + integrity (magic, version, whole-plan hash)
        ──▶ plan contract ──▶ decode ──▶ executor contract ──▶ one arena
Train   ──▶ feeder contract ──▶ N steps of {stage batch → execute stream}
Gate    ──▶ eval program before/after; no improvement → device untouched
Merge   ──▶ merge program materializes each Δ = (α/r)·A@B
Commit  ──▶ hash-checked source copy + deltas, fsync'd, atomically renamed
```

Let's take each phase seriously.

## engine/ — orchestration and verification

### Loading: paranoia as a protocol

What could a hostile — or merely corrupted — plan file do to a naive loader? Claim a section that extends past the file. Claim instruction counts that overflow when multiplied by 64. Reference memory outside the arena. Contain an opcode from a future version. The load path treats every one of these as a first-class possibility:

1. **Identity.** Check the magic and version; then recompute the FNV-1a hash of the whole blob (with the stored hash field zeroed) and compare. A single flipped bit anywhere in the plan fails this check. Cheap, and it converts "mysterious crash later" into "refused at load."
2. **`VerifyPlanContract`.** Prove the header is self-consistent with the blob: no section-size multiplication overflows, every section lies within the file, the arena dominates its persistent segment and initial image, and the input/label/loss I/O slots lie inside the mutable arena. This *re-proves at load time what the compiler's driver promised at emit time* — the device does not extend trust across the air gap.
3. **Decode**, then **`VerifyExecutorContract`.** Every operand of every instruction in all three programs is bounds-checked by the validator (next section), and every emit-table entry targets the arena. After this gate, `Execute()` dispatches the programs *blindly* — no per-instruction checks in the hot loop, because everything was proven once, up front.
4. **One arena.** A single 64-byte-aligned allocation of the compile-time-computed size, zeroed, with the persistent segment's initial image copied in. This is the only allocation the engine ever makes.

The engine also mirrors the compiler's outermost discipline: `Train` wraps `TrainImpl` exactly as the driver's `Compile` wraps `CompileImpl`, and any escaping error must be attributable to a unit registered in `diagnostics/` (`WellFormedDiagnostic`). Contract violations mean a corrupt or foreign artifact, or a misused subsystem, and report under the engine's own unit.

### Training: the loop that is barely a loop

Each step `s` (with `step_ = s + 1`, 1-indexed, because AdamW's bias correction needs a step number that starts at 1):

1. Ask the feeder for the next batch (which was staged *during* the previous step — see below).
2. `Execute(train_program)` — one pass over the fixed instruction stream is one full training step: forward, loss, backward, clip, optimizer.
3. Read the loss from its known arena slot and check `isfinite(loss)`. A NaN or infinity aborts the update immediately — and the kernels are written so NaN *propagates* to the loss rather than being masked (a NaN probability is never clamped away, only underflow is). Better a clean abort than 900 more steps of garbage. The model on disk is untouched by construction.
4. Optionally: log the loss, append to the loss curve, checkpoint every k steps.

The **learning-rate schedule** is evaluated per step from header fields. Constant, or cosine-with-warmup:

```
warmup   (step ≤ W):  lr = base · step / W                    (ramp in linearly)
decay    (t = (step − W)/horizon):  lr = floor + (base − floor)·½·(1 + cos(π·t))
beyond horizon:       lr = floor = base · min_lr_factor
```

Why warm up? Early AdamW steps operate on moment estimates built from almost no data; a few gentle steps let the statistics settle before full-size updates. Why cosine? It spends most of the budget at useful learning rates and glides smoothly to the floor, avoiding the sharp cliffs of step schedules.

### The gate: no improvement, no change

Would you install a software update that made your phone worse? Neither would SeeML. If a validation split exists, the engine runs the **eval program** — the compiler's snapshot of the adapted forward pass plus loss, *before* autodiff ever touched the graph — over the held-out set before training and again after (rewinding the dataset so both passes see the identical sample sequence). The update is accepted only if `val_final < val_initial`, strictly. Without a validation split, the fallback compares the average training loss over the first and last few steps (a window of up to 20). The device stays untouched on rejection — exit code 3, and the caller can decide `--force` is warranted.

### Merge and commit

`RunMerge` executes the merge program: each adapter's `Δ = (α/r)·A@B` materialized into its pinned arena slot. Then `CommitToModel`:

1. Read the source `.smf` file and recompute its `ContentHash64`; if the plan carries a source hash (nonzero), a mismatch **refuses the commit**. You cannot patch the wrong file, or a modified one.
2. For each emit-table entry — after bounds-checking it against the actual file size — add the delta onto the file's pristine f32 weights: `W′ᵢ = Wᵢ + Δᵢ`. Note the source: the *file's* weights, not the (possibly int8-quantized) rodata copy. Quantization error never reaches the committed model.
3. Write the result durably (fsync + atomic rename — see custodian below) to the output path. The source file itself is never modified.

## validator/ — the load-time proof

Here's the mindset shift that makes the executor simple: instead of checking bounds *during* execution (every kernel, every step, millions of times), prove the whole program safe *once*, before running any of it. `ValidateInstruction` (`runtime/validator/plan_validator.cc`) is that proof, run per instruction at load.

For each instruction, the validator knows — per opcode — exactly which operands are read, which are written, and how many elements each must span, with byte extents derived *exactly as the kernels derive their loop bounds* (a GEMM instruction with dims M, N, K implies extents M·K, K·N, M·N). It then checks every operand reference:

- Null refs and zero-extent operands are rejected (a zero-size claim would be an unchecked pointer).
- The element-count × element-size multiplication is done with `MulOk`, which fails on overflow rather than wrapping — `a > UINT64_MAX / b` asked *before* multiplying. `RangeOk(offset, bytes, size)` similarly refuses to wrap. These two helpers are the *single* way plan bounds math is written, anywhere.
- Offsets must be element-aligned, and the extent must lie inside the address space the ref's bit 63 selects — arena or rodata.
- **Writes may only target the mutable arena.** A write to rodata is a validation error, structurally impossible to express at runtime.
- Unknown opcodes are load errors, never silent skips — a plan from a newer compiler fails loudly.
- Finally, the **aliasing proof**: within an instruction, any two operand extents where at least one is a write must not overlap. This is what licenses the kernels' `restrict`-qualified pointers (and the vectorization they enable): the no-overlap guarantee isn't an assumption, it's a checked theorem.

An int8 GEMM gets one extra rule: its quantized B operand must live in rodata — quantized bytes are frozen bytes, by construction.

## executor/ — the kernel library

`update_kernels.h` is the façade; implementations are partitioned per kernel family, all sharing `kernel_policy.h`. Every kernel is allocation-free over caller-provided arena/rodata pointers. Two ideas run through the whole library — determinism and numerical care — so let's establish both before touring the families.

### Determinism: same bits, any thread count

Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)` can differ in the last bit. So if you split a sum across threads and combine "whenever threads finish," your result depends on scheduling — run twice, get two answers. SeeML forbids this categorically, with one policy (`kernel_policy.h` + `source/parallel/`):

- Work is split into chunks whose boundaries are a **pure function of the problem shape**: grain `g = max(caller_grain, ceil(n/256))`, chunk `c` covering `[c·g, min((c+1)·g, n))`. The thread count appears nowhere in that formula — 256 is the fixed chunk ceiling (`kMaxParallelChunks`), which also lets reductions size stack arrays with no allocation.
- Each chunk writes only its own output slice (exactly one writer per element — also why there are no data races), or its own partial-reduction slot.
- Reductions accumulate per-chunk partials in `double`, then combine them **in chunk order** — a fixed association, regardless of which thread computed what, when.

Threads then just *claim* chunks dynamically from an atomic counter (good load balancing), but since chunks are deterministic and combination order is fixed, the bits are identical on 1 core or 8. `SEEML_THREADS=1` never creates a thread at all. Thread count is a throughput knob, not a numerics knob — which is what makes "the loss curve from the dev board reproduces on the target" a guarantee rather than a hope.

The grain constants encode a cost model in two numbers: `kGrainCheap = 32768` elements for arithmetic-bound bodies, `kGrainMath = 4096` for transcendental-heavy ones — chosen so small tensors (LoRA adapters, loss scalars) never leave the calling thread.

### gemm — the workhorse

Matrix multiplication is `O(M·N·K)` arithmetic over `O(MN + NK + MK)` data — the rare kernel where compute can outrun memory, *if* you don't thrash the cache. The runtime's answer (`gemm.cc`) is two cache-blocked cores behind six public variants:

- **BlockedNN** computes `C += α·A@B` with B row-major, tiled `kTileK = 64` deep and `kTileN = 256` wide so the active B panel stays cache-resident; the inner loop is unrolled 4-wide over k — four independent multiply chains, four fewer C round-trips per pass.
- **BlockedNT** computes `C = α·(A row · B row)` dot products for the transposed-B case, four output columns per pass so the streamed A row is reused from L1 four times.

The six variants — `NN`, `NT`, `TN`, accumulating `NN`, and the two int8 forms `NNQ8`/`NTQ8` — are thin wrappers selecting core, transpose flag, and whether C is zeroed or accumulated. Recall from [compiler.md](compiler.md) why `NT` and `TN` exist at all: they are the two matmul VJPs (`dX = dC@Wᵀ`, `dW = Xᵀ@dC`). And the int8 variants dequantize *inside* the inner loop — the B element is `static_cast` from int8 and the per-tensor scale rides the existing α multiply, so dequantization costs literally zero extra instructions. All variants parallelize over C's rows; each worker owns a disjoint row slice whose arithmetic order matches the serial code exactly.

### loss — where numerical stability lives

First, the vocabulary, because it recurs everywhere in ML: a classifier's raw output scores are its **logits**; **softmax** converts logits into a probability distribution by exponentiating each and normalizing — `pᵢ = exp(zᵢ)/Σⱼ exp(zⱼ)`, so bigger scores get bigger shares and everything sums to 1; and **cross-entropy** scores the result as `−log p(correct class)` — nearly free when the model is confidently right, ruinous when it is confidently wrong, which is exactly the incentive you want a loss to create.

Now try computing that softmax naively with a logit of 800. `exp(800)` overflows to infinity; your probabilities become NaN. The classic fix (the **log-sum-exp trick**): subtract the row maximum first. Since softmax is invariant to shifting all logits, `exp(zᵢ − max)/Σexp(zⱼ − max)` is mathematically identical and never overflows (the largest exponent is exactly 0). Every softmax in `loss.cc` does this.

- **Softmax cross-entropy**: `L = −(1/N)·Σₙ log p(yₙ)`, accumulated in double, per-chunk, chunk-ordered. A NaN probability propagates (so the engine's finite-loss guard trips); an underflowed one is clamped to 10⁻¹² so `log` stays finite. Backward: the beautiful `dlogits = seed·(probs − onehot)/N`.
- **MSE**: `L = (1/n)·Σ(pᵢ − tᵢ)²`, backward `dp = 2·seed·(p − t)/n`.
- **KL distillation**: both student and teacher logits softmaxed at temperature T (dividing logits by T softens the distributions, exposing the teacher's "dark knowledge" — the relative probabilities of *wrong* classes); loss is the forward KL divergence `(1/N)·Σₙ Σ_c p_t·(log p_t − log p_s)` — **KL divergence** being the standard measure of how far one probability distribution strays from another, zero exactly when they agree; backward `dlogits = seed·(p_s − p_t)/(N·T)`.

### activation — forward/backward pairs

Each backward differentiates *exactly its forward's expression* — the discipline that makes gradient checking meaningful:

- **ReLU**: `max(x, 0)`; backward gates `dy` on `x > 0` (subgradient 0 at the kink).
- **GELU** (tanh approximation): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, and its exact derivative.
- **SiLU**: `x·σ(x)`, where `σ` is the *sigmoid* `1/(1+e⁻ˣ)` — the S-curve that squashes any real number into (0, 1); backward `dy·σ(x)·(1 + x·(1 − σ(x)))`.

### normalization — LayerNorm with receipts

Forward, per row of width d: mean μ, biased variance, `rstd = 1/√(var + 10⁻⁵)`, output `(x − μ)·rstd·γ + β` — and the per-row μ and rstd are **cached** into the two companion slots the compiler allocated. Backward then uses them directly: with `x̂ = (x − μ)·rstd` and `g = dy·γ`,

```
dx = rstd · (g − mean(g) − x̂·mean(g·x̂))
```

— the two mean-subtractions being exactly the correction terms from differentiating through a row's own statistics. Row sums accumulate in double. Only `dx` is computed; γ and β are frozen by design ([compiler.md](compiler.md)).

### optimizer — clip, SGD, AdamW

`ClipNorm` computes a tensor's L2 norm — its Euclidean length, `√(Σgᵢ²)` — with double partials, chunk-ordered (so even the *decision* to clip is deterministic), and rescales by `max_norm/‖g‖` only if `‖g‖ > max_norm`. `SgdStep` and `AdamWStep` apply the update equations derived in [compiler.md](compiler.md), in place, with the moments living in the persistent segment — which is precisely why a checkpoint can resume mid-optimization without losing Adam's memory.

## feeder/ — the corpus

### dataset — validation, then randomness done right

The SDS container is fully validated before the first sample is served — magic, version, and the payload-size arithmetic done overflow-safe, exactly in the spirit of the plan loader. Class labels are range-checked against the plan's softmax width up front, so a corrupt label can't index out of bounds mid-training.

Shuffling poses a question worth dwelling on: how do you shuffle *reproducibly, on every platform*? Not with `std::shuffle` — the standard deliberately doesn't pin down distribution algorithms across implementations. SeeML uses two classical pieces:

- **SplitMix64**, a tiny, fast PRNG — *pseudorandom number generator* — (add a golden-ratio-derived constant; xor-shift and multiply twice) with a fully specified, platform-independent output sequence.
- **Fisher–Yates**: for `i` from n−1 down to 1, swap element `i` with a random element in `[0, i]`. One pass, every permutation equally likely — the canonical unbiased shuffle.

Each **epoch** — one full pass through the dataset — ends by reshuffling with the PRNG's *advancing* state, so every epoch sees a fresh permutation, yet the entire sequence of permutations is a pure function of the initial seed. The validation split, if requested, is carved from the corpus *tail* before shuffling ever starts — held-out means held out.

### batch_pipeline — overlapping I/O with compute

While step s computes, someone could already be gathering batch s+1. That someone is the feeder thread, and the coordination is the classic **producer-consumer** pattern: one staging buffer, one mutex, one condition variable, one boolean `full`. The feeder fills the buffer while `full == false` and flips it true; `NextBatch` waits on `full == true`, copies out, flips it false, and wakes the feeder. Each side touches the buffer only in its own phase of the flag — that alternation *is* the correctness argument.

Two properties matter more than the concurrency: the staged sequence is **exactly the serial sequence** — pipelining changes *when* batches are materialized, never *which* — and the destructor joins the thread on every exit path (the engine scopes the pipeline so even an error return can't leak it). With `SEEML_THREADS=1`, no thread is created at all and `NextBatch` fills synchronously — serial mode isn't a special case, it's the same code with the overlap removed.

## custodian/ — durable state

### Why "just write the file" is a bug

Suppose the device loses power halfway through writing the updated model. What's on disk? Half a file. Worse: even a *completed* `write()` may still be sitting in OS caches, and the power cut eats it. It turns out crash-safe persistence has a standard liturgy, and `durable_io.cc` follows it exactly:

1. Write everything to a **sidecar** file (`path + ".tmp"`), looping on partial writes and retrying `EINTR`.
2. `fsync` the sidecar — force the bytes out of the cache onto the medium. Not optional; a failed fsync fails the operation.
3. `rename(tmp, path)` — POSIX guarantees this is **atomic**: any observer sees the old file or the new file, never a mixture.
4. `fsync` the *directory*, best-effort — the rename itself is a directory entry that also needs to reach the medium.

A power cut at any instant leaves either the complete old file or the complete new file. Never a torn one. This one function is the runtime's *only* path for bytes that must survive a power cut; commits and checkpoints both go through it. (The API takes a gather-list of byte spans, so header + payload are written without staging a concatenated copy.)

### checkpoint — resumable, but never confusable

A checkpoint (the SEKP container — [formats.md](formats.md)) is the arena's persistent segment — adapters plus optimizer moments — prefixed by a header carrying three protections: the **plan hash** (a checkpoint resumes only under the *exact* plan that produced it — offsets are meaningless under any other layout), the persistent **size** (must match), and a **payload hash** (a bit-flipped checkpoint is rejected before a single byte reaches the arena). Foreign, stale, or corrupt state never loads; the engine warns and trains from scratch instead. Resuming also restores the step counter, so AdamW's bias correction continues from the right t.

## diagnostics/ — errors by process

Mirrors `compiler/diagnostics/`: one-line `"<unit>: <message>"` diagnostics formed by header-only process modules over a shared core — message formation only, since the zero-dependency runtime has no logger.

| module | process delimited | units |
|---|---|---|
| `feeding/` | SDS decode + batch staging | `Dataset`, `BatchPipeline` |
| `validating/` | load-time plan verification | `PlanValidator` |
| `executing/` | engine lifecycle + dispatch | `UpdateEngine` |
| `persisting/` | checkpoints + atomic commits | `DurableIO`, `Checkpoint` |

## Vendoring

Every unit above (plus the `source/plan/` ABI headers, `source/identity/hash.h`, and the `source/parallel/` substrate) is vendored into the emitted package by the compiler's `native_emitter`, whose generated `build.sh` compiles the runtime translation units and links `model_update` — the package builds with no access to this repository. The runtime is not a library you install on the device; it's source that travels *with each update*.

## To recap

- The runtime is a small VM executing three compiler-fixed instruction streams with **one** allocation.
- Safety is front-loaded: hash the plan, prove the header, validate every operand's bounds *and* the no-aliasing theorem once at load — then dispatch blindly.
- Kernels are numerically careful (log-sum-exp, double partials, cached statistics) and **bitwise-deterministic** at any thread count, by chunking on problem shape and combining in fixed order.
- The corpus shuffles with SplitMix64 + Fisher–Yates for platform-independent reproducibility; a feeder thread hides I/O behind compute without changing the batch sequence.
- Nothing durable is ever half-written (fsync + atomic rename), nothing improves nothing changes (the eval gate), and nothing foreign is ever trusted (hash-bound commits and checkpoints).

## Testing

Runtime suites, organized to mirror this partition (`test/runtime/<subsystem>/*_test.cc` — see [test/README.md](../test/README.md)): `executor/kernels` (per-family numeric checks against references), `feeder/dataset` and `feeder/batch_pipeline` (the staged sequence is exactly the serial one), `validator/validator` (per-opcode bounds proofs, plus the regression that every compiled instruction validates), `custodian/custodian` (durable writes, checkpoint binding and corruption rejection), `engine/engine` (the boundary contracts and the unit registry), and `engine/update_engine` (the VM lifecycle end to end, gradient checks, checkpoint resume), plus the cross-half `system/update_system_test` and the emitter suite that verifies the vendored package layout.
