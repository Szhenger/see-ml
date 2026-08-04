# Roadmap — the three remaining architecture-review projects

The 2026-07 architecture review closed eight findings (alias validation,
exact memory gating, tiling wiring, CI, gate accuracy, version negotiation,
streaming/locked commit, the DCE optimization phase). Three findings remain,
each a genuine project rather than a fix. This outline scopes them: what to
build, in what order, against which seams in the current tree, and how each
is verified. Effort marks are relative (S/M/L per phase).

Cross-cutting ground rules, all now in place and to be leaned on:

- **ABI evolution** goes through the version-negotiation policy
  (`source/plan/schema.h`): additive zero-default fields bump `kSeeuVersion`
  only; anything an old runtime would *silently misread* must be gated by a
  version bump so the range check rejects it loudly.
- **New SIR rewrites** land in pass phase B/C under the `PassManager`
  verify gate, with `DeadCodeElimination` sweeping what they orphan.
- **New instruction semantics** must be provable by the validator
  (`runtime/validator/plan_validator.cc`), including its operand-overlap
  discipline, before `Execute()` may dispatch them blindly.
- **Bitwise determinism is a contract**: any change must either preserve
  per-element expression order exactly or be introduced as a new opcode the
  old path never emits.

---

## Project 1 — Operator fusion (efficiency: arena-bandwidth)

**Problem.** Every op materializes its full tensor to the arena; an MLP
layer costs three round-trips (GEMM → AddBias → activation) where one would
do. The runtime is CPU-bandwidth-bound (`runtime/executor/elementwise.cc`
says as much), so this is the largest efficiency lever left.

**Key constraint.** The backward pass consumes stored forward
intermediates (`ReluBwd(dy, x)` needs pre-activation `x`; `layer_norm_grad`
needs cached mean/rstd). Fusing away an intermediate that a backward op
reads is a correctness bug. Therefore fusion applies exactly where no
backward exists:

- the **eval program** (forward + loss only) — runs 2·⌈N/B⌉ times per
  update for the gate;
- the **teacher subgraph** of the train program (autodiff prunes it — no
  adjoints reach it) — runs every step under distillation.

**Phase 1a — GEMM epilogues (M).**

- ABI: use the dormant `UpdateInstruction::flags` (uint16) on the GEMM
  opcodes: `kEpilogueBias` (bias ref rides the free `in[3]` slot) and an
  activation selector (`kEpilogueRelu/Gelu/Silu`). Bump `kSeeuVersion` to 4
  — an old runtime ignores `flags` today and would silently skip the
  epilogue, exactly the "silent misread" case the version gate exists for.
  From v4 on, the validator **rejects unknown flag bits**, so all future
  flags fail loud instead.
- Compiler: a `FuseGemmEpilogue` pass in `analysis/` phase C (before DCE,
  which then sweeps the orphaned AddBias/activation ops). Legality: the
  GEMM's result and the bias sum must each have exactly one user (the next
  op in the chain) and no presence in the primal-root set *unless* the
  fused op itself replaces the rooted value identity (fuse-then-rebind).
  Run it only over blocks/regions marked no-backward: the driver knows the
  teacher prefix and the eval snapshot.
- Runtime: apply the epilogue in the blocked cores' write-back
  (`runtime/executor/gemm.cc`): per element, `c = act(acc + bias[n])` in
  register. Per-element expression order is identical to the unfused
  three-op sequence, so results stay **bitwise identical** — assert this in
  tests by compiling the same model fused and unfused and comparing eval
  losses bit-for-bit.
- Validator: epilogue flags imply `in[3]` bias bounds (N floats) and join
  the overlap discipline.

**Phase 1b — elementwise chain fusion (M, after 1a).** A general
`kFusedMap` opcode: a short micro-program (≤4 unary/binary stages encoded
in the aux words) applied per element in one pass. Grafts onto the same
no-backward legality analysis. Only pursue if profiling after 1a still
shows elementwise chains hot; 1a covers the dominant MLP pattern.

**Verification.** Bitwise-equality fused-vs-unfused (eval loss, probs
range); kernels_test golden refs for each epilogue; validator flag-bit
rejection tests; system test under distillation confirming teacher fusion
changes no loss bits; the DCE note reporting swept ops becomes >0 —
asserting the pass actually fired.

---

## Project 2 — Memory-efficiency training machinery

Three subprojects, ordered by value-per-effort. All three relieve the same
constraint the exact memory gate (`CheckPlanFitsLocally`) now enforces
honestly: the arena must fit the device.

**Phase 2a — Gradient accumulation / micro-batching (M).** Today the
optimizer step is fused into the train stream: effective batch ≡ compiled
batch, and activation memory scales with it.

- Compiler: compile at micro-batch `b`, accumulate `G` micro-steps.
  Autodiff output gains per-parameter accumulator buffers
  (`acc += grad` ops appended); the stream splits into a **grad program**
  (fwd + bwd + accumulate) and a **step program** (clip on acc, optimizer
  step on acc, `kFill` acc = 0). Additive header fields (v4): grad/step
  program offsets+counts and `grad_accum_steps`; a v3-style monolithic
  train program is still emitted for `G = 1`.
- Runtime: the step loop runs G grad executions per optimizer step; the
  LR schedule and AdamW timestep advance per *optimizer* step; checkpoints
  align to optimizer-step boundaries (the persistent segment already
  excludes activations, so nothing else changes).
- Memory math: activations scale with `b`, not `b·G`; cost is one extra
  grad-sized accumulator set — report both in the compile note and
  `report.json`.
- Verification: bitwise equivalence of `(b=32, G=1)` vs legacy path;
  statistical equivalence (loss trajectory) of `(b=8, G=4)` vs `(b=32)` is
  NOT expected bitwise (different batch partitioning) — test instead that
  accumulated gradients over G micro-batches equal the large-batch gradient
  bitwise when the samples are identical (sum order fixed), which they do
  if accumulation order is the micro-batch order.

**Phase 2b — Activation rematerialization (L).** Recompute-instead-of-store
for the backward pass, at segment granularity: the arena binder's liveness
scan identifies the high-water contributors; a compiler pass duplicates the
cheap forward segments (activations, bias adds — not GEMMs) immediately
before their backward consumers, letting `LinearScanTransients` free the
originals early. Purely a compile-time space/time trade, selectable
(`--remat auto|off`), reported by the resource analyzer. No ABI change —
the instruction stream just gets longer. Verify: gradients bitwise-equal
with remat on/off (recomputed expressions are identical per element);
arena high-water strictly smaller on the fixture MLPs.

**Phase 2c — bf16 storage precision (L, last).** SIR already declares
`BF16`; nothing uses it. Scope narrowly: bf16 **storage** for frozen rodata
and (optionally) activations, f32 **compute** with round-on-store — no loss
scaling needed at bf16 range. Touches the whole width of the stack (type
propagation, per-operand element width in lowering + validator, convert-on-
load in kernels, quantizer interplay), which is why it goes last, after 2a
/2b have paid down the same memory pressure more cheaply. Gate the whole
feature behind a config flag and a version bump; verify with the
finite-difference suite at loosened tolerances plus a drift budget against
the f32 reference.

---

## Project 3 — Convolution reachability (scope honesty)

**Problem.** `ConvLowering` and the conv/batch-norm `OpBuilder` path are
unreachable from any real input: SMF has no Conv op kind
(`source/language/model_format.h`), `export_model.py` exports only
Linear/activation/LayerNorm `nn.Sequential`s, and the whole frontend
assumes rank-2 `[batch, width]` activations. A documented pipeline stage is
test-only code.

**Decision first, code second.** Two coherent endpoints; pick by product
target:

- **(A) Make convolution real (L)** — only if on-device personalization of
  CNNs (vision) is on the product path:
  1. SMF v3: `kConv2d` (attrs: stride, pad; groups/dilation stay rejected,
     matching `ConvLowering`'s contract) and rank-4 non-constant tensors;
     writer/reader/fuzzer updated (`LoadSmf` is fuzzed — keep it that way).
  2. `export_model.py`: `nn.Conv2d` (+ the NCHW flatten/unflatten
     boundaries into the existing rank-2 head).
  3. Parser/sema: rank-4 shape rules for conv; `resource_analyzer` learns
     NCHW activation footprints (its width-propagation is rank-2 today).
  4. Driver: input handling generalizes from `dims.back()` to a shape
     descriptor; the feeder contract (`input_floats`) already carries a
     flat count, so SDS needs no format change.
  5. LoRA on conv: graft onto the im2col GEMM's filter matrix — which
     `ConvLowering` already materializes — so the adapter algebra is
     unchanged.
  6. End-to-end: a conv fixture through compile → train → gate → merge →
     commit, plus finite-difference gradients through the im2col path.
- **(B) Excise the dead path (S)** — otherwise: delete `ConvLowering`, the
  conv/batch-norm builders, their tests, and the `passing/` unit entry;
  the docs stop describing a stage that cannot run. Reversible via git if
  (A) is ever funded.

**Recommendation:** decide at the next planning point; default to (B) if no
CNN target exists within two quarters — unreachable subsystems rot, and the
review exists to prevent exactly that.

---

## Sequencing

1. **1a GEMM epilogues** — biggest win-per-effort, exercises the new
   flag/version machinery end to end.
2. **2a gradient accumulation** — unlocks larger effective batches on the
   devices the memory gate now honestly bounds.
3. **3 decision (A or B)** — cheap to decide, removes standing dishonesty
   either way.
4. **2b rematerialization**, then **1b chain fusion**, then **2c bf16** —
   each contingent on profiling evidence from the phases before it.
