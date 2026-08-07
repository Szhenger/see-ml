# The SeeML Update Compiler

## What is a compiler, anyway?

Recall that a compiler is a program that translates a *source language* into a *target language* — classically, C into machine code. But nothing about that definition requires the source to be C. A compiler is really just a promise: give me something declarative, and I will hand you back something executable, having made every decision that could possibly be made early, so that nothing is left to chance later.

SeeML's compiler makes exactly that promise about *training a neural network*. Its source language is a frozen model (an `.smf` file) plus a configuration — "adapt this model with rank-8 LoRA, cross-entropy loss, AdamW, 1,000 steps." Its target language is a `.seeu` **update plan**: three flat streams of 64-byte instructions (one to train, one to evaluate, one to merge), plus every constant and every byte of memory layout the job will ever need. The device that eventually runs the plan doesn't plan anything. It just executes.

Why go to all this trouble? Because everything a compiler decides ahead of time is something that *cannot go wrong on the device*. A shape mismatch, an out-of-memory surprise, a subtle difference between the graph you trained and the graph you evaluated — all of these become compile-time errors on your build machine instead of runtime failures in the field.

Here, then, is the plan of attack, which the rest of this document walks through stage by stage:

```
SMF ingest ──▶ feasibility gate ──▶ forward SIR (+ frozen teacher)
           ──▶ loss grafting
           ──▶ pass phase A: conv-lowering, lora-graft     (PassManager)
           ──▶ primal snapshot (becomes the eval program)
           ──▶ pass phase B: autodiff, optimizer synthesis (PassManager)
           ──▶ merge program (Δ = (α/r)·A@B)
           ──▶ int8 quantization review
           ──▶ arena binding: PERSISTENT | IO | TRANSIENT (+ rodata packing)
           ──▶ instruction lowering (train / eval / merge)
           ──▶ .seeu plan assembly ──▶ (optional) native package emission
```

## How the code is organized

`compiler/` is partitioned into five subsystems, each named for its role, and each subsystem into folders named for the discipline of the work inside. Where a folder splits its work across several files, a single *façade header* is the only include consumers need — the units behind it can be reorganized without churning the rest of the tree.

```
source/                     the source language (shared with the runtime; not a compiler stage)
compiler/
  driver/                   orchestrates the process, verifies every boundary
  frontend/                 SMF bytes -> forward SIR
    ingressor/  parser/  operator/  representation/
  analysis/                 forward SIR -> complete training program
    updater/  algebra/  calculus/  reviewer/
  backend/                  training program -> .seeu plan + native package
    trainer/  architecture/  tuner/
  diagnostics/              error handling, partitioned by process
```

One note before we dive in: `source/` is deliberately *not* under `compiler/`. It holds the abstractions of the source language itself and the substrate both halves of the product share — the SMF container structs (`source/language/`), the plan ABI (`source/plan/`), deterministic data-parallelism (`source/parallel/`), and content hashing (`source/identity/`). The compiler consumes these; it doesn't own them. The byte-level formats are specified in [formats.md](formats.md), and the parallelism and hashing machinery is taught in [runtime.md](runtime.md), where it matters most.

## The frontend: from untrusted bytes to a verified graph

### Ingestion, or: never trust a file

The first thing the compiler touches is a file, and files can lie. A header can claim four billion tensors; an offset can point past the end of the file; a length can be chosen so that `offset + length` overflows and wraps around to something small. So `model_reader.cc` (`compiler/frontend/ingressor/`) reads SMF the way you'd want any parser of untrusted input written: through a bounded `Reader` that checks every single read against the file size, rejects duplicate tensor names, range-checks every op kind, and — the subtle one — does its arithmetic with explicit overflow checks, so `data_offset > size || byte_size > size - data_offset` is asked in a form that *cannot* wrap. Only after everything validates does the loader copy weight payloads, and if there are at least 4 MiB of them across multiple tensors, it fans the copies out over `ParallelFor`.

While it's at it, the loader fingerprints the entire file with `ContentHash64` (a parallel, striped variant of FNV-1a — see [formats.md](formats.md)). That hash becomes the model's *identity*: the compiled plan carries it, and the device will later refuse to patch any file that doesn't match. Same idea as checking a checksum before installing a software update.

There's one more gate before any real work happens. `resource_analyzer.cc` estimates the training footprint — roughly,

```
weights     = Σ (bytes of every constant tensor)
activations = Σ over ops (batch × output_width × 4 bytes)   (+ 8·batch per LayerNorm)
```

(where a **batch** — here and throughout — is the small group of training samples processed together in one step, its size fixed at compile time, and **activations** are the intermediate outputs the network produces along the way — kept around because the backward pass will need them) — and refuses, up front, any model that provably cannot train within the memory budget (by default, the machine's physical RAM). Better to fail in one second on the build host than after an hour on the device. The estimate is a deliberate lower bound: if even the lower bound doesn't fit, nothing will.

### SIR: the intermediate representation

Here's a question worth pausing on: why do compilers bother with an *intermediate* representation at all? Why not translate the input directly to output instructions?

Because an IR is the one data structure every stage can agree on. The parser produces it, autodiff rewrites it, the memory planner walks it, the lowerer consumes it — and each of those stages can be written, tested, and verified against the IR's invariants alone, in blissful ignorance of the others. SeeML's IR is called **SIR**, and it lives in `compiler/frontend/representation/` (façade: `sir.h`, with `type` / `value` / `operation` / `block` behind it).

SIR is in **SSA form** — *static single assignment* — which sounds fancier than it is: every value is defined exactly once, and used any number of times afterward. Think of it as a spreadsheet where each cell is computed once from earlier cells and never overwritten. This one rule buys us a lot:

- A `Value` knows the single `Operation` that defines it, and keeps a **use-list** of every operation that reads it. Rewiring the graph (as LoRA grafting will, shortly) is just `replaceAllUsesWith`.
- "Defined before used" becomes a checkable invariant rather than a hope.
- A value's *lifetime* — from its definition to its last use — is a well-defined interval, which is exactly what the memory planner will need.

Ops are named with dialect prefixes that tell you at a glance which layer of the story you're in: `sc_mem.*` are storage declarations (frozen weights, trainable parameters), `sc_high.*` are the differentiable forward ops (the ones autodiff knows how to differentiate), and `sc_low.*` are the synthesized adjoints, optimizer steps, and merge kernels (the ones with runtime kernels). `sc_ctrl.*` is reserved for control flow.

The invariant gate the entire compiler leans on is `Block::verify()`. It checks, exhaustively: no duplicate value ids; every op's parent-block pointer is consistent; every operand is defined *before* the op that uses it (SSA dominance, in a single block: definition order); and — the paranoid one — every value's stored use-list agrees, as a multiset, with the use-list you'd derive by scanning all operand lists. That last check catches the classic silent corruption where a rewrite updates one side of the def-use bookkeeping but not the other.

Threading model, in one sentence: a block is built by one thread (use-lists are written during construction), then freely read by many.

### Parsing: semantic analysis and shape inference

`BuildForward` (`compiler/frontend/parser/`) turns the decoded SMF op list into SIR, but first `sema.cc` asks the whole-graph questions:

- **Is the op list topologically ordered?** SMF requires ops to appear in dependency order — like a course catalog where every prerequisite is listed before the course that needs it. The check is elegantly cheap: walk the ops in order, maintaining a set of names *bound so far*; if an op consumes a name that isn't bound yet but *is* produced by some later op, that's a use-before-produce error. O(tensors + ops), two set lookups per edge.
- **Does any op redefine an existing name?** (Outputs must be unique — this is SSA's "assigned once" rule, enforced at the source level.)
- **Is the declared model output actually produced by some op?**

Then come the per-op *shape semantics* — the type checking of this little language. `MatMul(x, W)` demands rank-2 operands with agreeing inner dimensions (`x:[N,K] @ W:[K,M] → [N,M]`); `AddBias` demands a rank-1 bias whose width matches the input's last dimension; `Mul` demands identical shapes; `LayerNorm(x, γ, β)` demands rank-2 input with γ and β matching the last dimension — and produces, alongside its output, two extra rank-1 values (`.mean` and `.rstd`, one per row), because the backward pass will want the row statistics the forward pass already computed. Saving them now is a classic space-time trade: a few bytes per row in exchange for not recomputing means and variances later.

Frozen weights are materialized lazily by `value_resolver.cc`: the first time an op references tensor `w3`, the resolver checks it's a constant tensor and emits an `sc_mem.weight` declaration carrying the tensor's absolute file offset as an attribute — a breadcrumb that survives all the way to the emit table, where it tells the device *which bytes of the original file* to patch. Every materialized weight is recorded in `GraphBuild::weight_sources` for rodata packing later (*rodata* — read-only data — is the immutable section of the eventual plan where frozen weights will live).

If a teacher model is in play — **distillation** trains a model to imitate a larger *teacher* model's output probabilities instead of (or alongside) hard labels — the same `BuildForward` runs again over the teacher's graph with every value id prefixed `t::`, sharing the student's input. That prefix is doing real work: it's how later stages (LoRA, autodiff) recognize and refuse to touch the teacher.

`operator/` rounds out the frontend with `OpBuilder`, typed constructors for compound ops (convolution / linear / normalization / activation families behind the `op_builder.h` façade) that wire operands and infer result shapes so no caller ever assembles an `Operation` by hand.

## The analysis phase: deriving the training program

This is the heart of the compiler, and the most mathematical part of SeeML. Everything in `compiler/analysis/` is re-exported by the `update_passes.h` façade. We'll take the passes in the order the driver runs them.

### The pass manager: trust, but re-verify

A **pass** is a named graph-to-graph transformation. `PassManager` (`analysis/updater/`) runs them in registration order, and after *every single pass* re-runs `Block::verify()`. Why so paranoid? Attribution. If a pass corrupts the graph, the error message names *that pass*, at the moment of the crime — not some innocent later stage that happened to trip over the wreckage. A pass's own error, meanwhile, propagates verbatim. (Cheap insurance: verification is linear in the graph, and graphs here are small.)

### Conv lowering: convolution is just matrix multiplication wearing a trench coat

`ConvLowering` rewrites any `sc_high.conv2d` into the **im2col** form. The insight, which is worth internalizing once in your life: a convolution slides a small filter over an image, computing a dot product at each position. If you *unroll* every filter-sized patch of the input into a row of a big matrix (that's "im2col" — image to columns), and flatten the filters into a second matrix, then the entire convolution collapses into one matrix multiplication:

```
cols = im2col(x)              [N·OH·OW, Cin·KH·KW]
wmat = filter_matrix(filter)  [Cin·KH·KW, Cout]
y    = col2im(cols @ wmat)    back to [N, Cout, OH, OW]
```

You pay memory (patches get duplicated) to buy the one operation the rest of the system already knows how to execute, differentiate, and tile: **GEMM** — *general matrix multiply*, the standard name for the workhorse operation `C = A@B`. Grouped and dilated convolutions don't fit this simple form, so the pass *rejects* them with a clear diagnostic rather than silently mis-lowering — a design rule you'll see everywhere in SeeML.

### LoRA grafting: the linear algebra of small updates

Now for the idea that makes on-device training feasible at all. Suppose a frozen weight matrix `W` is `[K, M]` — say, 1024×1024, so about a million parameters. Fine-tuning `W` directly means storing a million gradients, plus (for AdamW) two million optimizer-state floats. On a phone, that's the whole budget gone.

**LoRA** (Low-Rank Adaptation) observes that the *change* you need is usually far simpler than the weight itself — it has low *rank*. If linear algebra is a distant memory: rank measures how much genuinely independent information a matrix carries. A spreadsheet with a thousand columns, every one of which is some blend of the same eight underlying columns, has rank 8 no matter how wide it looks. And here's the key fact, checkable with dimensions alone: a product of a `[K, r]` matrix and an `[r, M]` matrix can never have rank greater than `r` — so, flipped around, any rank-r change can be *stored* as two skinny factors. Instead of learning a full ΔW, then, learn:

```
A: [K, r]      B: [r, M]      with r ≪ K, M   (default r = 8)
ΔW = (α/r) · A @ B
```

Count the parameters: `r·(K + M)` instead of `K·M`. For our 1024×1024 example with r = 8, that's 16,384 trainable parameters instead of 1,048,576 — a **64× reduction** — and the frozen `W` never needs a gradient at all. The scale `α/r` (defaults α = 16, r = 8, so 2.0) decouples the learning-rate tuning from the choice of rank.

`LoraGrafter` (`analysis/algebra/`) applies this to every eligible site: an `sc_high.matmul` whose weight-side operand is a frozen `sc_mem.weight`, excluding the teacher (`t::` prefix) and, optionally, filtered by `--targets` substring match. At each site, where the graph computed `C = X @ W`, it splices in:

```
t  = X @ A          [N, r]     ← project down to rank r
u  = t @ B          [N, M]     ← project back up
s  = (α/r) · u
C' = C + s          ← and every former consumer of C now reads C'
```

Note the order of operations: `(X@A)@B` costs `O(N·r·(K+M))`, whereas materializing `A@B` first would cost `O(K·r·M)` and produce a full-size matrix — the whole point is to never form ΔW during training.

Initialization is where the elegance shows. `A` is drawn from a Gaussian with standard deviation `1/√K` (mean 0, seeded deterministically per adapter as `seed + adapter_index`); `B` is **zeros**. Since `ΔW = (α/r)·A@B = 0` when `B = 0`, the grafted model at step 0 is *bit-identical* to the original. Training can only move it away from a known-good starting point. (The `1/√K` scale is the standard variance-preserving choice: a dot product of K terms, each with variance 1/K, has variance about 1.)

One subtlety: a weight *tied* across several matmuls gets a separate adapter per site (with unique ids `w`, `w@1`, …), because each site may need a different correction.

### Automatic differentiation: the chain rule, run in reverse, by a compiler

Here is the pass that earns the word "calculus" in its folder name. Training needs gradients — the direction in which each trainable parameter should move to reduce the loss. Frameworks like PyTorch compute these dynamically, by taping operations at runtime. SeeML cannot afford a tape (the device executes a *fixed* instruction stream), so `TrainableAutodiff` (`analysis/calculus/autodiff.cc`) does something more interesting: it **synthesizes the backward pass as more SIR**, at compile time. Differentiation becomes a graph rewrite.

First, the math. Recall the chain rule: if `L = f(g(h(x)))`, then `dL/dx = f′·g′·h′`. For a network, `L` is a scalar loss and there are many inputs, so the object of interest is the gradient `∂L/∂v` for every value `v` — called the **adjoint** of `v`. Reverse-mode autodiff computes adjoints by one backward sweep: start with `∂L/∂L = 1`, and walk the graph *backward*, applying to each op a **VJP** — vector-Jacobian product — rule that converts the adjoint of its output into adjoint contributions for its inputs. (The *Jacobian* is the matrix of every partial derivative of an op's outputs with respect to its inputs; a VJP multiplies the upstream adjoint through it without ever building it explicitly.) It turns out that one forward pass plus one backward pass gives you *every* parameter's gradient, no matter how many parameters there are — this is why reverse mode (and not forward mode) powers all of deep learning.

The implementation reads exactly like that description:

1. **Mark what needs gradients.** Seed a set with the trainables (the LoRA A's and B's), then sweep forward once: any op with a marked operand marks all its results. Everything *outside* this set — the entire frozen base model and the whole teacher branch — will generate **zero backward compute**. This pruning is the compile-time analog of `requires_grad`, and it's why adapting a model is so much cheaper than training one: the backward graph is proportional to the *adapted* part, not the whole network. (If the loss isn't in the set, the configuration is nonsense, and the pass says so.)
2. **Seed.** Emit `fill(1.0)` as the adjoint of the loss — `dL/dL = 1`.
3. **Sweep backward.** Snapshot the ops, iterate in reverse (SSA order is topological, so reverse order visits every consumer before its producer), skip storage decls, dead ops, and frozen ops, and apply each op's VJP rule from a registry. An op with a live adjoint but no rule is a hard error — "no VJP rule for …" — never a silent zero.
4. **Accumulate on fan-out.** If a value feeds two consumers, it receives two adjoint contributions, and the multivariate chain rule says: *add them*. The first contribution is stored directly; each later one splices in an `sc_high.add`.

The VJP registry is a table of small theorems. A few bits of vocabulary for the loss rows before you read it: **logits** are a classifier's raw output scores; **softmax** turns logits into probabilities (exponentiate, then normalize — [runtime.md](runtime.md) teaches it properly); `onehot(y)` is the vector with a 1 at the correct class and 0 everywhere else; **mse** is mean squared error; and `σ` is the sigmoid function `1/(1+e⁻ˣ)`. Now — the rules worth knowing by heart, for `C = X @ W` with upstream adjoint `dC`:

| Forward | Backward (the VJP) | Emitted as |
|---|---|---|
| `C = X @ W` | `dX = dC @ Wᵀ`, `dW = Xᵀ @ dC` | `sc_low.matmul_nt`, `sc_low.matmul_tn` |
| `C = A + B` | `dA = dC`, `dB = dC` | (adjoint reused) |
| `Y = X + b` (bias) | `dX = dY`, `db = Σ_rows dY` | `sc_low.reduce_rows` |
| `Y = relu(X)` | `dX = dY ⊙ 1[X > 0]` | `sc_low.relu_grad` |
| `Y = gelu(X)` | `dX = dY ⊙ gelu′(X)` (same tanh approximation as forward) | `sc_low.gelu_grad` |
| `Y = silu(X)` | `dX = dY ⊙ (σ(X)·(1 + X·(1 − σ(X))))` | `sc_low.silu_grad` |
| `C = X ⊙ Y` | `dX = dC ⊙ Y`, `dY = dC ⊙ X` | `sc_low.mul` |
| `Y = LN(X; γ, β)` | `dX` from cached mean/rstd (γ, β frozen by design) | `sc_low.layer_norm_grad` |
| `loss = xent(logits, y)` | `dlogits = seed·(softmax(logits) − onehot(y))/N` | `sc_low.softmax_xent_grad` |
| `loss = mse(p, t)` | `dp = seed·2(p − t)/count` | `sc_low.mse_grad` |
| `loss = KL(student ∥ teacher)` | `dlogits = seed·(p_s − p_t)/(N·T)` | `sc_low.kl_grad` |

Two of these deserve a remark. The softmax-cross-entropy gradient `probs − onehot` is one of the loveliest results in the field — the messy derivative of a log of a softmax collapses into a subtraction — and it's why the forward op keeps its probabilities around. And the matmul rules explain two-thirds of the GEMM variants the runtime carries: training needs `NT` (`dC @ Wᵀ`) and `TN` (`Xᵀ @ dC`) as first-class citizens, not just plain `NN`.

Each backward rule is the exact derivative of *its own forward's expression* — the GELU backward differentiates the tanh approximation the forward actually uses, not the "true" erf GELU. That discipline is what makes finite-difference gradient checking in the test suite meaningful.

Finally, the pass verifies every trainable actually *received* a gradient. A LoRA adapter the loss can't see is a configuration bug, and it's caught here.

### Optimizer synthesis: the update step is just more instructions

A training step isn't finished when gradients exist; the parameters must move. `OptimizerSynthesizer` (`analysis/calculus/optimizer.cc`) appends that movement as ordinary SIR ops, so that *one execution of the program is one complete training step* — forward, backward, clip, update, no interpreter in sight.

For each (parameter, gradient) pair — sorted by id, so emission order is deterministic — it appends:

- optionally, `sc_low.clip_norm` (if `--clip-norm` > 0), *before* the step: per-tensor gradient clipping, `g ← g · min(1, max_norm/‖g‖₂)`, where `‖g‖₂` is the gradient's *L2 norm* — its Euclidean length, the square root of the sum of squares. One pathological batch must be prevented from blowing up the parameters — or worse, poisoning AdamW's moment estimates, which have a long memory.
- the step itself. **SGD** (*stochastic gradient descent* — the plainest possible nudge) is one op: `p ← p − lr·(g + λ·p)`, with λ the weight decay. **AdamW** first declares two *persistent state tensors* per parameter — `p.adam_m` and `p.adam_v`, zero-initialized, checkpointed alongside the parameters — then emits `sc_low.adamw_step(p, g, m, v)`, which the runtime executes as:

```
m ← β₁·m + (1−β₁)·g                 (first moment: a running mean of gradients)
v ← β₂·v + (1−β₂)·g²                (second moment: a running mean of squares)
m̂ = m / (1 − β₁ᵗ)                   (bias correction: early on, m and v are
v̂ = v / (1 − β₂ᵗ)                    biased toward their zero init; divide it out)
p ← p − lr·( m̂/(√v̂ + ε) + λ·p )     (adaptive step + decoupled weight decay)
```

with defaults `lr = 10⁻³, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, λ = 0.01`, and `t` the 1-indexed step (`lr`, the **learning rate**, is the base step size of every update — the single most consequential knob in training). The `√v̂` denominator gives each parameter its own effective step size (parameters with consistently large gradients take smaller steps), and the *decoupled* weight decay — added to the update directly, not folded into the gradient — is precisely what distinguishes AdamW from the original Adam.

Note what the synthesizer does *not* bake in: the numeric **hyperparameters** (the tuning knobs *you* choose, as distinct from the parameters training learns). Learning rate, betas, epsilon, weight decay, and the LR schedule travel in the plan *header*, read by the runtime at dispatch. Only decisions that change the program's *structure* — which optimizer, whether clip ops exist — live in the instruction stream. One source of truth per fact.

### The merge program: materializing Δ

Training moves A and B; it never touches W. So how does the update reach the model file? A third, tiny program, built by `MergeBuilder` (`analysis/algebra/`): for each adapter, `fill(Δ, 0)` then one fused `sc_low.gemm_acc` computing `Δ += (α/r)·A@B` — this is the *one* place `A@B` is actually materialized at full `[K, M]` size. The device runs this program once, after training and after the improvement gate, and the commit step adds each Δ onto the pristine f32 weights of the source file. (Keep that phrase "pristine weights" in mind; it's about to matter.)

### Quantization review: shrinking the frozen base to int8

The frozen base weights dominate the plan's size, and during training they are only ever *read* — by matmuls. Can we store them smaller? `SelectQuantizedWeights` (`analysis/reviewer/quantization.cc`) says yes, carefully.

The scheme is **per-tensor symmetric int8**. For a weight tensor `w`:

```
scale = max|wᵢ| / 127         (or 1.0 for an all-zero tensor)
qᵢ    = clamp(round(wᵢ / scale), −127, +127)     stored as int8
```

Each float becomes one byte — a 4× shrink — and dequantization is a single multiply by `scale`, which the GEMM kernels fuse into their existing scaling multiply (so it's *free*). Why symmetric, and why 127 rather than 128? Because mapping `[−max, +max]` onto `[−127, +127]` keeps zero exactly representable and the two directions perfectly balanced; the asymmetric −128 slot buys one extra value at the cost of that symmetry, and isn't worth it here.

Eligibility is strict: *every* user of the tensor must be a matmul, with the tensor on the *weight* side. One use as an activation, or by any other op, disqualifies the whole tensor — those consumers would need dequantized floats, and there's no place to put them. LoRA adapters, gradients, and activations always stay f32; quantizing what you're *training* would be a very different (and lossier) design.

The max-abs sweep itself is a parallel chunked reduction, grain 65,536 elements per chunk, per-chunk maxima combined in fixed chunk order — the same determinism discipline as everything else (see [runtime.md](runtime.md)), and the same chunk geometry the backend's packer will use when it actually converts the bytes.

One more thing, and it's the punchline of the whole quantization story: the emit table maps each **delta** onto the *original file's f32 bytes*. Commit computes `W′ = W + Δ` from the pristine weights — so quantization error affects training dynamics (slightly), but is **never baked into the committed model**. The artifact you ship is the exact model you started with, plus exactly what was learned.

## The backend: from program to plan

### Arena binding: memory planning as a compile-time problem

Ask yourself: what does `malloc` cost you on a device? Not just cycles — *unpredictability*. Fragmentation, allocation failure at step 900 of 1,000, nondeterministic addresses. SeeML's answer is to compile memory away: `arena_binder.cc` (`backend/trainer/`) assigns every tensor a fixed byte offset in a single **arena**, sized at compile time, allocated exactly once on the device.

The arena has three segments, in order:

```
[ PERSISTENT | IO | TRANSIENT ]
```

- **PERSISTENT** holds what must survive across steps and be checkpointable: the LoRA A's and B's and the AdamW moments. It sits at offset 0, so a checkpoint is literally "write the first `persistent_size` bytes of the arena."
- **IO** holds the batch input and label slots the feeder writes into.
- **TRANSIENT** holds everything else — activations, adjoints, temporaries — and this is where the interesting algorithm lives.

(The frozen weights — rodata — are *not* in the arena at all. They're packed, possibly as int8, into a read-only section of the plan blob itself, and addressed through a separate address space. Mutable and immutable bytes never share a segment; the validator will later exploit that separation.)

Here's the transient problem, and it should ring a bell if you've seen register allocation: many values, one pool of memory, and values whose lifetimes don't overlap can share the same bytes. Since SSA gives every value a well-defined **liveness interval** — from the op that defines it to the last op that reads it — the binder can:

1. Walk the block once, numbering ops with a tick counter; record each transient value's interval `[birth, last_use]` (values that must outlive the program — the loss slot, parameter gradients, merge deltas — get `death = ∞`, i.e., pinned).
2. Sort intervals by start (a *stable* sort, so same-tick ties keep discovery order — a small detail that makes the resulting offsets, and hence the plan hash, reproducible across standard libraries).
3. Linear-scan with **first-fit**: for each interval, expire the active allocations whose lifetimes ended, then scan the active list in address order for the first gap large enough; extend past the last allocation if none fits.

Think of it as a hotel front desk: guests (values) with known check-in and check-out ticks, rooms (byte ranges) reused the moment a guest leaves. The high-water mark of this process *is* the transient segment's size — memory consumption is a compile-time constant, printed in the report, knowable before the device ever runs.

Everything is 64-byte aligned (`AlignUp(v) = (v + 63) & ~63`) — one cache line — so no tensor ever straddles a line boundary it didn't need to.

The merge program is bound into the *same* arena, with one twist: its deltas are placed *above* the training program's high-water mark. Why? The engine's public sequence permits Train → Merge → **Evaluate** → Commit — an evaluation may run after the deltas are materialized, and the evaluation program reuses transient memory freely. Placing deltas above the training high-water mark proves, by construction, that no evaluation can ever scribble over them.

### Instruction lowering: designing an ISA for training

`instruction_lowering.cc` translates each SIR op into exactly one `UpdateInstruction` — a fixed **64-byte** record (one cache line, not a coincidence):

```
u16 opcode | u16 flags | u32 pad | u64 in[4] | u64 out[3]
```

The instruction set has 31 opcodes — six GEMM variants (`NN`, `NT`, `TN`, accumulating `NN`, and two int8-dequantizing forms), elementwise ops, the activation forward/backward pairs, LayerNorm, the three loss families, the two optimizer steps, clip, fill, copy. The complete enumeration lives in `source/plan/instruction.h`, and [formats.md](formats.md) walks the encoding.

The addressing scheme is worth savoring for its economy. A tensor reference is a single 64-bit word: **bit 63 selects the address space** (0 = mutable arena, 1 = read-only rodata), bits 0–62 are a byte offset. That's the entire memory model — two flat spaces and an offset. No pointers, no relocation, and, on the device, one branchless test tells the validator which bounds to check. Scalars ride along bit-cast into spare operand slots (a GEMM's α, clip's max-norm, fill's value), and dimensions pack into the `out[]` words (a GEMM carries M, N, K; LayerNorm packs rows and columns into one word as `(N << 32) | D`).

### Host architecture and GEMM tiling: respecting the memory hierarchy

Recall the memory hierarchy: registers in a cycle, L1 in a few, L2 in a dozen or two, DRAM in hundreds. A naive triple-loop matmul of big matrices thrashes: by the time you need a row again, the cache has evicted it. The fix, known from the BLIS/GotoBLAS line of work, is **blocking**: choose tile sizes so that the pieces you reuse most *stay* in the levels closest to the core.

`DetectHostArch` (`backend/architecture/`) first learns the machine: the ISA (instruction set), SIMD width (*single instruction, multiple data* — how many floats one vector register can process at once), and FMA support (*fused multiply-add*, `a·b + c` in one instruction) are compile-time facts (host = target for this ahead-of-time compiler — they come from predefined macros); L1/L2 sizes, core count, and line size are runtime probes (`sysctlbyname` on Apple, `sysconf` elsewhere). Probes can fail — a container may hide `sysconf` values — and the policy is explicit: detection **never hard-fails**. It warns and falls back to conservative defaults (32 KiB L1, 512 KiB L2).

`SuggestGemmTiling` then derives block sizes `{mc, kc, nc}` as a *pure function* of that description — same inputs, same tiling, no measurement involved:

```
nc = min(4·simd, fit(L1/2 ÷ simd))     a register-blocked sweep of C columns
kc = min(fit(L1/2 ÷ nc), fit(L2/2 ÷ simd))   the kc×nc panel of B must live in half of L1
mc = fit(L2/2 ÷ kc)                    the mc×kc panel of A must live in half of L2
```

where `fit` rounds down to a SIMD-width multiple (never below it). The intent: while the kernel marches across `mc` rows of A, the B panel stays L1-resident and the A panel L2-resident, and only *half* of each cache is budgeted, leaving room for C and everything else. `ValidateGemmTiling` enforces this contract on any tiling handed in from outside — nonzero, SIMD-multiple, and the two half-cache inequalities (`kc·nc·4 ≤ L1/2`, `mc·kc·4 ≤ L2/2`) whenever the cache sizes are trustworthy.

(A precision worth stating: the *CPU* kernels in the runtime use fixed, conservative tiles baked into the code. The detected tiling feeds the GPU kernel emitter and the autotuner, below.)

### The autotuner: a multi-armed bandit, not a random search

Analytical models are good; measurements are better. But measurements cost time, so the question becomes: given a handful of candidate tilings and a limited budget of benchmark runs, how do you spend the budget wisely? This is the **multi-armed bandit** problem — named for slot machines: each "arm" pays out noisily, and you must balance *exploring* (trying arms you know little about) against *exploiting* (replaying the best arm so far).

`Ucb1Bandit` (`backend/tuner/`) implements the classic UCB1 answer. After pulling each arm once, always pull the arm maximizing:

```
score_i = mean_i + c·√(2·ln N / n_i)
```

where `mean_i` is arm i's average reward, `n_i` its pull count, `N` the total pulls, and `c` the exploration constant (default 1.0). Look at the two terms: the first says "this arm has paid well," the second — an upper confidence bound that *grows* for neglected arms and *shrinks* as ln N/n_i falls — says "but you haven't looked at this one lately, and your uncertainty about it is still large." Optimism in the face of uncertainty, with a provably small amount of regret.

Two implementation choices make it SeeML-flavored. There is **no randomness anywhere** — untried arms are taken lowest-index-first, and score ties break to the lowest index — so a tuning run is exactly reproducible. And the benchmark is **injected** as a function parameter (`double(const GemmTiling&)`, higher is better): tests hand in a synthetic reward function, so the tuner's logic is provable without a GPU or a wall clock.

`TilingCandidates` spans the search space frugally: the analytical hint, plus each dimension independently halved and doubled — at most 7 candidates. `AutotuneGemmTiling` clamps the budget up so every arm is measured at least once, runs the select/measure/update loop, and reports the winner *with full per-arm statistics* — you can audit exactly what it tried and what it saw.

### Emission: the plan, and a package that builds anywhere

Two emitters close out the backend. The plan itself is assembled by the driver (below) — header, three instruction streams, rodata, the persistent segment's initial image (zeros, except each `randn` parameter filled from its own seeded `mt19937_64` — deterministic bits, decided at compile time), and the emit table, each section 64-byte aligned, the whole sealed with a hash.

`native_emitter.cc` then writes a **self-contained package**: the plan as bytes *and* as an embedded C array, a generated driver `main` (flag parsing, the load→train→gate→merge→commit sequence, exit codes), the entire runtime vendored file-by-file, and a `build.sh` that compiles it all with nothing but a C++23 compiler. The package builds with no access to this repository — that's the deliverable you ship to a device vendor.

`kernel_emitter.cc` generates Metal GEMM kernel *source* (forward, both backward transposes, and the merge's scaled accumulate) from a `GpuTiling` clamped out of the host tiling into `{8, 16, 24, 32}` per dimension. Each kernel stages A- and B-tiles cooperatively into threadgroup memory, barriers, and accumulates with `fma` — the GPU expression of exactly the same blocking idea as above. It is text generation only, compiled by whoever integrates it.

## Diagnostics: errors as a designed surface

Every compiler diagnostic is one line, `"<unit>: <message>"`, and errors travel as `std::expected<T, std::string>` — no exceptions across subsystem boundaries. The core (`diagnostics/diagnostic.h`) provides three verbs over the thread-safe `Logger`: `Fail` (build an error), `Note` (progress), `Fallback` (degraded but continuing). Each *process* gets a header-only module owning its unit names and message shapes:

| module | process delimited | units |
|---|---|---|
| `tokenizing/` | SMF byte-stream decode/encode | `SMF`, `Ingressor` |
| `parsing/` | SMF graph → forward SIR | `Parser` |
| `passing/` | pass orchestration + lowering legality | `PassManager`, `ConvLowering` |
| `updating/` | the analytic methods | `TrainableAutodiff`, `LoraGrafter`, `MergeBuilder`, `OptimizerSynthesizer`, `GemmEpilogueFuser` |
| `architecting/` | local device analysis | `HostArch`, `Autotuner` |
| `generating/` | code generation + the driver | `UpdateCompiler`, `ArenaBinder`, `InstructionLowering`, `NativeEmitter` |

`architecting/` has a two-tier discipline worth noting: *detection* can never hard-fail (warn, fall back to conservative defaults — a missing sysctl shouldn't stop a compile), while *tiling-contract violations* are hard errors (a broken contract means broken code).

## The driver: orchestration under contract

`UpdateCompiler::Compile` (`compiler/driver/update_compiler.h`) owns the compilation *process* — the sequence at the top of this document — and nothing else. But sequencing is the boring half of its job. The interesting half is **design by contract**: at every subsystem boundary, the driver *verifies* that the subsystem was used correctly (`contract.h`):

- `VerifyFrontendContract` — after the forward builds: the graph build carries input/output, the SIR verifies, every weight source is a constant SMF tensor declared `sc_mem.weight`.
- `VerifyAnalysisContract` — after the merge program: adapters exist; every A/B is an `sc_mem.param` with a gradient reaching it (exactly two gradients per adapter); the merge program verifies and covers every adapter.
- `VerifyGeneratedPlan` — before returning: plan non-empty; all three programs lowered (eval ≤ train, since the eval program is the *primal* snapshot — the adapted forward pass plus loss, captured before autodiff appended any backward ops; at least one merge instruction per adapter); the arena contains its persistent segment; frozen weights reached rodata; debug hooks cover the trainable set.
- `WellFormedDiagnostic` — at the outer error boundary: any error escaping `Compile` must name a unit registered in `diagnostics/`. Even the *error messages* have a checked contract.

Why bother, when the subsystems have their own tests? Because contracts catch *integration* bugs — the driver wiring stage A's output into stage B wrongly — and they catch them with an honest confession: a contract violation reports under the driver's own unit, as a driver bug, never blamed on the user. And the runtime's engine mirrors this exact structure on the device ([runtime.md](runtime.md)), re-proving at load time what the driver proved at emit time. Trust, but verify — twice.

## To recap

- The **frontend** turns untrusted bytes into a verified SSA graph, with every semantic and shape error caught at the door.
- The **analysis** phase is compile-time mathematics: LoRA grafts rank-r adapters (`r(K+M)` parameters instead of `K·M`), autodiff synthesizes the backward pass by reverse-mode VJP rewriting pruned to the trainable set, the optimizer becomes instructions, the merge program materializes `Δ = (α/r)·A@B`, and the frozen base is reviewed for symmetric int8 storage.
- The **backend** binds every value to a byte offset (liveness + first-fit, like register allocation for tensors), lowers to a 31-opcode, 64-byte-instruction ISA with a two-address-space memory model, derives cache-respecting GEMM tilings analytically, refines them with a deterministic UCB1 bandit, and emits a package that builds anywhere.
- The **driver** sequences it all and verifies a contract at every seam.

## tool/ — the command-line surface

Not under `compiler/` for the same reason as `source/`: the tools are
consumers of the compiler and the formats, not stages of compilation.

- **`seeml_update_compile.cc`** — the compiler CLI; every flag is
  documented in [usage.md](usage.md). Argument parsing is strict: an
  unknown flag, a flag missing its value, or a numeric with trailing
  garbage is a hard error, never a silent default.
- **`seeml_seeu_dump.cc`** — the plan disassembler: header fields, arena
  segments, and (with `--instrs`) the decoded instruction streams.
- **`export_model.py`** — the PyTorch exporter producing SMF models and
  SDS corpora; the accepted module set is listed in [usage.md](usage.md).

## Testing

One SeeTest suite per module, organized to mirror this partition (`test/compiler/<subsystem>/*_test.cc` — see [test/README.md](../test/README.md)), run via `ctest` or directly from `build/` (see [usage.md](usage.md)). The compiler-side suites: `frontend/` (`model_io`, `resource_analyzer`, `sir`, `operator`, `parser`), `analysis/` (`update_passes`, `updater`, `reviewer`), `backend/` (`tuner`, `trainer`, `native_emitter`), `driver/` (`update_compiler`, `driver`), and `diagnostics/`, plus `source/` suites (`hash`, `parallel_for`) and the end-to-end `system/update_system_test`.
