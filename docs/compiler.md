# The SeeAI Update Compiler

## What is a compiler, anyway?

Recall that a compiler is a program that translates a *source language* into a *target language* — classically, C into machine code. But nothing about that definition requires the source to be C. A compiler is really just a promise: give me something declarative, and I will hand you back something executable, having made every decision that could possibly be made early, so that nothing is left to chance later.

SeeAI's compiler makes exactly that promise about *training a neural network*. Its source language is a frozen model (an `.smf` file) plus a configuration — "adapt this model with rank-8 LoRA, cross-entropy loss, AdamW, 1,000 steps." Its target language is a `.seeu` **update plan**: three flat streams of 64-byte instructions (one to train, one to evaluate, one to merge), plus every constant and every byte of memory layout the job will ever need. The device that eventually runs the plan doesn't plan anything. It just executes.

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
    ingressor/  accountant/  topology/  computation/  egressor/  tokenizer/  operator/  representation/
  analysis/                 forward SIR -> complete training program
    pass_manager  algebra/  calculus/  statistics/  topology/  optimization/
  backend/                  training program -> .seeu plan + native package
    architecture/  allocation/  selection/  packaging/
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

Because an IR is the one data structure every stage can agree on. The parser produces it, autodiff rewrites it, the memory planner walks it, the lowerer consumes it — and each of those stages can be written, tested, and verified against the IR's invariants alone, in blissful ignorance of the others. SeeAI's IR is called **SIR**, and it lives in `compiler/frontend/representation/` (façade: `sir.h`, with `type` / `value` / `operation` / `block` behind it).

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

This is the heart of the compiler, and the most mathematical part of SeeAI. Everything in `compiler/analysis/` is re-exported by the `update_passes.h` façade. We'll take the passes in the order the driver runs them.

### The pass manager: trust, but re-verify

A **pass** is a named graph-to-graph transformation. `PassManager` (`analysis/updater/`) runs them in registration order, and after *every single pass* re-runs `Block::verify()`. Why so paranoid? Attribution. If a pass corrupts the graph, the error message names *that pass*, at the moment of the crime — not some innocent later stage that happened to trip over the wreckage. A pass's own error, meanwhile, propagates verbatim. (Cheap insurance: verification is linear in the graph, and graphs here are small.)

The manager also **times every pass** (the pass plus its verify), and the driver times the phases between them — frontend, merge-and-review, arena binding, lowering, the persistent image, assembly, the seal. They travel in `CompiledUpdate::pass_timings` and in the compile report's `"passes"` array, so a compile-side performance claim is a measurement anyone can repeat (E5, #84). On an 800 MB decoder (Apple M5) the whole graph pipeline — 1,417 ops through eight passes — is under 4 ms; the compile is assembly (133 ms) and the seal's hash (32 ms), i.e. bytes, which is why E2 went after copies and not passes. Three companions from the same hygiene batch: **DCE marks, then compacts once** — a backward sweep over private use counts finds whole dead chains, and `Block::removeOps` unlinks them and compacts the op list in one pass, where one `removeOp` per dead op was O(removed × ops), a quadratic cliff the first time a pass dead-codes a real fraction of the graph; the **SIR dump is opt-in** (`UpdateConfig::dump_sir`, `--dump-sir out.txt`) instead of streamed and retained by every compile; and the dump changes no byte of the plan.

### Conv lowering: convolution is just matrix multiplication wearing a trench coat

`ConvLowering` rewrites any `sc_high.conv2d` into the **im2col** form. The insight, which is worth internalizing once in your life: a convolution slides a small filter over an image, computing a dot product at each position. If you *unroll* every filter-sized patch of the input into a row of a big matrix (that's "im2col" — image to columns), and flatten the filters into a second matrix, then the entire convolution collapses into one matrix multiplication:

```
cols = im2col(x)              [N·OH·OW, Cin·KH·KW]
wmat = filter_matrix(filter)  [Cin·KH·KW, Cout]
y    = col2im(cols @ wmat)    back to [N, Cout, OH, OW]
```

You pay memory (patches get duplicated) to buy the one operation the rest of the system already knows how to execute, differentiate, and tile: **GEMM** — *general matrix multiply*, the standard name for the workhorse operation `C = A@B`. Grouped and dilated convolutions don't fit this simple form, so the pass *rejects* them with a clear diagnostic rather than silently mis-lowering — a design rule you'll see everywhere in SeeAI.

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
ts = (α/r) · t      [N, r]     ← the scale, applied where the tensor is small
C' = C + ts @ B     [N, M]     ← project back up, summing into C as it goes
```

Every former consumer of `C` now reads `C'`.

Where the scale goes is not a detail. The textbook order — `u = t@B`, `s = (α/r)·u`, `C' = C + s` — spends two elementwise passes over an `[N, M]` activation to apply one constant and one sum, and autodiff then mirrors them: a full `(α/r)`-scaled copy of `dC` for the scale's adjoint, and a full `[N, K]` add where the adapter's `dX` joins the base path's. Four activation-sized passes per adapted layer per step, in a runtime whose elementwise kernels are limited by memory bandwidth, feeding GEMMs that are rank-r and tiny. Scaling `t` instead is `r/M` of the work, and the adjoints follow by themselves — `dB = tsᵀ @ dC`, `dt = (α/r)·(dC @ Bᵀ)` — with no activation-sized scale anywhere. The two sums then ride their GEMMs: `GemmAddendFuser` (below) folds a GEMM whose only reader is an add into that add, so `C + ts@B` and `dC@Wᵀ + dt@Aᵀ` are each written as the dot products come out of the core. `(α/r)·(t@B)` and `((α/r)·t)@B` are the same floats whenever `α/r` is a power of two — the default 16/8 is — and equal to rounding otherwise; when `α/r = 1` no scale is emitted at all.

Note the order of operations: `(X@A)@B` costs `O(N·r·(K+M))`, whereas materializing `A@B` first would cost `O(K·r·M)` and produce a full-size matrix — the whole point is to never form ΔW during training.

Initialization is where the elegance shows. `A` is drawn from a Gaussian with standard deviation `1/√K` (mean 0, seeded deterministically per adapter as `seed + adapter_index`); `B` is **zeros**. Since `ΔW = (α/r)·A@B = 0` when `B = 0`, the grafted model at step 0 is *bit-identical* to the original. Training can only move it away from a known-good starting point. (The `1/√K` scale is the standard variance-preserving choice: a dot product of K terms, each with variance 1/K, has variance about 1.)

One subtlety: a weight *tied* across several matmuls gets **one** adapter pair, shared by every site. Per-site pairs would train fine and then commit wrongly: the file holds a single `W`, so commit would write `W + ΣΔᵢ` and every site would inherit every other site's correction. With a shared pair each site computes `xᵢ@(W + Δ)` during training, autodiff sums the pair's gradient across the sites (fan-out accumulation), the merge builder rejects a duplicate emit target outright, and the committed weight is exactly the function that was trained and gated. A weight consumed any other way than as a MatMul's right operand — an embedding table doubling as the lm-head, say — is not adapted at all, with a note, for the same reason: committing `W + Δ` would silently change the site that read it differently.

### Automatic differentiation: the chain rule, run in reverse, by a compiler

Here is the pass that earns the word "calculus" in its folder name. Training needs gradients — the direction in which each trainable parameter should move to reduce the loss. Frameworks like PyTorch compute these dynamically, by taping operations at runtime. SeeAI cannot afford a tape (the device executes a *fixed* instruction stream), so `TrainableAutodiff` (`analysis/calculus/autodiff.cc`) does something more interesting: it **synthesizes the backward pass as more SIR**, at compile time. Differentiation becomes a graph rewrite.

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
| `loss = T²·KL(teacher ∥ student)` | `dlogits = seed·T·(p_s − p_t)/N` | `sc_low.kl_grad` |

Two of these deserve a remark. The softmax-cross-entropy gradient `probs − onehot` is one of the loveliest results in the field — the messy derivative of a log of a softmax collapses into a subtraction — and it's why the forward op keeps its probabilities around. And the matmul rules explain two-thirds of the GEMM variants the runtime carries: training needs `NT` (`dC @ Wᵀ`) and `TN` (`Xᵀ @ dC`) as first-class citizens, not just plain `NN`.

Each backward rule is the exact derivative of *its own forward's expression* — the GELU backward differentiates the tanh approximation the forward actually uses, not the "true" erf GELU. That discipline is what makes finite-difference gradient checking in the test suite meaningful.

Attention has two VJP rules, one per family, and which one runs is decided *before* the primal snapshot, by `AttentionTiling` (`analysis/algebra/`) on the step-0 memory gate's footprint (E11, #94). The cached rule above keeps the probability matrix `P` `[B·H·S, S]` alive from forward to backward and materializes `dP` and `dS` at the same size — at S = 2048, H = 8, 128 MiB per matrix per layer, the sequence-length ceiling on a device. The tiled rule (plan v15) keeps a stats row of four floats per query row instead — the score max, the inverse softmax denominator, and the softmax-backward rowsum `δ` — and emits three passes that each recompute the probabilities they need from Q, K and that row: `dQ = (P∘(dP − δ))K/√d` (which also writes `δ`), then `dK` and `dV` per key row. The tiled kernels evaluate the cached kernels' expressions in the cached kernels' orders, so the two families compute *identical bits* — the system suite trains both, at one thread and eight, and compares every byte — and the choice is a memory decision alone: `--attention auto` keeps the caches while all layers' caches fit the budget (`--attention-cache-budget-mib`, 256 by default) and the footprint fits local memory with them. `docs/benchmarks.md` records the cost: the tiled backward is about 3× the cached one, a whole step 1.2–1.7× slower, for 1,039 → 324 MiB at S = 2048.

One more thing the table implies: when a value feeds several ops, its gradient is the *sum* of what flows back from each, and the pass injects an `sc_high.add` for every such fan-out. At a LoRA site that sum is `dX = dC@Wᵀ + dt@Aᵀ` — an activation-sized add whose second operand is a rank-r GEMM with no other reader. `GemmAddendFuser` (`analysis/algebra/`, after autodiff so legality is read off the use-lists) folds exactly that shape — a plain f32 GEMM, one reader, nothing mutating storage in between — into the add, which lowers to the GEMM itself with `kFlagGemmAddend` (plan v14): `C = D + A@B`. The kernel applies the addend to each task's cell right after the core writes it, while the cell is still in cache, and each element is `d + s` over the *complete* dot product — the expression of the two instructions it replaces, so the fold never changes a bit (`--no-fuse-addend` compiles the unfolded program; the system suite trains both and compares every byte). When both operands qualify, the GEMM with the smaller inner dimension folds: the rank-r one, leaving the frozen-weight GEMM its own instruction and its int8 / bf16 forms.

Finally, the pass verifies every trainable actually *received* a gradient. A LoRA adapter the loss can't see is a configuration bug, and it's caught here.

### Optimizer synthesis: the update step is just more instructions

A training step isn't finished when gradients exist; the parameters must move. `OptimizerSynthesizer` (`analysis/calculus/optimizer.cc`) appends that movement as ordinary SIR ops, so that *one execution of the program is one complete training step* — forward, backward, clip, update, no interpreter in sight.

**Gradient accumulation** (`--grad-accum G`, roadmap 2a) keeps that shape and splits it in two. The plan compiles at the *micro-batch* `b`, so activation memory scales with `b`; the effective batch is `b·G`. Autodiff seeds `dL/dL = 1/G` instead of 1 (every micro-batch loss is a mean over its own rows, so `G` folded gradients sum to the mean over `b·G` rows — and a power-of-two `G` makes that scaling exact per element). The synthesizer then declares one persistent accumulator per parameter (`p.grad_acc`, zero-initialized, in the checkpointed segment like the AdamW moments) and appends `sc_low.accumulate(acc, grad)` for each — the tail of the **grad program** — before the clip, the step (now on the accumulator) and `sc_low.zero(acc)` — the **step program**. The driver splits the lowered stream at the first step instruction; the plan carries the grad program in its train section and the step program in a v9 section of its own, and the runtime runs `G` grad executions per optimizer step. With `G = 1` nothing changes: no accumulators, no step section, the very same monolithic program.

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

The scheme is **per-column symmetric int8** (since E12, #95; plan v17 — earlier plans used one scale per tensor). For a weight `W [K, M]`, each output column `m` gets its own scale:

```
scale[m] = max_k |W[k, m]| / 127     (1.0 for an all-zero column)
q[k, m]  = clamp(round(W[k, m] / scale[m]), −127, +127)     stored as int8
```

The `M` scales follow the int8 levels in rodata, and the GEMM carries a reference to them (`kFlagQ8ColScale`). Why per column: LLM projections carry a few outlier columns with |w| 10–50× the bulk, and a single max-abs scale for the whole tensor lets those set the step for every column, collapsing the bulk to a handful of levels. On a planted 50× outlier column the per-column form cuts the bulk's dequantization error 51× (RMS and mean-abs alike); the outlier column's own error cannot move — its scale is the tensor's either way — and does not. In the forward GEMM the scale belongs to the output column and joins the B panel as it widens; in the dX GEMM (`dC @ Wᵀ`) the same scale sits on the reduction axis, so each task scales its A rows once and runs the unchanged NT core. Measured at SmolLM-135M shapes the per-column form costs +0.9–2.3% of GEMM time, single- and 8-threaded.

Each float becomes one byte — a 4× shrink, plus four bytes per column — and dequantization rides the widening the kernels do anyway. Why symmetric, and why 127 rather than 128? Because mapping `[−max, +max]` onto `[−127, +127]` keeps zero exactly representable and the two directions perfectly balanced; the asymmetric −128 slot buys one extra value at the cost of that symmetry, and isn't worth it here.

**What the gate scores.** Training runs against the int8 (or bf16) copies — that is the memory they buy — but the commit patches the pristine f32 weights of the source file, so the function that ships is `W_f32 + Δ`, not `W_q + Δ`. Since E12 a plan compiled from a model file lowers its *eval* program against the f32 weights themselves: the student's frozen GEMMs become plain f32 GEMMs over source refs into that file (plan v17), mapped by the engine at run time. Every evaluation — the gate's before (exactly the source model at step 0, where `B = 0`) and after, and every best-state score — measures the model that ships. The teacher is not quantized at all: it is the distillation target, and it lives in a different file.

Eligibility is strict: *every* user of the tensor must be a matmul, with the tensor on the *weight* side. One use as an activation, or by any other op, disqualifies the whole tensor — those consumers would need dequantized floats, and there's no place to put them. LoRA adapters, gradients, and activations always stay f32; quantizing what you're *training* would be a very different (and lossier) design.

The max-abs sweep itself is a parallel chunked reduction, grain 65,536 elements per chunk, per-chunk maxima combined in fixed chunk order — the same determinism discipline as everything else (see [runtime.md](runtime.md)), and the same chunk geometry the backend's packer will use when it actually converts the bytes.

**bf16 storage** (`--bf16-base`, roadmap 2c) is the same review with a gentler rounding: every eligible weight — the same "only ever a matmul weight" rule, with no range condition since bfloat16 keeps float32's exponent — is packed as its f32 bits rounded to nearest-even at 16 bits (8 bits of mantissa, 2× smaller), and two widening GEMM opcodes (`kGemmNNBF16`, `kGemmNTBF16`, plan v10) put the bits back in the high half of an f32 on the way into the tile. Widening is exact, so the bf16 GEMMs are bit-for-bit the f32 GEMMs over the rounded matrix, and the f32 forward GEMM's fused epilogue still applies (its `in[3]` is free). Compute stays f32 throughout; the two storages are mutually exclusive per compile (one precision per weight). Choose int8 for 4× at the cost of outlier sensitivity, bf16 for 2× at a uniform 2⁻⁹ relative rounding.

One more thing, and it's the punchline of the whole quantization story: the emit table maps each **delta** onto the *original file's f32 bytes*. Commit computes `W′ = W + Δ` from the pristine weights — so quantization error affects training dynamics (slightly), but is **never baked into the committed model**. The artifact you ship is the exact model you started with, plus exactly what was learned.

## The backend: from program to plan

### Arena binding: memory planning as a compile-time problem

Ask yourself: what does `malloc` cost you on a device? Not just cycles — *unpredictability*. Fragmentation, allocation failure at step 900 of 1,000, nondeterministic addresses. SeeAI's answer is to compile memory away: `arena_binder.cc` (`backend/trainer/`) assigns every tensor a fixed byte offset in a single **arena**, sized at compile time, allocated exactly once on the device.

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

The instruction set has 45 opcodes — eight GEMM variants (`NN`, `NT`, `TN`, accumulating `NN`, two int8-dequantizing forms and two bf16-widening forms), elementwise ops, the activation forward/backward pairs, LayerNorm, the three loss families, the two optimizer steps, clip, fill, copy. The complete enumeration lives in `source/plan/instruction.h`, and [formats.md](formats.md) walks the encoding.

The addressing scheme is worth savoring for its economy. A tensor reference is a single 64-bit word: **bit 63 selects the address space** (0 = mutable arena, 1 = read-only rodata), bits 0–62 are a byte offset. That's the entire memory model — two flat spaces and an offset. No pointers, no relocation, and, on the device, one branchless test tells the validator which bounds to check. Scalars ride along bit-cast into spare operand slots (a GEMM's α, clip's max-norm, fill's value), and dimensions pack into the `out[]` words (a GEMM carries M, N, K; LayerNorm packs rows and columns into one word as `(N << 32) | D`).

### Precision: the relaxed family is a permission the compiler grants

Everything the compiler emits is exact by default. `--precision certified-bf16` (plan v18) sets `kFlagRelaxed` on the train and step programs' frozen-weight GEMMs — every `kGemmNN` / `kGemmNT` and their int8 and bf16 forms whose B operand is rodata — and on nothing else: the adapter products (A and B live in the arena), attention, norms, losses and the optimizer stay exact, and the eval program, lowered separately over the shipped f32 weights, carries no bit, so the gate scores what ships in reference arithmetic. The flag is a permission (the backend may run a certified relaxed kernel; the exact one is always legal), it is version-gated like every additive change, and the certificate that bounds what it permits binds to the plan's hash — so it lives beside the plan (`numerics_certificate.json`, written by `tool/certify_numerics.py`), and `tool/pack_update.py` refuses to package a relaxed plan without one. The compile report carries `precision` and `relaxed_gemms`.

### Host architecture and GEMM tiling: respecting the memory hierarchy

Recall the memory hierarchy: registers in a cycle, L1 in a few, L2 in a dozen or two, DRAM in hundreds. A naive triple-loop matmul of big matrices thrashes: by the time you need a row again, the cache has evicted it. The fix, known from the BLIS/GotoBLAS line of work, is **blocking**: choose tile sizes so that the pieces you reuse most *stay* in the levels closest to the core.

`DetectHostArch` (`backend/architecture/`) first learns the machine: the ISA (instruction set), SIMD width (*single instruction, multiple data* — how many floats one vector register can process at once), and FMA support (*fused multiply-add*, `a·b + c` in one instruction) are compile-time facts (host = target for this ahead-of-time compiler — they come from predefined macros); L1/L2 sizes, core count, and line size are runtime probes (`sysctlbyname` on Apple, `sysconf` elsewhere). Probes can fail — a container may hide `sysconf` values — and the policy is explicit: detection **never hard-fails**. It warns and falls back to conservative defaults (32 KiB L1, 512 KiB L2).

`SuggestGemmTiling` then derives block sizes `{mc, kc, nc}` as a *pure function* of that description — same inputs, same tiling, no measurement involved:

```
panel = min(L1/2 ÷ 4 bytes, 8192 floats)     the runtime's packed B panel (kDefaultGemmPanelFloats)
nc    = largest power of two ≤ √(2·panel)     the vectorized sweep: a LONG unit-stride loop
kc    = fit(panel ÷ nc)                       the rest of the panel, a multiple of the k-quad
mc    = fit(L2/2 ÷ kc)                        the band of A rows a task sweeps the panel across
```

where `fit` rounds down to a SIMD-width multiple (never below it). This is the geometry of the kernel that exists (`runtime.md`): each `kc × nc` tile of B is copied once into an L1-resident panel and swept by a four-row register block of C, so the panel is what must fit — in half of L1, and within the runtime's own panel — and the sweep wants to be long, while `kc` must not be starved (a 49k-vocabulary head lost 25% at `kc = 16`). On a 64 KiB L1 the model says 64 × 128, which is exactly what the runtime's defaults (64 × 256) are clamped to: model and kernel agree, by construction — the panel capacity is one constant in `source/plan/schema.h`, read by both planes. `ValidateGemmTiling` enforces the contract on any tiling handed in from outside — nonzero, SIMD-multiple, and the two half-cache inequalities (`kc·nc·4 ≤ L1/2`, `mc·kc·4 ≤ L2/2`) whenever the cache sizes are trustworthy.

It took a wrong model to get here, and the mistake is worth keeping (#90, the 2026-09 algorithm review). The first version defined `nc` as a **register width** — "4 vectors of C columns" — the right quantity for a BLIS microkernel the runtime did not have. The runtime read it as its N **cache tile**; the compiler baked `512 × 16` into every package's `build.sh`; packages ran 1.3–3.3× *slower* than the kernel's own defaults; and the nightly bench, built through CMake without those flags, measured a configuration no package shipped, so nothing noticed. Three things changed. Nothing emits a tiling any more: the geometry travels in the plan header (v11), decided from a measured table or left to the runtime defaults, and `seeml-bench` resolves the same policy the compiler would and records both the header's tiles and the tiles the core actually walks (`walked_gemm_tiles`), which the gate refuses to compare across. The kernel the model assumed was then built (E1, #80). And the model was re-derived against it, above. It is still a **hypothesis**, not a decision — one arm of the offline tuner's sweep (`analytic_gemm_tiles`), measured against the table rather than trusted.

### The kernel policy: measured offline, decided at compile time

Analytical models are good; measurements are better. But a measurement made *inside* the compiler is a measurement the compiler cannot reproduce or explain, and it costs every compile a wall-clock budget the build host may not have. So SeeAI draws the line where the Two-Plane Overhaul (`docs/next-project/`) draws it: **the compiler never measures.** Tuning lives on the build host, offline, in Python — `tool/autotune.py` — and the compiler consumes one decided fact per host.

The fact is the CPU GEMM **tile geometry**: the K panel a pass over C folds in and the N sweep width (`runtime/executor/kernel_policy.h`). It is a throughput knob and *only* a throughput knob: the N tile picks traversal order, not reduction grouping, and the K tile keeps the kernel's 4-wide unroll groups aligned as long as it stays a multiple of 4 — so every geometry the kernels accept computes **bit-identical** results (`kernels_test` proves it across a ragged shape at every tile boundary, `update_engine_test` across whole training runs). That is what lets a table pick among them without touching the determinism contract.

The tuner's method is the benchmark document's discipline made mechanical. Each *arm* is one geometry `KxN`; each measurement is one `seeml-bench` run at that arm (`--gemm-tiles`), medians of repeats by steps-regression per fixture and thread width; arms are visited round-robin for several rounds so thermal drift lands on every arm alike, and each arm's per-key result is the median over rounds — **medians of medians**. An arm's score is the geometric mean over every (fixture, threads) key of its rows/s relative to the default arm's, so no fixture outvotes another by being larger. Two arms are always in the sweep whatever the grid says: the kernel defaults (the arm to beat) and the analytic hypothesis above. The winner becomes the host's policy only when it beats the defaults by a margin (3 % — the benchmarks document's "signal, not noise" line); otherwise the defaults are recorded, with every arm's numbers beside them. Either way the decision is *measured*, and the table says so.

The table is keyed on the **host's identity** — `HostKey` in `backend/architecture/host_arch.h`: ISA, CPU model, physical cores, L1d and L2 bytes, SIMD width — every quantity that changes which tiling is fastest and nothing that does not. The bench prints the key, the tuner copies it, and the compiler recomputes it on the machine it runs on: one function, so the two planes cannot disagree about who a host is. The compiler's side of the seam is `backend/tuner/kernel_policy_table.cc`: a strict, purpose-built JSON reader (the compiler has no third-party dependencies, and a reader that guesses is a reader that misreads) and a resolution with three sources, in precedence order — `--gemm-tiles K,N` (explicit), `--kernel-policy table.json` (the entry for this host, or for `--target-host KEY` when cross-compiling), else the defaults. A table with no entry for the host is a note and the defaults; a table that does not parse, or names a geometry the kernels reject, is a hard error, because a bad policy silently costs every training step.

Where does the decision go? Into the **plan** — two header fields, `gemm_tile_k` and `gemm_tile_n` (v11; zero means the runtime's compiled-in default). This is the "decide early" principle applied to a knob that used to be a build flag: the compiler decides, the plan carries the decision, the runtime's load-time contract proves it (the K tile on the unroll), the CPU backend runs with it, and `seeml-seeu-dump` and the compile report both show it. Compile time is unchanged when no table is given — nothing is read.

### Emission: the plan, and a package that builds anywhere

Two emitters close out the backend. The plan itself is assembled by the driver (below) — header, three instruction streams, rodata, the persistent segment's initial image (zeros, except each `randn` parameter filled from its own seeded `mt19937_64` — deterministic bits, decided at compile time), and the emit table, each section 64-byte aligned, the whole sealed with a hash.

`native_emitter.cc` then writes a **self-contained package**: the plan as bytes *and* as an embedded C array, a generated driver `main` (flag parsing, the load→train→gate→merge→commit sequence, exit codes), the entire runtime vendored file-by-file, and a `build.sh` that compiles it all with nothing but a C++23 compiler. The package builds with no access to this repository — that's the deliverable you ship to a device vendor.

**How many copies of the weights does a compile hold?** About one (E2, #81). Until then the answer was three at plan assembly — the loaded model, a separately packed rodata vector, and the plan blob — on top of a whole-file read buffer during load, with the decimal TU adding ~4× the plan as strings: 4.5× the model at the peak (10.3× when embedding), on the hosts least able to spare it. Every piece of that was copy hygiene, and every fix is byte-neutral:

- **The model file is mapped, not read** (`model_reader.cc`, POSIX): file-backed pages are clean and evictable, so load costs the payloads and nothing else; elsewhere the read buffer is default-initialized rather than zero-filled one statement before the read overwrites it.
- **Payloads never zero-fill, and really leave when released.** `SmfBytes` is a byte vector whose growth default-initializes, and whose large buffers are their own anonymous mappings: `munmap` is unconditional, where macOS malloc keeps a freed 100 MB block resident in its reuse cache (that alone held the compile at 3.1×). Sizing without a fill also lets the load split its copy over **bytes, not tensors** — an embedding-dominated model used to copy its largest tensor on one thread.
- **Rodata is a layout, then one write.** `BindArena` records where each frozen weight goes (`RodataPack`); the driver assembles the plan by appending into one exact reservation and `PackRodata` writes each weight once, from the model straight to its place in the blob — raw f32 appended, int8 and bf16 computed in place. No intermediate section, no `resize` that zero-fills gigabytes it is about to overwrite.
- **The consuming compile** — `Compile(SmfModel&&)`, which the CLI uses — releases each source payload after its last pack (tied weights wait for the last of them), so the model shrinks as the plan grows. The const overload is unchanged and produces the same blob.
- **The decimal TU streams** in 16 MiB windows of parallel-rendered chunks, character-identical to the concatenated form.
- **The step-0 memory gate counts the merged deltas** (`EstimateLoraDeltaBytes`, the grafter's eligibility rule read off the SMF op list): they are full-size f32 whatever the base is stored as — 4× the weights under `--quantize-base` — and used to be left for the final gate to discover after all the work.

Measured on an 800 MB decoder (Apple M5, peak memory footprint ÷ model size): `--no-embed` 4.54× → **1.20×**, `--quantize-base` 2.93× → **1.20×**, with the decimal TU 10.34× → **1.20×**; wall time 2.8 → 1.2 s, 0.9 → 0.6 s and 11.9 → 7.6 s; every `.seeu`, embedded TU and `build.sh` byte-identical across a ten-plan fixture set (every loss, storage precision, teacher and accumulation form).

`kernel_emitter.cc` generates Metal GEMM kernel *source* (forward, both backward transposes, and the merge's scaled accumulate) from a `GpuTiling` clamped out of the host tiling into `{8, 16, 24, 32}` per dimension. Each kernel stages A- and B-tiles cooperatively into threadgroup memory, barriers, and accumulates with `fma` — the GPU expression of exactly the same blocking idea as above. It is text generation only, compiled by whoever integrates it.

## Diagnostics: errors as a designed surface

Every compiler diagnostic is one line, `"<unit>: <message>"`, and errors travel as `std::expected<T, std::string>` — no exceptions across subsystem boundaries. The core (`diagnostics/diagnostic.h`) provides three verbs over the thread-safe `Logger`: `Fail` (build an error), `Note` (progress), `Fallback` (degraded but continuing). Each *process* gets a header-only module owning its unit names and message shapes:

| module | process delimited | units |
|---|---|---|
| `tokenizing/` | SMF byte-stream decode/encode | `SMF`, `Ingressor` |
| `parsing/` | SMF graph → forward SIR | `Parser` |
| `passing/` | pass orchestration + lowering legality | `PassManager`, `ConvLowering` |
| `updating/` | the analytic methods | `TrainableAutodiff`, `LoraGrafter`, `MergeBuilder`, `OptimizerSynthesizer`, `GemmEpilogueFuser` |
| `architecting/` | local device analysis and the tuner's table | `HostArch`, `KernelPolicy` |
| `generating/` | code generation + the driver | `UpdateCompiler`, `ArenaBinder`, `InstructionLowering`, `NativeEmitter` |

`architecting/` has a two-tier discipline worth noting: *detection* can never hard-fail (warn, fall back to conservative defaults — a missing sysctl shouldn't stop a compile; a kernel-policy table with no entry for this host means the defaults), while *contract violations* — a tiling that lies about the cache hierarchy, a table that does not parse or names a K tile off the kernel's unroll — are hard errors (a broken contract means broken code, or every training step silently slower).

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
- The **backend** binds every value to a byte offset (liveness + first-fit, like register allocation for tensors), lowers to a 31-opcode, 64-byte-instruction ISA with a two-address-space memory model, writes the CPU GEMM tile geometry the offline tuner measured for the host (or the kernel defaults) into the plan, and emits a package that builds anywhere.
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
- **The Python plane** — `pack_update.py` (the `.incbin` package
  assembler), `autotune.py` (the offline kernel-policy tuner whose table
  this compiler reads), `bench_compare.py` (the Tier A gate),
  `frontier_exec.py` with `seeml_plan_probe.cc` (a second implementation of
  the whole instruction set, compared against the runtime one instruction
  at a time) and `certify_numerics.py` (the relaxed-reduction certificate).
  They run on the build host and are described in
  [tool/README.md](../tool/README.md). The division of labour is the
  two-plane doctrine: the compiler, the runtime and every emitted package
  are dependency-free C++; what the doctrine has no reason to constrain —
  packaging, measurement, tuning, certification, reference execution — is
  Python, and nothing Python ships on a device.

## Testing

One SeeTest suite per module, organized to mirror this partition (`test/compiler/<subsystem>/*_test.cc` — see [test/README.md](../test/README.md)), run via `ctest` or directly from `build/` (see [usage.md](usage.md)). The compiler-side suites: `frontend/` (`model_io`, `resource_analyzer`, `sir`, `operator`, `parser`), `analysis/` (`update_passes`, `updater`, `reviewer`), `backend/` (`tuner`, `trainer`, `native_emitter`), `driver/` (`update_compiler`, `driver`), and `diagnostics/`, plus `source/` suites (`hash`, `parallel_for`) and the end-to-end `system/update_system_test`.
