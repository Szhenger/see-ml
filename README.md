# SeeML: An ML Update Compiler (How to Train A Model)

SeeML treats a machine-learning fine-tune the way an operating system treats
a software update. On a build host, a compiler takes a frozen model, grafts
LoRA adapters onto it (LoRA — Low-Rank Adaptation — trains a pair of small
matrices beside each frozen weight instead of the weight itself), and
compiles the *entire training run* — forward pass, backward pass, optimizer,
memory plan — ahead of time into a single update plan. On the device, a small
zero-dependency runtime executes that plan, gates the result on a measurable
improvement, and commits the updated weights atomically — or leaves the
device untouched.

## Why compile a training run ahead of time?

On-device training normally means shipping a framework runtime: dynamic
allocation, dynamic shapes, kernel dispatch decided at run time, and numerics
that drift with thread count. SeeML moves every one of those decisions to
compile time, which buys three properties that matter when the thing being
trained is a device you don't control:

- **Verifiable before it runs.** The plan is a fixed instruction stream over
  a memory arena whose size and layout were decided by the compiler. At load
  time the runtime re-proves the bounds of every instruction operand before
  anything executes — a corrupt or foreign plan is rejected, never partially
  run.
- **Reproducible anywhere.** Parallel execution is bitwise-deterministic:
  work is chunked by problem shape, never by thread count, so the same plan,
  data, and seed produce the same bits on an 8-core dev board and a
  single-core target.
- **Safe to apply.** Like an OS update, the result is gated (validation loss
  must improve, or the device is untouched), resumable (checkpoints are
  hash-bound to their exact plan), and committed via fsync + atomic rename —
  a power cut leaves the old model or the new one, never a torn file.

The price of this is generality — see
[Scope and limitations](#scope-and-limitations).

## Quickstart

```bash
# 1. Build host: export a PyTorch model + corpus (or use the built-in demo)
python3 tool/export_model.py --demo out/

# 2. Build host: compile the update plan into a self-contained package
seeml-update-compile --source out/model.smf --out pkg/ \
  --loss xent --lora-rank 8 --steps 1000 --build

# 3. Device: run the update — train, gate, merge, commit
pkg/model_update --model out/model.smf --data out/corpus.sds \
  --out updated.smf
```

Full walkthrough, flag reference, and exit codes: [docs/usage.md](docs/usage.md).

## Documentation map

Read in this order:

| document | what it covers |
|---|---|
| [docs/usage.md](docs/usage.md) | the workflow: export, compile, run on-device; every CLI flag; threading and determinism |
| [docs/compiler.md](docs/compiler.md) | build-host compiler architecture: frontend, analysis passes, backend, driver contracts |
| [docs/runtime.md](docs/runtime.md) | on-device runtime architecture: engine, feeder, executor, validator, custodian |
| [docs/formats.md](docs/formats.md) | the binary formats: SMF models, SDS datasets, SEEU plans, SEKP checkpoints |
| [test/README.md](test/README.md) | the test tree, the in-repo SeeTest harness, and how suites mirror the code |
| [docs/workflows.md](docs/workflows.md) | continuous integration from first principles: what a workflow is, and what each of SeeML's asserts |
| [docs/roadmap.md](docs/roadmap.md) | the remaining architecture-review projects, scoped and sequenced |
| [docs/journey.md](docs/journey.md) | how the project got here: from a C compiler to an ML update compiler |

## Scope and limitations

What SeeML can train today, stated up front:

- **Models.** Feed-forward and decoder-transformer graphs over eleven
  operator kinds: `MatMul`, `AddBias`, `Relu`, `Gelu`, `Silu`, `Mul`,
  `LayerNorm` (SMF v2), plus `Add`, `RmsNorm`, `Rope`, and causal
  `Attention` (SMF v3, with model-level sequence geometry) — enough for
  pre-norm decoder blocks with SwiGLU MLPs. The PyTorch exporter accepts an
  `nn.Sequential` of `Linear`/activation/`LayerNorm` modules, and
  `export_decoder_smf` emits decoder stacks from plain weight arrays
  (embedding lookup stays outside the update: corpora carry pre-embedded
  rows). The compiler's intermediate representation additionally models 2-D
  convolution (lowered to matrix multiplication via im2col), but
  grouped/dilated forms are rejected and the SMF container does not yet
  carry convolutions.
- **Training method.** LoRA adapters on frozen `MatMul` weights only — the
  base model is never trained directly. Optimizers: SGD or AdamW, one
  parameter group. Losses: cross-entropy, MSE, KL distillation from a
  teacher model, or a weighted cross-entropy + KL composite.
- **Numerics.** Training is f32 throughout. `--quantize-base` stores the
  *frozen* weights as int8 in read-only data; because commit applies deltas
  to the pristine f32 source file, quantization error is never baked into
  the committed model.
- **Shapes.** The batch size is fixed at compile time and baked into the
  plan; the dataset must match the compiled geometry.
- **Target machine.** The compiler derives its cache tilings from the
  machine it runs on (host = target). The emitted package cross-compiles
  (set `CXX`), but tilings remain build-host-derived hints. Execution is
  CPU; on Apple hosts a hardware-validated Metal GEMM dispatch harness
  exists (roadmap Project 5), but the engine does not yet dispatch to it.
- **Integrity, not authenticity.** All hashing is FNV-1a — a corruption and
  mismatch detector, not a signature. Authenticate plans in your update
  transport.

## Prerequisites

- **Build and run:** a C++23 compiler (clang or gcc). CMake is supported but
  optional — `build/build.sh` drives a full build with `sh` alone.
- **Model export only:** Python 3 with PyTorch (`tool/export_model.py`).
- **On-device:** nothing. The runtime is zero-dependency and vendored into
  every emitted package; the package builds with no access to this
  repository.

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
# or, without CMake:
sh build/build.sh && for t in build/seeml_*_test; do "$t"; done
```

## Glossary

Terms and abbreviations used throughout the docs, defined once here. Format
names are specified byte-for-byte in [docs/formats.md](docs/formats.md).

| term | definition |
|---|---|
| **SMF** | SeeML Model Format (`.smf`) — the dependency-free container for source and teacher models |
| **SDS** | SeeML Dataset (`.sds`) — the fixed-shape training corpus streamed on-device |
| **SEEU** | SeeML Update plan (`.seeu`) — the fully compiled update: instruction streams, frozen weights, arena layout, emit table |
| **SEKP** | SeeML Checkpoint — the mid-training state container, hash-bound to its exact plan |
| **SIR** | SeeML Intermediate Representation — the compiler's in-memory program form, in SSA style (static single assignment: every value is defined exactly once) |
| **LoRA** | Low-Rank Adaptation — fine-tuning via a pair of small matrices `A` (n×r) and `B` (r×m) per frozen weight; the trained delta is `Δ = (α/r)·A@B` |
| **GEMM** | general matrix–matrix multiply — the workhorse kernel of both training and merging |
| **arena** | the runtime's single memory allocation, laid out entirely at compile time into read-only, persistent (checkpointed), I/O, and transient segments |
| **plan** | the `.seeu` artifact: everything the device needs to run one complete update |
| **AOT** | ahead-of-time — decided at compile time on the build host, not on the device |
| **subsystem** | a top-level folder of `compiler/` or `runtime/`, named for its role in the process |
| **discipline** | a folder inside a subsystem, named for the kind of work done by the units inside it |
| **unit** | one class or function family with its own header — the granularity of diagnostics and tests |
| **façade header** | the single include that re-exports a folder's units, so the files behind it can be reorganized without touching consumers |
