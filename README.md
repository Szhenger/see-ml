# SeeML: An ML Update Compiler (How to Train a Model)

## This is Machine Learning

Odds are, some device of yours updated an app this week. You barely noticed, and that's the point: the update arrived as a self-contained package, was verified before a single byte was trusted, applied atomically, and — had anything gone wrong — your device would have been left exactly as it was, as though nothing had happened at all.

Now consider the neural networks that increasingly live on those same devices. Suppose you'd like one of them to *learn* — to train, right there on the device, on the device's own data. Training is ordinarily a messy, dynamic affair: a Python interpreter, a framework of a few hundred megabytes, memory allocated on the fly, results that vary run to run. Nothing about it resembles the disciplined little package that updated your app.

So, a question: **how might we make training a model as safe, as small, and as boring as a software update?**

It turns out the answer is a compiler.

## Prerequisites of Machine Learning

- **Build and run:** a C++23 compiler (clang or gcc). CMake is supported but
  optional — `build/build.sh` drives a full build with `sh` alone.
- **Model export only:** Python 3 with PyTorch and NumPy
  (`tool/export_model.py`); `pip install -r tool/requirements.txt` installs
  both.
- **On-device:** nothing. The runtime is zero-dependency and vendored into
  every emitted package; the package builds with no access to this
  repository.

```bash
# 0. Build the tools (any C++23 compiler)
cmake -S . -B build && cmake --build build -j
# or, without CMake:
sh build/build.sh

# 1. Export a demo model, teacher, and corpus (build host, PyTorch)
python3 tool/export_model.py --demo out/

# 2. Compile an update plan and a self-contained native package
./build/seeml-update-compile \
  --source out/model.smf --out pkg/ \
  --data-batch 32 --loss xent --lora-rank 8 --lora-alpha 16 \
  --optimizer adamw --lr 1e-3 --steps 1000 --build

# 3. Run the update (this is what would run on the device)
./pkg/model_update --model out/model.smf --data out/corpus.sds \
  --out out/updated.smf --val-frac 0.1 --seed 7
```

Exit code `0` means the model improved and was committed. Exit code `3` means it didn't — and the device was left untouched. That, in miniature, is the entire philosophy.

## How to Read these Documents

These documents aim to be educational as well as descriptive. SeeML happens to be a small, complete instance of several of computer science's greatest hits — compilers, virtual machines, calculus done by a program, cache-aware algorithms, crash-safe storage — and the docs teach each idea from first principles before showing you SeeML's implementation of it. That includes the machine learning: if you've completed CS50x (or equivalent), you have every prerequisite; the ML itself is taught here, with the compiler as the textbook.

Read them in this order:

| Document | What it teaches |
|---|---|
| [docs/usage.md](docs/usage.md) | The three-step workflow, every flag explained, and what actually happens when you run an update. Start here. |
| [docs/compiler.md](docs/compiler.md) | The compiler, end to end — and most of the ML: what an intermediate representation is, the linear algebra of LoRA, how a program differentiates a program, what optimizers like AdamW actually compute, number formats and quantization, memory planning as register allocation, cache-aware matrix multiplication, and a multi-armed bandit that tunes it. |
| [docs/runtime.md](docs/runtime.md) | The on-device virtual machine: load-time validation as a safety proof, numerically stable kernels (and why naive formulas explode), deterministic parallelism, a producer-consumer pipeline, and storage that survives a power cut. |
| [docs/formats.md](docs/formats.md) | The four binary formats on disk — bytes, offsets, magic numbers, and hashes — and why each field is there. |
| [test/README.md](test/README.md) | How the test tree mirrors the code, and how you test calculus with arithmetic. |
| [docs/benchmarks.md](docs/benchmarks.md) | The metric program that gates frontier development: throughput, kernel, memory, lifecycle, and velocity tiers. |

## Design Principles of Software Systems

If you remember nothing else, remember these five, because every file in this repository is an application of one of them:

- **Decide early.** Anything decidable at compile time — shapes, offsets, instruction order, memory size — is decided at compile time. The device executes; it does not plan.
- **Verify at every boundary.** Each subsystem hands its output to the next only through an explicit contract that is checked, both in the compiler and again on the device. A plan is proven safe before it is run.
- **Same bits, any thread count.** Work is chunked by problem shape, never by thread count, so parallelism never changes results.
- **No improvement, no change.** Every update must prove itself on data it never trained on; a failed update leaves the device exactly as it was.
- **Power cuts are ordinary.** Every durable write is an fsync'd sidecar file plus an atomic rename. There is no torn state.

## Scope and Limitations

What SeeML can train today, stated up front:

- **Models.** Feed-forward and decoder-transformer graphs over twelve
  operator kinds: `MatMul`, `AddBias`, `Relu`, `Gelu`, `Silu`, `Mul`,
  `LayerNorm` (SMF v2), `Add`, `RmsNorm`, `Rope`, and causal `Attention`
  (SMF v3, with model-level sequence geometry), plus `Embedding` (SMF v4)
  — token-native decoders train end to end: the corpus carries i32 token
  ids (SDS v2), next-token labels are derived from the shifted view, and
  the frozen embedding gathers on-device. The PyTorch exporter accepts an
  `nn.Sequential` of `Linear`/activation/`LayerNorm` modules, and
  `export_decoder_smf` emits decoder stacks from plain weight arrays. 
  The compiler's intermediate representation additionally models 2-D convolution 
  (lowered to matrix multiplication via im2col), but
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

## Development

This project was developed at SmoothML! 


