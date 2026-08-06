# SeeML: An ML Update Compiler (How to Train A Model)

## This is SeeML

Odds are, some device of yours updated an app this week. You barely noticed, and that's the point: the update arrived as a self-contained package, was verified before a single byte was trusted, applied atomically, and — had anything gone wrong — your device would have been left exactly as it was, as though nothing had happened at all.

Now consider the neural networks that increasingly live on those same devices. Suppose you'd like one of them to *learn* — to train, right there on the device, on the device's own data. Training is ordinarily a messy, dynamic affair: a Python interpreter, a framework of a few hundred megabytes, memory allocated on the fly, results that vary run to run. Nothing about it resembles the disciplined little package that updated your app.

So, a question: **how might we make training a model as safe, as small, and as boring as a software update?**

It turns out the answer is a compiler.

## First, what does it even mean to train a model?

If you've taken CS50x but no machine-learning course, here is the entire idea, no mystery required. A neural network is just a function: input goes in (say, a few numbers describing a sensor reading), a prediction comes out. What makes it interesting is that the function's behavior is controlled by a large array of numbers called **weights** (or *parameters*) — and nobody sets them by hand. Instead, **training** is a loop:

1. Run the function on a few examples and measure *how wrong* the predictions were, as a single number called the **loss**. Smaller is better.
2. Using calculus, compute for every weight which direction of adjustment would reduce the loss. That per-weight list of directions is the **gradient**.
3. Nudge every weight a small step in its direction. The rule that decides exactly how big a step is called the **optimizer**.
4. Repeat, a few hundred or thousand times, until the loss stops improving.

That's it — measure, differentiate, nudge, repeat. Everything else in machine learning is refinement of that loop, and every piece of it (the loss, the calculus, the optimizer, the data handling) is something SeeML's documentation will teach you properly, using the code itself as the textbook.

## The big idea

SeeML treats one complete training job — that entire loop, plus evaluation and the final weight patch — as a *program to be compiled ahead of time*. On a build machine, the compiler:

1. reads a **frozen** model — one whose existing weights it will never modify,
2. grafts small trainable side-matrices called **LoRA adapters** onto it, so the update learns thousands of new numbers instead of retraining millions of old ones (the linear algebra that makes this work is taught in [docs/compiler.md](docs/compiler.md)),
3. *derives the calculus itself* — the compiler works out step 2 of the training loop from the model's structure, a technique called automatic differentiation, and appends the optimizer's nudge as ordinary instructions,
4. plans every byte of memory the job will ever touch, and
5. emits a single artifact: a `.seeu` **update plan** — three fixed instruction streams (train, evaluate, merge) plus data, every offset already decided.

On the device, a **runtime** of about a dozen C++ files — no framework, no dependencies — validates that plan, executes it with one memory allocation, checks whether the model actually improved, and only then patches the model file, atomically. If the loss didn't improve, or the power failed mid-write, the original model remains untouched, byte for byte.

And because every parallel computation in SeeML is **bitwise-deterministic** — the same plan, data, and seed produce the *same bits* on one core or eight — a training run on your laptop reproduces exactly on the device. Thread count is a throughput knob, not a numerics knob.

That's the whole product: `model.smf + corpus.sds → seeml-update-compile → update_plan.seeu → model_update → updated model, or no change at all`.

## What's in the box

```
source/     the source language: model format, plan ABI, parallel substrate, hashing
compiler/   the ahead-of-time compiler: frontend → analysis → backend, with a
            driver that verifies every subsystem boundary
runtime/    the zero-dependency on-device half: a little VM, its kernels, a
            corpus feeder, a load-time validator, and durable storage
tool/       export_model.py (PyTorch → SMF), the compiler CLI, a plan disassembler
test/       SeeTest, an in-tree harness, with one suite per module
docs/       what you're about to read
```

## Quick start

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

## How to read these docs

These documents aim to be educational as well as descriptive. SeeML happens to be a small, complete instance of several of computer science's greatest hits — compilers, virtual machines, calculus done by a program, cache-aware algorithms, crash-safe storage — and the docs teach each idea from first principles before showing you SeeML's implementation of it. That includes the machine learning: if you've completed CS50x (or equivalent), you have every prerequisite; the ML itself is taught here, with the compiler as the textbook.

Read them in this order:

| Document | What it teaches |
|---|---|
| [docs/usage.md](docs/usage.md) | The three-step workflow, every flag explained, and what actually happens when you run an update. Start here. |
| [docs/compiler.md](docs/compiler.md) | The compiler, end to end — and most of the ML: what an intermediate representation is, the linear algebra of LoRA, how a program differentiates a program, what optimizers like AdamW actually compute, number formats and quantization, memory planning as register allocation, cache-aware matrix multiplication, and a multi-armed bandit that tunes it. |
| [docs/runtime.md](docs/runtime.md) | The on-device virtual machine: load-time validation as a safety proof, numerically stable kernels (and why naive formulas explode), deterministic parallelism, a producer-consumer pipeline, and storage that survives a power cut. |
| [docs/formats.md](docs/formats.md) | The four binary formats on disk — bytes, offsets, magic numbers, and hashes — and why each field is there. |
| [test/README.md](test/README.md) | How the test tree mirrors the code, and how you test calculus with arithmetic. |

## Design principles, in one breath

If you remember nothing else, remember these five, because every file in this repository is an application of one of them:

- **Decide early.** Anything decidable at compile time — shapes, offsets, instruction order, memory size — is decided at compile time. The device executes; it does not plan.
- **Verify at every boundary.** Each subsystem hands its output to the next only through an explicit contract that is checked, both in the compiler and again on the device. A plan is proven safe before it is run.
- **Same bits, any thread count.** Work is chunked by problem shape, never by thread count, so parallelism never changes results.
- **No improvement, no change.** Every update must prove itself on data it never trained on; a failed update leaves the device exactly as it was.
- **Power cuts are ordinary.** Every durable write is an fsync'd sidecar file plus an atomic rename. There is no torn state.

Was this compiled for you? In a sense, yes — now go read [docs/usage.md](docs/usage.md).
