# SeeML: An ML Update Compiler (How to Train Your Local Model)

SeeML compiles an on-device model update **ahead of time**. On a build host,
it takes a frozen feed-forward model, grafts low-rank (LoRA) adapters onto
it, synthesizes the entire training computation — forward pass, loss,
reverse-mode gradients, optimizer — as a fixed instruction stream bound to a
pre-planned memory arena, and packages the result as a single `.seeu` plan.
On the device, a zero-dependency VM executes that plan against local data:
one plan = one complete, gated, resumable, atomically-committed update. The
device never links a training framework, never allocates beyond one arena,
and never commits a model that failed its validation gate.

## The workflow

```
 build host                                            device
┌─────────────────────────────────────────────┐   ┌──────────────────────────┐
│ 1. export       2. compile                  │   │ 3. update                │
│ PyTorch ──▶ .smf ──▶ seeml-update-compile ──┼──▶│ model_update             │
│ (tool/export_model.py)   │                  │   │  load ▶ train ▶ gate     │
│ corpus ──▶ .sds          ▼                  │   │  ▶ merge ▶ commit        │
│                 self-contained pkg/         │   │  ──▶ updated .smf        │
│                 (.seeu plan + vendored      │   │                          │
│                  runtime + build.sh)        │   │                          │
└─────────────────────────────────────────────┘   └──────────────────────────┘
```

**1. Export (build host, PyTorch).** `tool/export_model.py` converts a
`nn.Sequential` of Linear / ReLU / GELU / SiLU / LayerNorm modules into SMF
(`.smf`, a minimal dependency-free container, v2) and a training corpus into
SDS (`.sds`). `--demo` produces a ready-made model + teacher + corpus.

**2. Compile (build host).** `seeml-update-compile` runs the whole
ahead-of-time pipeline — every byte the runtime will touch is bound here:

```
SMF ingest ──▶ feasibility gate ──▶ forward SIR (+ frozen teacher)
           ──▶ loss grafting                (xent | mse | kl | xent+kl)
           ──▶ pass phase A: conv-lowering, LoRA grafting
           ──▶ primal snapshot              (becomes the eval program)
           ──▶ pass phase B: reverse-mode autodiff, optimizer synthesis
           ──▶ merge program                (Δ = (α/r)·A@B per adapter)
           ──▶ int8 quantization review     (--quantize-base)
           ──▶ pass phase C: GEMM-epilogue fusion + dead-code sweep
           ──▶ segmented arena binding      RODATA | PERSISTENT | IO | TRANSIENT
           ──▶ persistent image init        (deterministic counter-based randn)
           ──▶ instruction lowering         (train / eval / merge streams)
           ──▶ .seeu plan assembly          (v5, sealed with PlanSelfHash)
           ──▶ native package emission      (--build)
```

The stages, briefly: the **frontend** bounds-checks every SMF byte, refuses
models that provably cannot train in local memory, and parses the op list
into SIR — a small SSA IR whose structural verifier re-runs after every
pass. The **analysis** phase grafts the loss, adapts every eligible frozen
MatMul with a rank-r adapter pair (`A` randn-init, `B` zeros — step 0 is
bit-identical to the source model), differentiates only what can reach a
trainable parameter (frozen and teacher subgraphs get zero backward
compute), appends the optimizer as IR so one program execution is one full
training step, and fuses `GEMM → AddBias → activation` chains into
single-instruction epilogues wherever the use-lists prove no other reader.
The **backend** packs frozen weights into read-only data (optionally as
per-tensor symmetric int8), lays out one mutable arena with
liveness-scanned transient reuse, lowers the three programs to fixed
64-byte instructions, derives cache-aware GEMM tilings for the target, and
emits a **self-contained package**: the plan, a generated driver `main`,
the vendored runtime sources, and a `build.sh` that compiles `model_update`
on any machine with a C++23 compiler (set `CXX` to cross-compile).
Inspect any plan with `seeml-seeu-dump plan.seeu --instrs`.

**3. Update (device).** `model_update` runs the ML analog of an OS
software update:

```
Load    ──▶ magic/version negotiation, whole-plan hash, and a full
            bounds/alignment/aliasing proof of every instruction operand —
            after this the dispatcher executes the streams blindly
Train   ──▶ N steps of {stage batch ▶ fwd+bwd+clip+optimizer}, pipelined
            by a feeder thread; checkpoints are plan-hash-bound and
            fsync-durable (--checkpoint / --resume); non-finite loss aborts
Gate    ──▶ held-out validation loss before vs after (--val-frac); no
            improvement → exit 3, device untouched (--force overrides)
Merge   ──▶ the merge program materializes each Δ = (α/r)·A@B
Commit  ──▶ deltas are streamed onto the pristine f32 weights of a
            hash-verified copy of the source file, fsync'd, and renamed
            atomically — a quantized plan never bakes its quantization
            error into the committed model
```

Exit codes: `0` committed, `1` runtime error, `2` bad arguments,
`3` regression-gate rejection.

## Guarantees

- **Bitwise determinism.** Parallel work is split into chunks whose
  geometry depends only on the problem shape, never the thread count;
  reductions combine per-chunk partials in fixed order. The same plan,
  data, and seed produce the same bits at any `SEEML_THREADS` — thread
  count is a throughput knob, not a numerics knob.
- **Everything is hash-bound.** The plan seals itself (`PlanSelfHash`),
  binds to the exact source model file (`ContentHash64`), and checkpoints
  bind to both the plan and their own payload — wrong or corrupt artifacts
  are refused, loudly, before any byte moves.
- **Validated, then trusted.** Every instruction operand is bounds-,
  alignment-, and overlap-proven at load; version negotiation accepts the
  readable range and rejects everything else; unknown opcodes and unknown
  flag bits are load errors, never silent skips.
- **The device is never left worse.** The regression gate, the atomic
  commit, and merge-staleness tracking (any parameter mutation invalidates
  materialized deltas) mean a failed or interrupted update leaves the
  source model untouched.

## Repository layout

```
source/     the shared substrate: SMF model format, the .seeu plan ABI,
            deterministic ParallelFor, integrity hashes
compiler/   driver / frontend / analysis / backend / diagnostics —
            SMF bytes to sealed plan (docs/compiler.md)
runtime/    engine / feeder / executor / validator / custodian —
            the zero-dependency on-device VM (docs/runtime.md)
tool/       seeml-update-compile, seeml-seeu-dump, export_model.py
test/       one SeeTest suite per module, mirroring the partition
docs/       compiler.md · runtime.md · formats.md · usage.md · roadmap.md
```

## Building and testing

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
# without CMake:
sh build/build.sh && for t in build/seeml_*_test; do "$t"; done
```

Full usage, flags, and the binary format specifications live in
[docs/usage.md](docs/usage.md) and [docs/formats.md](docs/formats.md);
planned work is scoped in [docs/roadmap.md](docs/roadmap.md).
