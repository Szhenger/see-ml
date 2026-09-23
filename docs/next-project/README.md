# The Two-Plane Overhaul — the next development project

*Compiled 2026-09-01 from the two audits of 2026-08-31 — the **SeeML
Performance Audit** (subsystem-by-subsystem, code-derived) and the **SeeML
Frontier Bridge** (architecture audit against `torch.compile` CPU fine-tuning
and MLX-LM LoRA on Apple-Silicon GPU) — plus the Nightly #20–#23 bench-gate
incident analysis. This document merges the deliverables of both reports into
the objectives of one project, and partitions every finding by its root
cause. The issue bodies live in [`issues/`](issues/);
[`tool/create_next_project.sh`](../../tool/create_next_project.sh)
materializes them as GitHub Issues and a GitHub Project board.*

## The paradigm change

Today's doctrine line is drawn at the on-ramp: `export_model.py` is "the
only Python in the product, and everything downstream is dependency-free
C++" (`tool/README.md`). The Frontier Bridge showed what that line costs:
both residual performance ceilings — the AMX coprocessor (≈1.8×, reachable
only through an Accelerate dependency) and reduction reassociation (up to
≈2×, forbidden by the bitwise contract) — are **doctrinal**, not
engineering, and no amount of kernel work inside the current design removes
them. Meanwhile the Performance Audit showed the dominant *compile-side*
costs (the 16.5-minute decimal-TU `--build`, the dead autotuner, the
unmeasured gate) are things a build-host tool does badly in doctrinal C++
and well in Python.

The overhaul redraws the line by **where the doctrine actually applies**:

| Plane | Lives | Language | Doctrine |
|---|---|---|---|
| **Core plane** (device) | the emitted package, the runtime, the plan ISA | C++, zero-dependency | **intact and non-negotiable**: bitwise determinism per backend, one allocation, load-time validation, atomic gated commit |
| **Python plane** (build host) | packaging, measurement, autotuning, numerics certification, frontier reference execution | Python (`tool/`), free to depend on PyTorch/MLX/NumPy | doctrine is *spent here deliberately* — the build host was never doctrine-bound; only the artifact it emits is |

The partition rule for every audit finding:

> **A doctrinal root cause is handled by coupling with a Python subsystem
> on the build host** — the capability the doctrine forbids in the core
> moves to where the doctrine does not apply, and anything that relaxes a
> core guarantee ships only as a *measured, certified, opt-in* plan the
> default path never emits.
>
> **A purely algorithmic root cause gets a better design inside the core**
> — every such fix in this project is bitwise-safe unless explicitly
> marked otherwise.

## Objectives (the merged deliverables)

1. **Compile any model the device can train.** Collapse the 3× plan-assembly
   residency and the ≈11× emission RSS to ~1×/2×; a 7B-f32 compile stops
   being an OOM (Performance Audit §02–03).
2. **Make `--build` seconds, not minutes.** Package assembly leaves the C++
   compiler: a Python packer emits `.seeu` + `.incbin`, implementing G1c
   (#64) without a multi-GB translation unit (16.5 min at 135M today).
3. **Lift CPU MFU from 4–13% into the 20–30%-of-NEON-peak band** with the
   GEMM redesign (m-hoist → microkernel + arena-planned packing → 2D
   partition), the bitwise kernel batch, and — profiling-gated — chain
   fusion (Frontier Bridge §02).
4. **Turn both doctrinal ceilings into measured product decisions.** The
   frontier executor prices AMX/GPU upside per plan on our own shapes; the
   numerics-certification subsystem makes relaxed reductions an opt-in
   opcode family with a per-plan tolerance certificate — never a silent
   default (Frontier Bridge §02, conclusion).
5. **Make the measurement plane trustworthy.** Host-keyed autotuning
   replaces the dead in-tree bandit; the nightly gate stops failing on
   runner speed (Nightly #20–#23 were false positives at the same SHA as
   green #24).
6. **GPU bridge unchanged** — it is already owned by epic #59 (G1b-1→4,
   G1c, v1.3.0 gate #65); the project board includes those issues rather
   than duplicating them.

## The partition

### Doctrinal root causes → Python-plane subsystems

| Issue | Root cause (doctrine that binds) | Python subsystem |
|---|---|---|
| P1 | Zero-dependency vendoring → 3.9× decimal TU, `--build` dominates all compile time, ≈11× emission RSS | `tool/pack_update.py` package assembler (implements G1c #64) |
| P2 | No compile-time measurement (torch `max-autotune` parity); in-tree UCB1 is dead code with no persistence | `tool/autotune.py`, host-keyed kernel-policy tables |
| P3 | Bitwise contract forbids reduction reassociation (≤2× on every reduction; attention QK^T is the largest) | `tool/certify_numerics.py` + opt-in relaxed-reduction opcode family |
| P4 | Zero-dependency forbids Accelerate → AMX ceiling ≈1.8× is unpriced on our shapes | `tool/frontier_exec.py` — plan-level PyTorch/MLX reference executor + differential-test oracle |
| P5 | Determinism-grade regression gate undone by runner variance (Nightly #20–#23) | calibration-ratio normalization in `tool/bench_compare.py` |
| P6 | Every Python↔C++ contract (SMF, SDS, SEEU, `bench.json`, `--report`) is two hand-maintained copies checked only by one PyTorch demo; Python has no test gate while C++ has six (seam review, 2026-09-03) | ABI manifest + golden bytes, `tool/seeml/` package with dependency tiers, Python CI parity, `seeml-seeu-dump --json` as P4's only decoder |

### Algorithmic root causes → core-plane redesign

| Issue | Root cause | Fix |
|---|---|---|
| E1 | GEMM streams C ⌈K/64⌉×, no microkernel, no packing, small-M serializes | m-hoist (bitwise) → MR×NR microkernel with **arena-planned** pack buffers → 2D C partition; #66 stays the `GemmNT` fix |
| E2 | Compile holds 3 weight copies; zero-fills, no `reserve`, non-streaming emission | single-residency assembly + streaming emission + zero-fill removal (all bitwise-safe) |
| E3 | Per-element transcendentals, false sharing, 3× optimizer streaming, serial argmax | RoPE table, chunk-local `ReduceRows`, clip-fused step, parallel argmax (all bitwise) |
| E4 | One fusion pattern vs Inductor's arbitrary chains; runtime is bandwidth-bound | `kFusedMap` (roadmap Phase 1b, profiling-gated) |
| E5 | Latent O(ops²) cliffs, unmeasured passes, debug output always-on | graph/infra hygiene batch |
| E6 | Four doc claims the code contradicts; `tool/README.md` doctrine line | doc corrections + the Python-plane doctrine update |

## Sequencing

P6 lands first or alongside P1 — its ABI manifest, golden fixtures, and
Python CI job are what every later Python-plane subsystem is verified
against (and it keeps P1's default compile path Python-free: `.incbin` is
a C++ emitter change; the packer is for the G1c extras). Then
P1 and E2 (they interact at emission — coordinate, don't collide),
with E1 stage 1 (the free bitwise m-hoist) alongside; then P5/P2 so every
later claim is measured trustworthily; E3/E5 as fill; P4 before P3 (the
executor is the oracle certification needs); E4 last, per the roadmap's own
profiling gate. The GPU track (#59–#65) proceeds independently as
sequenced in `docs/roadmap.md`.

## Status

- **P1 shipped** (2026-09-11, #88): `tool/pack_update.py`, `--no-embed`.
- **P2 shipped** (2026-09-15, #76): `tool/autotune.py` and the host-keyed
  kernel-policy table; the compiler's `--kernel-policy` / `--gemm-tiles` /
  `--target-host`; the CPU GEMM tiles became a plan-header property (v11)
  the runtime proves and the CPU backend runs, so the bench sweeps arms
  in-process and measures what ships; the in-tree UCB1 bandit is retired
  (git keeps it); `seeml-bench` schema 3 records policy, host key and
  host; `bench_compare.py` refuses cross-policy comparisons. E7 (#90)
  steps 1 and 2 landed with it — packages no longer bake the analytic
  tiling, and the bench runs the geometry a package ships with — leaving
  E7's contract rewrite and E1's re-premising.

## Materializing the board

```sh
brew install gh          # if needed
gh auth login            # then: gh auth refresh -s project
tool/create_next_project.sh
```

The script reconciles, so the bodies stay the source of truth after the
first run: it upserts every label and milestone the bodies name, files one
GitHub Issue per body in `issues/` and **edits** an existing one whose live
body, labels or milestone differ from the file (`--dry-run` lists what
would change), creates the
**“SeeML Two-Plane Overhaul”** GitHub Project with `Plane` / `Origin` /
`Priority` fields, and adds both the new issues and the existing
#59–#67/#69–#71 to the board.
