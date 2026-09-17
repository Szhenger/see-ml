# SeeML Technical Documentation

This document enumerates every programming language, framework, library, system
interface, and piece of tooling used to build SeeML, and describes the compiler
and runtime architectures at the level a contributing (or forking) software
engineer needs. It is a technology specification, not a machine-learning
tutorial — for the concepts the system implements, see `docs/compiler.md` and
`docs/runtime.md`.

Repository shape: ~225 tracked files — 83 `.cc`, 69 `.h`, 1 `.mm`, 1 `.py`,
1 `.sh`, 3 `.yml`. Top-level directories: `compiler/`, `runtime/`, `source/`,
`tool/`, `test/`, `build/`, `docs/`, `.github/`.

---

## 1. Languages

### C++23 (ISO, no extensions)

The entire product — compiler, runtime, and both CLI tools — is C++23,
compiled with `-std=c++23 -O2 -Wall -Wextra -Werror -pthread` (strict ISO:
`CMAKE_CXX_EXTENSIONS OFF`; MSVC arms use `/W4 /WX`). The standard was chosen
for exactly one feature: **`std::expected`**, which is the universal error
idiom — every fallible API returns `std::expected<T, std::string>`, and error
builders return `std::unexpected` (`compiler/diagnostics/diagnostic.h`).

Features actually in use, so you know the house style before writing code:

| Standard | Features used |
|---|---|
| C++23 | `std::expected` / `std::unexpected` (pervasive; the reason for C++23) |
| C++20 | `std::span`, `std::source_location` (logging/diagnostics), `std::bit_cast`, `std::endian::native` static asserts (one per binary format), designated initializers (idiomatic, ~64 sites), `starts_with`/`contains`, `std::erase_if`; exactly one `concept` (`Streamable`, in the test framework) |
| C++17 and earlier | `std::string_view`, `std::optional`, `std::variant`, `std::filesystem` (compiler/tools/tests only — **never** in the vendored runtime), `<thread>`/`<mutex>`/`<atomic>`/`<condition_variable>` |

Deliberately **not** used: `<ranges>`, `<format>`/`std::print` (formatting is
`printf`-family with `<cinttypes>` macros, or `ostringstream`), `<mdspan>`,
`std::jthread`, coroutines, `<=>`, SIMD intrinsics (no `<immintrin.h>` /
`<arm_neon.h>` — vectorization is left to the optimizer, aided by a
`SEEML_RESTRICT` macro wrapping `__restrict__`). Exceptions and RTTI are on;
the thread pool propagates exceptions via `std::exception_ptr`.

Naming and layout are Google-style: `kConstant`, `PascalCase` functions,
trailing-underscore members, anonymous namespaces for TU-local state,
repo-root-relative includes (`#include "compiler/frontend/..."`, enabled by
`-I.`). Where a discipline spans several files, a single façade header
re-exports it with `// IWYU pragma: export` (`sir.h`, `update_passes.h`,
`update_types.h`, `update_kernels.h`, `seetest.h`, `builders.h`). A `.clangd`
file configures the language server with the same flags.

### Objective-C++ and Metal Shading Language (Apple-only)

Two `.mm` files — `runtime/executor/metal_backend.mm` (the executor
backend, §7) and `runtime/executor/metal_gemm.mm` (the G1a correctness
harness) — compiled with ARC (`-fobjc-arc`) and linked against
`-framework Metal -framework Foundation`, gated on Apple hosts in both build
drivers and in the emitted package's `build.sh` (`SEEML_NO_METAL=1` opts
out; `metal_backend_stub.cc` takes the backend's place everywhere else, so no
non-Apple translation unit references a Metal symbol). MSL kernels are
**not** checked in as `.metal` files: the backend's kernel library is a C++
string literal owned by the runtime (`runtime/executor/metal_kernels.h`, so
CPU and GPU kernel semantics are versioned together), the harness's four
GEMMs are emitted by `compiler/backend/trainer/kernel_emitter.cc`, and both
are JIT-compiled at runtime via `newLibraryWithSource:options:error:` — no
Metal toolchain is needed to build anything.

### POSIX shell

Both build scripts are `#!/bin/sh` with `set -e` — POSIX, not bash: the
in-tree driver `build/build.sh` and the *generated* per-package `build.sh`
(emitted from a template in `compiler/backend/trainer/native_emitter.cc`,
which uses POSIX `${VAR-default}` expansion for the tile-flag override).

### Python 3

Build host only — the **Python plane** of the two-plane design
(`docs/next-project/README.md`): the emitted package and the runtime are
dependency-free C++ under the full doctrine, and the build host carries
Python for what that doctrine has no reason to constrain — export,
packaging, measurement, autotuning, certification, reference execution.
Three dependency tiers. Standard library only, under a bare interpreter:
`tool/pack_update.py` (the package assembler), `tool/autotune.py` (the
offline kernel-policy tuner), `tool/bench_compare.py` (the bench gate) and
`certify_numerics.py verify`. NumPy: `tool/frontier_exec.py` (the plan
interpreter and differential oracle), `tool/certify_numerics.py certify`,
and `export_model.py --hf` / `--demo-decoder`. torch (and optionally MLX):
`tool/export_model.py`'s `nn.Module` path and the frontier backends, all
imported function-locally so every tool byte-compiles without them. The Python plane is developed and gated on the
pinned stack in `tool/requirements-pinned.txt` — **CPython 3.14.7, torch
2.14.0, NumPy 2.5.3** — and stays runnable down to the floors in
`tool/requirements.txt` (Python 3.9, NumPy 1.17, torch 1.7): no 3.10+
syntax in any tool, and CI runs the tools on both 3.9 and 3.14.7
(`python-tools`), the exporter's byte-oracle suite on NumPy 1.x/3.9 and
NumPy 2.5/3.14.7 (`exporter-compat`), and the e2e job on the pinned stack.
Nothing on the device path touches Python: the packer's output is an
assembly stub the package's own `build.sh` assembles.

---

## 2. Dependencies

### C++: zero third-party dependencies

Every non-project include in `compiler/`, `runtime/`, `source/`, `tool/`, and
`test/` is a C++/C standard header, a POSIX header, a Win32 header, or an
Apple framework header. There is no vendored third-party code, no
`FetchContent`, no submodules, and the only `find_package` is
`Threads`. This is a design pillar: the emitted package must build on a
machine that has never seen this repository, so the runtime's dependency
surface is the C++23 standard library plus POSIX file I/O.

### Python: two required packages, build host only

`tool/requirements.txt` states the floors: `torch>=1.7` (bound set by
`nn.SiLU`) and `numpy>=1.17` (bound set by `np.random.Generator`).
`tool/requirements-pinned.txt` states the gated stack (`torch==2.14.0`,
`numpy==2.5.3`, on CPython 3.14.7). The exporter writes every multi-byte
field with an explicit little-endian dtype, streams the SMF container
(header, then each tensor straight from the array that holds it, at its
64-byte offset) and writes SDS corpora as chunked packed records, so a
model exports at about its own size in memory and a corpus at constant
memory; `test/tool/export_model_test.py` holds the original per-row and
whole-blob algorithms as byte oracles, and pins the default demos'
digests (`test/tool/demo_digests.json`).

### System interfaces (all direct, no wrapper libraries)

| Interface | Where | Purpose |
|---|---|---|
| `std::thread` + `<mutex>`/`<condition_variable>`/`<atomic>` | `source/parallel/parallel_for.cc`, `runtime/feeder/batch_pipeline.cc` | Worker pool and feeder thread. No direct `pthread_*` calls; `-pthread` at compile and link. |
| POSIX file I/O: `open`/`write`/`fsync`/`close`, `rename`, directory fsync, `flock(LOCK_EX\|LOCK_NB)`, `fseeko`, `getpid` | `runtime/custodian/durable_io.cc` | Durable sidecar-then-atomic-rename writes, commit lock, random-access durable edits. Win32 mirror (`CreateFileA`, `MoveFileExA(MOVEFILE_REPLACE_EXISTING\|MOVEFILE_WRITE_THROUGH)`, …) exists but is not CI-tested. |
| `sysctlbyname("hw.l1dcachesize"/"hw.l2cachesize"/"hw.physicalcpu"/"hw.cachelinesize"/"hw.memsize")` | `compiler/backend/architecture/host_arch.cc`, `compiler/frontend/ingressor/resource_analyzer.cc` | Apple host cache/memory detection for GEMM tiling and memory gating. |
| `sysconf(_SC_LEVEL*_CACHE*, _SC_NPROCESSORS_ONLN, _SC_PHYS_PAGES)` + sysfs `/sys/devices/system/cpu/*/topology/` scan | same | Linux equivalents (topology scan is Linux-only). Fallback: `std::thread::hardware_concurrency()`. |
| `std::aligned_alloc(64, …)` | `runtime/engine/update_engine.cc` | The single arena allocation. (Unavailable on MSVC — one reason Windows is untested.) |
| `isatty(1)`, `localtime_r`/`localtime_s` | test runner, logger | Color gating, timestamps. |

`mmap` appears on the build host only: the compiler maps the source model
read-only (`compiler/frontend/ingressor/model_reader.cc`) and gives large
tensor payloads their own anonymous mappings (`source/language/model_format.h`)
so that releasing one really returns its pages; both fall back to the heap
off POSIX. The device runtime maps nothing — a plan file is read whole into
one heap buffer, or borrowed from the binary image. ISA detection
is compile-time (`__aarch64__`, `__AVX2__`, `__AVX512F__`, `__FMA__` macros),
not runtime CPUID.

---

## 3. Build systems

Two coequal build paths produce identical artifacts; CI treats the shell
driver as canonical.

**CMake ≥ 3.20** (`CMakeLists.txt`): `LANGUAGES CXX` only — the `.mm` file is
compiled through the CXX driver by source property rather than enabling
OBJCXX. The release version is *extracted* from
`source/identity/version.h` (`kSeemlVersion`, the single source of truth) with
`file(STRINGS …)` + regex, failing configuration if unparseable. Targets:
four static libraries (`seeml_parallel`, `seeml_sir`, `seeml_update`,
`seeml_update_rt`), two executables (`seeml-update-compile`,
`seeml-seeu-dump`), a `seeml_testing` support library, 25 test executables
registered with `add_test`, and an opt-in fuzz target. Options:

- `SEEML_SANITIZE` (semicolon list → `-fsanitize=<list> -fno-omit-frame-pointer -g`),
  used as `"address;undefined"` and `"thread"` in CI.
- `SEEML_FUZZ=ON` → `seeml_fuzz_formats` with `-fsanitize=fuzzer,address` (clang only).
- `SEEML_SOURCE_DIR` — repo path baked into test binaries so the emitter
  suite can vendor real runtime sources.

**`build/build.sh`**: pure POSIX sh, `CXX="${CXX:-c++}"`, compiles ~60
translation units serially into `build/*.o` and links loose objects (no `ar`)
into the same tools and 25 suites. This is the path used by the CI matrix and
by CodeQL's manual build tracing, and the only path on hosts without CMake.

**Toolchain floor** (documented in `README.md`, enforced by CI pins): GCC 13+,
Apple Clang 15+, or Clang 19+ with libstdc++ — libstdc++ hides `<expected>`
behind `__cpp_concepts >= 202002L`, a macro Clang only defines from 19.

**The generated package build** — the third, outward-facing build system:
`EmitNativePackage` writes `update_plan.seeu`, the same plan embedded as an
aligned C array TU (emission parallelized, byte-identical to serial), a
generated driver `update_main.cc`, 35 vendored runtime/`source/` files, and a
`build.sh` that compiles them with `-std=c++23 -O2 -pthread` (no `-Werror` in
the field). Host-derived GEMM tile geometry is baked in as
`-DSEEML_GEMM_TILE_K/N` defines, overridable via `SEEML_TILE_FLAGS` when
cross-compiling with `CXX`. The Metal backend *is* vendored: `build.sh`
compiles its Objective-C++ unit and links the frameworks on Darwin, and the
stub elsewhere, so a Linux package is byte-for-byte what it was.

---

## 4. Compiler architecture (`compiler/`)

Layout is subsystems-by-role: `frontend/` → `analysis/` → `backend/` →
`driver/`, plus `diagnostics/`.

- **Frontend.** `ingressor/` reads and writes the SMF model container with a
  never-trust-a-file discipline (bounds and arithmetic-overflow checks before
  any allocation) and a `resource_analyzer` that gates infeasible memory
  footprints against host RAM. `representation/` is **SIR**, the in-memory
  IR: typed values, operations with string mnemonics (e.g. `sc_high.conv2d`),
  attribute maps, and blocks; operator builders live in `operator/`.
  `parser/` performs semantic analysis and shape inference, turning an
  ingested container into a verified SIR graph.
- **Analysis.** A `pass_manager` that re-verifies invariants between passes.
  Passes: `conv_lowering` (conv2d → im2col + GEMM rewrite; grouped/dilated
  forms rejected), `dce`, `epilogue_fuser`, `lora_grafter` (adapter
  insertion), `merge_builder` (the delta-materialization program),
  `autodiff` (reverse-mode differentiation over SIR), `optimizer` synthesis
  (SGD/AdamW as instructions), and a `quantization` reviewer (int8 frozen
  base).
- **Backend.** `architecture/host_arch` detects cache sizes, core counts
  and the CPU model (§2), forms the host key a kernel-policy table is keyed
  on, and derives the analytic GEMM tiling (a measured arm, not a
  decision); `tuner/kernel_policy_table` reads the host-keyed table the
  offline tuner (`tool/autotune.py`) wrote — a strict purpose-built JSON
  reader — and resolves the CPU GEMM tiles the driver writes into the plan
  header (v11): `--gemm-tiles`, then the table, then the runtime defaults;
  `trainer/` binds every tensor to a compile-time
  arena layout (`arena_binder`), lowers SIR to the fixed ~35-opcode
  instruction ISA (`instruction_lowering`), emits MSL kernel source
  (`kernel_emitter`), and emits the self-contained package
  (`native_emitter`).
- **Driver.** `update_compiler.cc` orchestrates the phases under explicit
  checked contracts (`contract.cc`).
- **Diagnostics.** Six header-only process modules (`tokenizing`, `parsing`,
  `passing`, `updating`, `architecting`, `generating`); every failure is one
  line, `"<unit>: <message>"`, returned as `std::unexpected` — plus the one
  stateful `logger` (atomic level, ANSI color, mutex only around the write).

## 5. Runtime architecture (`runtime/`)

A zero-dependency virtual machine executing the compiled plan. Subsystems:

- **engine/** — loads the plan (hash check first), runs the validator, makes
  the single `std::aligned_alloc(64, …)` arena allocation, then executes the
  three straight-line instruction programs (train / eval / merge) and
  orchestrates gate → merge → commit.
- **validator/** — load-time proof: every instruction operand is
  bounds-checked against arena/rodata geometry *before* execution; `Execute()`
  then runs unchecked by contract.
- **executor/** — the kernel library (`gemm`, `elementwise`, `activation`,
  `normalization`, `loss`, `optimizer`, `attention`), each parallelized over
  the shared `ParallelFor` substrate with shape-derived chunking; grain
  policy in `kernel_policy.h` (`kGrainCheap=32768`, `kGrainMath=4096`,
  `RowGrain`). The CPU GEMM tile geometry (`GemmTiles`) arrives per plan
  from the header (v11) through `ExecutorBackend::Configure`; zero fields
  select the compiled-in defaults (`-DSEEML_GEMM_TILE_K/N`, 64 and 256).
  Tiles change throughput only, never bits.
- **feeder/** — `dataset` (SDS validation, seeded per-epoch permutation) and
  `batch_pipeline` (one producer thread staging batch *s+1* during step *s*
  over a mutex + condvar; provably identical batch sequence to serial, no
  thread at width 1).
- **custodian/** — `durable_io` (fsync'd sidecar + atomic rename, commit
  lock, durable random-access edit) and `checkpoint` (hash-bound, resumable).
- **diagnostics/** — a runtime-local mirror of the compiler's process-module
  idiom, kept separate so vendored packages never include compiler headers.

### The concurrency substrate (`source/parallel/`)

`ParallelFor(n, grain, body)` is shared by both halves and defines the
determinism contract: chunk boundaries are a pure function of `(n, grain)` —
never thread count — capped at `kMaxParallelChunks = 256` so reductions use
fixed stack arrays; partials are folded in chunk order. Thread count comes
from `SetParallelThreadCount` > `SEEML_THREADS` (parsed with `strtoll`
specifically to reject negatives that `strtoul` would wrap) >
`hardware_concurrency()`. The pool is a persistent, intentionally leaked
singleton; the calling thread participates; chunks are claimed by atomic
counter (dynamic load balance without affecting results); nested calls run
inline via `thread_local` flags; a failed worker spawn degrades width rather
than aborting. `SEEML_THREADS=1` never creates a thread. `SEEML_THREADS` is
the only environment variable the C++ reads.

### Shared vocabulary (`source/`)

`source/` exists so the halves agree on bytes without depending on each
other: `identity/` (release version; FNV-1a hashing), `language/` (SMF
container structs), `plan/` (the SEEU ABI: `#pragma pack(1)` structs with
size static-asserts), `parallel/` (above).

---

## 6. Binary formats and integrity

Four little-endian formats, each with a four-byte ASCII magic and an explicit
version-negotiation policy (additive changes bump the version; semantic breaks
raise the oldest-readable floor; newer-than-reader is always rejected):

| Format | Magic | Current version | Implemented in |
|---|---|---|---|
| SMF (model container) | `"SMF1"` | v5 (readers accept v1–v5; writers emit the lowest version the model needs) | `source/language/model_format.*`, `compiler/frontend/ingressor/model_{reader,writer}.cc`, Python writer in `tool/export_model.py` |
| SDS (dataset) | `"SDS1"` | v1 (feature rows) / v2 (token records) | `runtime/feeder/dataset.{h,cc}`, Python writer |
| SEEU (update plan) | `"SEEU"` | v14, oldest-readable v4 | written by `compiler/backend/trainer/*` + driver; read/validated by `runtime/validator` + `runtime/engine`; disassembled by `seeml-seeu-dump` |
| SEKP (checkpoint) | `"SEKP"` | v4, oldest-readable v3 | `runtime/custodian/checkpoint_format.h`, `runtime/custodian/checkpoint.cc` |

The Python plane restates these layouts exactly once, in
`tool/seeml/formats.py`, and that restatement is held to the headers by
machine: `seeml-abi` prints every magic, version, enum and packed-struct
offset from the C++ side, the output is committed as `tool/seeml/abi.json`,
CI regenerates it and fails on a diff, and `test/tool/formats_test.py`
compares the Python declarations — and this table's versions — against it.
Golden SMF/SDS files under `test/fixtures/golden/` are read value-for-value
by a C++ suite and byte-compared against a fresh export by the Python one;
plans, whose persist-init bytes depend on the host's libm, are instead
compiled on the spot and decoded twice (`seeml-seeu-dump --json` and
`frontier_exec.Plan`), field for field.

Integrity is 64-bit FNV-1a in three forms (`source/identity/hash.h`): plain
incremental; `StripedFnv1a64` (8 interleaved lanes to break the serial
multiply dependency chain); and `ContentHash64` (1 MiB chunks over
`ParallelFor`, folded in chunk order — bitwise-identical at any thread
count), plus `PlanSelfHash` (the seal field zeroed within the sealed bytes)
and a streaming file variant with one-chunk peak memory. Explicitly
non-cryptographic: corruption/mismatch detection only; authentication belongs
to the update transport.

---

## 7. GPU status (Apple Metal)

The Metal integration is an opt-in **executor backend** behind the
`ExecutorBackend` seam (`runtime/executor/backend.h`): `model_update
--backend cpu|metal|auto` (or `$SEEML_BACKEND`; default `cpu`). The engine
decodes, validates and sequences the plan; the backend executes it.

- **Residency.** The engine's arena is page-aligned (16 KiB) and wrapped
  once as a shared `MTLBuffer` — the CPU and GPU read and write the same
  pages, so the loss slot, checkpoints and merge deltas need no copies. The
  plan's rodata section starts on a page boundary and the blob is page-padded
  (`kSeeuRodataAlignment`, a layout property, no version bump), so a
  page-aligned, writable embedded plan (the `.incbin` stub, `__DATA,__data`
  on Apple) is wrapped zero-copy too; a heap-resident plan is copied once at
  load, and the device label says which.
- **Batching.** Consecutive GPU instructions encode into one command buffer
  through one serial compute encoder; a CPU-resident instruction (the three
  loss families, the embedding gather) waits only for pending GPU work it
  depends on, decided from the validator's operand extents
  (`DescribeInstruction`) — dependency tracking is never looser than the
  bounds proof. A command buffer that ends in any state but Completed is an
  executor diagnostic, never a silent fallback.
- **Coverage.** The GEMM family (NN/NT/TN/accumulate, the int8 and bf16
  variants, fused bias+activation epilogues) runs as `simdgroup_matrix`
  64×64 tiles — register-prefetched 4-vector loads, K split across
  threadgroups when a shape has too few tiles (partials summed in a fixed
  order) — or, for skinny shapes (an adapter's rank-8 factors, a K=8
  product), as one of three register-blocked forms with fixed shuffle-tree
  reductions. Every GEMM kernel is strided and batched, and the attention
  family is expressed through it: `Q Kᵀ`, `P V`, `dO Vᵀ`, `Pᵀ dO`, `dS K`,
  `dSᵀ Q` are batched GEMMs over the `[B·S, H·d]` activations plus a causal
  row-softmax pair. Elementwise, RoPE, LayerNorm/RMSNorm (one simdgroup
  per row), ReduceRows, ClipNorm (the CPU's chunk geometry, partials
  combined in chunk order), SGD and AdamW complete the library; the loss
  families and the embedding gather stay on the CPU.
- **Determinism is per-backend.** A backend is bitwise-reproducible against
  itself (tested run-to-run); CPU and GPU compare at tolerance (tested on
  every program family the compiler emits), and the backend is recorded in
  the banner, the gate line, `--report`, and `bench.json`. `--backend cpu`
  is the reference and is bit-identical to the pre-backend runtime.

The G1a copy-in/copy-out harness (`metal_gemm.mm`) remains as the hardware
test of the compiler-emitted GEMM source.

---

## 8. Testing and quality tooling

- **SeeTest** (`test/framework/`) — the in-tree, few-hundred-line test
  framework (the zero-dependency doctrine applied to testing): static-object
  test registration defeating init-order issues, GoogleTest-shaped output,
  `EXPECT_*`/`ASSERT_*` including `std::expected`-aware `_OK`/`_ERROR`
  variants and `ASSERT_OK_AND_ASSIGN`, a `--filter` with wildcard/substring
  semantics, ANSI color via `isatty`. Exit codes: 0 pass / 1 fail / 2 usage
  or empty selection. 25 suite executables mirror the source tree
  one-to-one; 345 tests at v1.2.3.
- **Fixtures** (`test/support/`) — seeded in-process builders for models,
  corpora, and arena probes; a RAII temp-dir. There are no checked-in binary
  golden files; determinism claims are tested as serial-vs-parallel bitwise
  equality, and autodiff is verified against central finite differences.
- **Sanitizers** — ASan+UBSan per-PR, TSan nightly (the mechanical proof the
  pool and feeder are race-free), via `SEEML_SANITIZE`.
- **Fuzzing** — one libFuzzer target (`test/fuzz/binary_formats.cc`) whose
  first input byte selects among four arms: SMF loader, SEEU loader+validator
  (re-sealing the mutated blob's hash so the integrity gate doesn't starve
  the deeper validators), SDS loader, and full compile-of-loaded-SMF.
  Contract: hostile bytes may be rejected, never crash.
- **CodeQL** — `security-and-quality` C/C++ queries over a manual-mode traced
  `build/build.sh`, per-PR and weekly.

## 9. Continuous integration (GitHub Actions)

`ci.yml` (PR + main, cancel-in-progress concurrency):
`build-and-test` matrix {ubuntu/g++-14, ubuntu/clang++-19, macOS/clang++};
`determinism` (full suites at `SEEML_THREADS` ∈ {1, 3, 8});
`asan-ubsan`; `fuzz-smoke` (90 s, crash artifacts uploaded);
`e2e-package` (Python 3.14.7 + the pinned CPU torch 2.14 stack: exporter
suite → export → compile → package
build → plan-seal grep → device run → serial re-run → `cmp` bitwise, with an
exit-3 gate-rejection retry path, then the `--no-embed` + `pack_update.py`
route whose stub-built binary must commit the same bytes); `python-tools`
(torch-free `py_compile` of every script + the packer's unit suite, on
3.9 and 3.14.7); `exporter-compat` (NumPy-only exporter suite on NumPy 1.x
/ 3.9 and NumPy 2.5 / 3.14.7). Linux jobs install clang 19 from apt.llvm.org (see toolchain
floor, §3). `nightly.yml`: TSan full suite; 900 s fuzz with an
ever-accumulating corpus via `actions/cache` restore-key chaining. There is
no release automation; versioning is a manual edit of
`source/identity/version.h`, from which everything else derives.

## 10. Command-line tools

- **`seeml-update-compile`** — the compiler driver. Strict argument cursor
  (unknown flag, missing value, trailing garbage, or numeric overflow is a
  hard error, never a default), optional machine-readable JSON `--report`
  (written with `ferror`+`fclose` checked so a truncated report cannot exit
  0), `--build` shelling out (`std::system`, the codebase's only process
  spawn) to the generated package script.
- **`seeml-seeu-dump`** — plan disassembler: verifies the integrity seal,
  prints header/sections, and with `--instrs` disassembles the three
  instruction streams with symbolic opcode names. Depends only on
  `source/plan` + `source/identity` (and the parallel lib for the hash) so it
  builds anywhere.
- **`tool/pack_update.py`** — the package assembler (Two-Plane Overhaul
  P1): embeds `update_plan.seeu` as a page-aligned `.incbin` assembly stub
  (`update_plan_embedded.S`) in place of the decimal C-array TU the compiler
  emits by default (`seeml-update-compile --no-embed` skips that TU), and
  with `--build` drives the package's `build.sh`, which assembles the stub
  when present. Stdlib only; strict CLI (exit 2); atomic temp-and-rename
  writes; exit 1 on any packaging or build failure.
- **`tool/autotune.py`** — the offline autotuner (Two-Plane Overhaul P2):
  sweeps CPU GEMM tile arms through `seeml-bench --gemm-tiles`, round-robin
  over rounds, scores each arm by the geometric mean of its rows/s relative
  to the kernel defaults (medians of medians), and persists the winner —
  or the defaults, when nothing beats them by `--min-gain` — in a JSON
  table keyed on the bench-reported host key; `show` prints a table.
  Stdlib only; exit 2 on usage, 1 on a failed bench run or an unreadable
  table; atomic writes.
- **`tool/export_model.py`** — PyTorch/NumPy → SMF/SDS exporters
  (`export_smf`, `export_decoder_smf`, `export_token_decoder_smf`,
  `export_sds`, `export_token_sds`) plus `--demo` / `--demo-decoder`
  generators; unsupported modules raise a loud `ValueError`. All byte packing
  is explicit little-endian `struct` format strings.

## 11. On-device contract (what an emitted package guarantees)

The generated `model_update` binary encodes the update lifecycle: load +
hash-verify + validate → split/shuffle (seeded, thread-count-invariant) →
train (interruptible; hash-bound fsync-durable checkpoints; bitwise-identical
resume) → regression gate → merge + atomic commit. Exit codes are the
orchestration API: `0` committed, `1` runtime error, `2` bad arguments, `3`
gate rejection with the device left untouched. Determinism across
`SEEML_THREADS` is a tested contract, not an aspiration.

## 12. Platform support matrix

| Component | Platforms |
|---|---|
| Compiler + runtime + tools | macOS (Apple Clang 15+), Linux (GCC 13+ / Clang 19+ with libstdc++) — both CI-tested |
| Metal executor backend (`--backend metal|auto`), Metal GEMM harness, `sysctlbyname` detection | Apple only |
| sysfs core-topology scan, `_SC_LEVEL*` cache queries | Linux only |
| Windows | Code paths exist in `durable_io.cc` and MSVC flag arms in the build, but untested in CI, and `std::aligned_alloc` is unavailable on MSVC — treat as unsupported |
| Emitted packages | Any C++23 toolchain + POSIX I/O; cross-compile with `CXX=` and `SEEML_TILE_FLAGS=` |
