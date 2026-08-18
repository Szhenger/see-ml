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

One `.mm` file — `runtime/executor/metal_gemm.mm` — compiled with ARC
(`-fobjc-arc`) and linked against `-framework Metal -framework Foundation`,
gated on Apple hosts in both build drivers. MSL kernels are **not** checked in
as `.metal` files: the compiler emits MSL source as C++ string literals
(`compiler/backend/trainer/kernel_emitter.cc` — four kernels: `seeml_matmul`,
`seeml_matmul_nt`, `seeml_matmul_tn`, `seeml_gemm_acc`), and the harness JITs
them at runtime via `newLibraryWithSource:options:error:`. See §7 for scope.

### POSIX shell

Both build scripts are `#!/bin/sh` with `set -e` — POSIX, not bash: the
in-tree driver `build/build.sh` and the *generated* per-package `build.sh`
(emitted from a template in `compiler/backend/trainer/native_emitter.cc`,
which uses POSIX `${VAR-default}` expansion for the tile-flag override).

### Python 3

One file: `tool/export_model.py` (model export, build host only). CI pins
Python 3.12; the module imports `torch`/`numpy` function-locally so it
byte-compiles without them (CI exploits this with a torch-free
`py_compile` job). Nothing on the device path touches Python.

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

### Python: two packages, export-only

`tool/requirements.txt`: `torch>=1.7` (bound set by `nn.SiLU`) and
`numpy>=1.17` (bound set by `np.random.Generator`).

### System interfaces (all direct, no wrapper libraries)

| Interface | Where | Purpose |
|---|---|---|
| `std::thread` + `<mutex>`/`<condition_variable>`/`<atomic>` | `source/parallel/parallel_for.cc`, `runtime/feeder/batch_pipeline.cc` | Worker pool and feeder thread. No direct `pthread_*` calls; `-pthread` at compile and link. |
| POSIX file I/O: `open`/`write`/`fsync`/`close`, `rename`, directory fsync, `flock(LOCK_EX\|LOCK_NB)`, `fseeko`, `getpid` | `runtime/custodian/durable_io.cc` | Durable sidecar-then-atomic-rename writes, commit lock, random-access durable edits. Win32 mirror (`CreateFileA`, `MoveFileExA(MOVEFILE_REPLACE_EXISTING\|MOVEFILE_WRITE_THROUGH)`, …) exists but is not CI-tested. |
| `sysctlbyname("hw.l1dcachesize"/"hw.l2cachesize"/"hw.physicalcpu"/"hw.cachelinesize"/"hw.memsize")` | `compiler/backend/architecture/host_arch.cc`, `compiler/frontend/ingressor/resource_analyzer.cc` | Apple host cache/memory detection for GEMM tiling and memory gating. |
| `sysconf(_SC_LEVEL*_CACHE*, _SC_NPROCESSORS_ONLN, _SC_PHYS_PAGES)` + sysfs `/sys/devices/system/cpu/*/topology/` scan | same | Linux equivalents (topology scan is Linux-only). Fallback: `std::thread::hardware_concurrency()`. |
| `std::aligned_alloc(64, …)` | `runtime/engine/update_engine.cc` | The single arena allocation. (Unavailable on MSVC — one reason Windows is untested.) |
| `isatty(1)`, `localtime_r`/`localtime_s` | test runner, logger | Color gating, timestamps. |

Notably absent: no `mmap` (plans are read with `std::ifstream`); ISA detection
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
cross-compiling with `CXX`. The Metal path is *not* vendored.

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
- **Backend.** `architecture/host_arch` detects cache sizes and core counts
  (§2) and derives GEMM tilings; `tuner/` is a multi-armed-bandit autotuner
  over candidate tilings; `trainer/` binds every tensor to a compile-time
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
  `RowGrain`). GEMM tile sizes come from the baked `-DSEEML_GEMM_TILE_K/N`.
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
| SMF (model container) | `"SMF1"` | v4 (readers accept v1–v4) | `source/language/model_format.*`, `compiler/frontend/ingressor/model_{reader,writer}.cc`, Python writer in `tool/export_model.py` |
| SDS (dataset) | `"SDS1"` | v1 (feature rows) / v2 (token records) | `runtime/feeder/dataset.cc`, Python writer |
| SEEU (update plan) | `"SEEU"` | v7, oldest-readable v4 | written by `compiler/backend/trainer/*` + driver; read/validated by `runtime/validator` + `runtime/engine`; disassembled by `seeml-seeu-dump` |
| SEKP (checkpoint) | `"SEKP"` | v3 | `runtime/custodian/checkpoint.cc` |

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

The Metal integration is a hardware-validated **correctness harness**, not a
dispatch path: synchronous copy-in/copy-out GEMM via shared-mode
`MTLBuffer`s, one command buffer per call, MSL JIT-compiled from the
compiler-emitted source string. Contracts: bitwise-reproducible on the same
device, but *not* bitwise-equal to CPU kernels (different FMA contraction) —
cross-backend comparison is tolerance-based. The training engine does not yet
dispatch to it, and the file is excluded from vendored packages.

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
`e2e-package` (Python 3.12 + CPU torch wheels: export → compile → package
build → plan-seal grep → device run → serial re-run → `cmp` bitwise, with an
exit-3 gate-rejection retry path); `exporter-syntax` (torch-free
`py_compile`). Linux jobs install clang 19 from apt.llvm.org (see toolchain
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
| Metal GEMM harness, `sysctlbyname` detection | Apple only |
| sysfs core-topology scan, `_SC_LEVEL*` cache queries | Linux only |
| Windows | Code paths exist in `durable_io.cc` and MSVC flag arms in the build, but untested in CI, and `std::aligned_alloc` is unavailable on MSVC — treat as unsupported |
| Emitted packages | Any C++23 toolchain + POSIX I/O; cross-compile with `CXX=` and `SEEML_TILE_FLAGS=` |
