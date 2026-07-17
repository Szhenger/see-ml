# SeeML Test Module

A dependency-free testing framework and per-module test suites for the update
compiler and runtime. No external test library is required — the framework
(`SeeTest`) is ~300 lines of in-tree C++23 matching the codebase's
`std::expected` error-handling idiom.

## Layout

```
test/
├── framework/            The SeeTest framework
│   ├── seetest.h         TEST() macro, EXPECT_*/ASSERT_* assertions
│   ├── seetest.cc        registry + runner (--list, --filter=PATTERN)
│   └── seetest_main.cc   main() linked into every suite executable
├── support/              Shared fixtures
│   ├── scoped_temp_dir.* per-test unique temp directories (auto-cleaned)
│   └── builders.*        deterministic MLPs, synthetic datasets, arena helpers
├── source/
│   └── smf_test.cc           SMF container: round-trips, corrupt-file rejection
├── compiler/
│   ├── sir_test.cc            SIR core: shapes, use-def chains, block surgery
│   ├── forward_builder_test.cc SMF→SIR import + semantic analysis
│   ├── update_passes_test.cc   LoRA grafting, autodiff, optimizer, merge passes
│   ├── update_compiler_test.cc plan assembly, per-loss config, error surface
│   └── native_emitter_test.cc  native package emission
├── runtime/
│   ├── kernels_test.cc        every reference kernel vs. hand-computed values
│   ├── dataset_test.cc        SDS format: batching, validation, file I/O
│   └── update_engine_test.cc  plan ingestion hardening, checkpoints, commit
└── system/
    └── update_system_test.cc  end-to-end numerics: step-0 identity,
                               finite-difference gradient check,
                               train→merge→commit, MSE / KL / composite losses
```

## Running

```sh
cmake -B build && cmake --build build
ctest --test-dir build                    # all suites
ctest --test-dir build -R seeml_smf_test  # one suite via CTest

./build/seeml_kernels_test                        # run a suite directly
./build/seeml_kernels_test --list                 # enumerate its tests
./build/seeml_kernels_test --filter=Gemm          # substring filter
./build/seeml_smf_test '--filter=Smf.*RoundTrip'  # '*' wildcards
```

## Writing tests

```cpp
#include "test/framework/seetest.h"

TEST(MySuite, MyCase) {
  auto model = LoadSmf(path);
  ASSERT_OK(model);                          // std::expected-aware; fatal
  ASSERT_OK_AND_ASSIGN(auto data, Dataset::LoadFromFile(p));  // unwraps
  EXPECT_EQ(model->tensors.size(), 5u);      // non-fatal, mixed-sign safe
  EXPECT_NEAR(loss, 1.386f, 1e-4);
  EXPECT_ERROR_CONTAINS(LoadSmf("nope"), "cannot open");
  EXPECT_STR_CONTAINS(compiled.sir_dump, "sc_high.matmul");
}
```

- `EXPECT_*` records the failure and continues; `ASSERT_*` returns from the
  test function, so use it only in the TEST body itself.
- Tests must be deterministic: seed every RNG, and use `ScopedTempDir` for
  any file I/O so suites can run in parallel.
- New suites: add the `.cc` under the matching module directory and register
  it in the top-level `CMakeLists.txt` with `seeml_add_test(...)`.

Not yet covered: the `seeml-update-compile` CLI driver (`tool/`) and the
generated package's `build.sh` (both need a shell-level harness), and
`tool/export_model.py` (needs a Python + PyTorch environment).
