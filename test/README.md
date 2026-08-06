# The SeeML Test Tree

## How do you trust a compiler?

Think about what SeeML claims: that it can differentiate your model correctly, plan every byte of memory without collisions, survive power cuts, and produce identical bits at any thread count. Extraordinary claims, and each one is *checkable* — which is what this tree is for. It's organized in the same fashion as the `compiler/` and `runtime/` subsystems: the harness and fixtures are partitioned per discipline behind façade headers, and the suites mirror the subsystem partition of the code they verify — **a suite lives where its subject lives**, so if you can find the code, you can find its tests, and vice versa.

```
test/
  framework/            SeeTest, the in-tree harness (façade: seetest.h)
    registry.{h,cc}     test registration + the runner (--list, --filter)
    matcher.h           value formatting, safe comparisons, failure messages
    assert.h            the TEST / EXPECT_* / ASSERT_* macro surface
    seetest_main.cc     the runner main linked into every suite
  support/              shared fixtures (façade: builders.h)
    models.cc           deterministic SMF model builders + BaseConfig
    corpora.cc          synthetic datasets (classification/regression/unlabeled)
    probes.cc           engine-arena introspection + the test-run environment
    scoped_temp_dir     filesystem sandbox for I/O suites
  source/               one folder per source-language subsystem
    parallel/           parallel_for
    identity/           hash
  compiler/             one folder per compiler subsystem
    frontend/           model_io  resource_analyzer  sir  operator  parser
    analysis/           update_passes  updater  reviewer
    backend/            tuner  trainer  native_emitter
    driver/             update_compiler  driver
    diagnostics/        diagnostics
  runtime/              one folder per runtime subsystem
    feeder/             dataset  batch_pipeline
    executor/           kernels
    validator/          validator
    custodian/          custodian
    engine/             engine  update_engine
  system/               the cross-half end-to-end update
    update_system_test
  fuzz/                 libFuzzer harnesses (built with -DSEEML_FUZZ=ON)
    binary_formats.cc
```

## Why an in-tree harness?

Fair question — GoogleTest exists. But recall the product's core constraint: the runtime is *zero-dependency*, and the project prizes knowing exactly what every byte of its code does. SeeTest is a few hundred lines implementing precisely what the suites need — a `TEST` macro that self-registers (each `TEST` block constructs a static registrar object before `main` runs, the classic trick), `EXPECT_*` (record the failure, keep going) versus `ASSERT_*` (this test can't meaningfully continue), a matcher layer that formats both sides of a failed comparison so failures read like sentences, and a runner with `--list` and `--filter`. Nothing more, because nothing more is needed — and when a test fails on some exotic target, the entire harness is right there to read.

The `support/` fixtures earn their keep the same way: **deterministic** model builders and synthetic corpora (seeded, so a failure reproduces exactly), `probes.cc` for peeking inside an engine's arena from a test, and `scoped_temp_dir`, which sandboxes every I/O suite into a temp directory that cleans itself up — tests that touch the filesystem must not be able to touch *your* filesystem.

## What the suites actually prove

Worth knowing the flavors, because they answer different questions:

- **Unit suites** pin one module's behavior — `sir` proves `Block::verify()` catches each corruption it claims to; `validator` proves each opcode's bounds math, including the overflow cases; `hash` pins the digests so an accidental algorithm change can't slip by.
- **Numeric suites** check kernels against independent references — and the autodiff machinery is verified by **finite-difference gradient checking**: nudge a parameter by ε, watch the loss, and compare `(L(θ+ε) − L(θ−ε)) / 2ε` against the gradient the compiled backward pass computes. Calculus, cross-examined by arithmetic. (This is why the compiler can emit a plan *without* the optimizer step — a pure forward+backward program exists precisely to be gradient-checked.)
- **Property/regression suites** pin the invariants the docs promise: the pipelined batch sequence is *exactly* the serial one; every instruction the compiler emits passes the runtime's validator; corrupted checkpoints are rejected; parallel results match serial bit-for-bit.
- **The system suite** (`system/update_system_test`) runs the whole story — export-shaped model in, train, gate, merge, commit — proving the two halves agree about every format and contract between them.
- **Fuzzing** (`fuzz/binary_formats.cc`, libFuzzer) feeds the SMF/SDS/plan parsers adversarial garbage for hours on end, hunting for any input that makes a bounds check lie. Run under sanitizers, it's the sharpest tool we have against parser bugs.

## Running things

Every suite is one executable (`seeml_<basename>`), built by both `build/build.sh` and CMake/CTest; run one directly with e.g.

```bash
./build/seeml_updater_test --filter=ConvLowering
```

Suites include only the façades (`test/framework/seetest.h`, `test/support/builders.h`); the split units behind them can be reorganized without touching any suite — the same façade discipline as the code under test.
