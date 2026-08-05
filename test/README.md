# The SeeML Test Tree

Organized in the same fashion as the `compiler/` and `runtime/` subsystems:
the harness and fixtures are partitioned per discipline behind façade
headers (structural vocabulary — *subsystem*, *discipline*, *unit*,
*façade header* — as defined in [docs/compiler.md](../docs/compiler.md)
and the [README glossary](../README.md#glossary)), and the suites mirror
the subsystem partition of the code they verify — a suite lives where its
subject lives.

```
test/
  framework/            SeeTest, the in-tree harness (façade: seetest.h)
    registry.{h,cc}     test registration + the runner (--list, --filter)
    matcher.h           value formatting, safe comparisons, failure messages
    assert.h            the TEST / EXPECT_* / ASSERT_* macro surface
    seetest_main.cc     the runner main linked into every suite
  support/              shared fixtures (façade: builders.h)
    models.cc           deterministic SMF (SeeML Model Format) model builders + BaseConfig
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

Every suite is one executable (`seeml_<basename>`), built by both
`build/build.sh` and CMake/CTest; run one directly with e.g.

```bash
./build/seeml_updater_test --filter=ConvLowering
```

Suites include only the façades (`test/framework/seetest.h`,
`test/support/builders.h`); the split units behind them can be reorganized
without touching any suite.
