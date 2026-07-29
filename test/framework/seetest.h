#ifndef SEEML_TEST_FRAMEWORK_SEETEST_H_
#define SEEML_TEST_FRAMEWORK_SEETEST_H_

// =============================================================================
// SeeTest — the SeeML testing framework.
//
// A dependency-free harness in the spirit of GoogleTest, matching the
// codebase's C++23 / std::expected idiom:
//
//   TEST(Suite, Name) {
//     auto model = LoadSmf(path);
//     ASSERT_OK(model);                    // std::expected-aware
//     EXPECT_EQ(model->tensors.size(), 5u);
//     EXPECT_NEAR(loss, 1.386f, 1e-4);
//   }
//
// This is the façade; the framework is partitioned per discipline, in the
// fashion of the compiler and runtime subsystems:
//   registry.{h,cc}  test registration and the runner (--list, --filter,
//                    per-test failure state, the process exit code)
//   matcher.h        value formatting, warning-safe comparisons, and
//                    failure-message construction
//   assert.h         the TEST / EXPECT_* / ASSERT_* macro surface
//
// Tests self-register via static initializers; link a suite's .cc files
// with seetest_main.cc to obtain the runner. Suites include only this
// header.
// =============================================================================

#include "test/framework/assert.h"    // IWYU pragma: export
#include "test/framework/matcher.h"   // IWYU pragma: export
#include "test/framework/registry.h"  // IWYU pragma: export

#endif  // SEEML_TEST_FRAMEWORK_SEETEST_H_
