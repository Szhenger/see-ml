#ifndef SEEML_TEST_FRAMEWORK_ASSERT_H_
#define SEEML_TEST_FRAMEWORK_ASSERT_H_

#include <cmath>
#include <string>
#include <string_view>
#include <utility>

#include "test/framework/matcher.h"
#include "test/framework/registry.h"

// =============================================================================
// assert/ discipline of SeeTest: the macro surface suites write against —
// TEST registration, boolean/binary/proximity assertions, and the
// std::expected-aware forms matching the codebase's error-handling idiom.
// EXPECT_* records the failure and continues; ASSERT_* records and returns
// from the test function (so it must be used in the TEST body itself, not
// in helpers with non-void returns).
// =============================================================================

#define SEETEST_CAT_INNER_(a, b) a##b
#define SEETEST_CAT_(a, b) SEETEST_CAT_INNER_(a, b)

#define TEST(Suite, Name)                                                    \
  static void SeeTest_##Suite##_##Name();                                    \
  static const ::seeml::testing::internal::Registrar                        \
      seetest_registrar_##Suite##_##Name{#Suite, #Name,                      \
                                         &SeeTest_##Suite##_##Name};         \
  static void SeeTest_##Suite##_##Name()

// --- Explicit failures ---------------------------------------------------------

#define ADD_FAILURE(msg) \
  ::seeml::testing::ReportFailure(__FILE__, __LINE__, (msg))

#define FAIL(msg)         \
  do {                    \
    ADD_FAILURE(msg);     \
    return;               \
  } while (0)

// --- Boolean assertions ----------------------------------------------------------

#define SEETEST_BOOL_(cond, text, fatal)  \
  do {                                    \
    if (!(cond)) {                        \
      ADD_FAILURE(text);                  \
      if (fatal) return;                  \
    }                                     \
  } while (0)

#define EXPECT_TRUE(cond) SEETEST_BOOL_((cond), "expected true: " #cond, false)
#define ASSERT_TRUE(cond) SEETEST_BOOL_((cond), "expected true: " #cond, true)
#define EXPECT_FALSE(cond) \
  SEETEST_BOOL_(!(cond), "expected false: " #cond, false)
#define ASSERT_FALSE(cond) \
  SEETEST_BOOL_(!(cond), "expected false: " #cond, true)

// --- Binary comparisons ------------------------------------------------------------

#define SEETEST_CMP_(fn, sym, a, b, fatal)                                  \
  do {                                                                      \
    const auto& seetest_a_ = (a);                                           \
    const auto& seetest_b_ = (b);                                           \
    if (!::seeml::testing::internal::fn(seetest_a_, seetest_b_)) {          \
      ADD_FAILURE(::seeml::testing::internal::BinaryFailure(                \
          #a, #b, sym, seetest_a_, seetest_b_));                            \
      if (fatal) return;                                                    \
    }                                                                       \
  } while (0)

#define EXPECT_EQ(a, b) SEETEST_CMP_(CmpEq, "==", a, b, false)
#define ASSERT_EQ(a, b) SEETEST_CMP_(CmpEq, "==", a, b, true)
#define EXPECT_NE(a, b) SEETEST_CMP_(CmpNe, "!=", a, b, false)
#define ASSERT_NE(a, b) SEETEST_CMP_(CmpNe, "!=", a, b, true)
#define EXPECT_LT(a, b) SEETEST_CMP_(CmpLt, "<", a, b, false)
#define ASSERT_LT(a, b) SEETEST_CMP_(CmpLt, "<", a, b, true)
#define EXPECT_LE(a, b) SEETEST_CMP_(CmpLe, "<=", a, b, false)
#define ASSERT_LE(a, b) SEETEST_CMP_(CmpLe, "<=", a, b, true)
#define EXPECT_GT(a, b) SEETEST_CMP_(CmpGt, ">", a, b, false)
#define ASSERT_GT(a, b) SEETEST_CMP_(CmpGt, ">", a, b, true)
#define EXPECT_GE(a, b) SEETEST_CMP_(CmpGe, ">=", a, b, false)
#define ASSERT_GE(a, b) SEETEST_CMP_(CmpGe, ">=", a, b, true)

// --- Floating-point proximity --------------------------------------------------------

#define SEETEST_NEAR_(a, b, tol, fatal)                                     \
  do {                                                                      \
    const double seetest_a_ = static_cast<double>(a);                       \
    const double seetest_b_ = static_cast<double>(b);                       \
    const double seetest_tol_ = static_cast<double>(tol);                   \
    if (!(std::fabs(seetest_a_ - seetest_b_) <= seetest_tol_)) {            \
      ADD_FAILURE(::seeml::testing::internal::NearFailure(                  \
          #a, #b, seetest_a_, seetest_b_, seetest_tol_));                   \
      if (fatal) return;                                                    \
    }                                                                       \
  } while (0)

#define EXPECT_NEAR(a, b, tol) SEETEST_NEAR_(a, b, tol, false)
#define ASSERT_NEAR(a, b, tol) SEETEST_NEAR_(a, b, tol, true)

// --- std::expected assertions ------------------------------------------------------

#define SEETEST_OK_(expr, fatal)                                            \
  do {                                                                      \
    const auto& seetest_r_ = (expr);                                        \
    if (!seetest_r_.has_value()) {                                          \
      ADD_FAILURE(std::string("expected success: " #expr "\n  error: ") +   \
                  ::seeml::testing::internal::Describe(seetest_r_.error())); \
      if (fatal) return;                                                    \
    }                                                                       \
  } while (0)

#define EXPECT_OK(expr) SEETEST_OK_(expr, false)
#define ASSERT_OK(expr) SEETEST_OK_(expr, true)

#define SEETEST_ERROR_(expr, fatal)                                         \
  do {                                                                      \
    const auto& seetest_r_ = (expr);                                        \
    if (seetest_r_.has_value()) {                                           \
      ADD_FAILURE("expected failure: " #expr);                              \
      if (fatal) return;                                                    \
    }                                                                       \
  } while (0)

#define EXPECT_ERROR(expr) SEETEST_ERROR_(expr, false)
#define ASSERT_ERROR(expr) SEETEST_ERROR_(expr, true)

/// Expects `expr` to fail with an error message containing `substr`.
#define EXPECT_ERROR_CONTAINS(expr, substr)                                 \
  do {                                                                      \
    const auto& seetest_r_ = (expr);                                        \
    if (seetest_r_.has_value()) {                                           \
      ADD_FAILURE("expected failure: " #expr);                              \
    } else if (std::string_view(seetest_r_.error()).find(substr) ==         \
               std::string_view::npos) {                                    \
      ADD_FAILURE(std::string("error message mismatch for " #expr) +        \
                  "\n  error = " +                                          \
                  ::seeml::testing::internal::Describe(seetest_r_.error()) + \
                  "\n  expected substring = \"" + (substr) + "\"");         \
    }                                                                       \
  } while (0)

/// Unwraps a successful std::expected into `lhs` or fails the test:
///   ASSERT_OK_AND_ASSIGN(auto model, LoadSmf(path));
#define ASSERT_OK_AND_ASSIGN(lhs, expr)                                     \
  auto SEETEST_CAT_(seetest_res_, __LINE__) = (expr);                       \
  if (!SEETEST_CAT_(seetest_res_, __LINE__).has_value()) {                  \
    ADD_FAILURE(std::string("expected success: " #expr "\n  error: ") +     \
                ::seeml::testing::internal::Describe(                       \
                    SEETEST_CAT_(seetest_res_, __LINE__).error()));         \
    return;                                                                 \
  }                                                                         \
  lhs = std::move(*SEETEST_CAT_(seetest_res_, __LINE__))

// --- String containment --------------------------------------------------------------

#define EXPECT_STR_CONTAINS(haystack, needle)                               \
  do {                                                                      \
    const std::string seetest_h_ = (haystack);                              \
    const std::string seetest_n_ = (needle);                                \
    if (seetest_h_.find(seetest_n_) == std::string::npos) {                 \
      ADD_FAILURE(std::string("expected " #haystack " to contain \"") +     \
                  seetest_n_ + "\"");                                       \
    }                                                                       \
  } while (0)

#endif  // SEEML_TEST_FRAMEWORK_ASSERT_H_
