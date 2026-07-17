#ifndef SEEML_TEST_FRAMEWORK_SEETEST_H_
#define SEEML_TEST_FRAMEWORK_SEETEST_H_

#include <cmath>
#include <concepts>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// =============================================================================
// SeeTest — the SeeML testing framework.
//
// A single-header, dependency-free harness in the spirit of GoogleTest,
// matching the codebase's C++23 / std::expected idiom:
//
//   TEST(Suite, Name) {
//     auto model = LoadSmf(path);
//     ASSERT_OK(model);                    // std::expected-aware
//     EXPECT_EQ(model->tensors.size(), 5u);
//     EXPECT_NEAR(loss, 1.386f, 1e-4);
//   }
//
// Tests self-register via static initializers; link a suite's .cc files with
// seetest_main.cc to obtain a runner supporting:
//   --list             print registered tests without running them
//   --filter=PATTERN   run matching tests ('*' wildcards; a pattern without
//                      '*' matches as a substring of "Suite.Name")
//
// EXPECT_* records the failure and continues; ASSERT_* records and returns
// from the test function (so it must be used in the TEST body itself, not in
// helpers with non-void returns). The process exits non-zero if any selected
// test failed.
// =============================================================================

namespace seeml::testing {

using TestFn = void (*)();

/// Registers a test. Invoked by TEST() through static Registrar objects; the
/// runner executes tests in registration order (deterministic within a TU).
void RegisterTest(const char* suite, const char* name, TestFn fn);

/// Records a failure against the currently running test and prints it.
/// Aborts if no test is running (assertion used outside the harness).
void ReportFailure(const char* file, int line, const std::string& message);

/// Runs every registered test selected by argv. Returns the process exit
/// code: 0 iff every selected test passed.
int RunAllTests(int argc, char** argv);

namespace internal {

struct Registrar {
  Registrar(const char* suite, const char* name, TestFn fn) {
    RegisterTest(suite, name, fn);
  }
};

// --- Value formatting ---------------------------------------------------------

template <typename T>
concept Streamable = requires(std::ostream& os, const T& v) { os << v; };

template <typename T>
std::string Describe(const T& v) {
  using D = std::remove_cvref_t<T>;
  if constexpr (std::same_as<D, bool>) {
    return v ? "true" : "false";
  } else if constexpr (std::convertible_to<const D&, std::string_view>) {
    std::string s = "\"";
    s += std::string_view(v);
    s += "\"";
    return s;
  } else if constexpr (std::is_enum_v<D>) {
    std::ostringstream os;
    os << static_cast<long long>(v);
    return os.str();
  } else if constexpr (Streamable<D>) {
    std::ostringstream os;
    os << std::setprecision(9) << v;
    return os.str();
  } else {
    return "<unprintable>";
  }
}

// --- Comparison helpers ---------------------------------------------------------
// Mixed-signedness integer comparisons go through std::cmp_* so that
// EXPECT_EQ(u64_value, 4) is both warning-free under -Werror and correct.

template <typename A, typename B>
inline constexpr bool kSafeIntegerCompare =
    std::integral<std::remove_cvref_t<A>> &&
    std::integral<std::remove_cvref_t<B>> &&
    !std::same_as<std::remove_cvref_t<A>, bool> &&
    !std::same_as<std::remove_cvref_t<B>, bool> &&
    !std::same_as<std::remove_cvref_t<A>, char> &&
    !std::same_as<std::remove_cvref_t<B>, char>;

template <typename A, typename B>
bool CmpEq(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>)
    return std::cmp_equal(a, b);
  else
    return a == b;
}

template <typename A, typename B>
bool CmpNe(const A& a, const B& b) {
  return !CmpEq(a, b);
}

template <typename A, typename B>
bool CmpLt(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>)
    return std::cmp_less(a, b);
  else
    return a < b;
}

template <typename A, typename B>
bool CmpLe(const A& a, const B& b) {
  return !CmpLt(b, a);
}

template <typename A, typename B>
bool CmpGt(const A& a, const B& b) {
  return CmpLt(b, a);
}

template <typename A, typename B>
bool CmpGe(const A& a, const B& b) {
  return !CmpLt(a, b);
}

// --- Failure message builders ----------------------------------------------------

template <typename A, typename B>
std::string BinaryFailure(const char* a_text, const char* b_text,
                          const char* op, const A& a, const B& b) {
  std::string msg = "expected: ";
  msg += a_text;
  msg += " ";
  msg += op;
  msg += " ";
  msg += b_text;
  msg += "\n  lhs = " + Describe(a);
  msg += "\n  rhs = " + Describe(b);
  return msg;
}

inline std::string NearFailure(const char* a_text, const char* b_text,
                               double a, double b, double tol) {
  std::ostringstream os;
  os << "expected: |" << a_text << " - " << b_text << "| <= " << tol
     << "\n  lhs = " << std::setprecision(12) << a
     << "\n  rhs = " << std::setprecision(12) << b
     << "\n  |lhs - rhs| = " << std::fabs(a - b);
  return os.str();
}

}  // namespace internal
}  // namespace seeml::testing

// =============================================================================
// Macros
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

#endif  // SEEML_TEST_FRAMEWORK_SEETEST_H_
