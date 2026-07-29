#ifndef SEEML_TEST_FRAMEWORK_MATCHER_H_
#define SEEML_TEST_FRAMEWORK_MATCHER_H_

#include <cmath>
#include <concepts>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// =============================================================================
// matcher/ discipline of SeeTest: value formatting, warning-safe
// comparisons, and failure-message construction — everything the assertion
// macros (assert.h) evaluate. Internal to the framework; suites reach it
// only through the macros.
// =============================================================================

namespace seeml::testing::internal {

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

}  // namespace seeml::testing::internal

#endif  // SEEML_TEST_FRAMEWORK_MATCHER_H_
