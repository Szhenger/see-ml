#ifndef SEEML_TEST_FRAMEWORK_MATCHER_H_
#define SEEML_TEST_FRAMEWORK_MATCHER_H_

#include <cmath>
#include <concepts>
#include <cstring>
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
    // A null char* must not reach the string_view constructor (UB): the
    // comparison layer tolerates null C-strings, so the failure reporter
    // has to as well. (Arrays can't be null; is_pointer_v guards the check
    // so no tautological comparison is emitted for them.)
    if constexpr (std::is_pointer_v<D>)
      if (v == nullptr) return "(null)";
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

// std::cmp_* mandates against bool and ALL character types (char, wchar_t,
// char8_t/16_t/32_t) — those fall through to the plain operators instead.
template <typename T>
inline constexpr bool kCmpSafeInteger =
    std::integral<std::remove_cvref_t<T>> &&
    !std::same_as<std::remove_cvref_t<T>, bool> &&
    !std::same_as<std::remove_cvref_t<T>, char> &&
    !std::same_as<std::remove_cvref_t<T>, wchar_t> &&
    !std::same_as<std::remove_cvref_t<T>, char8_t> &&
    !std::same_as<std::remove_cvref_t<T>, char16_t> &&
    !std::same_as<std::remove_cvref_t<T>, char32_t>;

template <typename A, typename B>
inline constexpr bool kSafeIntegerCompare =
    kCmpSafeInteger<A> && kCmpSafeInteger<B>;

// C-string operands (pointers or string literals) compare by content:
// Describe() prints their contents, and a matcher whose message shows two
// identical strings "not equal" because their storage differs would be
// actively misleading.
template <typename T>
inline constexpr bool kIsCString =
    std::same_as<std::decay_t<T>, char*> ||
    std::same_as<std::decay_t<T>, const char*>;

template <typename A, typename B>
bool CmpEq(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>) {
    return std::cmp_equal(a, b);
  } else if constexpr (kIsCString<A> && kIsCString<B>) {
    const char* pa = a;
    const char* pb = b;
    if (pa == nullptr || pb == nullptr) return pa == pb;
    return std::strcmp(pa, pb) == 0;
  } else {
    return a == b;
  }
}

template <typename A, typename B>
bool CmpNe(const A& a, const B& b) {
  return !CmpEq(a, b);
}

// The ordered comparisons are each written directly, never derived by
// negating another: with a NaN operand every ordering is false, so a
// negation-derived CmpLe/CmpGe would return true — silently green-lighting
// exactly the NaN-divergence regression (EXPECT_LE(loss, bound)) an ML
// test suite most needs to catch.
template <typename A, typename B>
bool CmpLt(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>)
    return std::cmp_less(a, b);
  else
    return a < b;
}

// LE/GE are evaluated directly, never as negated LT: that identity only
// holds for total orders, and IEEE-754 floats are not one. With a NaN
// operand every comparison is false, so !(b < a) would PASS the assertion —
// EXPECT_LE(loss, 2.0) silently green on a diverged NaN loss is exactly the
// failure an ML test framework exists to catch.
template <typename A, typename B>
bool CmpLe(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>)
    return std::cmp_less_equal(a, b);
  else
    return a <= b;
}

template <typename A, typename B>
bool CmpGt(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>)
    return std::cmp_greater(a, b);
  else
    return a > b;
}

template <typename A, typename B>
bool CmpGe(const A& a, const B& b) {
  if constexpr (kSafeIntegerCompare<A, B>)
    return std::cmp_greater_equal(a, b);
  else
    return a >= b;
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
