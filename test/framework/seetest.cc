#include "test/framework/seetest.h"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace seeml::testing {

namespace {

struct TestInfo {
  const char* suite;
  const char* name;
  TestFn fn;
  bool selected = true;
  bool failed = false;

  std::string FullName() const { return std::string(suite) + "." + name; }
};

// Function-local static: safe against static-initialization order across the
// registrar objects in other translation units.
std::vector<TestInfo>& Registry() {
  static std::vector<TestInfo>* tests = new std::vector<TestInfo>();
  return *tests;
}

struct RunState {
  TestInfo* current = nullptr;
};

RunState& State() {
  static RunState state;
  return state;
}

bool UseColor() {
  static const bool color = ::isatty(1) != 0;
  return color;
}

const char* Green() { return UseColor() ? "\x1b[32m" : ""; }
const char* Red() { return UseColor() ? "\x1b[31m" : ""; }
const char* Reset() { return UseColor() ? "\x1b[0m" : ""; }

/// Glob-style match: '*' matches any (possibly empty) substring.
bool Matches(std::string_view pattern, std::string_view text) {
  if (pattern.empty()) return text.empty();
  if (pattern.front() == '*')
    return Matches(pattern.substr(1), text) ||
           (!text.empty() && Matches(pattern, text.substr(1)));
  return !text.empty() && pattern.front() == text.front() &&
         Matches(pattern.substr(1), text.substr(1));
}

void PrintIndented(const std::string& message) {
  size_t start = 0;
  while (start <= message.size()) {
    const size_t end = message.find('\n', start);
    const std::string_view line(message.data() + start,
                                (end == std::string::npos ? message.size()
                                                          : end) -
                                    start);
    std::printf("    %.*s\n", static_cast<int>(line.size()), line.data());
    if (end == std::string::npos) break;
    start = end + 1;
  }
}

}  // namespace

void RegisterTest(const char* suite, const char* name, TestFn fn) {
  Registry().push_back({suite, name, fn, /*selected=*/true, /*failed=*/false});
}

void ReportFailure(const char* file, int line, const std::string& message) {
  std::printf("%s  FAILURE%s at %s:%d\n", Red(), Reset(), file, line);
  PrintIndented(message);
  RunState& state = State();
  if (!state.current) {
    std::fprintf(stderr,
                 "seetest: assertion used outside a running test — aborting\n");
    std::abort();
  }
  state.current->failed = true;
}

int RunAllTests(int argc, char** argv) {
  std::string filter = "*";
  bool list_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--list") {
      list_only = true;
    } else if (arg.starts_with("--filter=")) {
      filter = arg.substr(9);
    } else if (arg == "--help" || arg == "-h") {
      std::printf("usage: %s [--list] [--filter=PATTERN]\n", argv[0]);
      std::printf(
          "  PATTERN matches \"Suite.Name\"; '*' is a wildcard; a pattern\n"
          "  without '*' matches as a substring.\n");
      return 0;
    } else {
      std::fprintf(stderr, "seetest: unknown argument '%s'\n", argv[i]);
      return 2;
    }
  }
  // A wildcard-free pattern is a substring query.
  if (filter.find('*') == std::string::npos && filter != "*")
    filter = "*" + filter + "*";

  std::vector<TestInfo>& tests = Registry();
  size_t selected = 0;
  for (TestInfo& t : tests) {
    t.selected = Matches(filter, t.FullName());
    selected += t.selected ? 1 : 0;
  }

  if (list_only) {
    for (const TestInfo& t : tests)
      if (t.selected) std::printf("%s\n", t.FullName().c_str());
    return 0;
  }
  if (selected == 0) {
    std::fprintf(stderr, "seetest: no tests match filter '%s'\n",
                 filter.c_str());
    return 2;
  }

  std::printf("[==========] running %zu of %zu registered test(s)\n", selected,
              tests.size());
  const auto suite_start = std::chrono::steady_clock::now();
  size_t failed_count = 0;

  for (TestInfo& t : tests) {
    if (!t.selected) continue;
    std::printf("[ RUN      ] %s\n", t.FullName().c_str());
    State().current = &t;
    const auto start = std::chrono::steady_clock::now();
    try {
      t.fn();
    } catch (const std::exception& e) {
      ReportFailure("<seetest>", 0,
                    std::string("uncaught exception: ") + e.what());
    } catch (...) {
      ReportFailure("<seetest>", 0, "uncaught non-standard exception");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    State().current = nullptr;
    if (t.failed) {
      ++failed_count;
      std::printf("[%s  FAILED  %s] %s (%lld ms)\n", Red(), Reset(),
                  t.FullName().c_str(), static_cast<long long>(elapsed));
    } else {
      std::printf("[%s       OK %s] %s (%lld ms)\n", Green(), Reset(),
                  t.FullName().c_str(), static_cast<long long>(elapsed));
    }
  }

  const auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - suite_start)
                         .count();
  std::printf("[==========] %zu test(s) ran (%lld ms total)\n", selected,
              static_cast<long long>(total));
  std::printf("[%s  PASSED  %s] %zu test(s)\n", Green(), Reset(),
              selected - failed_count);
  if (failed_count > 0) {
    std::printf("[%s  FAILED  %s] %zu test(s), listed below:\n", Red(), Reset(),
                failed_count);
    for (const TestInfo& t : tests)
      if (t.selected && t.failed)
        std::printf("[%s  FAILED  %s] %s\n", Red(), Reset(),
                    t.FullName().c_str());
  }
  return failed_count == 0 ? 0 : 1;
}

}  // namespace seeml::testing
