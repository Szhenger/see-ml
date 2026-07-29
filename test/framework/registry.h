#ifndef SEEML_TEST_FRAMEWORK_REGISTRY_H_
#define SEEML_TEST_FRAMEWORK_REGISTRY_H_

#include <string>

// =============================================================================
// registry/ discipline of SeeTest: test registration and the runner. Tests
// self-register via static Registrar objects (constructed by the TEST macro
// in assert.h); the runner executes them in registration order and owns the
// per-test failure state that ReportFailure records into.
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

}  // namespace internal
}  // namespace seeml::testing

#endif  // SEEML_TEST_FRAMEWORK_REGISTRY_H_
