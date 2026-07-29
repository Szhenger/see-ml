#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "test/support/builders.h"

// =============================================================================
// probes/ discipline of the shared fixtures: direct engine-arena
// introspection for gradient and merge verification, plus the test-run
// environment (the repository checkout the binaries were configured from).
// =============================================================================

namespace seeml::testing {

using update_rt::UpdateEngine;

std::string RepoRoot() {
#ifdef SEEML_SOURCE_DIR
  return SEEML_SOURCE_DIR;
#else
  return ".";
#endif
}

float ReadArenaF32(UpdateEngine& e, uint64_t ref, uint64_t index) {
  return reinterpret_cast<const float*>(e.arena() +
                                        update::RefOffset(ref))[index];
}

void WriteArenaF32(UpdateEngine& e, uint64_t ref, uint64_t index, float v) {
  reinterpret_cast<float*>(e.arena() + update::RefOffset(ref))[index] = v;
}

void FillSlots(UpdateEngine& e, const std::vector<float>& x,
               const std::vector<int32_t>& labels) {
  if (x.size() != e.header().input_floats ||
      (!labels.empty() &&
       labels.size() * sizeof(int32_t) != e.header().label_bytes)) {
    std::fprintf(stderr, "FillSlots: batch does not match the plan header\n");
    std::abort();
  }
  std::memcpy(e.arena() + update::RefOffset(e.header().input_ref), x.data(),
              x.size() * sizeof(float));
  if (!labels.empty())
    std::memcpy(e.arena() + update::RefOffset(e.header().label_ref),
                labels.data(), labels.size() * sizeof(int32_t));
}

}  // namespace seeml::testing
