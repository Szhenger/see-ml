// =============================================================================
// libFuzzer harness over the three attacker/corruption-facing parsers:
//
//   [0] LoadSmf                 — the model container reader
//   [1] UpdateEngine::Initialize — the .seeu plan loader + validator
//   [2] Dataset::LoadFromFile    — the corpus reader
//
// The first input byte selects the parser; the rest is the file image. The
// contract under test: arbitrary bytes may be REJECTED but must never crash,
// overflow, or over-read — the runtime trusts these validators completely
// (Execute() runs unchecked after Initialize() accepts a plan).
//
// Build: cmake -DSEEML_FUZZ=ON (clang), then run seeml_fuzz_formats.
// =============================================================================

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "compiler/backend/update_types.h"
#include "runtime/dataset.h"
#include "runtime/update_engine.h"
#include "source/hash.h"
#include "source/smf.h"

namespace {

std::string TempPath(const char* tag) {
  return std::string("/tmp/seeml_fuzz_") + tag + "_" +
         std::to_string(::getpid());
}

void WriteBytes(const std::string& path, const uint8_t* data, size_t size) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  if (size) std::fwrite(data, 1, size, f);
  std::fclose(f);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1) return 0;
  const uint8_t selector = data[0] % 3;
  const uint8_t* body = data + 1;
  const size_t body_size = size - 1;

  switch (selector) {
    case 0: {
      const std::string path = TempPath("smf");
      WriteBytes(path, body, body_size);
      auto r = seeml::update::LoadSmf(path);
      (void)r;
      std::remove(path.c_str());
      break;
    }
    case 1: {
      // The integrity hash would reject virtually every mutated input at the
      // first gate and starve the deeper validators of coverage. The hash
      // exists to catch corruption, not adversaries — so the harness seals
      // the blob the same way the compiler does, letting the fuzzer attack
      // the section/instruction/bounds validation underneath.
      std::vector<uint8_t> plan(body, body + body_size);
      if (plan.size() >= sizeof(seeml::update::PlanHeader)) {
        constexpr size_t kHashAt =
            offsetof(seeml::update::PlanHeader, plan_hash);
        std::memset(plan.data() + kHashAt, 0, sizeof(uint64_t));
        const uint64_t h = seeml::update::Fnv1a64(plan.data(), plan.size());
        std::memcpy(plan.data() + kHashAt, &h, sizeof(h));
      }
      seeml::update_rt::UpdateEngine engine;
      auto r = engine.LoadFromMemory(plan.data(), plan.size());
      (void)r;
      break;
    }
    case 2: {
      const std::string path = TempPath("sds");
      WriteBytes(path, body, body_size);
      auto r = seeml::update_rt::Dataset::LoadFromFile(path);
      (void)r;
      std::remove(path.c_str());
      break;
    }
  }
  return 0;
}
