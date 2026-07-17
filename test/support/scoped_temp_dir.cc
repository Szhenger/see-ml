#include "test/support/scoped_temp_dir.h"

#include <unistd.h>

#include <atomic>
#include <string>

namespace seeml::testing {

ScopedTempDir::ScopedTempDir() {
  static std::atomic<uint64_t> counter{0};
  const uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
  path_ = std::filesystem::temp_directory_path() /
          ("seeml_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(id));
  std::filesystem::create_directories(path_);
}

ScopedTempDir::~ScopedTempDir() {
  std::error_code ec;  // best effort: never throw from a destructor
  std::filesystem::remove_all(path_, ec);
}

}  // namespace seeml::testing
