#ifndef SEEML_TEST_SUPPORT_SCOPED_TEMP_DIR_H_
#define SEEML_TEST_SUPPORT_SCOPED_TEMP_DIR_H_

#include <filesystem>
#include <string>
#include <string_view>

namespace seeml::testing {

/// A uniquely named directory under the system temp root, created on
/// construction and removed (recursively) on destruction. Each test that
/// touches the filesystem owns its own instance, so parallel ctest runs and
/// repeated invocations never collide.
class ScopedTempDir {
 public:
  ScopedTempDir();
  ~ScopedTempDir();

  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  const std::filesystem::path& path() const { return path_; }

  /// Absolute path of `name` inside the directory.
  std::string File(std::string_view name) const {
    return (path_ / name).string();
  }

 private:
  std::filesystem::path path_;
};

}  // namespace seeml::testing

#endif  // SEEML_TEST_SUPPORT_SCOPED_TEMP_DIR_H_
