#ifndef SEEML_RUNTIME_DIAGNOSTICS_FEEDING_ERROR_H_
#define SEEML_RUNTIME_DIAGNOSTICS_FEEDING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "runtime/diagnostics/diagnostic.h"

// =============================================================================
// feeding/ — errors formed while decoding the SDS dataset and staging
// batches (runtime/feeder/). The lexical layer of the update: nothing here
// understands the plan, only the corpus — magic, geometry, label bounds.
// Failure discipline: reject the file before the first sample is served;
// the pipeline itself cannot fail (staging is allocation-free and joined on
// every exit path), so every feeding error is a dataset error.
// =============================================================================

namespace seeml::update_rt::diag::feeding {

inline constexpr std::string_view kDataset = "Dataset";
inline constexpr std::string_view kBatchPipeline = "BatchPipeline";

/// Corpus-level failure: "Dataset: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view message) {
  return Fail(kDataset, message);
}

/// File-shaped failure: "Dataset: <what> '<path>'".
[[nodiscard]] inline std::unexpected<std::string> FileError(std::string_view what,
                                                            std::string_view path) {
  std::string m;
  m.reserve(what.size() + path.size() + 3);
  m.append(what).append(" '").append(path).append("'");
  return Fail(kDataset, m);
}

}  // namespace seeml::update_rt::diag::feeding

#endif  // SEEML_RUNTIME_DIAGNOSTICS_FEEDING_ERROR_H_
