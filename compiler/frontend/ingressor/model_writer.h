#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_WRITER_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_WRITER_H_

#include <expected>
#include <string>

#include "compiler/frontend/ingressor/model_format.h"

// =============================================================================
// SMF writer — serializes an in-memory model to an SMF container file.
//
// Two-pass layout: a probe serialization of the metadata section determines
// where the 64-aligned data section starts, then the final pass writes both.
// On success the model's data offsets and content_hash are updated to match
// the saved bytes, binding its identity to the file on disk.
// =============================================================================

namespace seeml::update {

/// Serializes a model. Constant tensors must carry their data in
/// SmfTensor::data; data_offset/byte_size are (re)computed during the write.
[[nodiscard]] std::expected<void, std::string> SaveSmf(const std::string& path,
                                                       SmfModel& model);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_WRITER_H_
