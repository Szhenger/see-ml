#ifndef SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_READER_H_
#define SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_READER_H_

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "compiler/frontend/ingressor/model_format.h"

// =============================================================================
// SMF reader — deserializes and validates an SMF container file.
//
// Every field of a hostile file is bounds-checked: magic/version gating,
// bounded primitive reads, dim positivity and volume-overflow checks, and
// exact agreement between a constant tensor's dims and its byte_size (the
// instruction stream sizes reads from dims while rodata packing sizes from
// byte_size — they must not disagree). Fuzzed by test/fuzz_binary_formats.cc.
//
// The load is two-phase for throughput: a serial metadata scan validates
// everything, then the byte-heavy work — constant payload copies (disjoint
// per tensor, fanned over ParallelFor past a size threshold) and the
// ContentHash64 identity hash (8-lane ILP striping + deterministic chunking)
// — runs on the validated layout.
// =============================================================================

namespace seeml::update {

[[nodiscard]] std::expected<SmfModel, std::string> LoadSmf(
    const std::string& path);

/// Loads several model files concurrently (one loader thread per path, e.g.
/// student + teacher). Task-level concurrency on top of LoadSmf's internal
/// data parallelism: one model's serial file read overlaps another's hashing
/// and payload copies, while the worker pool serializes the data-parallel
/// phases themselves — so every returned model is bitwise-identical to a
/// sequential LoadSmf of the same path. Results are in input order; on
/// failure, the error of the first failing path (by input order) is returned.
[[nodiscard]] std::expected<std::vector<SmfModel>, std::string> LoadSmfMany(
    std::span<const std::string> paths);

}  // namespace seeml::update

#endif  // SEEML_COMPILER_FRONTEND_INGRESSOR_MODEL_READER_H_
