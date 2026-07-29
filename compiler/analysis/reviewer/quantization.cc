#include "compiler/analysis/reviewer/quantization.h"

#include <algorithm>
#include <cmath>
#include <string_view>

#include "source/parallel/parallel_for.h"

namespace seeml::update {

namespace sir = seeml::sir;

std::unordered_map<const sir::Value*, float> SelectQuantizedWeights(
    sir::Block& block, const GraphBuild& build) {
  std::unordered_map<const sir::Value*, float> scales;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.weight") return;
    const sir::Value* v = op->result(0);
    auto src = build.weight_sources.find(v);
    if (src == build.weight_sources.end()) return;

    for (const sir::Operation* user : v->users()) {
      const std::string_view m = user->mnemonic();
      if (m != "sc_high.matmul" && m != "sc_low.matmul_nt") return;
      // v must be exactly the weight operand, never the activation side.
      if (user->numOperands() != 2 || user->operand(1) != v ||
          user->operand(0) == v)
        return;
    }

    const auto* data = reinterpret_cast<const float*>(src->second->data.data());
    const size_t count = src->second->byte_size / sizeof(float);
    // Chunked max reduction; max is order-insensitive, and the fixed chunk
    // geometry keeps even the combining order thread-count-independent.
    float partials[kMaxParallelChunks] = {};
    ParallelFor(count, kWeightSweepGrain, [&](size_t b, size_t e, size_t c) {
      float m = 0.0f;
      for (size_t i = b; i < e; ++i) m = std::max(m, std::fabs(data[i]));
      partials[c] = m;
    });
    float max_abs = 0.0f;
    const size_t chunks = ParallelChunkCount(count, kWeightSweepGrain);
    for (size_t c = 0; c < chunks; ++c) max_abs = std::max(max_abs, partials[c]);
    // An all-zero weight quantizes to zeros under any scale; 1.0 keeps the
    // reciprocal finite.
    scales[v] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
  });
  return scales;
}

}  // namespace seeml::update
