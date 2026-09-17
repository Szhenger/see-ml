#include <cstring>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Elementwise / broadcast family: pointwise arithmetic, the bias broadcast,
// the row reduction, and the serial utility fills and copies.
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

void AddEW(const float* x, const float* y, float* out, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = x[i] + y[i];
  });
}

void MulEW(const float* x, const float* y, float* out, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = x[i] * y[i];
  });
}

void AddBias(const float* x, const float* b, float* out, size_t rows,
             size_t cols) {
  up::ParallelFor(rows, RowGrain(cols, kGrainCheap),
                  [&](size_t r0, size_t r1, size_t) {
                    for (size_t r = r0; r < r1; ++r)
                      for (size_t c = 0; c < cols; ++c)
                        out[r * cols + c] = x[r * cols + c] + b[c];
                  });
}

void Accumulate(float* dst, const float* src, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dst[i] += src[i];
  });
}

void Scale(const float* x, float* out, float alpha, size_t n) {
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) out[i] = alpha * x[i];
  });
}

// The fused chain runs BLOCK-wise, stage by stage: each stage is its own
// loop over a block that lives in L1, evaluating the very expression the
// standalone kernel evaluates (the Expr functions of kernel_policy.h, x + y,
// x * y, alpha * x). Keeping the stages as separate loops is what makes
// fusion bitwise-safe — the compiler contracts a multiply into an add only
// within one expression, so no stage can fuse arithmetically with its
// neighbour — and keeping the block in L1 is what makes it worth doing:
// the instruction sequence it replaces wrote every intermediate out to the
// arena and read it back, a full-tensor round trip per folded op.
namespace {

/// One stage over one block. kInPlace: a middle stage, reading and writing
/// the block through one pointer; otherwise source and destination are
/// distinct (x -> block, block -> out, or x -> out) and restrict-qualified
/// — without that the compiler must assume every store may feed the next
/// load, and the activation stages ran 20% slower than standalone.
template <bool kInPlace>
void RunFusedStage(uint8_t stage, const float* src, float* dst,
                   const float* const others[3], size_t b0, size_t w,
                   const float imm[2]) {
  auto apply = [&](auto&& expr) {
    if constexpr (kInPlace) {
      for (size_t i = 0; i < w; ++i) dst[i] = expr(dst[i], i);
    } else {
      const float* SEEML_RESTRICT s = src;
      float* SEEML_RESTRICT d = dst;
      for (size_t i = 0; i < w; ++i) d[i] = expr(s[i], i);
    }
  };
  const uint8_t arg = up::FusedStageArg(stage);
  const bool run_is_right = (stage & up::kFusedStageRunIsRight) != 0;
  switch (up::FusedStageKind(stage)) {
    case up::FusedStage::kAdd: {
      const float* y = others[arg] + b0;
      if (run_is_right)
        apply([y](float v, size_t i) { return y[i] + v; });
      else
        apply([y](float v, size_t i) { return v + y[i]; });
      break;
    }
    case up::FusedStage::kMul: {
      const float* y = others[arg] + b0;
      if (run_is_right)
        apply([y](float v, size_t i) { return y[i] * v; });
      else
        apply([y](float v, size_t i) { return v * y[i]; });
      break;
    }
    case up::FusedStage::kScale: {
      const float alpha = imm[arg];
      apply([alpha](float v, size_t) { return alpha * v; });
      break;
    }
    case up::FusedStage::kRelu:
      apply([](float v, size_t) { return ReluExpr(v); });
      break;
    case up::FusedStage::kGelu:
      apply([](float v, size_t) { return GeluExpr(v); });
      break;
    case up::FusedStage::kSilu:
      apply([](float v, size_t) { return SiluExpr(v); });
      break;
    case up::FusedStage::kEnd:
      break;
  }
}

}  // namespace

void FusedMap(const float* x, const float* const others[3], float* out,
              size_t n, uint64_t stages, const float imm[2]) {
  constexpr size_t kBlock = 1024;  // 4 KiB: the block and its sources in L1
  uint8_t prog[up::kFusedMapMaxStages] = {};
  size_t count = 0;
  bool transcendental = false;
  for (; count < up::kFusedMapMaxStages; ++count) {
    prog[count] = static_cast<uint8_t>(stages >> (8 * count));
    const auto kind = up::FusedStageKind(prog[count]);
    if (kind == up::FusedStage::kEnd) break;
    transcendental |= kind == up::FusedStage::kGelu ||
                      kind == up::FusedStage::kSilu;
  }
  if (count == 0) return;  // the validator never admits an empty program
  up::ParallelFor(n, transcendental ? kGrainMath : kGrainCheap,
                  [&](size_t begin, size_t end, size_t) {
    float block[kBlock];
    for (size_t b0 = begin; b0 < end; b0 += kBlock) {
      const size_t w = MinZ(kBlock, end - b0);
      // First stage: x -> block (or straight to out for a one-stage
      // program); middle stages: the block in place; last: block -> out.
      float* first_dst = count == 1 ? out + b0 : block;
      RunFusedStage<false>(prog[0], x + b0, first_dst, others, b0, w, imm);
      for (size_t s = 1; s + 1 < count; ++s)
        RunFusedStage<true>(prog[s], block, block, others, b0, w, imm);
      if (count > 1)
        RunFusedStage<false>(prog[count - 1], block, out + b0, others, b0, w,
                             imm);
    }
  });
}

void EmbedFwd(const int32_t* tokens, const float* table, float* out,
              size_t rows, size_t dim) {
  up::ParallelFor(rows, RowGrain(dim, kGrainCheap),
                  [&](size_t r0, size_t r1, size_t) {
    for (size_t r = r0; r < r1; ++r)
      std::memcpy(out + r * dim,
                  table + static_cast<size_t>(tokens[r]) * dim,
                  dim * sizeof(float));
  });
}

void ReduceRows(const float* dy, float* db, size_t rows, size_t cols) {
  // Partitioned over db's columns: each chunk owns a column slice and walks
  // the rows in order, so per-column accumulation order — and therefore the
  // result — is bit-identical to the serial loop.
  //
  // The sums accumulate in a chunk-local tile and reach db once, when the
  // tile is done (E3, #82). Accumulating in db itself made every row a
  // read-modify-write of the slice's boundary cache lines, which the
  // neighbouring chunk's thread was rewriting at the same cadence — false
  // sharing once per row. A column's additions are the same floats in the
  // same order whether the running sum lives in db or on this stack, so
  // the tile changes memory traffic, never bits.
  constexpr size_t kTile = 256;  // 1 KiB of L1, and no allocation
  up::ParallelFor(cols, RowGrain(rows, kGrainCheap),
                  [&](size_t c0, size_t c1, size_t) {
                    for (size_t t0 = c0; t0 < c1; t0 += kTile) {
                      const size_t width = MinZ(kTile, c1 - t0);
                      float acc[kTile] = {};
                      for (size_t r = 0; r < rows; ++r) {
                        const float* SEEML_RESTRICT row = dy + r * cols + t0;
                        for (size_t c = 0; c < width; ++c) acc[c] += row[c];
                      }
                      std::memcpy(db + t0, acc, width * sizeof(float));
                    }
                  });
}

// Fill and Copy stay serial: memset/memcpy-class loops saturate memory
// bandwidth from one core on the device classes the runtime targets, so
// splitting them buys contention, not throughput.
void Fill(float* dst, float value, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = value;
}

void Copy(const float* src, float* dst, size_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}

}  // namespace seeml::update_rt::kernels
