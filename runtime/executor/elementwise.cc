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
