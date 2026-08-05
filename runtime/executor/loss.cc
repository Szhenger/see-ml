#include <cmath>

#include "runtime/executor/kernel_policy.h"
#include "runtime/executor/update_kernels.h"
#include "source/parallel/parallel_for.h"

// =============================================================================
// Loss family: softmax cross-entropy, MSE, and KL distillation, each a
// forward-backward pair. Every loss reduces with one partial per chunk,
// combined in chunk order — bitwise-deterministic for every thread count
// (chunk geometry is a pure function of the shape).
// =============================================================================

namespace seeml::update_rt::kernels {

namespace up = seeml::update;

namespace {

// Numerically stable in-place row softmax with optional temperature.
void RowSoftmax(const float* logits, float* probs, size_t C, float inv_T) {
  float max_v = logits[0] * inv_T;
  for (size_t c = 1; c < C; ++c) max_v = std::fmax(max_v, logits[c] * inv_T);
  float sum = 0.0f;
  for (size_t c = 0; c < C; ++c) {
    probs[c] = std::exp(logits[c] * inv_T - max_v);
    sum += probs[c];
  }
  const float inv_sum = 1.0f / sum;
  for (size_t c = 0; c < C; ++c) probs[c] *= inv_sum;
}

}  // namespace

void SoftmaxXEntFwd(const float* logits, const int32_t* labels, float* loss,
                    float* probs, size_t N, size_t C) {
  const size_t grain = RowGrain(C, kGrainMath);
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(N, grain, [&](size_t r0, size_t r1, size_t chunk) {
    double total = 0.0;
    for (size_t n = r0; n < r1; ++n) {
      RowSoftmax(logits + n * C, probs + n * C, C, 1.0f);
      const float p = probs[n * C + static_cast<size_t>(labels[n])];
      // fmax would silently scrub a NaN probability (fmax(NaN, x) == x) and
      // report a large-but-finite loss while the gradients poison the
      // parameters. Propagate NaN so the engine's finite-loss guard trips;
      // the clamp only rescues genuine underflow.
      const double safe = std::isnan(p)
                              ? static_cast<double>(p)
                              : static_cast<double>(std::fmax(p, 1e-12f));
      total -= std::log(safe);
    }
    partials[chunk] = total;
  });
  double total = 0.0;
  const size_t chunks = up::ParallelChunkCount(N, grain);
  for (size_t c = 0; c < chunks; ++c) total += partials[c];
  *loss = static_cast<float>(total / static_cast<double>(N));
}

void SoftmaxXEntBwd(const float* probs, const int32_t* labels,
                    const float* seed, float* dlogits, size_t N, size_t C) {
  const float scale = *seed / static_cast<float>(N);
  up::ParallelFor(N, RowGrain(C, kGrainCheap),
                  [&](size_t r0, size_t r1, size_t) {
                    for (size_t n = r0; n < r1; ++n) {
                      for (size_t c = 0; c < C; ++c)
                        dlogits[n * C + c] = scale * probs[n * C + c];
                      dlogits[n * C + static_cast<size_t>(labels[n])] -= scale;
                    }
                  });
}

void MseFwd(const float* pred, const float* target, float* loss, size_t n) {
  const size_t grain = kGrainCheap;
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(n, grain, [&](size_t b, size_t e, size_t chunk) {
    double total = 0.0;
    for (size_t i = b; i < e; ++i) {
      const double d = static_cast<double>(pred[i]) - target[i];
      total += d * d;
    }
    partials[chunk] = total;
  });
  double total = 0.0;
  const size_t chunks = up::ParallelChunkCount(n, grain);
  for (size_t c = 0; c < chunks; ++c) total += partials[c];
  *loss = static_cast<float>(total / static_cast<double>(n));
}

void MseBwd(const float* pred, const float* target, const float* seed,
            float* dpred, size_t n) {
  const float scale = 2.0f * *seed / static_cast<float>(n);
  up::ParallelFor(n, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dpred[i] = scale * (pred[i] - target[i]);
  });
}

void KLDistillFwd(const float* s_logits, const float* t_logits, float* loss,
                  float* p_s, float* p_t, size_t N, size_t C, float T) {
  const float inv_T = 1.0f / T;
  const size_t grain = RowGrain(C, kGrainMath);
  double partials[up::kMaxParallelChunks] = {};
  up::ParallelFor(N, grain, [&](size_t r0, size_t r1, size_t chunk) {
    double total = 0.0;
    for (size_t n = r0; n < r1; ++n) {
      RowSoftmax(s_logits + n * C, p_s + n * C, C, inv_T);
      RowSoftmax(t_logits + n * C, p_t + n * C, C, inv_T);
      for (size_t c = 0; c < C; ++c) {
        const float pt = p_t[n * C + c];
        if (pt <= 0.0f) continue;
        // Same NaN discipline as SoftmaxXEntFwd: fmax(NaN, x) == x would
        // scrub a NaN student probability into a large-but-finite loss
        // while the backward pass poisons the parameters — the engine's
        // finite-loss guard must trip instead. The clamp only rescues
        // genuine underflow. (A NaN teacher probability already propagates:
        // `pt <= 0.0f` is false for NaN, and log(NaN) is NaN.)
        const float ps_raw = p_s[n * C + c];
        const double ps = std::isnan(ps_raw)
                              ? static_cast<double>(ps_raw)
                              : static_cast<double>(std::fmax(ps_raw, 1e-12f));
        total += static_cast<double>(pt) *
                 (std::log(static_cast<double>(pt)) - std::log(ps));
      }
    }
    partials[chunk] = total;
  });
  double total = 0.0;
  const size_t chunks = up::ParallelChunkCount(N, grain);
  for (size_t c = 0; c < chunks; ++c) total += partials[c];
  *loss = static_cast<float>(total / static_cast<double>(N));
}

void KLDistillBwd(const float* p_s, const float* p_t, const float* seed,
                  float* dlogits, size_t N, size_t C, float T) {
  const float scale = *seed / (static_cast<float>(N) * T);
  up::ParallelFor(N * C, kGrainCheap, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) dlogits[i] = scale * (p_s[i] - p_t[i]);
  });
}

}  // namespace seeml::update_rt::kernels
