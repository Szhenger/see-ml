#include "compiler/frontend/ingressor/resource_analyzer.h"

#include <string_view>
#include <unordered_map>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#include "compiler/diagnostics/tokenizing/error.h"

namespace seeml::update {

namespace {

uint64_t SatAdd(uint64_t a, uint64_t b) {
  return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

uint64_t SatMul(uint64_t a, uint64_t b) {
  if (b != 0 && a > UINT64_MAX / b) return UINT64_MAX;
  return a * b;
}

std::string FormatMiB(uint64_t bytes) {
  // Round up without the additive form: bytes near UINT64_MAX (exactly what
  // SatAdd/SatMul saturate to) would wrap and print "0 MiB".
  const uint64_t whole = bytes >> 20;
  const uint64_t frac = bytes & ((uint64_t{1} << 20) - 1);
  return std::to_string(whole + (frac != 0 ? 1 : 0)) + " MiB";
}

}  // namespace

uint64_t TrainingFootprint::total_bytes() const {
  return SatAdd(weight_bytes, activation_bytes);
}

TrainingFootprint& TrainingFootprint::operator+=(const TrainingFootprint& o) {
  weight_bytes = SatAdd(weight_bytes, o.weight_bytes);
  activation_bytes = SatAdd(activation_bytes, o.activation_bytes);
  return *this;
}

namespace {

/// Shared walk behind both estimators. `sum_activations` selects the
/// training model (every activation is cached for the backward pass — sum
/// them) or the frozen-forward model (no backward consumes anything, so
/// transients die at their single reader and the arena binder reuses their
/// slots — the honest lower bound is the single widest live term).
TrainingFootprint EstimateFootprintImpl(const SmfModel& model, int64_t batch,
                                        bool sum_activations) {
  TrainingFootprint fp;
  uint64_t peak_activation = 0;
  auto charge = [&](uint64_t term) {
    if (sum_activations)
      fp.activation_bytes = SatAdd(fp.activation_bytes, term);
    else
      peak_activation = std::max(peak_activation, term);
  };
  const uint64_t rows = batch > 0 ? static_cast<uint64_t>(batch) : 0;

  // Frozen weights: one resident copy each (they at least fill rodata).
  // Track each tensor's last dimension as its "width" to seed propagation;
  // the reader has already validated dims, so widths are positive (or the
  // dynamic batch dim, which never appears last).
  std::unordered_map<std::string_view, const SmfTensor*> tensors;
  std::unordered_map<std::string_view, uint64_t> width;
  std::unordered_map<std::string_view, uint64_t> row_count;
  tensors.reserve(model.tensors.size());
  for (const SmfTensor& t : model.tensors) {
    tensors[t.name] = &t;
    if (t.is_const) fp.weight_bytes = SatAdd(fp.weight_bytes, t.byte_size);
    if (!t.dims.empty() && t.dims.back() > 0)
      width[t.name] = static_cast<uint64_t>(t.dims.back());
    // Row seed: a constant tensor keeps its declared row count (the parser
    // sizes a MatMul from its actual LHS, which may be a const with rows !=
    // batch); anything else is served at the compiled batch. Over-counting
    // rows would break the lower-bound guarantee rejections rely on.
    row_count[t.name] =
        t.is_const ? (t.dims.size() >= 2 && t.dims.front() > 0
                          ? static_cast<uint64_t>(t.dims.front())
                          : 1)
                   : rows;
  }

  // Forward activations: propagate output widths and row counts with the
  // parser's shape rules. Every activation is [rows, width] f32 and is
  // cached for the backward pass. An unresolvable width contributes zero —
  // the estimate must stay a lower bound.
  for (const SmfOp& op : model.ops) {
    uint64_t w = 0;
    uint64_t r = rows;
    if (!op.inputs.empty())
      if (auto it = row_count.find(op.inputs[0]); it != row_count.end())
        r = it->second;
    switch (op.kind) {
      case SmfOpKind::kMatMul:
      case SmfOpKind::kEmbedding: {
        // Both produce [rows, dims[1] of the second operand]: the matmul's
        // B width, or the embedding table's dim.
        if (op.inputs.size() != 2) break;
        auto it = tensors.find(op.inputs[1]);
        if (it != tensors.end() && it->second->dims.size() == 2 &&
            it->second->dims[1] > 0)
          w = static_cast<uint64_t>(it->second->dims[1]);
        break;
      }
      case SmfOpKind::kAddBias:
      case SmfOpKind::kRelu:
      case SmfOpKind::kGelu:
      case SmfOpKind::kSilu:
      case SmfOpKind::kMul:
      case SmfOpKind::kLayerNorm:
      case SmfOpKind::kAdd:
      case SmfOpKind::kRmsNorm:
      case SmfOpKind::kRope:
      case SmfOpKind::kAttention: {
        if (op.inputs.empty()) break;
        if (auto it = width.find(op.inputs[0]); it != width.end())
          w = it->second;
        // LayerNorm additionally caches per-row mean/rstd for the backward
        // kernel (two f32 per row); RMSNorm caches rstd (one).
        if (op.kind == SmfOpKind::kLayerNorm)
          charge(SatMul(2 * sizeof(float), r));
        if (op.kind == SmfOpKind::kRmsNorm) charge(SatMul(sizeof(float), r));
        // Attention caches the probability matrix P[B,H,S,S] — flattened
        // [rows * heads, seq_len] — for the backward primitives; usually
        // the dominant transformer activation. heads/seq_len of zero mean
        // an invalid model the parser will reject; contribute nothing here
        // so the estimate stays a lower bound.
        if (op.kind == SmfOpKind::kAttention && op.attr0 > 0 &&
            model.seq_len > 0)
          charge(SatMul(SatMul(SatMul(r, op.attr0), model.seq_len),
                        sizeof(float)));
        break;
      }
    }
    if (w == 0) continue;
    width[op.output] = w;
    row_count[op.output] = r;
    charge(SatMul(SatMul(r, w), sizeof(float)));
  }
  if (!sum_activations) fp.activation_bytes = peak_activation;
  return fp;
}

}  // namespace

TrainingFootprint EstimateTrainingFootprint(const SmfModel& model,
                                            int64_t batch) {
  return EstimateFootprintImpl(model, batch, /*sum_activations=*/true);
}

TrainingFootprint EstimateFrozenForwardFootprint(const SmfModel& model,
                                                 int64_t batch) {
  return EstimateFootprintImpl(model, batch, /*sum_activations=*/false);
}

uint64_t DetectLocalMemoryBytes() {
#if defined(__APPLE__)
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) return 0;
  return mem;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || page_size <= 0) return 0;
  return SatMul(static_cast<uint64_t>(pages), static_cast<uint64_t>(page_size));
#endif
}

std::expected<void, std::string> CheckTrainableLocally(
    const TrainingFootprint& footprint, uint64_t budget_bytes) {
  const uint64_t budget =
      budget_bytes != 0 ? budget_bytes : DetectLocalMemoryBytes();
  if (budget == 0) return {};  // cannot prove infeasibility — do not reject
  const uint64_t need = footprint.total_bytes();
  if (need <= budget) return {};
  return seeml::diag::tokenizing::IngressError(
      "model is too big to train locally: weights " +
      FormatMiB(footprint.weight_bytes) + " + activations " +
      FormatMiB(footprint.activation_bytes) + " need at least " +
      FormatMiB(need) + ", but the local memory budget is " +
      FormatMiB(budget));
}

std::expected<void, std::string> CheckPlanFitsLocally(uint64_t arena_bytes,
                                                      uint64_t plan_bytes,
                                                      uint64_t budget_bytes) {
  const uint64_t budget =
      budget_bytes != 0 ? budget_bytes : DetectLocalMemoryBytes();
  if (budget == 0) return {};  // cannot prove infeasibility — do not reject
  const uint64_t need = SatAdd(arena_bytes, plan_bytes);
  if (need <= budget) return {};
  return seeml::diag::tokenizing::IngressError(
      "compiled update cannot run locally: arena " + FormatMiB(arena_bytes) +
      " + plan " + FormatMiB(plan_bytes) + " need " + FormatMiB(need) +
      ", but the local memory budget is " + FormatMiB(budget));
}

}  // namespace seeml::update
