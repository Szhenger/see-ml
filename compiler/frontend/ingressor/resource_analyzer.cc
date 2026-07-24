#include "compiler/frontend/ingressor/resource_analyzer.h"

#include <string_view>
#include <unordered_map>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

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
  return std::to_string((bytes + (1 << 20) - 1) >> 20) + " MiB";
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

TrainingFootprint EstimateTrainingFootprint(const SmfModel& model,
                                            int64_t batch) {
  TrainingFootprint fp;
  const uint64_t rows = batch > 0 ? static_cast<uint64_t>(batch) : 0;

  // Frozen weights: one resident copy each (they at least fill rodata).
  // Track each tensor's last dimension as its "width" to seed propagation;
  // the reader has already validated dims, so widths are positive (or the
  // dynamic batch dim, which never appears last).
  std::unordered_map<std::string_view, const SmfTensor*> tensors;
  std::unordered_map<std::string_view, uint64_t> width;
  tensors.reserve(model.tensors.size());
  for (const SmfTensor& t : model.tensors) {
    tensors[t.name] = &t;
    if (t.is_const) fp.weight_bytes = SatAdd(fp.weight_bytes, t.byte_size);
    if (!t.dims.empty() && t.dims.back() > 0)
      width[t.name] = static_cast<uint64_t>(t.dims.back());
  }

  // Forward activations: propagate output widths with the parser's shape
  // rules. Every activation is [batch, width] f32 and is cached for the
  // backward pass. An unresolvable width contributes zero — the estimate
  // must stay a lower bound.
  for (const SmfOp& op : model.ops) {
    uint64_t w = 0;
    switch (op.kind) {
      case SmfOpKind::kMatMul: {
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
      case SmfOpKind::kLayerNorm: {
        if (op.inputs.empty()) break;
        if (auto it = width.find(op.inputs[0]); it != width.end())
          w = it->second;
        // LayerNorm additionally caches per-row mean/rstd for the backward
        // kernel: two f32 per batch row.
        if (op.kind == SmfOpKind::kLayerNorm)
          fp.activation_bytes =
              SatAdd(fp.activation_bytes, SatMul(2 * sizeof(float), rows));
        break;
      }
    }
    if (w == 0) continue;
    width[op.output] = w;
    fp.activation_bytes = SatAdd(fp.activation_bytes,
                                 SatMul(SatMul(rows, w), sizeof(float)));
  }
  return fp;
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
  return std::unexpected(
      "Ingressor: model is too big to train locally: weights " +
      FormatMiB(footprint.weight_bytes) + " + activations " +
      FormatMiB(footprint.activation_bytes) + " need at least " +
      FormatMiB(need) + ", but the local memory budget is " +
      FormatMiB(budget));
}

}  // namespace seeml::update
