#include "compiler/backend/allocation/arena_binder.h"

#include "source/plan/bf16.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

// For kWeightSweepGrain: the int8 pack below sweeps with the same chunk
// geometry as the quantization review's max-abs scan that selected the weights.
#include "compiler/analysis/statistics/quantization.h"
#include "compiler/diagnostics/generating/error.h"
#include "source/parallel/parallel_for.h"

namespace seeml::update {

namespace sir = seeml::sir;

namespace {

uint64_t ValueBytes(const sir::Value* v) {
  return AlignUp(v->shape().byteSize(v->dtype()));
}

}  // namespace

uint64_t LinearScanTransients(
    sir::Block& block, uint64_t base,
    const std::unordered_map<const sir::Value*, uint64_t>& already_bound,
    const std::unordered_set<const sir::Value*>& pinned,
    std::unordered_map<const sir::Value*, uint64_t>& refs_out) {
  struct Interval {
    const sir::Value* value;
    size_t start, end;
    uint64_t bytes;
  };

  std::unordered_map<const sir::Value*, size_t> birth, death;
  std::vector<const sir::Value*> order;
  size_t tick = 0;
  block.walk([&](sir::Operation* op) {
    if (!op->mnemonic().starts_with("sc_mem.")) {
      for (const auto& res : op->results()) {
        if (already_bound.contains(res.get())) continue;
        birth[res.get()] = tick;
        death[res.get()] = tick;
        order.push_back(res.get());
      }
    }
    for (sir::Value* operand : op->operands())
      if (auto it = death.find(operand); it != death.end()) it->second = tick;
    ++tick;
  });

  std::vector<Interval> intervals;
  intervals.reserve(order.size());
  for (const sir::Value* v : order)
    intervals.push_back({v, birth[v],
                         pinned.contains(v) ? SIZE_MAX : death[v],
                         ValueBytes(v)});
  // stable_sort, not sort: same-tick births (a LayerNorm's result + mean +
  // rstd, say) tie on start, and an unspecified equal-key order would let
  // first-fit offsets — and with them the plan bytes and plan_hash — differ
  // across standard-library implementations for identical input. Stability
  // pins ties to the deterministic discovery order.
  std::stable_sort(intervals.begin(), intervals.end(),
                   [](const Interval& a, const Interval& b) {
                     return a.start < b.start;
                   });

  struct ActiveBlock {
    uint64_t start, end;
    size_t free_after;
  };
  // `active` is kept sorted by start at all times: expiry (erase_if) is
  // order-preserving and each new block is inserted at its sorted position,
  // so no per-interval re-sort is needed.
  std::vector<ActiveBlock> active;
  active.reserve(intervals.size());
  uint64_t high_water = base;

  for (const Interval& iv : intervals) {
    std::erase_if(active, [&](const ActiveBlock& ab) {
      return ab.free_after != SIZE_MAX && ab.free_after < iv.start;
    });
    uint64_t offset = base;
    for (const ActiveBlock& ab : active) {
      if (offset + iv.bytes <= ab.start) break;  // first fit
      offset = std::max(offset, ab.end);
    }
    refs_out[iv.value] = MakeArenaRef(offset);
    const ActiveBlock fresh{offset, offset + iv.bytes, iv.end};
    active.insert(std::upper_bound(active.begin(), active.end(), fresh,
                                   [](const ActiveBlock& a,
                                      const ActiveBlock& b) {
                                     return a.start < b.start;
                                   }),
                  fresh);
    high_water = std::max(high_water, offset + iv.bytes);
  }
  return high_water;
}

/// Per-output-column int8 scales of W [K, M] (plan v17): column m's
/// max |w| / 127. An all-zero column takes 1 (its levels are all zero under
/// any scale); a column so small that max/127 is subnormal takes FLT_MIN,
/// which keeps the pack's divide finite and rounds the column to at most
/// its 127 levels. Max is order-insensitive, and the row chunks are the
/// fixed ParallelFor geometry, so the scales are thread-count-invariant.
std::vector<float> ColumnScales(const SmfTensor& w) {
  const auto* data = reinterpret_cast<const float*>(w.data.data());
  const size_t count = w.byte_size / sizeof(float);
  const size_t cols =
      w.dims.size() == 2 ? static_cast<size_t>(w.dims[1]) : count;
  const size_t rows = cols ? count / cols : 0;
  const size_t grain = std::max<size_t>(1, kWeightSweepGrain / std::max<size_t>(1, cols));
  const size_t chunks = ParallelChunkCount(rows, grain);
  std::vector<float> partial(chunks * cols, 0.0f);
  ParallelFor(rows, grain, [&](size_t r0, size_t r1, size_t c) {
    float* m = partial.data() + c * cols;
    for (size_t r = r0; r < r1; ++r)
      for (size_t j = 0; j < cols; ++j)
        m[j] = std::max(m[j], std::fabs(data[r * cols + j]));
  });
  std::vector<float> scales(cols, 0.0f);
  for (size_t c = 0; c < chunks; ++c)
    for (size_t j = 0; j < cols; ++j)
      scales[j] = std::max(scales[j], partial[c * cols + j]);
  for (float& s : scales)
    s = s == 0.0f ? 1.0f
                  : std::max(s / 127.0f, std::numeric_limits<float>::min());
  return scales;
}

void PackRodata(const RodataPack& pack, uint8_t* dst) {
  const auto* data = reinterpret_cast<const float*>(pack.source->data.data());
  const size_t count = pack.source->byte_size / sizeof(float);
  switch (pack.storage) {
    case RodataPack::Storage::kF32:
      std::memcpy(dst, pack.source->data.data(), pack.bytes);
      return;
    case RodataPack::Storage::kInt8: {
      // Divide, don't multiply by a hoisted reciprocal: for a denormal
      // scale 1/scale is +Inf, which clamps every nonzero element to +-127
      // and turns zeros into clamp(NaN) — UB on the int8 cast. The
      // quantizer refuses denormal scales, but the pack must not rely on
      // that upstream discipline for memory safety.
      auto* out = reinterpret_cast<int8_t*>(dst);
      if (!pack.column_scales.empty()) {
        // Per column (v17): element i of W [K, M] is column i % M, divided
        // by that column's scale; the scales follow the levels, f32.
        const float* cs = pack.column_scales.data();
        const size_t cols = pack.column_scales.size();
        ParallelFor(count, kWeightSweepGrain, [&](size_t b, size_t e, size_t) {
          for (size_t i = b; i < e; ++i) {
            const float r = std::round(data[i] / cs[i % cols]);
            out[i] = static_cast<int8_t>(std::clamp(r, -127.0f, 127.0f));
          }
        });
        std::memcpy(dst + (pack.scales_offset - pack.offset), cs,
                    cols * sizeof(float));
        return;
      }
      const float scale = pack.scale;
      ParallelFor(count, kWeightSweepGrain, [&](size_t b, size_t e, size_t) {
        for (size_t i = b; i < e; ++i) {
          const float r = std::round(data[i] / scale);
          out[i] = static_cast<int8_t>(std::clamp(r, -127.0f, 127.0f));
        }
      });
      return;
    }
    case RodataPack::Storage::kBf16: {
      // Round-to-nearest-even of the f32 bits, 2 bytes per element; the
      // kernels widen exactly. Same chunk geometry as the int8 pack.
      auto* out = reinterpret_cast<uint16_t*>(dst);
      ParallelFor(count, kWeightSweepGrain, [&](size_t b, size_t e, size_t) {
        for (size_t i = b; i < e; ++i) out[i] = Float32ToBf16Bits(data[i]);
      });
      return;
    }
  }
}

std::expected<ArenaBinding, std::string> BindArena(
    sir::Block& train_block, const GraphBuild& build,
    const std::unordered_set<const sir::Value*>& pinned,
    const std::unordered_map<const sir::Value*, float>& quant_scales,
    const std::unordered_set<const sir::Value*>& bf16_weights) {
  ArenaBinding binding;

  // --- PERSISTENT segment: trainable adapters + optimizer state at offset 0.
  uint64_t cursor = 0;
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.param") return;
    const sir::Value* v = op->result(0);
    const uint64_t bytes = ValueBytes(v);
    binding.refs[v] = MakeArenaRef(cursor);
    binding.params.push_back(
        {.value = v,
         .offset = cursor,
         .bytes = bytes,
         .init = op->getAttrAs<std::string>("init").value_or("zeros"),
         .std = op->getAttrAs<float>("std").value_or(0.02f),
         .seed = static_cast<uint64_t>(
             op->getAttrAs<int64_t>("seed").value_or(0))});
    cursor += bytes;
  });
  binding.persistent_size = cursor;

  // --- IO segment: batch input + label slots, right after PERSISTENT.
  for (const auto& arg : train_block.arguments()) {
    binding.refs[arg.get()] = MakeArenaRef(cursor);
    cursor += ValueBytes(arg.get());
  }
  binding.io_end = cursor;

  // --- RODATA: lay out every frozen weight, dedup-free and sequential.
  // Weights selected for quantization take one int8 per element (4x
  // smaller), bf16 weights two bytes; everything else raw f32. Layout
  // only — PackRodata writes the bytes, once, into the plan.
  uint64_t rodata_cursor = 0;
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.weight") return;
    const sir::Value* v = op->result(0);
    auto src = build.weight_sources.find(v);
    if (src == build.weight_sources.end()) return;  // caught below
    RodataPack pack{.source = src->second, .offset = AlignUp(rodata_cursor)};
    const uint64_t count = src->second->byte_size / sizeof(float);
    if (auto q = quant_scales.find(v); q != quant_scales.end()) {
      pack.storage = RodataPack::Storage::kInt8;
      pack.scale = q->second;
      // v17: one scale per output column of W [K, M], right after the
      // int8 levels (f32-aligned); the pack's bytes cover both.
      pack.column_scales = ColumnScales(*src->second);
      pack.scales_offset = (pack.offset + count + 3) & ~uint64_t{3};
      pack.bytes = pack.scales_offset - pack.offset +
                   pack.column_scales.size() * sizeof(float);
      binding.quant_column_scales[v] = MakeRodataRef(pack.scales_offset);
    } else if (bf16_weights.contains(v)) {
      pack.storage = RodataPack::Storage::kBf16;
      pack.bytes = count * sizeof(uint16_t);
    } else {
      pack.bytes = src->second->byte_size;
    }
    rodata_cursor = pack.offset + pack.bytes;
    binding.refs[v] = MakeRodataRef(pack.offset);
    binding.rodata_packs.push_back(pack);
  });
  binding.rodata_size = rodata_cursor;

  bool missing_source = false;
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_mem.weight" &&
        !binding.refs.contains(op->result(0)))
      missing_source = true;
  });
  if (missing_source)
    return seeml::diag::generating::Error(
        seeml::diag::generating::kArenaBinder,
        "frozen weight without SMF backing data");

  // --- TRANSIENT segment: liveness-scanned workspace after the IO prefix.
  binding.arena_size =
      LinearScanTransients(train_block, AlignUp(binding.io_end), binding.refs,
                           pinned, binding.refs);
  return binding;
}

}  // namespace seeml::update
