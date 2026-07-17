#include "compiler/backend/arena_binder.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace seeml::update {

namespace sir = seecpp::sir;

namespace {

uint64_t ValueBytes(const sir::Value* v) {
  return AlignUp(v->shape().byteSize(v->dtype()));
}

}  // namespace

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
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i)
      max_abs = std::max(max_abs, std::fabs(data[i]));
    // An all-zero weight quantizes to zeros under any scale; 1.0 keeps the
    // reciprocal finite.
    scales[v] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
  });
  return scales;
}

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
  std::sort(intervals.begin(), intervals.end(),
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

std::expected<ArenaBinding, std::string> BindArena(
    sir::Block& train_block, const GraphBuild& build,
    const std::unordered_set<const sir::Value*>& pinned,
    const std::unordered_map<const sir::Value*, float>& quant_scales) {
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

  // --- RODATA: pack every frozen weight, dedup-free sequential layout.
  // Weights selected for quantization pack as per-tensor symmetric int8
  // (4x smaller); everything else as raw f32.
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() != "sc_mem.weight") return;
    const sir::Value* v = op->result(0);
    auto src = build.weight_sources.find(v);
    if (src == build.weight_sources.end()) return;  // caught below
    const uint64_t offset = AlignUp(binding.rodata.size());
    if (auto q = quant_scales.find(v); q != quant_scales.end()) {
      const auto* data =
          reinterpret_cast<const float*>(src->second->data.data());
      const size_t count = src->second->byte_size / sizeof(float);
      binding.rodata.resize(offset + count, 0);
      auto* dst = reinterpret_cast<int8_t*>(binding.rodata.data() + offset);
      const float inv_scale = 1.0f / q->second;
      for (size_t i = 0; i < count; ++i) {
        const float r = std::round(data[i] * inv_scale);
        dst[i] = static_cast<int8_t>(std::clamp(r, -127.0f, 127.0f));
      }
    } else {
      binding.rodata.resize(offset + src->second->byte_size, 0);
      std::memcpy(binding.rodata.data() + offset, src->second->data.data(),
                  src->second->byte_size);
    }
    binding.refs[v] = MakeRodataRef(offset);
  });

  bool missing_source = false;
  train_block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_mem.weight" &&
        !binding.refs.contains(op->result(0)))
      missing_source = true;
  });
  if (missing_source)
    return std::unexpected(
        "UpdateCompiler: frozen weight without SMF backing data");

  // --- TRANSIENT segment: liveness-scanned workspace after the IO prefix.
  binding.arena_size =
      LinearScanTransients(train_block, AlignUp(binding.io_end), binding.refs,
                           pinned, binding.refs);
  return binding;
}

}  // namespace seeml::update
