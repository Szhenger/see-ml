// =============================================================================
// seeml-plan-probe — the C++ half of the frontier executor's differential
// oracle (tool/frontier_exec.py, P4 #78). Usage:
//
//   seeml-plan-probe --plan update_plan.seeu
//                    --section train|eval|merge|step
//                    --arena-in arena.bin --arena-out arena.bin
//                    [--trace trace.bin | --time N] [--lr-bits 0xHHHHHHHH]
//                    [--step N] [--backend cpu|metal|auto] [--threads N]
//                    [--version]
//
// A pure function of its inputs: (plan, section, arena image, step scalars)
// -> (arena image, write trace). It binds one executor backend to the
// caller's arena image and the plan's rodata and dispatches one section's
// instruction stream through ExecutorBackend::Execute — the very call the
// engine makes — with no feeder, no schedule, no gate and no checkpoint in
// the way. Python stages the arena (the persistent image, a batch in the
// I/O slots), names the step scalars the optimizer instructions read, and
// compares what comes back against its own, independently written
// implementation of every opcode.
//
// --trace records, after EVERY instruction (the backend is flushed first,
// so a deferred GPU instruction is complete), each extent the validator
// says the instruction writes:
//
//   u32 magic "SEPT"; u32 version 1; u64 instruction_count
//   per instruction: u32 index; u16 opcode; u16 extent_count;
//                    per extent: u64 arena_offset; u64 bytes; the bytes
//
// --time N is the pricing mode: one warm-up execution of the section, then
// N timed ones (each flushed) against the same staged arena, and one JSON
// line on stdout with the median / min / max milliseconds — the C++ number
// frontier_exec.py sets beside PyTorch's and MLX's for the same plan. It
// measures the instruction stream alone (no feeder, no evaluation), which
// is also all the Python side times.
//
// The extents come from DescribeInstruction — the bounds proof itself — so
// the trace can never claim less than the instruction was allowed to write.
// Every instruction is validated before anything executes, and the plan's
// self-hash is verified: the probe refuses what the engine would refuse.
// Strict arguments, as in every SeeML tool: exit 2 for a bad command line,
// 1 for a bad input.
// =============================================================================

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "runtime/executor/backend.h"
#include "runtime/validator/plan_validator.h"
#include "source/identity/hash.h"
#include "source/identity/version.h"
#include "source/parallel/parallel_for.h"
#include "source/plan/update_types.h"

namespace {

namespace up = seeml::update;
namespace rt = seeml::update_rt;

int Usage(const std::string& msg) {
  std::fprintf(stderr, "seeml-plan-probe: %s\n", msg.c_str());
  return 2;
}
int Fail(const std::string& msg) {
  std::fprintf(stderr, "seeml-plan-probe: %s\n", msg.c_str());
  return 1;
}

bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamoff size = f.tellg();
  if (size < 0) return false;
  out->resize(static_cast<size_t>(size));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out->data()), size);
  return static_cast<bool>(f);
}

bool ParseU64(const std::string& s, int base, uint64_t* out) {
  if (s.empty() || s[0] == '-' || s[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, base);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

template <typename T>
void Put(std::ofstream& f, T v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

}  // namespace

int main(int argc, char** argv) {
  std::string plan_path, section, arena_in, arena_out, trace_path;
  std::string backend_name = "cpu";
  uint64_t lr_bits = 0, step = 1, threads = 0, timed = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--version") {
      std::printf("seeml-plan-probe %s\n", up::kSeemlVersion);
      return 0;
    }
    if (i + 1 >= argc) return Usage(a + " is missing its value");
    const std::string v = argv[++i];
    if (a == "--plan") plan_path = v;
    else if (a == "--section") section = v;
    else if (a == "--arena-in") arena_in = v;
    else if (a == "--arena-out") arena_out = v;
    else if (a == "--trace") trace_path = v;
    else if (a == "--backend") backend_name = v;
    else if (a == "--lr-bits") {
      if (!ParseU64(v, 16, &lr_bits) || lr_bits > 0xFFFFFFFFull)
        return Usage("--lr-bits wants the f32 bit pattern in hex");
    } else if (a == "--step") {
      if (!ParseU64(v, 10, &step) || step == 0)
        return Usage("--step wants a 1-indexed optimizer step");
    } else if (a == "--time") {
      if (!ParseU64(v, 10, &timed) || timed == 0)
        return Usage("--time wants a positive execution count");
    } else if (a == "--threads") {
      if (!ParseU64(v, 10, &threads) || threads == 0)
        return Usage("--threads wants a positive integer");
    } else {
      return Usage("unknown flag '" + a + "'");
    }
  }
  if (plan_path.empty() || section.empty() || arena_in.empty() ||
      arena_out.empty())
    return Usage("--plan, --section, --arena-in and --arena-out are required");
  if (timed && !trace_path.empty())
    return Usage("--time and --trace are mutually exclusive (tracing "
                 "flushes after every instruction)");
  const auto kind = rt::ParseBackendKind(backend_name);
  if (!kind) return Usage("--backend wants cpu, metal or auto");
  if (threads) up::SetParallelThreadCount(static_cast<size_t>(threads));

  std::vector<uint8_t> plan;
  if (!ReadFile(plan_path, &plan)) return Fail("cannot read " + plan_path);
  if (plan.size() < sizeof(up::PlanHeader)) return Fail("plan is truncated");
  up::PlanHeader h;
  std::memcpy(&h, plan.data(), sizeof(h));
  if (h.magic != up::kSeeuMagic) return Fail("not a .seeu plan");
  if (h.version < up::kSeeuOldestReadable || h.version > up::kSeeuVersion)
    return Fail("unsupported plan version " + std::to_string(h.version));
  if (up::PlanSelfHash(plan.data(), plan.size(),
                       offsetof(up::PlanHeader, plan_hash)) != h.plan_hash)
    return Fail("plan integrity check failed (plan_hash)");

  uint64_t off = 0, count = 0;
  if (section == "train") off = h.train_instr_offset, count = h.train_instr_count;
  else if (section == "eval") off = h.eval_instr_offset, count = h.eval_instr_count;
  else if (section == "merge") off = h.merge_instr_offset, count = h.merge_instr_count;
  else if (section == "step") off = h.step_instr_offset, count = h.step_instr_count;
  else return Usage("--section wants train, eval, merge or step");
  uint64_t bytes = 0;
  if (!rt::MulOk(count, sizeof(up::UpdateInstruction), &bytes) ||
      !rt::RangeOk(off, bytes, plan.size()) ||
      !rt::RangeOk(h.rodata_offset, h.rodata_size, plan.size()))
    return Fail("a plan section lies outside the file");
  std::vector<up::UpdateInstruction> program(static_cast<size_t>(count));
  if (count) std::memcpy(program.data(), plan.data() + off, bytes);

  std::vector<rt::InstructionExtents> extents;
  extents.reserve(program.size());
  for (size_t i = 0; i < program.size(); ++i) {
    auto e = rt::DescribeInstruction(program[i], h.arena_size, h.rodata_size,
                                     h.version);
    if (!e)
      return Fail("instruction " + std::to_string(i) + ": " + e.error());
    extents.push_back(*e);
  }

  std::vector<uint8_t> image;
  if (!ReadFile(arena_in, &image)) return Fail("cannot read " + arena_in);
  if (image.size() != h.arena_size)
    return Fail("the arena image is " + std::to_string(image.size()) +
                " bytes; the plan declares " + std::to_string(h.arena_size));
  const size_t aligned =
      (static_cast<size_t>(h.arena_size) + rt::kArenaAlignment - 1) /
      rt::kArenaAlignment * rt::kArenaAlignment;
  struct Free {
    void operator()(uint8_t* p) const { std::free(p); }
  };
  std::unique_ptr<uint8_t, Free> arena(static_cast<uint8_t*>(
      std::aligned_alloc(rt::kArenaAlignment, aligned ? aligned
                                                      : rt::kArenaAlignment)));
  if (!arena) return Fail("cannot allocate the arena");
  std::memset(arena.get(), 0, aligned);
  std::memcpy(arena.get(), image.data(), image.size());

  auto selected = rt::CreateBackend(*kind);
  if (!selected) return Fail(selected.error());
  rt::ExecutorBackend& backend = *selected->backend;
  if (auto r = backend.Bind(arena.get(), aligned,
                            plan.data() + h.rodata_offset, h.rodata_size,
                            plan.size() - h.rodata_offset);
      !r)
    return Fail("bind: " + r.error());
  rt::kernels::KernelPolicy policy;
  if (h.gemm_tile_k) policy.gemm_tiles.k = h.gemm_tile_k;
  if (h.gemm_tile_n) policy.gemm_tiles.n = h.gemm_tile_n;
  backend.Configure(policy);

  float lr = 0.0f;
  const uint32_t lr32 = static_cast<uint32_t>(lr_bits);
  std::memcpy(&lr, &lr32, sizeof(lr));
  const rt::StepParams params{.lr = lr, .beta1 = h.beta1, .beta2 = h.beta2,
                              .eps = h.eps, .weight_decay = h.weight_decay,
                              .step = step};

  if (timed) {
    using Clock = std::chrono::steady_clock;
    std::vector<double> ms;
    for (uint64_t run = 0; run <= timed; ++run) {  // run 0 is the warm-up
      const auto t0 = Clock::now();
      for (size_t i = 0; i < program.size(); ++i)
        if (auto r = backend.Execute(program[i], params); !r)
          return Fail("instruction " + std::to_string(i) + ": " + r.error());
      if (auto r = backend.Flush(); !r) return Fail("flush: " + r.error());
      if (run)
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() -
                                                               t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const size_t n = ms.size();
    std::printf("{\"backend\": \"%s\", \"section\": \"%s\", "
                "\"executions\": %zu, \"ms_median\": %.4f, "
                "\"ms_min\": %.4f, \"ms_max\": %.4f}\n",
                backend.name(), section.c_str(), n,
                n % 2 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]),
                ms.front(), ms.back());
  }

  std::ofstream trace;
  if (!timed && !trace_path.empty()) {
    trace.open(trace_path, std::ios::binary | std::ios::trunc);
    if (!trace) return Fail("cannot open " + trace_path);
    Put<uint32_t>(trace, 0x54504553u);  // "SEPT"
    Put<uint32_t>(trace, 1);
    Put<uint64_t>(trace, count);
  }
  for (size_t i = 0; !timed && i < program.size(); ++i) {
    if (auto r = backend.Execute(program[i], params); !r)
      return Fail("instruction " + std::to_string(i) + ": " + r.error());
    if (!trace.is_open()) continue;
    if (auto r = backend.Flush(); !r) return Fail("flush: " + r.error());
    uint16_t writes = 0;
    for (size_t k = 0; k < extents[i].count; ++k)
      writes += extents[i].ranges[k].write ? 1 : 0;
    Put<uint32_t>(trace, static_cast<uint32_t>(i));
    Put<uint16_t>(trace, program[i].opcode);
    Put<uint16_t>(trace, writes);
    for (size_t k = 0; k < extents[i].count; ++k) {
      const rt::OperandExtent& e = extents[i].ranges[k];
      if (!e.write) continue;
      Put<uint64_t>(trace, e.off);
      Put<uint64_t>(trace, e.bytes);
      trace.write(reinterpret_cast<const char*>(arena.get() + e.off),
                  static_cast<std::streamsize>(e.bytes));
    }
  }
  if (auto r = backend.Flush(); !r) return Fail("flush: " + r.error());
  if (trace.is_open()) {
    trace.close();
    if (trace.fail()) return Fail("short write to " + trace_path);
  }
  std::ofstream out(arena_out, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(arena.get()),
            static_cast<std::streamsize>(h.arena_size));
  out.close();
  if (out.fail()) return Fail("short write to " + arena_out);
  std::fprintf(stderr, "seeml-plan-probe: %s, %" PRIu64 " instruction(s), %s\n",
               section.c_str(), count, backend.name());
  return 0;
}
