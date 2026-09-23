// =============================================================================
// seeml-plan-probe — the C++ half of the frontier executor's differential
// oracle (tool/frontier_exec.py, P4 #78). Usage:
//
//   seeml-plan-probe --plan update_plan.seeu
//                    --section train|eval|merge|step
//                    --arena-in arena.bin --arena-out arena.bin
//                    [--trace trace.bin | --time N | --profile N]
//                    [--lr-bits 0xHHHHHHHH]
//                    [--step N] [--backend cpu|metal|auto] [--threads N]
//                    [--source model.smf]
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
// --profile N is Tier B's question — why did Tier A move — asked of one
// plan: N executions (after a warm-up) with the backend flushed after every
// instruction, wall time accumulated per opcode and, for the GEMM family,
// per shape; one JSON object on stdout, largest first. Flushing per
// instruction serializes a GPU backend, so the numbers attribute time,
// they do not predict the unprofiled step.
//
// The extents come from DescribeInstruction — the bounds proof itself — so
// the trace can never claim less than the instruction was allowed to write.
// Every instruction is validated before anything executes, and the plan's
// self-hash is verified: the probe refuses what the engine would refuse.
// Strict arguments, as in every SeeML tool: exit 2 for a bad command line,
// 1 for a bad input.
//
// Paths. This is a build-host tool run with its invoker's own privileges,
// so a path names whatever that user may already read or write; what the
// probe adds is that it never acts on a path it has not classified. Inputs
// must be existing REGULAR files (not a directory, FIFO or device that
// would block or stream forever). Outputs are staged beside their
// destination and renamed into place only when complete, the destination
// must be absent or a regular file, and a symlink there is refused rather
// than followed — a failed run leaves no truncated image a later step
// could mistake for a result (the discipline seeml-bench keeps for
// bench.json).
// =============================================================================

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "runtime/executor/backend.h"
#include "runtime/validator/plan_validator.h"
#include "source/identity/hash.h"
#include "source/identity/version.h"
#include "source/parallel/parallel_for.h"
#include "source/plan/update_types.h"
#include "tool/probe_trace.h"

namespace {

namespace fs = std::filesystem;
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

/// Strict argument cursor, the seeml-update-compile idiom: every flag must
/// be known and every value present; an argv slot no Take* call consumed
/// is an error, never a default.
class Args {
 public:
  Args(int argc, char** argv)
      : argc_(argc), argv_(argv), taken_(static_cast<size_t>(argc), false) {}

  bool Take(const char* flag) {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i] && std::strcmp(argv_[i], flag) == 0) {
        taken_[i] = true;
        return true;
      }
    return false;
  }

  /// The value after `flag`, consuming both slots. A following "--flag" is
  /// a missing value, not a value.
  std::optional<std::string> TakeValue(const char* flag) {
    for (int i = 1; i < argc_; ++i) {
      if (taken_[i] || std::strcmp(argv_[i], flag) != 0) continue;
      if (i + 1 >= argc_ || std::strncmp(argv_[i + 1], "--", 2) == 0) {
        if (!missing_value_) missing_value_ = flag;
        return std::nullopt;
      }
      taken_[i] = taken_[i + 1] = true;
      return std::string(argv_[i + 1]);
    }
    return std::nullopt;
  }

  const std::optional<std::string>& MissingValue() const {
    return missing_value_;
  }
  std::optional<std::string> FirstUnknown() const {
    for (int i = 1; i < argc_; ++i)
      if (!taken_[i]) return std::string(argv_[i]);
    return std::nullopt;
  }

 private:
  int argc_;
  char** argv_;
  std::vector<bool> taken_;
  std::optional<std::string> missing_value_;
};

bool ParseU64(const std::string& s, int base, uint64_t* out) {
  if (s.empty() || s[0] == '-' || s[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, base);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

/// Everything the command line says, parsed and range-checked.
struct Options {
  fs::path plan, arena_in, arena_out, trace;  // trace empty = no trace
  fs::path source;  // v17: the model file an eval program reads (E12)
  std::string section;
  rt::BackendKind backend = rt::BackendKind::kCpu;
  uint32_t lr_bits = 0;
  uint64_t step = 1, threads = 0, timed = 0, profiled = 0;
};

/// An input path the probe will read: it must name an existing regular
/// file. Returns the canonical path (symlinks resolved), so the file that
/// was classified is the file that is opened.
std::optional<fs::path> InputFile(const std::string& arg, std::string* why) {
  std::error_code ec;
  const fs::path canonical = fs::canonical(fs::path(arg), ec);
  if (ec) {
    *why = "cannot resolve '" + arg + "': " + ec.message();
    return std::nullopt;
  }
  if (!fs::is_regular_file(canonical, ec) || ec) {
    *why = "'" + arg + "' is not a regular file";
    return std::nullopt;
  }
  return canonical;
}

/// An output path the probe will create or replace: its directory must
/// exist, and the destination must be absent or a regular file — never a
/// symlink (not followed), a directory or a device. Returns the path
/// inside the canonical directory.
std::optional<fs::path> OutputFile(const std::string& arg, std::string* why) {
  std::error_code ec;
  const fs::path given(arg);
  if (!given.has_filename()) {
    *why = "'" + arg + "' names no file";
    return std::nullopt;
  }
  const fs::path parent = given.has_parent_path() ? given.parent_path()
                                                  : fs::path(".");
  const fs::path dir = fs::canonical(parent, ec);
  if (ec || !fs::is_directory(dir, ec) || ec) {
    *why = "the directory of '" + arg + "' does not exist";
    return std::nullopt;
  }
  const fs::path out = dir / given.filename();
  const fs::file_status st = fs::symlink_status(out, ec);
  if (fs::exists(st) && !fs::is_regular_file(st)) {
    *why = "'" + arg + "' exists and is not a regular file (a symlink is "
           "refused, not followed)";
    return std::nullopt;
  }
  return out;
}

/// Parses argv into `opts`. Returns 0, or the exit code after reporting.
int ParseOptions(int argc, char** argv, Options* opts) {
  Args args(argc, argv);
  const auto plan = args.TakeValue("--plan");
  const auto section = args.TakeValue("--section");
  const auto arena_in = args.TakeValue("--arena-in");
  const auto arena_out = args.TakeValue("--arena-out");
  const auto trace = args.TakeValue("--trace");
  const auto backend = args.TakeValue("--backend");
  const auto lr_bits = args.TakeValue("--lr-bits");
  const auto step = args.TakeValue("--step");
  const auto threads = args.TakeValue("--threads");
  const auto timed = args.TakeValue("--time");
  const auto profiled = args.TakeValue("--profile");
  const auto source = args.TakeValue("--source");
  if (const auto& flag = args.MissingValue())
    return Usage(*flag + " is missing its value");
  if (const auto unknown = args.FirstUnknown())
    return Usage("unknown argument '" + *unknown + "'");
  if (!plan || !section || !arena_in || !arena_out)
    return Usage("--plan, --section, --arena-in and --arena-out are required");
  if (*section != "train" && *section != "eval" && *section != "merge" &&
      *section != "step")
    return Usage("--section wants train, eval, merge or step");
  opts->section = *section;
  if (backend) {
    const auto kind = rt::ParseBackendKind(*backend);
    if (!kind) return Usage("--backend wants cpu, metal or auto");
    opts->backend = *kind;
  }
  uint64_t bits = 0;
  if (lr_bits && (!ParseU64(*lr_bits, 16, &bits) || bits > 0xFFFFFFFFull))
    return Usage("--lr-bits wants the f32 bit pattern in hex");
  opts->lr_bits = static_cast<uint32_t>(bits);
  if (step && (!ParseU64(*step, 10, &opts->step) || opts->step == 0))
    return Usage("--step wants a 1-indexed optimizer step");
  if (threads && (!ParseU64(*threads, 10, &opts->threads) ||
                  opts->threads == 0 || opts->threads > 4096))
    return Usage("--threads wants an integer in [1, 4096]");
  if (timed && (!ParseU64(*timed, 10, &opts->timed) || opts->timed == 0 ||
                opts->timed > 1000000))
    return Usage("--time wants an execution count in [1, 1000000]");
  if (profiled && (!ParseU64(*profiled, 10, &opts->profiled) ||
                   opts->profiled == 0 || opts->profiled > 1000000))
    return Usage("--profile wants an execution count in [1, 1000000]");
  if ((timed ? 1 : 0) + (trace ? 1 : 0) + (profiled ? 1 : 0) > 1)
    return Usage("--time, --trace and --profile are mutually exclusive");

  // Classify every path before anything is opened (see the banner).
  std::string why;
  const auto in_plan = InputFile(*plan, &why);
  if (!in_plan) return Fail(why);
  const auto in_arena = InputFile(*arena_in, &why);
  if (!in_arena) return Fail(why);
  const auto out_arena = OutputFile(*arena_out, &why);
  if (!out_arena) return Fail(why);
  opts->plan = *in_plan;
  opts->arena_in = *in_arena;
  opts->arena_out = *out_arena;
  if (source) {
    if (*section != "eval")
      return Usage("--source applies to --section eval only");
    const auto in_source = InputFile(*source, &why);
    if (!in_source) return Fail(why);
    opts->source = *in_source;
  }
  if (trace) {
    const auto out_trace = OutputFile(*trace, &why);
    if (!out_trace) return Fail(why);
    opts->trace = *out_trace;
  }
  return 0;
}

bool ReadFile(const fs::path& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamoff size = f.tellg();
  if (size < 0) return false;
  out->resize(static_cast<size_t>(size));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out->data()), size);
  return static_cast<bool>(f);
}

/// A file written beside its destination and renamed into place by
/// Commit(); destroyed uncommitted, it removes what it staged.
class StagedFile {
 public:
  explicit StagedFile(const fs::path& destination)
      : destination_(destination), staging_(destination) {
    staging_ += ".tmp";
    stream_.open(staging_, std::ios::binary | std::ios::trunc);
  }
  ~StagedFile() {
    if (committed_) return;
    stream_.close();
    std::error_code ec;
    fs::remove(staging_, ec);
  }
  StagedFile(const StagedFile&) = delete;
  StagedFile& operator=(const StagedFile&) = delete;

  bool ok() const { return static_cast<bool>(stream_); }
  void Write(const void* data, size_t bytes) {
    stream_.write(static_cast<const char*>(data),
                  static_cast<std::streamsize>(bytes));
  }
  template <typename T>
  void Put(T value) {
    Write(&value, sizeof(value));
  }
  /// Flushes, closes and renames; false if any byte failed to land.
  bool Commit() {
    stream_.close();
    if (stream_.fail()) return false;
    std::error_code ec;
    fs::rename(staging_, destination_, ec);
    committed_ = !ec;
    return committed_;
  }

 private:
  fs::path destination_, staging_;
  std::ofstream stream_;
  bool committed_ = false;
};

/// One validated section of one plan, ready to dispatch.
struct Program {
  up::PlanHeader header;
  std::vector<up::UpdateInstruction> instructions;
  std::vector<rt::InstructionExtents> extents;  // what each may write
};

/// Verifies the plan as the engine's loader would (magic, version window,
/// self-hash, section bounds) and validates every instruction of `section`
/// before any of them runs. Returns an error message, or empty.
std::string LoadProgram(const std::vector<uint8_t>& plan,
                        const std::string& section, Program* out) {
  if (plan.size() < sizeof(up::PlanHeader)) return "plan is truncated";
  up::PlanHeader& h = out->header;
  std::memcpy(&h, plan.data(), sizeof(h));
  if (h.magic != up::kSeeuMagic) return "not a .seeu plan";
  if (h.version < up::kSeeuOldestReadable || h.version > up::kSeeuVersion)
    return "unsupported plan version " + std::to_string(h.version);
  if (up::PlanSelfHash(plan.data(), plan.size(),
                       offsetof(up::PlanHeader, plan_hash)) != h.plan_hash)
    return "plan integrity check failed (plan_hash)";

  uint64_t off = h.train_instr_offset, count = h.train_instr_count;
  if (section == "eval") off = h.eval_instr_offset, count = h.eval_instr_count;
  if (section == "merge")
    off = h.merge_instr_offset, count = h.merge_instr_count;
  if (section == "step") off = h.step_instr_offset, count = h.step_instr_count;
  uint64_t bytes = 0;
  if (!rt::MulOk(count, sizeof(up::UpdateInstruction), &bytes) ||
      !rt::RangeOk(off, bytes, plan.size()) ||
      !rt::RangeOk(h.rodata_offset, h.rodata_size, plan.size()))
    return "a plan section lies outside the file";
  out->instructions.resize(static_cast<size_t>(count));
  if (count) std::memcpy(out->instructions.data(), plan.data() + off, bytes);

  out->extents.reserve(out->instructions.size());
  for (size_t i = 0; i < out->instructions.size(); ++i) {
    auto e = rt::DescribeInstruction(out->instructions[i], h.arena_size,
                                     h.rodata_size, h.version,
                                     /*allow_source=*/section == "eval");
    if (!e) return "instruction " + std::to_string(i) + ": " + e.error();
    out->extents.push_back(*e);
  }
  return {};
}

/// Pricing mode: one warm-up execution, then `timed` flushed executions
/// against the same arena; one JSON line on stdout.
std::string TimeProgram(rt::ExecutorBackend& backend, const Program& program,
                        const rt::StepParams& params,
                        const std::string& section, uint64_t timed) {
  using Clock = std::chrono::steady_clock;
  std::vector<double> ms;
  for (uint64_t run = 0; run <= timed; ++run) {  // run 0 is the warm-up
    const auto t0 = Clock::now();
    for (size_t i = 0; i < program.instructions.size(); ++i)
      if (auto r = backend.Execute(program.instructions[i], params); !r)
        return "instruction " + std::to_string(i) + ": " + r.error();
    if (auto r = backend.Flush(); !r) return "flush: " + r.error();
    if (run)
      ms.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  const size_t n = ms.size();
  std::printf("{\"backend\": \"%s\", \"section\": \"%s\", "
              "\"executions\": %zu, \"ms_median\": %.4f, "
              "\"ms_min\": %.4f, \"ms_max\": %.4f}\n",
              backend.name(), section.c_str(), n,
              n % 2 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]),
              ms.front(), ms.back());
  return {};
}

/// Profile mode: per-opcode (and per-GEMM-shape) wall time over `runs`
/// executions, each instruction flushed; one JSON object on stdout.
std::string ProfileProgram(rt::ExecutorBackend& backend,
                           const Program& program,
                           const rt::StepParams& params, uint64_t runs) {
  using Clock = std::chrono::steady_clock;
  std::map<std::string, std::pair<double, uint64_t>> rows;  // ms, count
  auto key_of = [](const up::UpdateInstruction& ins) {
    std::string key = "op" + std::to_string(ins.opcode);
    const auto op = static_cast<up::OpCode>(ins.opcode);
    const bool gemm =
        op == up::OpCode::kGemmNN || op == up::OpCode::kGemmNT ||
        op == up::OpCode::kGemmTN || op == up::OpCode::kGemmAccNN ||
        op == up::OpCode::kGemmNNQ8 || op == up::OpCode::kGemmNTQ8 ||
        op == up::OpCode::kGemmNNBF16 || op == up::OpCode::kGemmNTBF16;
    if (gemm)
      key += " M" + std::to_string(ins.out[0]) + " N" +
             std::to_string(ins.out[1]) + " K" + std::to_string(ins.out[2]);
    return key;
  };
  double total = 0.0;
  for (uint64_t run = 0; run <= runs; ++run) {  // run 0 is the warm-up
    for (size_t i = 0; i < program.instructions.size(); ++i) {
      const auto t0 = Clock::now();
      if (auto r = backend.Execute(program.instructions[i], params); !r)
        return "instruction " + std::to_string(i) + ": " + r.error();
      if (auto r = backend.Flush(); !r) return "flush: " + r.error();
      if (run == 0) continue;
      const double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      auto& row = rows[key_of(program.instructions[i])];
      row.first += ms;
      ++row.second;
      total += ms;
    }
  }
  std::vector<std::pair<std::string, std::pair<double, uint64_t>>> sorted(
      rows.begin(), rows.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second.first > b.second.first;
  });
  const double n = static_cast<double>(runs);
  std::printf("{\"backend\": \"%s\", \"executions\": %" PRIu64
              ", \"ms_per_execution\": %.4f, \"rows\": [",
              backend.name(), runs, total / n);
  for (size_t i = 0; i < sorted.size(); ++i)
    std::printf("%s{\"key\": \"%s\", \"ms\": %.4f, \"calls\": %" PRIu64 "}",
                i ? ", " : "", sorted[i].first.c_str(),
                sorted[i].second.first / n,
                sorted[i].second.second / runs);
  std::printf("]}\n");
  return {};
}

/// Oracle mode: executes the section once; with `trace`, flushes after
/// every instruction and records each extent it was allowed to write.
std::string RunProgram(rt::ExecutorBackend& backend, const Program& program,
                       const rt::StepParams& params, const uint8_t* arena,
                       StagedFile* trace) {
  if (trace) {
    trace->Put<uint32_t>(seeml::tool::kProbeTraceMagic);
    trace->Put<uint32_t>(seeml::tool::kProbeTraceVersion);
    trace->Put<uint64_t>(program.instructions.size());
  }
  for (size_t i = 0; i < program.instructions.size(); ++i) {
    if (auto r = backend.Execute(program.instructions[i], params); !r)
      return "instruction " + std::to_string(i) + ": " + r.error();
    if (!trace) continue;
    if (auto r = backend.Flush(); !r) return "flush: " + r.error();
    const rt::InstructionExtents& ext = program.extents[i];
    uint16_t writes = 0;
    for (size_t k = 0; k < ext.count; ++k) writes += ext.ranges[k].write ? 1 : 0;
    trace->Put<uint32_t>(static_cast<uint32_t>(i));
    trace->Put<uint16_t>(program.instructions[i].opcode);
    trace->Put<uint16_t>(writes);
    for (size_t k = 0; k < ext.count; ++k) {
      const rt::OperandExtent& e = ext.ranges[k];
      if (!e.write) continue;
      trace->Put<uint64_t>(e.off);
      trace->Put<uint64_t>(e.bytes);
      trace->Write(arena + e.off, static_cast<size_t>(e.bytes));
    }
  }
  if (auto r = backend.Flush(); !r) return "flush: " + r.error();
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  // --version short-circuits, as in every SeeML tool.
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--version") == 0) {
      std::printf("seeml-plan-probe %s\n", up::kSeemlVersion);
      return 0;
    }
  Options opts;
  if (const int rc = ParseOptions(argc, argv, &opts); rc != 0) return rc;
  if (opts.threads)
    up::SetParallelThreadCount(static_cast<size_t>(opts.threads));

  // The plan: verified and validated before anything executes.
  std::vector<uint8_t> plan;
  if (!ReadFile(opts.plan, &plan)) return Fail("cannot read the plan");
  Program program;
  if (const std::string err = LoadProgram(plan, opts.section, &program);
      !err.empty())
    return Fail(err);
  const up::PlanHeader& h = program.header;

  // The arena: the caller's image in a backend-aligned allocation.
  std::vector<uint8_t> image;
  if (!ReadFile(opts.arena_in, &image))
    return Fail("cannot read the arena image");
  if (image.size() != h.arena_size)
    return Fail("the arena image is " + std::to_string(image.size()) +
                " bytes; the plan declares " + std::to_string(h.arena_size));
  const size_t aligned =
      std::max<size_t>(1, (static_cast<size_t>(h.arena_size) +
                           rt::kArenaAlignment - 1) / rt::kArenaAlignment) *
      rt::kArenaAlignment;
  struct Free {
    void operator()(uint8_t* p) const { std::free(p); }
  };
  std::unique_ptr<uint8_t, Free> arena(
      static_cast<uint8_t*>(std::aligned_alloc(rt::kArenaAlignment, aligned)));
  if (!arena) return Fail("cannot allocate the arena");
  std::memset(arena.get(), 0, aligned);
  std::memcpy(arena.get(), image.data(), image.size());

  // The backend: bound and configured exactly as the engine binds it.
  auto selected = rt::CreateBackend(opts.backend);
  if (!selected) return Fail(selected.error());
  rt::ExecutorBackend& backend = *selected->backend;
  if (auto r = backend.Bind(arena.get(), aligned,
                            plan.data() + h.rodata_offset, h.rodata_size,
                            plan.size() - h.rodata_offset);
      !r)
    return Fail("bind: " + r.error());
  // v17: an eval program that reads the source model needs the file, and
  // the file must be the one the plan was compiled from and cover every
  // extent the program reads — the engine's BindSourceModel checks, here.
  std::vector<uint8_t> source_bytes;
  uint64_t source_needed = 0;
  for (const rt::InstructionExtents& ex : program.extents)
    for (size_t i = 0; i < ex.count; ++i)
      if (ex.ranges[i].source)
        source_needed =
            std::max(source_needed, ex.ranges[i].off + ex.ranges[i].bytes);
  if (source_needed > 0) {
    if (opts.source.empty())
      return Fail("the eval program reads the source model: pass --source");
    if (!ReadFile(opts.source, &source_bytes))
      return Fail("cannot read --source");
    if (up::ContentHash64(source_bytes.data(), source_bytes.size()) !=
        h.source_model_hash)
      return Fail("--source does not match the plan's source_model_hash");
    if (source_bytes.size() < source_needed)
      return Fail("--source is shorter than the eval program reads");
    if (auto r = backend.BindSource(source_bytes.data(), source_bytes.size());
        !r)
      return Fail("bind source: " + r.error());
  }
  rt::kernels::KernelPolicy policy;
  if (h.gemm_tile_k) policy.gemm_tiles.k = h.gemm_tile_k;
  if (h.gemm_tile_n) policy.gemm_tiles.n = h.gemm_tile_n;
  backend.Configure(policy);

  float lr = 0.0f;
  std::memcpy(&lr, &opts.lr_bits, sizeof(lr));
  const rt::StepParams params{.lr = lr, .beta1 = h.beta1, .beta2 = h.beta2,
                              .eps = h.eps, .weight_decay = h.weight_decay,
                              .step = opts.step};

  if (opts.timed) {
    if (const std::string err =
            TimeProgram(backend, program, params, opts.section, opts.timed);
        !err.empty())
      return Fail(err);
  } else if (opts.profiled) {
    if (const std::string err =
            ProfileProgram(backend, program, params, opts.profiled);
        !err.empty())
      return Fail(err);
  } else {
    std::optional<StagedFile> trace;
    if (!opts.trace.empty()) {
      trace.emplace(opts.trace);
      if (!trace->ok()) return Fail("cannot stage the trace");
    }
    if (const std::string err = RunProgram(backend, program, params,
                                           arena.get(),
                                           trace ? &*trace : nullptr);
        !err.empty())
      return Fail(err);
    if (trace && !trace->Commit()) return Fail("short write to the trace");
  }

  // The arena image out — after a timed run too: the caller's contract is
  // that --arena-out always exists on exit 0.
  StagedFile out(opts.arena_out);
  if (!out.ok()) return Fail("cannot stage the arena image");
  out.Write(arena.get(), static_cast<size_t>(h.arena_size));
  if (!out.Commit()) return Fail("short write to the arena image");
  std::fprintf(stderr, "seeml-plan-probe: %s, %zu instruction(s), %s\n",
               opts.section.c_str(), program.instructions.size(),
               backend.name());
  return 0;
}
