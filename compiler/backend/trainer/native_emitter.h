#ifndef SEEML_COMPILER_BACKEND_TRAINER_NATIVE_EMITTER_H_
#define SEEML_COMPILER_BACKEND_TRAINER_NATIVE_EMITTER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

// =============================================================================
// NativeEmitter — packages a compiled Update Plan as a native update program.
//
// Emits into an output directory:
//   update_plan.seeu          the raw plan (also runnable via LoadFromFile)
//   update_plan_embedded.cc   the plan as a 64-byte-aligned byte array TU
//                             (skipped with EmitOptions::embed_plan_tu =
//                             false: tool/pack_update.py then embeds the
//                             plan as an .incbin assembly stub instead)
//   update_main.cc            the generated driver (arg parsing, train loop,
//                             regression gate, merge, atomic commit)
//   build.sh                  compiles the TUs + the update runtime into a
//                             set of object files and links `model_update`;
//                             it assembles update_plan_embedded.S when the
//                             packer has written one, else compiles the .cc
//
// The result is the deliverable the user ships to devices: a self-contained
// C++ executable whose training program, memory plan, and adapter
// initialization are baked in as data — no compiler, no allocator, no
// framework on the device.
// =============================================================================

namespace seeml::update {

struct EmitPaths {
  std::string plan_file;
  std::string embedded_tu;  // empty when the decimal TU was not emitted
  std::string main_tu;
  std::string build_script;
  // Every runtime source vendored into the package, relative to its root,
  // in emission order — what a packer or an auditor needs to know the
  // package's exact contents without listing the directory (P6, #86).
  // The `{}` matters: designated initializers omit this field, and -Wextra
  // flags an omitted field unless it has a default initializer.
  std::vector<std::string> vendored_sources{};
};

struct EmitOptions {
  /// Render the plan as the decimal byte-array TU. The default keeps every
  /// package buildable with a C++ compiler alone; false leaves the plan on
  /// disk once (the .seeu) for tool/pack_update.py to embed with .incbin —
  /// the only route that stays seconds and ~1x plan size at 100M+ params.
  bool embed_plan_tu = true;
  /// The plan carries relaxed GEMMs (v18, --precision certified-bf16): the
  /// build script then compiles the CPU backend with SEEML_ACCELERATE on
  /// Apple hosts (F4), so those GEMMs run on Accelerate's matrix units —
  /// SEEML_NO_ACCELERATE=1 keeps the portable kernels. An exact plan's
  /// script never links Accelerate.
  bool relaxed_plan = false;
};

[[nodiscard]] std::expected<EmitPaths, std::string> EmitNativePackage(
    const std::vector<uint8_t>& plan, const std::string& out_dir,
    const std::string& repo_root, const EmitOptions& options = {});

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_TRAINER_NATIVE_EMITTER_H_
