#ifndef SEEML_UPDATE_NATIVE_EMITTER_H_
#define SEEML_UPDATE_NATIVE_EMITTER_H_

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
//   update_main.cc            the generated driver (arg parsing, train loop,
//                             regression gate, merge, atomic commit)
//   build.sh                  compiles the TUs + the update runtime into a
//                             set of object files and links `model_update`
//
// The result is the deliverable the user ships to devices: a self-contained
// C++ executable whose training program, memory plan, and adapter
// initialization are baked in as data — no compiler, no allocator, no
// framework on the device.
// =============================================================================

namespace seeml::update {

struct EmitPaths {
  std::string plan_file;
  std::string embedded_tu;
  std::string main_tu;
  std::string build_script;
};

[[nodiscard]] std::expected<EmitPaths, std::string> EmitNativePackage(
    const std::vector<uint8_t>& plan, const std::string& out_dir,
    const std::string& repo_root);

}  // namespace seeml::update

#endif  // SEEML_UPDATE_NATIVE_EMITTER_H_
