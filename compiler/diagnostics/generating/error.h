#ifndef SEEML_COMPILER_DIAGNOSTICS_GENERATING_ERROR_H_
#define SEEML_COMPILER_DIAGNOSTICS_GENERATING_ERROR_H_

#include <expected>
#include <string>
#include <string_view>

#include "compiler/diagnostics/diagnostic.h"

// =============================================================================
// generating/ — errors formed while generating code (compiler/backend/
// trainer/ and the AOT driver that sequences it): arena binding, instruction
// lowering, plan assembly, and native package emission. By this stage the
// program is analytically correct; what can fail is materialization — an
// unbound value, an op with no encoding, a package directory that cannot be
// written. Failure discipline: binding and lowering errors name the value or
// mnemonic that lacked a home; emission errors name the file path, since the
// fix is on disk, not in the model.
// =============================================================================

namespace seeml::diag::generating {

/// The AOT driver (update_compiler) speaks under its own name.
inline constexpr std::string_view kDriver = "UpdateCompiler";
inline constexpr std::string_view kArenaBinder = "ArenaBinder";
inline constexpr std::string_view kInstructionLowering = "InstructionLowering";
inline constexpr std::string_view kNativeEmitter = "NativeEmitter";

/// Codegen failure in `unit`: "<unit>: <message>".
[[nodiscard]] inline std::unexpected<std::string> Error(std::string_view unit,
                                                        std::string_view message) {
  return Fail(unit, message);
}

/// Emission failure against the filesystem: "<unit>: <what> '<path>'".
[[nodiscard]] inline std::unexpected<std::string> FileError(std::string_view unit,
                                                            std::string_view what,
                                                            std::string_view path) {
  std::string m;
  m.reserve(what.size() + path.size() + 3);
  m.append(what).append(" '").append(path).append("'");
  return Fail(unit, m);
}

}  // namespace seeml::diag::generating

#endif  // SEEML_COMPILER_DIAGNOSTICS_GENERATING_ERROR_H_
