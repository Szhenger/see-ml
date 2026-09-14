#include "runtime/executor/backend.h"

#include "runtime/executor/metal_backend.h"

namespace seeml::update_rt {

const char* BackendKindName(BackendKind kind) {
  switch (kind) {
    case BackendKind::kCpu:
      return "cpu";
    case BackendKind::kMetal:
      return "metal";
    case BackendKind::kAuto:
      return "auto";
  }
  return "?";
}

std::optional<BackendKind> ParseBackendKind(std::string_view text) {
  if (text == "cpu") return BackendKind::kCpu;
  if (text == "metal") return BackendKind::kMetal;
  if (text == "auto") return BackendKind::kAuto;
  return std::nullopt;
}

std::expected<BackendSelection, std::string> CreateBackend(
    BackendKind requested) {
  BackendSelection sel;
  switch (requested) {
    case BackendKind::kCpu:
      sel.backend = CreateCpuBackend();
      sel.resolved = BackendKind::kCpu;
      return sel;
    case BackendKind::kMetal: {
      auto metal = CreateMetalBackend();
      if (!metal) return std::unexpected(metal.error());
      sel.backend = std::move(*metal);
      sel.resolved = BackendKind::kMetal;
      return sel;
    }
    case BackendKind::kAuto: {
      if (MetalBackendAvailable()) {
        auto metal = CreateMetalBackend();
        if (metal) {
          sel.backend = std::move(*metal);
          sel.resolved = BackendKind::kMetal;
          return sel;
        }
        sel.note = "backend auto: Metal device present but unusable (" +
                   metal.error() + "); using cpu";
      } else {
        sel.note = "backend auto: no Metal device on this host; using cpu";
      }
      sel.backend = CreateCpuBackend();
      sel.resolved = BackendKind::kCpu;
      return sel;
    }
  }
  return std::unexpected("unknown backend kind");
}

}  // namespace seeml::update_rt
