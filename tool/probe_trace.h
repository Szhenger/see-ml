/// The seeml-plan-probe trace container — the one byte seam that exists only
/// between two tools (the probe writes it, tool/frontier_exec.py reads it).
/// It lives in a header so seeml-abi can publish it like every other format.
///
///   u32 magic "SEPT" | u32 version | u64 instruction count
///   per instruction: u32 index | u16 opcode | u16 extents
///     per written extent: u64 arena offset | u64 bytes | the bytes
#pragma once

#include <cstdint>

namespace seeml::tool {

inline constexpr uint32_t kProbeTraceMagic = 0x54504553u;  // "SEPT"
inline constexpr uint32_t kProbeTraceVersion = 1;

}  // namespace seeml::tool
