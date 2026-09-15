#ifndef SEEML_SOURCE_PLAN_BF16_H_
#define SEEML_SOURCE_PLAN_BF16_H_

#include <bit>
#include <cstdint>

// =============================================================================
// bfloat16 storage conversions (plan v10, roadmap 2c). bf16 is float32 with
// the low 16 mantissa bits dropped: same exponent range, 8 bits of
// precision. Widening is exact (place the 16 bits in the high half);
// narrowing rounds to nearest-even, with NaN kept quiet. Shared by the
// compiler's rodata packer, the runtime's widening GEMMs and the tests, so
// every side of the plan converts identically.
// =============================================================================

namespace seeml::update {

inline constexpr uint16_t Float32ToBf16Bits(float f) {
  const uint32_t bits = std::bit_cast<uint32_t>(f);
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u)  // NaN: keep it a quiet NaN
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline constexpr float Bf16BitsToFloat32(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

/// The element type of a bf16 rodata tensor as the GEMM templates see it:
/// static_cast<float> widens exactly, so the f32, int8 and bf16 variants
/// share one blocked core.
struct Bf16 {
  uint16_t bits;
  explicit constexpr operator float() const { return Bf16BitsToFloat32(bits); }
};
static_assert(sizeof(Bf16) == 2, "bf16 storage is two bytes per element");

}  // namespace seeml::update

#endif  // SEEML_SOURCE_PLAN_BF16_H_
