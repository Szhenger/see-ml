#ifndef SEEML_SOURCE_IDENTITY_VERSION_H_
#define SEEML_SOURCE_IDENTITY_VERSION_H_

// =============================================================================
// The release version, in exactly one place.
//
// Everything that reports a version derives it from this constant: CMake
// extracts it for project(VERSION ...) at configure time, and both CLI
// tools print it for --version. The wire-format versions (SMF, SDS, SEEU,
// checkpoint) are deliberately separate — they gate compatibility and move
// only when a format changes; this constant names the release.
// =============================================================================

namespace seeml::update {

inline constexpr char kSeemlVersion[] = "1.0.2";

}  // namespace seeml::update

#endif  // SEEML_SOURCE_IDENTITY_VERSION_H_
