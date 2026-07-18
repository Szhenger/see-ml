cmake_minimum_required(VERSION 3.20)
project(seeml VERSION 1.2.0 LANGUAGES CXX)

# ==============================================================================
# Global Configuration
# ==============================================================================

# C++23 for std::expected (matching the codebase's error-handling idiom).
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Werror)
elseif(MSVC)
    add_compile_options(/W4 /WX)
endif()

# CI hardening: -DSEEML_SANITIZE=address;undefined (or thread).
set(SEEML_SANITIZE "" CACHE STRING "Semicolon-separated sanitizer list")
if(SEEML_SANITIZE)
    string(REPLACE ";" "," _san "${SEEML_SANITIZE}")
    add_compile_options(-fsanitize=${_san} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${_san})
endif()

# libFuzzer harnesses over the binary-format parsers (clang only):
# -DSEEML_FUZZ=ON builds seeml_fuzz_formats.
option(SEEML_FUZZ "Build the libFuzzer format-parser harness" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads REQUIRED)

# ==============================================================================
# Target 0: The parallel substrate (seeml_parallel)
# Deterministic chunked ParallelFor over a persistent worker pool; shared by
# the compiler's byte-heavy passes and the runtime's kernels. Chunk geometry
# is thread-count-independent, so parallel results are bitwise-reproducible.
# ==============================================================================

add_library(seeml_parallel STATIC
    source/parallel_for.cc
)
target_link_libraries(seeml_parallel PUBLIC Threads::Threads)

# ==============================================================================
# Target 1: SIR — the SeeML Intermediate Representation (+ shared utilities)
# ==============================================================================

add_library(seeml_sir STATIC
    compiler/frontend/sir.cc
    compiler/diagnostics/logger.cc
)

# ==============================================================================
# Target 2: The Update Compiler (seeml_update)
# Heavyweight AOT target: front end (SMF import), analysis (LoRA / autodiff /
# optimizer / merge passes), back end (arena binding, lowering, plan assembly,
# native packaging).
# ==============================================================================

add_library(seeml_update STATIC
    source/smf.cc
    compiler/frontend/forward_builder.cc
    compiler/analysis/update_passes.cc
    compiler/backend/arena_binder.cc
    compiler/backend/instruction_lowering.cc
    compiler/backend/update_compiler.cc
    compiler/backend/native_emitter.cc
)
target_link_libraries(seeml_update PUBLIC seeml_sir seeml_parallel)

# ==============================================================================
# Target 3: The Update Runtime (seeml_update_rt)
# Ultra-lightweight target deployed to the device: the training-loop VM,
# reference kernels, dataset feeder. Zero compiler dependencies.
# ==============================================================================

add_library(seeml_update_rt STATIC
    runtime/update_kernels.cc
    runtime/dataset.cc
    runtime/batch_pipeline.cc
    runtime/durable_io.cc
    runtime/plan_validator.cc
    runtime/checkpoint.cc
    runtime/update_engine.cc
)
target_link_libraries(seeml_update_rt PUBLIC seeml_parallel)

# ==============================================================================
# Tools
# ==============================================================================

add_executable(seeml-update-compile tool/seeml_update_compile.cc)
target_link_libraries(seeml-update-compile PRIVATE seeml_update)

add_executable(seeml-seeu-dump tool/seeml_seeu_dump.cc)

# ==============================================================================
# Tests
#
# seeml_testing is the in-tree test framework (test/framework/) plus shared
# fixtures (test/support/). Each suite is one executable per module, linked
# against the framework's main() and registered with CTest. Run everything
# with `ctest`; run one suite directly with e.g.
#   ./seeml_smf_test --filter=Smf.RoundTrip
# ==============================================================================

enable_testing()

add_library(seeml_testing STATIC
    test/framework/seetest.cc
    test/support/scoped_temp_dir.cc
    test/support/builders.cc
)
target_link_libraries(seeml_testing PUBLIC seeml_update seeml_update_rt)
# The emitter suite vendors runtime sources out of the repository checkout.
target_compile_definitions(seeml_testing
    PUBLIC SEEML_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

function(seeml_add_test name src)
    add_executable(${name} ${src} test/framework/seetest_main.cc)
    target_link_libraries(${name} PRIVATE seeml_testing)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

seeml_add_test(seeml_smf_test             test/source/smf_test.cc)
seeml_add_test(seeml_parallel_for_test    test/source/parallel_for_test.cc)
seeml_add_test(seeml_sir_test             test/compiler/sir_test.cc)
seeml_add_test(seeml_forward_builder_test test/compiler/forward_builder_test.cc)
seeml_add_test(seeml_update_passes_test   test/compiler/update_passes_test.cc)
seeml_add_test(seeml_update_compiler_test test/compiler/update_compiler_test.cc)
seeml_add_test(seeml_native_emitter_test  test/compiler/native_emitter_test.cc)
seeml_add_test(seeml_kernels_test         test/runtime/kernels_test.cc)
seeml_add_test(seeml_dataset_test         test/runtime/dataset_test.cc)
seeml_add_test(seeml_update_engine_test   test/runtime/update_engine_test.cc)
seeml_add_test(seeml_update_system_test   test/system/update_system_test.cc)

if(SEEML_FUZZ)
    add_executable(seeml_fuzz_formats test/fuzz_binary_formats.cc)
    target_compile_options(seeml_fuzz_formats PRIVATE -fsanitize=fuzzer,address)
    target_link_options(seeml_fuzz_formats PRIVATE -fsanitize=fuzzer,address)
    target_link_libraries(seeml_fuzz_formats PRIVATE seeml_update seeml_update_rt)
endif()
