cmake_minimum_required(VERSION 3.20)
project(seeml VERSION 1.1.0 LANGUAGES CXX)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# Target 1: SIR — the SeeML Intermediate Representation (+ shared utilities)
# ==============================================================================

add_library(seeml_sir STATIC
    compiler/frontend/sir.cc
    compiler/diagnostics/logger.cc
)

# ==============================================================================
# Target 2: The Update Compiler (seeml_update)
# Heavyweight AOT target: front end (SMF import), trainer (LoRA / autodiff /
# optimizer / merge passes), back end (arena binding, lowering, plan assembly,
# native packaging).
# ==============================================================================

add_library(seeml_update STATIC
    source/smf.cc
    compiler/frontend/forward_builder.cc
    compiler/trainer/update_passes.cc
    compiler/backend/update_compiler.cc
    compiler/backend/native_emitter.cc
)
target_link_libraries(seeml_update PUBLIC seeml_sir)

# ==============================================================================
# Target 3: The Update Runtime (seeml_update_rt)
# Ultra-lightweight target deployed to the device: the training-loop VM,
# reference kernels, dataset feeder. Zero compiler dependencies.
# ==============================================================================

add_library(seeml_update_rt STATIC
    runtime/update_kernels.cc
    runtime/dataset.cc
    runtime/update_engine.cc
)

# ==============================================================================
# Tools
# ==============================================================================

add_executable(seeml-update-compile tool/seeml_update_compile.cc)
target_link_libraries(seeml-update-compile PRIVATE seeml_update)

# ==============================================================================
# Tests
# ==============================================================================

enable_testing()
add_executable(seeml_update_tests test/test_update_system.cc)
target_link_libraries(seeml_update_tests PRIVATE seeml_update seeml_update_rt)
add_test(NAME update_system COMMAND seeml_update_tests)
