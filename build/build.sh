#!/bin/sh
# Direct build driver for hosts without CMake. Mirrors CMakeLists.txt:
#   seeml_sir + seeml_update + seeml_update_rt + tools + tests.
set -e
cd "$(dirname "$0")/.."
CXX="${CXX:-c++}"
FLAGS="-std=c++23 -O2 -Wall -Wextra -Werror -I."

compile() { echo "  CXX $1"; $CXX $FLAGS -c "$1" -o "build/$2"; }

compile compiler/frontend/sir.cc              sir.o
compile compiler/diagnostics/logger.cc        logger.o
compile source/smf.cc                         smf.o
compile compiler/frontend/forward_builder.cc  forward_builder.o
compile compiler/trainer/update_passes.cc     update_passes.o
compile compiler/backend/update_compiler.cc   update_compiler.o
compile compiler/backend/native_emitter.cc    native_emitter.o
compile runtime/update_kernels.cc             update_kernels.o
compile runtime/dataset.cc                    dataset.o
compile runtime/update_engine.cc              update_engine.o
compile tool/seeml_update_compile.cc          seeml_update_compile.o
compile tool/seeml_seeu_dump.cc               seeml_seeu_dump.o
compile test/test_update_system.cc            test_update_system.o

echo "  LINK seeml-update-compile"
$CXX build/seeml_update_compile.o build/smf.o build/forward_builder.o \
     build/update_passes.o build/update_compiler.o build/native_emitter.o \
     build/sir.o build/logger.o -o build/seeml-update-compile

echo "  LINK seeml-seeu-dump"
$CXX build/seeml_seeu_dump.o -o build/seeml-seeu-dump

echo "  LINK seeml_update_tests"
$CXX build/test_update_system.o build/smf.o build/forward_builder.o \
     build/update_passes.o build/update_compiler.o build/native_emitter.o \
     build/sir.o build/logger.o build/update_kernels.o build/dataset.o \
     build/update_engine.o -o build/seeml_update_tests
echo "build complete"
