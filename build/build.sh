#!/bin/sh
# Direct build driver for hosts without CMake. Mirrors CMakeLists.txt:
#   seeml_sir + seeml_update + seeml_update_rt + tools + per-module suites.
set -e
cd "$(dirname "$0")/.."
CXX="${CXX:-c++}"
FLAGS="-std=c++23 -O2 -Wall -Wextra -Werror -I. -DSEEML_SOURCE_DIR='\"$(pwd)\"'"

compile() { echo "  CXX $1"; eval "$CXX $FLAGS -c '$1' -o 'build/$2'"; }

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
compile test/framework/seetest.cc             seetest.o
compile test/framework/seetest_main.cc        seetest_main.o
compile test/support/scoped_temp_dir.cc       scoped_temp_dir.o
compile test/support/builders.cc              builders.o

LIBS="build/smf.o build/forward_builder.o build/update_passes.o \
      build/update_compiler.o build/native_emitter.o build/sir.o \
      build/logger.o build/update_kernels.o build/dataset.o \
      build/update_engine.o"
TESTING="build/seetest.o build/seetest_main.o build/scoped_temp_dir.o \
         build/builders.o"

echo "  LINK seeml-update-compile"
eval "$CXX build/seeml_update_compile.o $LIBS -o build/seeml-update-compile"
echo "  LINK seeml-seeu-dump"
eval "$CXX build/seeml_seeu_dump.o -o build/seeml-seeu-dump"

for suite in \
    source/smf_test compiler/sir_test compiler/forward_builder_test \
    compiler/update_passes_test compiler/update_compiler_test \
    compiler/native_emitter_test runtime/kernels_test runtime/dataset_test \
    runtime/update_engine_test system/update_system_test; do
  name="seeml_$(basename "$suite")"
  echo "  CXX+LINK $name"
  eval "$CXX $FLAGS -c 'test/$suite.cc' -o 'build/$name.o'"
  eval "$CXX 'build/$name.o' $TESTING $LIBS -o 'build/$name'"
done
echo "build complete"
