#!/bin/sh
# Direct build driver for hosts without CMake. Mirrors CMakeLists.txt:
#   seeml_sir + seeml_update + seeml_update_rt + tools + per-module suites.
set -e
cd "$(dirname "$0")/.."
CXX="${CXX:-c++}"
FLAGS="-std=c++23 -O2 -Wall -Wextra -Werror -pthread -I. -DSEEML_SOURCE_DIR='\"$(pwd)\"'"

compile() { echo "  CXX $1"; eval "$CXX $FLAGS -c '$1' -o 'build/$2'"; }

compile compiler/frontend/representation/type.cc      sir_type.o
compile compiler/frontend/representation/value.cc     sir_value.o
compile compiler/frontend/representation/operation.cc sir_operation.o
compile compiler/frontend/representation/block.cc     sir_block.o
compile compiler/frontend/operator/convolution.cc     op_convolution.o
compile compiler/frontend/operator/linear.cc          op_linear.o
compile compiler/frontend/operator/normalization.cc   op_normalization.o
compile compiler/frontend/operator/activation.cc      op_activation.o
compile compiler/diagnostics/logger.cc        logger.o
compile source/model_format.cc model_format.o
compile compiler/frontend/ingressor/model_reader.cc model_reader.o
compile compiler/frontend/ingressor/model_writer.cc model_writer.o
compile compiler/frontend/ingressor/resource_analyzer.cc resource_analyzer.o
compile source/parallel_for.cc                parallel_for.o
compile compiler/frontend/parser/value_resolver.cc value_resolver.o
compile compiler/frontend/parser/sema.cc      sema.o
compile compiler/frontend/parser/parser.cc    parser.o
compile compiler/analysis/updater/pass_manager.cc  pass_manager.o
compile compiler/analysis/updater/conv_lowering.cc conv_lowering.o
compile compiler/analysis/algebra/lora_grafter.cc  lora_grafter.o
compile compiler/analysis/algebra/merge_builder.cc merge_builder.o
compile compiler/analysis/calculus/autodiff.cc     autodiff.o
compile compiler/analysis/calculus/optimizer.cc    optimizer_synth.o
compile compiler/analysis/reviewer/quantization.cc quantization.o
compile compiler/backend/arena_binder.cc      arena_binder.o
compile compiler/backend/instruction_lowering.cc instruction_lowering.o
compile compiler/backend/update_compiler.cc   update_compiler.o
compile compiler/backend/native_emitter.cc    native_emitter.o
compile runtime/update_kernels.cc             update_kernels.o
compile runtime/dataset.cc                    dataset.o
compile runtime/batch_pipeline.cc             batch_pipeline.o
compile runtime/durable_io.cc                 durable_io.o
compile runtime/plan_validator.cc             plan_validator.o
compile runtime/checkpoint.cc                 checkpoint.o
compile runtime/update_engine.cc              update_engine.o
compile tool/seeml_update_compile.cc          seeml_update_compile.o
compile tool/seeml_seeu_dump.cc               seeml_seeu_dump.o
compile test/framework/seetest.cc             seetest.o
compile test/framework/seetest_main.cc        seetest_main.o
compile test/support/scoped_temp_dir.cc       scoped_temp_dir.o
compile test/support/builders.cc              builders.o

LIBS="build/model_format.o build/model_reader.o build/model_writer.o \
      build/resource_analyzer.o \
      build/value_resolver.o build/sema.o build/parser.o \
      build/pass_manager.o build/conv_lowering.o build/lora_grafter.o \
      build/merge_builder.o build/autodiff.o build/optimizer_synth.o \
      build/quantization.o \
      build/arena_binder.o build/instruction_lowering.o \
      build/update_compiler.o build/native_emitter.o \
      build/sir_type.o build/sir_value.o build/sir_operation.o \
      build/sir_block.o \
      build/op_convolution.o build/op_linear.o build/op_normalization.o \
      build/op_activation.o \
      build/logger.o build/update_kernels.o build/dataset.o \
      build/batch_pipeline.o build/durable_io.o build/plan_validator.o \
      build/checkpoint.o build/update_engine.o build/parallel_for.o"
TESTING="build/seetest.o build/seetest_main.o build/scoped_temp_dir.o \
         build/builders.o"

echo "  LINK seeml-update-compile"
eval "$CXX -pthread build/seeml_update_compile.o $LIBS -o build/seeml-update-compile"
echo "  LINK seeml-seeu-dump"
eval "$CXX build/seeml_seeu_dump.o -o build/seeml-seeu-dump"

for suite in \
    compiler/model_io_test source/hash_test source/parallel_for_test \
    compiler/sir_test \
    compiler/resource_analyzer_test compiler/parser_test \
    compiler/update_passes_test compiler/updater_test \
    compiler/update_compiler_test \
    compiler/native_emitter_test runtime/kernels_test runtime/dataset_test \
    runtime/update_engine_test system/update_system_test; do
  name="seeml_$(basename "$suite")"
  echo "  CXX+LINK $name"
  eval "$CXX $FLAGS -c 'test/$suite.cc' -o 'build/$name.o'"
  eval "$CXX -pthread 'build/$name.o' $TESTING $LIBS -o 'build/$name'"
done
echo "build complete"
