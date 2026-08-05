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
compile source/language/model_format.cc model_format.o
compile compiler/frontend/ingressor/model_reader.cc model_reader.o
compile compiler/frontend/ingressor/model_writer.cc model_writer.o
compile compiler/frontend/ingressor/resource_analyzer.cc resource_analyzer.o
compile source/parallel/parallel_for.cc       parallel_for.o
compile compiler/frontend/parser/value_resolver.cc value_resolver.o
compile compiler/frontend/parser/sema.cc      sema.o
compile compiler/frontend/parser/parser.cc    parser.o
compile compiler/analysis/updater/pass_manager.cc  pass_manager.o
compile compiler/analysis/updater/conv_lowering.cc conv_lowering.o
compile compiler/analysis/updater/dce.cc           dce.o
compile compiler/analysis/algebra/epilogue_fuser.cc epilogue_fuser.o
compile compiler/analysis/algebra/lora_grafter.cc  lora_grafter.o
compile compiler/analysis/algebra/merge_builder.cc merge_builder.o
compile compiler/analysis/calculus/autodiff.cc     autodiff.o
compile compiler/analysis/calculus/optimizer.cc    optimizer_synth.o
compile compiler/analysis/reviewer/quantization.cc quantization.o
compile compiler/backend/architecture/host_arch.cc host_arch.o
compile compiler/backend/tuner/bandit.cc      bandit.o
compile compiler/backend/tuner/autotuner.cc   autotuner.o
compile compiler/backend/trainer/arena_binder.cc arena_binder.o
compile compiler/backend/trainer/instruction_lowering.cc instruction_lowering.o
compile compiler/backend/trainer/kernel_emitter.cc kernel_emitter.o
compile compiler/driver/contract.cc           driver_contract.o
compile compiler/driver/update_compiler.cc    update_compiler.o
compile compiler/backend/trainer/native_emitter.cc native_emitter.o
compile runtime/executor/gemm.cc              rt_gemm.o
compile runtime/executor/elementwise.cc       rt_elementwise.o
compile runtime/executor/activation.cc        rt_activation.o
compile runtime/executor/normalization.cc     rt_normalization.o
compile runtime/executor/loss.cc              rt_loss.o
compile runtime/executor/optimizer.cc         rt_optimizer.o
compile runtime/executor/attention.cc         rt_attention.o

# Metal GEMM dispatch (G1a) exists only on Apple hosts; elsewhere the
# hardware-gated suite compiles to zero tests and nothing links the runner.
METAL_OBJS=""
METAL_LDFLAGS=""
if [ "$(uname)" = "Darwin" ]; then
  echo "  OBJCXX runtime/executor/metal_gemm.mm"
  eval "$CXX $FLAGS -x objective-c++ -fobjc-arc -c runtime/executor/metal_gemm.mm -o build/rt_metal_gemm.o"
  METAL_OBJS="build/rt_metal_gemm.o"
  METAL_LDFLAGS="-framework Metal -framework Foundation"
fi
compile runtime/feeder/dataset.cc             dataset.o
compile runtime/feeder/batch_pipeline.cc      batch_pipeline.o
compile runtime/custodian/durable_io.cc       durable_io.o
compile runtime/validator/plan_validator.cc   plan_validator.o
compile runtime/custodian/checkpoint.cc       checkpoint.o
compile runtime/engine/contract.cc            engine_contract.o
compile runtime/engine/update_engine.cc       update_engine.o
compile tool/seeml_update_compile.cc          seeml_update_compile.o
compile tool/seeml_seeu_dump.cc               seeml_seeu_dump.o
compile test/framework/registry.cc            seetest_registry.o
compile test/framework/seetest_main.cc        seetest_main.o
compile test/support/scoped_temp_dir.cc       scoped_temp_dir.o
compile test/support/models.cc                fixtures_models.o
compile test/support/corpora.cc               fixtures_corpora.o
compile test/support/probes.cc                fixtures_probes.o

LIBS="build/model_format.o build/model_reader.o build/model_writer.o \
      build/resource_analyzer.o \
      build/value_resolver.o build/sema.o build/parser.o \
      build/pass_manager.o build/conv_lowering.o build/dce.o \
      build/epilogue_fuser.o build/lora_grafter.o \
      build/merge_builder.o build/autodiff.o build/optimizer_synth.o \
      build/quantization.o \
      build/arena_binder.o build/instruction_lowering.o \
      build/host_arch.o build/bandit.o build/autotuner.o \
      build/kernel_emitter.o \
      build/driver_contract.o build/update_compiler.o build/native_emitter.o \
      build/sir_type.o build/sir_value.o build/sir_operation.o \
      build/sir_block.o \
      build/op_convolution.o build/op_linear.o build/op_normalization.o \
      build/op_activation.o \
      build/logger.o \
      build/rt_gemm.o build/rt_elementwise.o build/rt_activation.o \
      build/rt_normalization.o build/rt_loss.o build/rt_optimizer.o \
      build/rt_attention.o \
      build/dataset.o \
      build/batch_pipeline.o build/durable_io.o build/plan_validator.o \
      build/checkpoint.o build/engine_contract.o build/update_engine.o \
      build/parallel_for.o"
TESTING="build/seetest_registry.o build/seetest_main.o \
         build/scoped_temp_dir.o build/fixtures_models.o \
         build/fixtures_corpora.o build/fixtures_probes.o"

echo "  LINK seeml-update-compile"
eval "$CXX -pthread build/seeml_update_compile.o $LIBS -o build/seeml-update-compile"
echo "  LINK seeml-seeu-dump"
# PlanSelfHash (the v4 integrity seal) runs on the parallel substrate.
eval "$CXX -pthread build/seeml_seeu_dump.o build/parallel_for.o -o build/seeml-seeu-dump"

for suite in \
    source/identity/hash_test source/parallel/parallel_for_test \
    compiler/frontend/model_io_test compiler/frontend/resource_analyzer_test \
    compiler/frontend/sir_test compiler/frontend/operator_test \
    compiler/frontend/parser_test \
    compiler/analysis/update_passes_test compiler/analysis/updater_test \
    compiler/analysis/reviewer_test \
    compiler/backend/tuner_test compiler/backend/trainer_test \
    compiler/backend/native_emitter_test \
    compiler/driver/update_compiler_test compiler/driver/driver_test \
    compiler/diagnostics/diagnostics_test \
    runtime/feeder/dataset_test runtime/feeder/batch_pipeline_test \
    runtime/executor/kernels_test runtime/executor/metal_gemm_test \
    runtime/validator/validator_test \
    runtime/custodian/custodian_test \
    runtime/engine/engine_test runtime/engine/update_engine_test \
    system/update_system_test; do
  name="seeml_$(basename "$suite")"
  echo "  CXX+LINK $name"
  eval "$CXX $FLAGS -c 'test/$suite.cc' -o 'build/$name.o'"
  eval "$CXX -pthread 'build/$name.o' $TESTING $LIBS $METAL_OBJS $METAL_LDFLAGS -o 'build/$name'"
done
echo "build complete"
