#ifndef SEEML_SOURCE_PLAN_UPDATE_TYPES_H_
#define SEEML_SOURCE_PLAN_UPDATE_TYPES_H_

// =============================================================================
// The update-plan ABI — the language both halves of the product speak: the
// compiler writes it, the device runtime reads it, and neither may depend
// on the other's internals. This is the façade; the ABI is partitioned per
// discipline, in the fashion of the compiler and runtime subsystems:
//   config.h       the compilation request (loss, LoRA, optimizer, budget)
//                  — consumed by the compiler only, its consequences
//                  compiled into the PlanHeader
//   instruction.h  the instruction set: tensor references, the opcode
//                  vocabulary, the fixed 64-byte UpdateInstruction
//   schema.h       the .seeu container: magic/version, the master
//                  PlanHeader, the delta-to-file EmitEntry table
// The byte format is documented in docs/formats.md.
// =============================================================================

#include "source/plan/config.h"       // IWYU pragma: export
#include "source/plan/instruction.h"  // IWYU pragma: export
#include "source/plan/schema.h"       // IWYU pragma: export

#endif  // SEEML_SOURCE_PLAN_UPDATE_TYPES_H_
