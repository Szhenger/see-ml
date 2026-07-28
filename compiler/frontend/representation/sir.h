#ifndef SEEML_COMPILER_FRONTEND_REPRESENTATION_SIR_H_
#define SEEML_COMPILER_FRONTEND_REPRESENTATION_SIR_H_

// =============================================================================
// SIR — the SeeML intermediate representation. This is the façade header:
// consumers include it and get the whole representation; each core
// definition lives in its own header/impl pair so a debugger, a diff, or a
// stack trace points at exactly one group:
//
//   type.h/.cc        DataType, Shape, AttributeValue — the type system
//   value.h/.cc       Value — SSA values and their use-lists
//   operation.h/.cc   Operation — instructions, operands, results, attributes
//   block.h/.cc       Block & Region — ownership, ordering, the verifier
//
// Threading model: SIR is a single-writer structure. A Block — and every
// Operation and Value it owns — may be mutated by one thread at a time;
// nothing here locks. Distinct Blocks are independent and may be built
// concurrently (auto-generated result ids come from one atomic counter), with
// one caveat: a Value's use-list is written by its *readers* (addOperand /
// setOperand / removeOp), so two threads may not build ops that reference the
// same Value, even from different blocks. Once construction stops, any number
// of threads may traverse concurrently.
// =============================================================================

#include "compiler/frontend/representation/type.h"       // IWYU pragma: export
#include "compiler/frontend/representation/value.h"      // IWYU pragma: export
#include "compiler/frontend/representation/operation.h"  // IWYU pragma: export
#include "compiler/frontend/representation/block.h"      // IWYU pragma: export

#endif // SEEML_COMPILER_FRONTEND_REPRESENTATION_SIR_H_
