#ifndef SEEML_COMPILER_FRONTEND_REPRESENTATION_VALUE_H_
#define SEEML_COMPILER_FRONTEND_REPRESENTATION_VALUE_H_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/frontend/representation/type.h"

// =============================================================================
// Value — an SSA value: the typed, shaped result of an operation or a block
// argument, carrying the use-list that links it to its readers. Part of the
// SIR core; see sir.h for the threading model (use-lists are written by a
// value's *readers*, which is what forbids two threads sharing a Value).
// =============================================================================

namespace seeml::sir {

class Operation;

class Value {
 public:
    Value(std::string id, DataType dt, Shape sh, Operation* def_op);
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    std::string_view id() const { return id_; }
    DataType dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    Operation* definingOp() { return defining_op_; }
    const Operation* definingOp() const { return defining_op_; }

    bool isBlockArgument() const { return defining_op_ == nullptr; }
    std::span<Operation* const> users() const { return users_; }
    bool hasOneUse() const { return users_.size() == 1; }
    bool hasNoUses() const { return users_.empty(); }

    void replaceAllUsesWith(Value* newVal);
    void addUser(Operation* op);
    void removeUser(Operation* op);
    void setShape(Shape sh) { shape_ = std::move(sh); }

 private:
    std::string id_;
    DataType dtype_;
    Shape shape_;
    Operation* defining_op_;
    std::vector<Operation*> users_;
};

} // namespace seeml::sir

#endif // SEEML_COMPILER_FRONTEND_REPRESENTATION_VALUE_H_
