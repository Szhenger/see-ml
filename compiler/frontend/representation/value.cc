#include "compiler/frontend/representation/value.h"

#include <algorithm>
#include <cassert>

#include "compiler/frontend/representation/operation.h"

namespace seeml::sir {

Value::Value(std::string id, DataType dt, Shape sh, Operation* def_op)
    : id_(std::move(id)), dtype_(dt), shape_(std::move(sh)), defining_op_(def_op) {}

void Value::replaceAllUsesWith(Value* newVal) {
    assert(newVal && "replaceAllUsesWith called with null value");
    assert(newVal != this && "replaceAllUsesWith called with same value");

    // setOperand() rewires both user lists (removeUser on this, addUser on
    // newVal), so iterate over a snapshot rather than the live vector.
    const std::vector<Operation*> users_snapshot(users_.begin(), users_.end());
    for (Operation* user : users_snapshot) {
        for (size_t i = 0; i < user->numOperands(); ++i) {
            if (user->operand(i) == this)
                user->setOperand(i, newVal);
        }
    }
    users_.clear();
}

void Value::addUser(Operation* op) {
    users_.push_back(op);
}

void Value::removeUser(Operation* op) {
    auto it = std::find(users_.begin(), users_.end(), op);
    if (it != users_.end())
        users_.erase(it);
}

} // namespace seeml::sir
