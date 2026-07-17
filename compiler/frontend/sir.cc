#include "compiler/frontend/sir.h"

#include <atomic>
#include <ostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstdio>
#include <cstdlib>

namespace seecpp::sir {

namespace {

// Structural-invariant violations abort in every build mode: the callers of
// these APIs (the grafting and autodiff passes) mutate live graphs, and a
// compiled-out assert would turn a pass bug into silent memory corruption
// (an increment past end(), a dereference of end()) instead of a diagnosis.
[[noreturn]] void FatalInvariant(const char* what) {
    std::fprintf(stderr, "sir: fatal invariant violation: %s\n", what);
    std::abort();
}

}  // namespace

// =============================================================================
// 1. Shape
// =============================================================================

bool Shape::isFullyStatic() const {
    return std::none_of(dims.begin(), dims.end(), [](int64_t d) { return d == kDynamic; });
}

int64_t Shape::volume() const {
    int64_t v = 1;
    for (auto d : dims) {
        if (d == kDynamic) return kDynamic;
        v *= d;
    }
    return v;
}

size_t Shape::byteSize(DataType dt) const {
    auto vol = volume();
    return (vol == kDynamic) ? 0 : static_cast<size_t>(vol) * dtypeByteWidth(dt);
}

// =============================================================================
// 2. Value
// =============================================================================

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

// =============================================================================
// 3. Operation
// =============================================================================

std::atomic<size_t> Operation::id_counter_{0};

Operation::Operation(std::string mnemonic, Block* parent)
    : mnemonic_(std::move(mnemonic)), parent_block_(parent) {}

void Operation::addOperand(Value* v) {
    assert(v && "addOperand: null Value*");
    operands_.push_back(v);
    v->addUser(this);
}

void Operation::setOperand(size_t i, Value* newVal) {
    assert(i < operands_.size() && "setOperand: index out of range");
    assert(newVal && "setOperand: null Value*");

    operands_[i]->removeUser(this);
    operands_[i] = newVal;
    newVal->addUser(this);
}

Value* Operation::addResult(std::string id, DataType dt, Shape sh) {
    if (id.empty())
        id = "%" + std::to_string(id_counter_.fetch_add(1, std::memory_order_relaxed));

    auto val = std::make_unique<Value>(std::move(id), dt, std::move(sh), this);
    results_.push_back(std::move(val));
    return results_.back().get();
}

void Operation::setAttribute(std::string key, AttributeValue val) {
    attributes_[std::move(key)] = std::move(val);
}

const AttributeValue* Operation::getAttribute(std::string_view key) const {
    if (auto it = attributes_.find(std::string(key)); it != attributes_.end())
        return &it->second;
    return nullptr;
}

void Operation::print(std::ostream& os) const {
    if (!results_.empty()) {
        for (size_t i = 0; i < results_.size(); ++i) {
            if (i) os << ", ";
            os << results_[i]->id() << " : " << dtypeName(results_[i]->dtype());
            const auto& dims = results_[i]->shape().dims;
            if (!dims.empty()) {
                os << "<";
                for (size_t d = 0; d < dims.size(); ++d) {
                    if (d) os << "x";
                    if (dims[d] == Shape::kDynamic) os << "?";
                    else os << dims[d];
                }
                os << ">";
            }
        }
        os << " = ";
    }

    os << mnemonic_ << "(";
    for (size_t i = 0; i < operands_.size(); ++i) {
        if (i) os << ", ";
        os << operands_[i]->id();
    }
    os << ")";

    if (!attributes_.empty()) {
        os << " {";
        bool first = true;
        for (const auto& [k, v] : attributes_) {
            if (!first) os << ", ";
            first = false;
            os << k << " = ";
            std::visit([&os](const auto& val) {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    os << '"' << val << '"';
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                                     std::is_same_v<T, std::vector<float>>) {
                    os << "[";
                    for (size_t i = 0; i < val.size(); ++i) {
                        if (i) os << ", ";
                        os << val[i];
                    }
                    os << "]";
                } else {
                    os << val;
                }
            }, v);
        }
        os << "}";
    }
}

std::string Operation::toString() const {
    std::ostringstream oss;
    print(oss);
    return oss.str();
}

// =============================================================================
// 4. Block & Region
// =============================================================================

Value* Block::addArgument(DataType dt, Shape sh) {
    std::string id = "%" + std::to_string(Operation::id_counter_.fetch_add(1, std::memory_order_relaxed));
    auto val = std::make_unique<Value>(std::move(id), dt, std::move(sh), nullptr);
    args_.push_back(std::move(val));
    return args_.back().get();
}

Operation* Block::appendOp(std::string name) {
    auto op = std::make_unique<Operation>(std::move(name), this);
    ops_.push_back(std::move(op));
    return ops_.back().get();
}

Operation* Block::appendOp(std::unique_ptr<Operation> op) {
    assert(op && "appendOp: null operation");
    op->setParentBlock(this);
    ops_.push_back(std::move(op));
    return ops_.back().get();
}

void Block::insertOpsAfter(Operation* anchor,
                           std::vector<std::unique_ptr<Operation>> new_ops) {
    auto it = std::find_if(ops_.begin(), ops_.end(),
                           [anchor](const auto& p) { return p.get() == anchor; });
    if (it == ops_.end())
        FatalInvariant("insertOpsAfter: anchor not found in block");

    for (auto& op : new_ops)
        op->setParentBlock(this);

    ops_.insert(std::next(it),
                std::make_move_iterator(new_ops.begin()),
                std::make_move_iterator(new_ops.end()));
}

std::unique_ptr<Operation> Block::removeOp(Operation* op) {
    auto it = std::find_if(ops_.begin(), ops_.end(), [op](const auto& p) { return p.get() == op; });
    if (it == ops_.end())
        FatalInvariant("removeOp: operation not found in block");

    for (size_t i = 0; i < op->numOperands(); ++i)
        op->operand(i)->removeUser(op);

    auto owned = std::move(*it);
    ops_.erase(it);
    owned->setParentBlock(nullptr);
    return owned;
}

bool Block::validate() const {
    std::unordered_set<const Value*> defined;
    for (const auto& arg : args_) defined.insert(arg.get());

    for (const auto& op : ops_) {
        for (const Value* operand : op->operands()) {
            if (defined.find(operand) == defined.end()) {
                return false;
            }
        }
        for (const auto& res : op->results())
            defined.insert(res.get());
    }

    is_validated_ = true;
    return true;
}

void Block::walk(std::function<void(Operation*)> fn) {
    for (auto& op : ops_) fn(op.get());
}

void Block::walkReverse(std::function<void(Operation*)> fn) {
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) fn(it->get());
}

void Block::print(std::ostream& os) const {
    if (!args_.empty()) {
        os << "(";
        for (size_t i = 0; i < args_.size(); ++i) {
            if (i) os << ", ";
            os << args_[i]->id() << ": " << dtypeName(args_[i]->dtype());
        }
        os << "):\n";
    }
    for (const auto& op : ops_) {
        os << "  ";
        op->print(os);
        os << "\n";
    }
}

Block* Region::addBlock() {
    blocks_.push_back(std::make_unique<Block>());
    return blocks_.back().get();
}

Block* Region::entryBlock() {
    if (blocks_.empty())
        FatalInvariant("entryBlock: region has no blocks");
    return blocks_.front().get();
}

// =============================================================================
// 5. OpBuilder
// =============================================================================

std::unique_ptr<Operation> OpBuilder::conv2d(
    Value* input, Value* filter, Value* bias,
    std::vector<int64_t> strides, std::vector<int64_t> pads,
    std::vector<int64_t> dilations, int64_t group) {
    assert(input  && "conv2d: null input");
    assert(filter && "conv2d: null filter");

    auto op = std::make_unique<Operation>("sc_high.conv2d");
    op->addOperand(input);
    op->addOperand(filter);
    if (bias) op->addOperand(bias);

    op->setAttribute("strides",   std::move(strides));
    op->setAttribute("pads",      std::move(pads));
    op->setAttribute("dilations", std::move(dilations));
    op->setAttribute("group",     group);

    op->addResult("", input->dtype(), Shape{});
    return op;
}

std::unique_ptr<Operation> OpBuilder::batchNorm(
    Value* input, Value* scale, Value* bias,
    Value* running_mean, Value* running_var, float epsilon) {
    assert(input        && "batchNorm: null input");
    assert(scale        && "batchNorm: null scale");
    assert(bias         && "batchNorm: null bias");
    assert(running_mean && "batchNorm: null running_mean");
    assert(running_var  && "batchNorm: null running_var");

    auto op = std::make_unique<Operation>("sc_high.batch_norm");
    op->addOperand(input);
    op->addOperand(scale);
    op->addOperand(bias);
    op->addOperand(running_mean);
    op->addOperand(running_var);
    op->setAttribute("epsilon", epsilon);

    op->addResult("", input->dtype(), input->shape());
    return op;
}

std::unique_ptr<Operation> OpBuilder::gemm(
    Value* A, Value* B, Value* bias, bool trans_a, bool trans_b) {
    assert(A && "gemm: null A");
    assert(B && "gemm: null B");

    auto op = std::make_unique<Operation>("sc_high.gemm");
    op->addOperand(A);
    op->addOperand(B);
    if (bias) op->addOperand(bias);

    op->setAttribute("trans_a", static_cast<int64_t>(trans_a));
    op->setAttribute("trans_b", static_cast<int64_t>(trans_b));
    op->addResult("", A->dtype(), Shape{});
    return op;
}

std::unique_ptr<Operation> OpBuilder::relu(Value* input) {
    assert(input && "relu: null input");
    auto op = std::make_unique<Operation>("sc_high.relu");
    op->addOperand(input);
    op->addResult("", input->dtype(), input->shape());
    return op;
}

std::unique_ptr<Operation> OpBuilder::im2col(
    Value* input, std::vector<int64_t> kernel_shape,
    std::vector<int64_t> strides, std::vector<int64_t> pads) {
    assert(input && "im2col: null input");
    auto op = std::make_unique<Operation>("sc_low.im2col");
    op->addOperand(input);
    op->setAttribute("kernel_shape", std::move(kernel_shape));
    op->setAttribute("strides",      std::move(strides));
    op->setAttribute("pads",         std::move(pads));
    op->addResult("", input->dtype(), Shape{});
    return op;
}

} // namespace seecpp::sir
