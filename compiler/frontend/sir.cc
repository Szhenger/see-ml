#include "compiler/frontend/sir.h"

#include <atomic>
#include <ostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace seeml::sir {

// =============================================================================
// 1. Shape
// =============================================================================

bool Shape::isFullyStatic() const {
    return std::none_of(dims.begin(), dims.end(), [](int64_t d) { return d == kDynamic; });
}

int64_t Shape::volume() const {
    int64_t v = 1;
    for (auto d : dims) {
        // Any negative dim (kDynamic or invalid) and any product that would
        // overflow int64 saturate to "unknown" — signed overflow here would
        // be UB, and a wrapped element count is worse than no count.
        if (d < 0) return kDynamic;
        if (d != 0 && v > INT64_MAX / d) return kDynamic;
        v *= d;
    }
    return v;
}

size_t Shape::byteSize(DataType dt) const {
    const int64_t vol = volume();
    if (vol < 0) return 0;
    const size_t width = dtypeByteWidth(dt);
    if (width != 0 && static_cast<size_t>(vol) > SIZE_MAX / width) return 0;
    return static_cast<size_t>(vol) * width;
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
    // std::less<> makes the map's find heterogeneous: look up by
    // string_view directly, no temporary std::string per attribute query.
    if (auto it = attributes_.find(key); it != attributes_.end())
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
    assert(it != ops_.end() && "insertOpsAfter: anchor not found in block");

    for (auto& op : new_ops)
        op->setParentBlock(this);

    ops_.insert(std::next(it),
                std::make_move_iterator(new_ops.begin()),
                std::make_move_iterator(new_ops.end()));
}

std::unique_ptr<Operation> Block::removeOp(Operation* op) {
    auto it = std::find_if(ops_.begin(), ops_.end(), [op](const auto& p) { return p.get() == op; });
    assert(it != ops_.end() && "removeOp: operation not found in block");
#ifndef NDEBUG
    for (const auto& res : op->results())
        assert(res->hasNoUses() &&
               "removeOp: results are still in use — replaceAllUsesWith the "
               "consumers first");
#endif

    for (size_t i = 0; i < op->numOperands(); ++i)
        op->operand(i)->removeUser(op);

    auto owned = std::move(*it);
    ops_.erase(it);
    owned->setParentBlock(nullptr);
    return owned;
}

std::expected<void, std::string> Block::verify() const {
    std::unordered_set<const Value*> defined;
    std::unordered_set<std::string_view> ids;
    defined.reserve(args_.size() + ops_.size() * 2);  // ~2 results/op typical
    ids.reserve(args_.size() + ops_.size() * 2);

    auto define = [&](const Value* v) -> bool {
        defined.insert(v);
        return ids.insert(v->id()).second;
    };

    for (const auto& arg : args_)
        if (!define(arg.get()))
            return std::unexpected("SIR verify: duplicate value id '" +
                                   std::string(arg->id()) + "'");

    // The users each value SHOULD have, rebuilt from the operand lists; the
    // stored use-lists are checked against this below.
    std::unordered_map<const Value*, std::vector<const Operation*>> expected;

    for (const auto& op : ops_) {
        if (op->parentBlock() != this)
            return std::unexpected("SIR verify: op '" +
                                   std::string(op->mnemonic()) +
                                   "' has a stale parent block");
        for (const Value* operand : op->operands()) {
            if (defined.find(operand) == defined.end())
                return std::unexpected("SIR verify: op '" + op->toString() +
                                       "' uses '" + std::string(operand->id()) +
                                       "' before its definition");
            expected[operand].push_back(op.get());
        }
        for (const auto& res : op->results())
            if (!define(res.get()))
                return std::unexpected("SIR verify: duplicate value id '" +
                                       std::string(res->id()) + "'");
    }

    // Use-list symmetry: drift between a value's stored users and the ops
    // that actually reference it is how a buggy rewrite corrupts later
    // passes silently. Multiset comparison — an op referencing a value in
    // two operand slots must appear twice.
    auto users_agree = [&](const Value* v) {
        std::vector<const Operation*> stored(v->users().begin(),
                                             v->users().end());
        std::vector<const Operation*> derived = std::move(expected[v]);
        std::sort(stored.begin(), stored.end());
        std::sort(derived.begin(), derived.end());
        return stored == derived;
    };
    for (const auto& arg : args_)
        if (!users_agree(arg.get()))
            return std::unexpected("SIR verify: use-list of '" +
                                   std::string(arg->id()) +
                                   "' disagrees with the operations that "
                                   "reference it");
    for (const auto& op : ops_)
        for (const auto& res : op->results())
            if (!users_agree(res.get()))
                return std::unexpected("SIR verify: use-list of '" +
                                       std::string(res->id()) +
                                       "' disagrees with the operations that "
                                       "reference it");
    return {};
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
    assert(!blocks_.empty());
    return blocks_.front().get();
}

} // namespace seeml::sir
