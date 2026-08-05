#include "compiler/frontend/representation/block.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ostream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace seeml::sir {

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

void Block::moveOpBefore(Operation* op, Operation* anchor) {
    assert(op != anchor && "moveOpBefore: op and anchor are the same");
    assert(op->numOperands() == 0 &&
           "moveOpBefore: only operand-less ops (storage declarations) can "
           "be hoisted without an SSA-order proof");
    auto from = std::find_if(ops_.begin(), ops_.end(),
                             [op](const auto& p) { return p.get() == op; });
    assert(from != ops_.end() && "moveOpBefore: op not found in block");
    auto owned = std::move(*from);
    ops_.erase(from);
    auto to = std::find_if(ops_.begin(), ops_.end(),
                           [anchor](const auto& p) { return p.get() == anchor; });
    assert(to != ops_.end() && "moveOpBefore: anchor not found in block");
    ops_.insert(to, std::move(owned));
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
