#include "compiler/frontend/representation/operation.h"

#include <cassert>
#include <ostream>
#include <sstream>
#include <type_traits>

namespace seeml::sir {

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

} // namespace seeml::sir
