#ifndef SEEML_COMPILER_FRONTEND_REPRESENTATION_OPERATION_H_
#define SEEML_COMPILER_FRONTEND_REPRESENTATION_OPERATION_H_

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "compiler/frontend/representation/type.h"
#include "compiler/frontend/representation/value.h"

// =============================================================================
// Operation — one instruction of the graph: a mnemonic, its operands (with
// use registration), the result values it owns, and its attribute map. Part
// of the SIR core; see sir.h for the threading model.
// =============================================================================

namespace seeml::sir {

class Block;

class Operation {
 public:
    explicit Operation(std::string mnemonic, Block* parent = nullptr);
    Operation(const Operation&) = delete;

    std::string_view mnemonic() const { return mnemonic_; }
    Block* parentBlock() { return parent_block_; }
    const Block* parentBlock() const { return parent_block_; }
    void setParentBlock(Block* b) { parent_block_ = b; }

    bool isHighLevel() const { return mnemonic_.starts_with("sc_high."); }
    bool isLowLevel() const { return mnemonic_.starts_with("sc_low."); }
    bool isMemoryOp() const { return mnemonic_.starts_with("sc_mem."); }
    bool isControlFlow() const { return mnemonic_.starts_with("sc_ctrl."); }

    std::span<Value* const> operands() const { return operands_; }
    Value* operand(size_t i) const { return operands_.at(i); }
    size_t numOperands() const { return operands_.size(); }

    void addOperand(Value* v);
    void setOperand(size_t i, Value* newVal);

    std::span<const std::unique_ptr<Value>> results() const { return results_; }
    Value* result(size_t i = 0) const { return results_.at(i).get(); }
    size_t numResults() const { return results_.size(); }

    Value* addResult(std::string id, DataType dt, Shape sh);

    void setAttribute(std::string key, AttributeValue val);
    const AttributeValue* getAttribute(std::string_view key) const;
    bool hasAttribute(std::string_view key) const { return getAttribute(key) != nullptr; }

    template <typename T>
    std::optional<T> getAttrAs(std::string_view key) const {
        if (auto* av = getAttribute(key))
            if (auto* v = std::get_if<T>(av))
                return *v;
        return std::nullopt;
    }

    void print(std::ostream& os) const;
    std::string toString() const;

 private:
    std::string mnemonic_;
    std::vector<Value*> operands_;
    std::vector<std::unique_ptr<Value>> results_;
    std::map<std::string, AttributeValue, std::less<>> attributes_;
    Block* parent_block_ = nullptr;

    static std::atomic<size_t> id_counter_;
    friend class Block;
};

} // namespace seeml::sir

#endif // SEEML_COMPILER_FRONTEND_REPRESENTATION_OPERATION_H_
