#ifndef SEEML_COMPILER_FRONTEND_SIR_H_
#define SEEML_COMPILER_FRONTEND_SIR_H_

#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <map>
#include <variant>
#include <optional>
#include <span>
#include <cstdint>
#include <expected>
#include <functional>
#include <atomic>
#include <unordered_set>

// =============================================================================
// Threading model: SIR is a single-writer structure. A Block — and every
// Operation and Value it owns — may be mutated by one thread at a time;
// nothing here locks. Distinct Blocks are independent and may be built
// concurrently (auto-generated result ids come from one atomic counter), with
// one caveat: a Value's use-list is written by its *readers* (addOperand /
// setOperand / removeOp), so two threads may not build ops that reference the
// same Value, even from different blocks. Once construction stops, any number
// of threads may traverse concurrently.
// =============================================================================

namespace seeml::sir {

class Operation;
class Block;
class Region;

enum class DataType : uint8_t {
    F16, BF16, F32, F64,
    I8, I32, I64,
    Bool
};

constexpr size_t dtypeByteWidth(DataType dt) {
    switch (dt) {
        case DataType::Bool:
        case DataType::I8:   return 1;
        case DataType::F16:
        case DataType::BF16: return 2;
        case DataType::F32:
        case DataType::I32:  return 4;
        case DataType::F64:
        case DataType::I64:  return 8;
        default: return 0;
    }
}

constexpr std::string_view dtypeName(DataType dt) {
    switch (dt) {
        case DataType::F16:  return "f16";
        case DataType::BF16: return "bf16";
        case DataType::F32:  return "f32";
        case DataType::F64:  return "f64";
        case DataType::I8:   return "i8";
        case DataType::I32:  return "i32";
        case DataType::I64:  return "i64";
        case DataType::Bool: return "bool";
        default: return "unknown";
    }
}

struct Shape {
    static constexpr int64_t kDynamic = -1;

    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<int64_t> d) : dims(d) {}

    static Shape scalar() { return Shape{}; }

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool isScalar() const { return dims.empty(); }
    bool isFullyStatic() const;
    /// Element count; kDynamic when any dimension is dynamic/negative or the
    /// product would overflow int64 ("unknown" saturates, never wraps).
    int64_t volume() const;
    /// Bytes at dtype `dt`; 0 when the volume is unknown or the byte count
    /// would overflow size_t.
    size_t byteSize(DataType dt) const;

    bool operator==(const Shape& o) const = default;
};

using AttributeValue = std::variant<
    int64_t, float, double, std::string,
    std::vector<int64_t>, std::vector<float>
>;

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

class Block {
 public:
    Block() = default;
    Block(const Block&) = delete;

    Value* addArgument(DataType dt, Shape sh);
    std::span<const std::unique_ptr<Value>> arguments() const { return args_; }
    std::span<const std::unique_ptr<Operation>> operations() const { return ops_; }
    size_t numOps() const { return ops_.size(); }

    Operation* appendOp(std::string name);
    Operation* appendOp(std::unique_ptr<Operation> op);

    /// Detaches `op` and returns ownership, fully unlinked: its operands'
    /// use-lists drop it, and none of its results may still have uses
    /// (asserted) — replaceAllUsesWith the consumers first. A removed op
    /// must not be re-inserted; its use registration is gone.
    std::unique_ptr<Operation> removeOp(Operation* op);

    /// Inserts a sequence of operations immediately after `anchor`, preserving
    /// their relative order. Required by graph-rewriting passes (e.g. LoRA
    /// grafting) that must splice new computation between a producer and its
    /// existing consumers. `anchor` must belong to this block.
    void insertOpsAfter(Operation* anchor,
                        std::vector<std::unique_ptr<Operation>> new_ops);

    /// The structural verifier — the invariant gate passes rerun after
    /// mutating the graph. Checks, with a diagnostic naming the first
    /// violation: SSA order (every operand defined by an earlier op or an
    /// argument), value-id uniqueness, parent-block consistency, and
    /// use-list symmetry (every value's use-list matches exactly the ops in
    /// this block that reference it — drift here is how a buggy rewrite
    /// corrupts later passes silently). Assumes the block is self-contained:
    /// values defined here are only used by ops appended here.
    [[nodiscard]] std::expected<void, std::string> verify() const;

    /// verify() as a predicate.
    bool validate() const { return verify().has_value(); }

    /// Traversals are templates rather than std::function sinks: walk() is
    /// the single hottest entry point of every compiler pass, and the
    /// template form lets each lambda inline into the loop with no
    /// type-erasure allocation or indirect call per operation.
    template <typename Fn>
    void walk(Fn&& fn) {
        for (auto& op : ops_) fn(op.get());
    }
    template <typename Fn>
    void walkReverse(Fn&& fn) {
        for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) fn(it->get());
    }

    void print(std::ostream& os) const;

 private:
    std::vector<std::unique_ptr<Value>> args_;
    std::vector<std::unique_ptr<Operation>> ops_;
};

class Region {
 public:
    Block* addBlock();
    Block* entryBlock();
    std::span<const std::unique_ptr<Block>> blocks() const { return blocks_; }

 private:
    std::vector<std::unique_ptr<Block>> blocks_;
};

} // namespace seeml::sir

#endif // SEEML_COMPILER_FRONTEND_SIR_H_
