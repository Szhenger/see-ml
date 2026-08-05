#ifndef SEEML_COMPILER_FRONTEND_REPRESENTATION_BLOCK_H_
#define SEEML_COMPILER_FRONTEND_REPRESENTATION_BLOCK_H_

#include <expected>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "compiler/frontend/representation/operation.h"
#include "compiler/frontend/representation/value.h"

// =============================================================================
// Block & Region — ownership and ordering: a Block owns its arguments and
// operations in program order and carries the structural verifier; a Region
// owns blocks. Part of the SIR core; see sir.h for the threading model (a
// Block is the unit of single-writer construction).
// =============================================================================

namespace seeml::sir {

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

    /// Moves `op` immediately before `anchor` (both must belong to this
    /// block). Only sound when every operand of `op` remains defined before
    /// the new position; the canonical caller hoists an operand-less storage
    /// declaration (`sc_mem.*`) above a consumer it is being fused into
    /// (asserted). No use-lists change — only program order does.
    void moveOpBefore(Operation* op, Operation* anchor);

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

#endif // SEEML_COMPILER_FRONTEND_REPRESENTATION_BLOCK_H_
