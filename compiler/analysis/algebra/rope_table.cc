#include "compiler/analysis/algebra/rope_table.h"

#include <bit>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "compiler/diagnostics/updating/error.h"
#include "source/language/model_format.h"

namespace seeml::update {

namespace sir = seeml::sir;
namespace updating = seeml::diag::updating;

std::expected<size_t, std::string> RopeTableHoister::Run(sir::Block& block) {
  std::vector<sir::Operation*> ropes;
  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == "sc_high.rope") ropes.push_back(op);
  });

  // One table per (seq, head width, base bits): layers share geometry, so
  // a decoder of any depth declares exactly one.
  std::map<std::tuple<int64_t, int64_t, uint32_t>, sir::Value*> tables;
  for (sir::Operation* rope : ropes) {
    if (rope->numOperands() != 1)
      return updating::Error(updating::kRopeTable,
                             "rope op already carries a table operand");
    const int64_t heads = rope->getAttrAs<int64_t>("heads").value_or(0);
    const int64_t seq = rope->getAttrAs<int64_t>("seq").value_or(0);
    const auto& dims = rope->operand(0)->shape().dims;
    if (heads <= 0 || seq <= 0 || dims.size() != 2 || dims[1] % heads != 0)
      return updating::Error(updating::kRopeTable,
                             "malformed sequence geometry on a rope op");
    const int64_t width = dims[1] / heads;
    const float base =
        rope->getAttrAs<float>("base").value_or(kSmfDefaultRopeBase);
    const auto key = std::make_tuple(seq, width, std::bit_cast<uint32_t>(base));

    auto it = tables.find(key);
    if (it == tables.end()) {
      // Declared at the end, then moved above its first user: the op has
      // no operands, so any earlier position is SSA-sound.
      sir::Operation* table = block.appendOp("sc_low.rope_table");
      table->setAttribute("seq", seq);
      table->setAttribute("width", width);
      table->setAttribute("base", base);
      sir::Value* result = table->addResult(
          "rope_table." + std::to_string(tables.size()), sir::DataType::F32,
          sir::Shape{{seq, width}});
      block.moveOpBefore(table, rope);
      it = tables.emplace(key, result).first;
    }
    rope->addOperand(it->second);
  }
  return tables.size();
}

}  // namespace seeml::update
