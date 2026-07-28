#include "compiler/analysis/calculus/autodiff.h"

#include <functional>
#include <initializer_list>
#include <string_view>
#include <unordered_set>

#include "compiler/diagnostics/logger.h"

namespace seeml::update {

namespace sir = seeml::sir;
using seecpp::utility::Logger;

namespace {

/// Shared state threaded through the VJP rules.
struct AdContext {
  sir::Block* block;
  // needs_grad: values on a path from a trainable parameter to the loss.
  std::unordered_set<const sir::Value*> needs;
  // adjoint environment: primal value -> current accumulated gradient value.
  std::unordered_map<sir::Value*, sir::Value*> adjoint;
  size_t fresh_counter = 0;

  bool Needs(const sir::Value* v) const { return needs.contains(v); }

  sir::Value* GradOf(sir::Value* primal) {
    auto it = adjoint.find(primal);
    return it == adjoint.end() ? nullptr : it->second;
  }

  /// Accumulates `grad` into `primal`'s adjoint. On fan-out an sc_high.add is
  /// injected — the summation of the multivariable chain rule.
  void Accumulate(sir::Value* primal, sir::Value* grad) {
    if (!Needs(primal)) return;  // pruned: cannot reach a trainable parameter
    auto it = adjoint.find(primal);
    if (it == adjoint.end()) {
      adjoint[primal] = grad;
      return;
    }
    sir::Operation* sum = block->appendOp("sc_high.add");
    sum->addOperand(it->second);
    sum->addOperand(grad);
    sir::Value* acc = sum->addResult(
        std::string(primal->id()) + ".grad_acc" +
            std::to_string(fresh_counter++),
        grad->dtype(), grad->shape());
    it->second = acc;
  }
};

/// A VJP rule synthesizes the adjoint ops for one primal operation.
using VjpRule = std::function<bool(sir::Operation*, AdContext&)>;

sir::Value* Emit1(AdContext& ctx, const char* mnemonic,
                  std::initializer_list<sir::Value*> operands,
                  const std::string& id, const sir::Shape& shape) {
  sir::Operation* op = ctx.block->appendOp(mnemonic);
  for (sir::Value* v : operands) op->addOperand(v);
  return op->addResult(id, sir::DataType::F32, shape);
}

const std::unordered_map<std::string_view, VjpRule>& VjpRegistry() {
  static const auto* const kRegistry = new std::unordered_map<std::string_view,
                                                              VjpRule>{
      // C[N,M] = X[N,K] @ W[K,M]
      // dX = dC @ W^T (GemmNT), dW = X^T @ dC (GemmTN)
      {"sc_high.matmul",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* w = op->operand(1);
         sir::Value* dc = ctx.GradOf(op->result(0));
         if (ctx.Needs(x)) {
           sir::Value* dx = Emit1(
               ctx, "sc_low.matmul_nt", {dc, w},
               std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
               x->shape());
           ctx.Accumulate(x, dx);
         }
         if (ctx.Needs(w)) {
           sir::Value* dw = Emit1(
               ctx, "sc_low.matmul_tn", {x, dc},
               std::string(w->id()) + ".d" + std::to_string(ctx.fresh_counter++),
               w->shape());
           ctx.Accumulate(w, dw);
         }
         return true;
       }},

      // C = A + B (same shape): gradient flows to both, pruning decides which.
      {"sc_high.add",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* dc = ctx.GradOf(op->result(0));
         ctx.Accumulate(op->operand(0), dc);
         ctx.Accumulate(op->operand(1), dc);
         return true;
       }},

      // Y[N,M] = X + b[M]: dX = dY; db = sum over rows of dY.
      {"sc_high.add_bias",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* dy = ctx.GradOf(op->result(0));
         ctx.Accumulate(op->operand(0), dy);
         sir::Value* b = op->operand(1);
         if (ctx.Needs(b)) {
           sir::Value* db = Emit1(
               ctx, "sc_low.reduce_rows", {dy},
               std::string(b->id()) + ".d" + std::to_string(ctx.fresh_counter++),
               b->shape());
           ctx.Accumulate(b, db);
         }
         return true;
       }},

      // Y = max(X, 0): dX = dY * 1[X > 0]  (subgradient 0 at the kink)
      {"sc_high.relu",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* dy = ctx.GradOf(op->result(0));
         sir::Value* dx = Emit1(
             ctx, "sc_low.relu_grad", {dy, x},
             std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
             x->shape());
         ctx.Accumulate(x, dx);
         return true;
       }},

      // Y = gelu(X) (tanh approximation): dX = dY * gelu'(X). The backward
      // kernel differentiates the same approximation the forward evaluates,
      // so the pair is exactly consistent under finite differences.
      {"sc_high.gelu",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* dy = ctx.GradOf(op->result(0));
         sir::Value* dx = Emit1(
             ctx, "sc_low.gelu_grad", {dy, x},
             std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
             x->shape());
         ctx.Accumulate(x, dx);
         return true;
       }},

      // Y = X * sigmoid(X): dX = dY * (s(X) * (1 + X * (1 - s(X)))).
      {"sc_high.silu",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* dy = ctx.GradOf(op->result(0));
         sir::Value* dx = Emit1(
             ctx, "sc_low.silu_grad", {dy, x},
             std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
             x->shape());
         ctx.Accumulate(x, dx);
         return true;
       }},

      // C = X * Y (elementwise): dX = dC * Y, dY = dC * X.
      {"sc_high.mul",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* y = op->operand(1);
         sir::Value* dc = ctx.GradOf(op->result(0));
         if (ctx.Needs(x)) {
           sir::Value* dx = Emit1(
               ctx, "sc_low.mul", {dc, y},
               std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
               x->shape());
           ctx.Accumulate(x, dx);
         }
         if (ctx.Needs(y)) {
           sir::Value* dy = Emit1(
               ctx, "sc_low.mul", {dc, x},
               std::string(y->id()) + ".d" + std::to_string(ctx.fresh_counter++),
               y->shape());
           ctx.Accumulate(y, dy);
         }
         return true;
       }},

      // (Y, mean, rstd) = layer_norm(X, gamma, beta). Only dX is synthesized:
      // gamma/beta are frozen base weights (LoRA adapts MatMuls only), so an
      // adjoint can never be demanded for them by construction.
      {"sc_high.layer_norm",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* gamma = op->operand(1);
         if (ctx.Needs(gamma) || ctx.Needs(op->operand(2)))
           return false;  // trainable affine params are unsupported
         sir::Value* dy = ctx.GradOf(op->result(0));
         if (!ctx.Needs(x)) return true;
         sir::Value* dx = Emit1(
             ctx, "sc_low.layer_norm_grad",
             {dy, x, gamma, op->result(1), op->result(2)},
             std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
             x->shape());
         ctx.Accumulate(x, dx);
         return true;
       }},

      // Y = alpha * X: dX = alpha * dY.
      {"sc_high.scale",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* x = op->operand(0);
         sir::Value* dy = ctx.GradOf(op->result(0));
         sir::Operation* g = ctx.block->appendOp("sc_low.scale");
         g->setAttribute("alpha", op->getAttrAs<float>("alpha").value_or(1.0f));
         g->addOperand(dy);
         sir::Value* dx = g->addResult(
             std::string(x->id()) + ".d" + std::to_string(ctx.fresh_counter++),
             sir::DataType::F32, x->shape());
         ctx.Accumulate(x, dx);
         return true;
       }},

      // (loss, probs) = softmax_xent(logits, labels)
      // dlogits = seed * (probs - onehot(labels)) / N
      {"sc_high.softmax_xent",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* logits = op->operand(0);
         sir::Value* labels = op->operand(1);
         sir::Value* probs = op->result(1);
         sir::Value* seed = ctx.GradOf(op->result(0));
         if (!ctx.Needs(logits)) return true;
         sir::Value* dl =
             Emit1(ctx, "sc_low.softmax_xent_grad", {probs, labels, seed},
                   std::string(logits->id()) + ".d" +
                       std::to_string(ctx.fresh_counter++),
                   logits->shape());
         ctx.Accumulate(logits, dl);
         return true;
       }},

      // loss = mean((pred - target)^2): dpred = seed * 2(pred - target)/count
      {"sc_high.mse",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* pred = op->operand(0);
         sir::Value* target = op->operand(1);
         sir::Value* seed = ctx.GradOf(op->result(0));
         if (!ctx.Needs(pred)) return true;
         sir::Value* dp =
             Emit1(ctx, "sc_low.mse_grad", {pred, target, seed},
                   std::string(pred->id()) + ".d" +
                       std::to_string(ctx.fresh_counter++),
                   pred->shape());
         ctx.Accumulate(pred, dp);
         return true;
       }},

      // (loss, p_s, p_t) = kl_distill(student_logits, teacher_logits)
      // dstudent = seed * (p_s - p_t) / (N * T); the teacher branch is frozen
      // by construction, so no adjoint ever flows into it.
      {"sc_high.kl_distill",
       [](sir::Operation* op, AdContext& ctx) {
         sir::Value* s_logits = op->operand(0);
         sir::Value* p_s = op->result(1);
         sir::Value* p_t = op->result(2);
         sir::Value* seed = ctx.GradOf(op->result(0));
         if (!ctx.Needs(s_logits)) return true;
         sir::Operation* g = ctx.block->appendOp("sc_low.kl_grad");
         g->setAttribute("temperature",
                         op->getAttrAs<float>("temperature").value_or(1.0f));
         g->addOperand(p_s);
         g->addOperand(p_t);
         g->addOperand(seed);
         sir::Value* ds = g->addResult(
             std::string(s_logits->id()) + ".d" +
                 std::to_string(ctx.fresh_counter++),
             sir::DataType::F32, s_logits->shape());
         ctx.Accumulate(s_logits, ds);
         return true;
       }},
  };
  return *kRegistry;
}

}  // namespace

std::expected<std::unordered_map<sir::Value*, sir::Value*>, std::string>
TrainableAutodiff::Run(sir::Block& block, sir::Value* loss,
                       const std::vector<sir::Value*>& trainables) {
  if (!loss) return std::unexpected("TrainableAutodiff: null loss value");
  if (trainables.empty())
    return std::unexpected("TrainableAutodiff: empty trainable set");

  AdContext ctx{.block = &block};

  // --- Forward pass: propagate needs_grad from the trainable set. ---------
  // A value needs a gradient iff it is trainable or is computed from a value
  // that needs one. Everything else — the entire frozen base model and the
  // teacher — is excluded, so no backward compute is ever synthesized for it.
  ctx.needs.reserve(trainables.size() + block.numOps());
  for (sir::Value* p : trainables) ctx.needs.insert(p);
  block.walk([&](sir::Operation* op) {
    bool any = false;
    for (sir::Value* operand : op->operands())
      if (ctx.Needs(operand)) any = true;
    if (any)
      for (const auto& res : op->results()) ctx.needs.insert(res.get());
  });

  if (!ctx.Needs(loss))
    return std::unexpected(
        "TrainableAutodiff: loss does not depend on any trainable parameter");

  // --- Seed dL/dL = 1.0 ----------------------------------------------------
  sir::Operation* fill = block.appendOp("sc_low.fill");
  fill->setAttribute("value", 1.0f);
  sir::Value* seed = fill->addResult(std::string(loss->id()) + ".seed",
                                     sir::DataType::F32, loss->shape());
  ctx.adjoint[loss] = seed;

  // --- Reverse topological sweep over a snapshot of the primal ops. -------
  std::vector<sir::Operation*> primal_ops;
  primal_ops.reserve(block.numOps());
  block.walk([&](sir::Operation* op) { primal_ops.push_back(op); });

  const auto& registry = VjpRegistry();
  for (auto it = primal_ops.rbegin(); it != primal_ops.rend(); ++it) {
    sir::Operation* op = *it;
    const std::string_view mnemonic = op->mnemonic();
    if (mnemonic.starts_with("sc_mem.")) continue;  // storage declarations

    // Skip ops whose outputs carry no adjoint (dead w.r.t. the loss) or whose
    // inputs cannot reach a trainable parameter (frozen subgraphs).
    if (op->numResults() == 0 || !ctx.GradOf(op->result(0))) continue;
    bool any_operand_needs = false;
    for (sir::Value* operand : op->operands())
      if (ctx.Needs(operand)) any_operand_needs = true;
    if (!any_operand_needs) continue;

    auto rule = registry.find(mnemonic);
    if (rule == registry.end())
      return std::unexpected("TrainableAutodiff: no VJP rule for '" +
                             std::string(mnemonic) + "'");
    if (!rule->second(op, ctx))
      return std::unexpected("TrainableAutodiff: VJP failed for '" +
                             std::string(mnemonic) + "'");
  }

  // --- Collect the parameter gradients. ------------------------------------
  std::unordered_map<sir::Value*, sir::Value*> param_grads;
  param_grads.reserve(trainables.size());
  for (sir::Value* p : trainables) {
    sir::Value* g = ctx.GradOf(p);
    if (!g)
      return std::unexpected("TrainableAutodiff: no gradient reached '" +
                             std::string(p->id()) + "'");
    param_grads[p] = g;
  }

  Logger::Info("TrainableAutodiff: built adjoints for " +
               std::to_string(param_grads.size()) + " trainable parameter(s)");
  return param_grads;
}

}  // namespace seeml::update
