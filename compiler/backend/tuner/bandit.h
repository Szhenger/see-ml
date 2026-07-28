#ifndef SEEML_COMPILER_BACKEND_TUNER_BANDIT_H_
#define SEEML_COMPILER_BACKEND_TUNER_BANDIT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

// =============================================================================
// Ucb1Bandit — the tuner's reinforcement-learning core: a multi-armed
// bandit with the UCB1 selection rule (Auer et al.). Each arm is one
// discrete configuration choice; each pull's reward is a measured outcome
// (e.g. benchmark throughput). UCB1 balances trying under-explored arms
// against exploiting the best mean seen so far:
//
//   select  argmax_i  mean_i + c * sqrt(2 ln(total_pulls) / pulls_i)
//
// with every untried arm pulled once first. Selection is fully
// deterministic — no RNG: untried arms and score ties resolve to the lowest
// index — so a tuning run is reproducible given the same reward sequence.
// =============================================================================

namespace seeml::update {

class Ucb1Bandit {
 public:
  /// `num_arms` must be at least 1. `exploration` scales the confidence
  /// bonus; sqrt(2)/2 ~ 0.7 explores less, 2.0 explores more.
  explicit Ucb1Bandit(size_t num_arms, double exploration = 1.0);

  /// The arm to pull next: the lowest-index untried arm, else the highest
  /// UCB score (ties to the lowest index).
  size_t Select() const;

  /// Records `reward` (higher is better) for a pull of `arm`.
  void Update(size_t arm, double reward);

  /// The arm with the best observed mean reward (ties to the lowest index).
  /// Meaningful once every arm has been pulled at least once.
  size_t BestArm() const;

  size_t numArms() const { return arms_.size(); }
  uint64_t totalPulls() const { return total_pulls_; }
  uint64_t pulls(size_t arm) const { return arms_.at(arm).pulls; }
  double meanReward(size_t arm) const { return arms_.at(arm).mean; }

 private:
  struct ArmStats {
    uint64_t pulls = 0;
    double mean = 0.0;  // incremental mean of observed rewards
  };

  std::vector<ArmStats> arms_;
  uint64_t total_pulls_ = 0;
  double exploration_;
};

}  // namespace seeml::update

#endif  // SEEML_COMPILER_BACKEND_TUNER_BANDIT_H_
