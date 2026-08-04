#include "compiler/backend/tuner/bandit.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace seeml::update {

Ucb1Bandit::Ucb1Bandit(size_t num_arms, double exploration)
    : arms_(num_arms), exploration_(exploration) {
  assert(num_arms > 0 && "Ucb1Bandit: need at least one arm");
}

size_t Ucb1Bandit::Select() const {
  // Every arm is pulled once before any confidence bound is trusted.
  for (size_t i = 0; i < arms_.size(); ++i)
    if (arms_[i].pulls == 0) return i;

  // -inf, not any finite sentinel: rewards are arbitrary "higher is better"
  // measurements, so every score can legitimately be far below zero.
  size_t best = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  const double log_total = std::log(static_cast<double>(total_pulls_));
  for (size_t i = 0; i < arms_.size(); ++i) {
    const double bonus = exploration_ *
        std::sqrt(2.0 * log_total / static_cast<double>(arms_[i].pulls));
    const double score = arms_[i].mean + bonus;
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

void Ucb1Bandit::Update(size_t arm, double reward) {
  ArmStats& s = arms_.at(arm);
  ++s.pulls;
  ++total_pulls_;
  s.mean += (reward - s.mean) / static_cast<double>(s.pulls);
}

size_t Ucb1Bandit::BestArm() const {
  size_t best = 0;
  for (size_t i = 1; i < arms_.size(); ++i)
    if (arms_[i].pulls > 0 &&
        (arms_[best].pulls == 0 || arms_[i].mean > arms_[best].mean))
      best = i;
  return best;
}

}  // namespace seeml::update
