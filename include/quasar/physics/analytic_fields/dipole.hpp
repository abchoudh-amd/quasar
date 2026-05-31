#pragma once

#include "quasar/numerics/field_evaluator.hpp"

namespace quasar::analytic_fields {

class DipoleEvaluator final : public numerics::IFieldEvaluator {
 public:
  DipoleEvaluator() = default;
  DipoleEvaluator(Vec3 moment, Vec3 origin = Vec3{0, 0, 0})
    : moment_{moment}, origin_{origin} {}

  // Deck params: "moment" (Vec3 A*m^2, default (0,0,1)), "origin" (Vec3 m).
  void configure(const numerics::EvaluatorParams& p) override;

  Field<Vec3> evaluate_B(const core::IFieldSource&,
                         const core::PointCloud& observations) const override;
  // evaluate_grad_B uses the base-class zero default (no analytic Jacobian).

 private:
  Vec3 moment_{0, 0, 1};
  Vec3 origin_{0, 0, 0};
};

}  // namespace quasar::analytic_fields
