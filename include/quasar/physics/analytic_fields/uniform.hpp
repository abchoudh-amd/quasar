#pragma once

#include "quasar/numerics/field_evaluator.hpp"

namespace quasar::analytic_fields {

class UniformEvaluator final : public numerics::IFieldEvaluator {
 public:
  UniformEvaluator() = default;
  explicit UniformEvaluator(Vec3 b0, Vec3 e0 = Vec3{0, 0, 0}) : b0_{b0}, e0_{e0} {}

  Field<Vec3> evaluate_B(const core::IFieldSource&,
                         const core::PointCloud& observations) const override;
  Field<Vec3> evaluate_E(const core::IFieldSource&,
                         const core::PointCloud& observations) const override;
  // evaluate_grad_B uses the base-class zero default (no analytic Jacobian).

 private:
  Vec3 b0_{0, 0, 0};
  Vec3 e0_{0, 0, 0};
};

}  // namespace quasar::analytic_fields
