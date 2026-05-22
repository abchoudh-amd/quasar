#pragma once

#include "quasar/numerics/field_evaluator.hpp"

namespace quasar::analytic_fields {

class GradientEvaluator final : public numerics::IFieldEvaluator {
 public:
  GradientEvaluator() = default;
  GradientEvaluator(Vec3 b0, Mat3x3 grad, Vec3 origin = Vec3{0, 0, 0})
    : b0_{b0}, grad_{grad}, origin_{origin} {}

  Field<Vec3> evaluate_B(const magnetostatics::ConductorSystem&,
                         const magnetostatics::PointCloud& observations) const override;
  Field<Mat3x3> evaluate_grad_B(const magnetostatics::ConductorSystem&,
                                const magnetostatics::PointCloud& observations) const override;

 private:
  Vec3 b0_{0, 0, 0};
  Mat3x3 grad_{};
  Vec3 origin_{0, 0, 0};
};

}  // namespace quasar::analytic_fields
