#pragma once

#include "quasar/numerics/field_evaluator.hpp"

namespace quasar::analytic_fields {

class GradientEvaluator final : public numerics::IFieldEvaluator {
 public:
  GradientEvaluator() = default;
  GradientEvaluator(Vec3 b0, Mat3x3 grad, Vec3 origin = Vec3{0, 0, 0});

  // Deck params: "b0" (Vec3 tesla), "grad" (Mat3x3 row-major T/m), "origin"
  // (Vec3 m). All default to zero. Maxwell's div(B)=0 requires trace(grad)=0;
  // non-solenoidal matrices are rejected. Symmetry is not required because a
  // general region may carry current (curl(B) != 0).
  void configure(const numerics::EvaluatorParams& p) override;

  core::DeviceVectorField evaluate_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;
  bool provides_grad_B() const noexcept override { return true; }
  core::DeviceTensorField evaluate_grad_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;

 private:
  Vec3 b0_{0, 0, 0};
  Mat3x3 grad_{};
  Vec3 origin_{0, 0, 0};
};

}  // namespace quasar::analytic_fields
