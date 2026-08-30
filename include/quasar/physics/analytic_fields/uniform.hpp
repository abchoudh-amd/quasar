#pragma once

#include "quasar/numerics/field_evaluator.hpp"

namespace quasar::analytic_fields {

class UniformEvaluator final : public numerics::IFieldEvaluator {
 public:
  UniformEvaluator() = default;
  explicit UniformEvaluator(Vec3 b0, Vec3 e0 = Vec3{0, 0, 0});

  // Deck params: "b0" (Vec3 tesla), "e0" (Vec3 V/m). Both default to zero.
  void configure(const numerics::EvaluatorParams& p) override;

  core::DeviceVectorField evaluate_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;
  core::DeviceVectorField evaluate_E(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;
  // The base-class zero Jacobian is the exact gradient of this evaluator.
  bool provides_grad_B() const noexcept override { return true; }

 private:
  Vec3 b0_{0, 0, 0};
  Vec3 e0_{0, 0, 0};
};

}  // namespace quasar::analytic_fields
