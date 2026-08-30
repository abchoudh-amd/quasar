#pragma once

#include "quasar/numerics/field_evaluator.hpp"

namespace quasar::analytic_fields {

class DipoleEvaluator final : public numerics::IFieldEvaluator {
 public:
  DipoleEvaluator() = default;
  DipoleEvaluator(Vec3 moment, Vec3 origin = Vec3{0, 0, 0});

  // Deck params: "moment" (Vec3 A*m^2, default (0,0,1)), "origin" (Vec3 m).
  void configure(const numerics::EvaluatorParams& p) override;

  core::DeviceVectorField evaluate_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;
  bool provides_grad_B() const noexcept override { return true; }
  core::DeviceTensorField evaluate_grad_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;

 private:
  Vec3 moment_{0, 0, 1};
  Vec3 origin_{0, 0, 0};
};

}  // namespace quasar::analytic_fields
