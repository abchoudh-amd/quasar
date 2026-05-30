#pragma once

#include "quasar/numerics/field_evaluator.hpp"

#include <string>

namespace quasar::analytic_fields {

class FileGridEvaluator final : public numerics::IFieldEvaluator {
 public:
  FileGridEvaluator() = default;
  explicit FileGridEvaluator(std::string path) : path_{std::move(path)} {}

  Field<Vec3> evaluate_B(const magnetostatics::ConductorSystem&,
                         const core::PointCloud& observations) const override;
  Field<Mat3x3> evaluate_grad_B(const magnetostatics::ConductorSystem&,
                                const core::PointCloud& observations) const override;

  const std::string& path() const noexcept { return path_; }

 private:
  std::string path_{};
};

}  // namespace quasar::analytic_fields
