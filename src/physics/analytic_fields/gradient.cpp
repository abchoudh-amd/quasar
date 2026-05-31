#include "quasar/physics/analytic_fields/gradient.hpp"

#include "quasar/core/observations.hpp"

namespace quasar::analytic_fields {

Field<Vec3> GradientEvaluator::evaluate_B(const core::IFieldSource&,
                                          const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  const auto& pts = obs.points();
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = b0_ + grad_ * (pts[i] - origin_);
  }
  return out;
}

Field<Mat3x3> GradientEvaluator::evaluate_grad_B(const core::IFieldSource&,
                                                 const core::PointCloud& obs) const {
  Field<Mat3x3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = grad_;
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("gradient", GradientEvaluator)

}  // namespace quasar::analytic_fields
