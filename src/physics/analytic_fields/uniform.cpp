#include "quasar/physics/analytic_fields/uniform.hpp"

#include "quasar/core/observations.hpp"

namespace quasar::analytic_fields {

Field<Vec3> UniformEvaluator::evaluate_B(const magnetostatics::ConductorSystem&,
                                         const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = b0_;
  return out;
}

Field<Vec3> UniformEvaluator::evaluate_E(const magnetostatics::ConductorSystem&,
                                         const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = e0_;
  return out;
}

Field<Mat3x3> UniformEvaluator::evaluate_grad_B(const magnetostatics::ConductorSystem&,
                                                const core::PointCloud& obs) const {
  Field<Mat3x3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = Mat3x3{};
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("uniform", UniformEvaluator)

}  // namespace quasar::analytic_fields
