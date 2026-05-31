#include "quasar/physics/analytic_fields/uniform.hpp"

#include "quasar/core/observations.hpp"

namespace quasar::analytic_fields {

void UniformEvaluator::configure(const numerics::EvaluatorParams& p) {
  b0_ = numerics::param_vec3(p, "b0", b0_);
  e0_ = numerics::param_vec3(p, "e0", e0_);
}

Field<Vec3> UniformEvaluator::evaluate_B(const core::IFieldSource&,
                                         const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = b0_;
  return out;
}

Field<Vec3> UniformEvaluator::evaluate_E(const core::IFieldSource&,
                                         const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = e0_;
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("uniform", UniformEvaluator)

}  // namespace quasar::analytic_fields
