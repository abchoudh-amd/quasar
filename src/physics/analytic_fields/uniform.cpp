#include "quasar/physics/analytic_fields/uniform.hpp"

#include "quasar/core/observations.hpp"

#include <cmath>
#include <stdexcept>

namespace quasar::analytic_fields {

namespace {

bool finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void validate(Vec3 b0, Vec3 e0) {
  if (!finite(b0) || !finite(e0)) {
    throw std::invalid_argument{
        "UniformEvaluator: electric and magnetic fields must have finite components"};
  }
}

}  // namespace

UniformEvaluator::UniformEvaluator(Vec3 b0, Vec3 e0) : b0_{b0}, e0_{e0} {
  validate(b0_, e0_);
}

void UniformEvaluator::configure(const numerics::EvaluatorParams& p) {
  numerics::reject_unknown_params(p, {"b0", "e0"}, "UniformEvaluator");
  const Vec3 b0 = numerics::param_vec3(p, "b0", b0_);
  const Vec3 e0 = numerics::param_vec3(p, "e0", e0_);
  validate(b0, e0);
  b0_ = b0;
  e0_ = e0;
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
