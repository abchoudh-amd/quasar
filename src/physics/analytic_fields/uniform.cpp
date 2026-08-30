#include "quasar/physics/analytic_fields/uniform.hpp"

#include "quasar/core/device_observations.hpp"
#include "quasar/physics/analytic_fields/kernels.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

int checked_point_count(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "UniformEvaluator: observation count exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

// Both evaluate_B and evaluate_E are the same constant fill; only the vector
// differs. The components are validated at configure time, so this needs no
// status word.
core::DeviceVectorField fill(Vec3 value, const core::DevicePointCloud& obs) {
  const int M = checked_point_count(obs.size());
  core::DeviceVectorField out(obs.size(), backend::uninitialized);
  ::launch_analytic_uniform_fill(value.x, value.y, value.z, M, out.x(), out.y(),
                                 out.z(), nullptr);
  return out;
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

core::DeviceVectorField UniformEvaluator::evaluate_B(
    const core::IFieldSource&, const core::DevicePointCloud& obs) const {
  return fill(b0_, obs);
}

core::DeviceVectorField UniformEvaluator::evaluate_E(
    const core::IFieldSource&, const core::DevicePointCloud& obs) const {
  return fill(e0_, obs);
}

QUASAR_REGISTER_FIELD_EVALUATOR("uniform", UniformEvaluator)

}  // namespace quasar::analytic_fields
