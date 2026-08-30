#include "quasar/physics/analytic_fields/dipole.hpp"

#include "quasar/core/device_observations.hpp"
#include "quasar/physics/analytic_fields/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quasar::analytic_fields {

namespace {

bool finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void validate(Vec3 moment, Vec3 origin) {
  if (!finite(moment) || !finite(origin)) {
    throw std::invalid_argument{
        "DipoleEvaluator: moment and origin must have finite components"};
  }
}

bool is_zero(Vec3 v) noexcept {
  return v.x == Real{0} && v.y == Real{0} && v.z == Real{0};
}

int checked_point_count(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "DipoleEvaluator: observation count exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

QuasarAfDipoleParams make_params(Vec3 moment, Vec3 origin) {
  QuasarAfDipoleParams params{};
  params.moment[0] = moment.x;
  params.moment[1] = moment.y;
  params.moment[2] = moment.z;
  params.origin[0] = origin.x;
  params.origin[1] = origin.y;
  params.origin[2] = origin.z;
  params.mu0_over_4pi = mu0_over_4pi;
  return params;
}

}  // namespace

DipoleEvaluator::DipoleEvaluator(Vec3 moment, Vec3 origin)
    : moment_{moment}, origin_{origin} {
  validate(moment_, origin_);
}

void DipoleEvaluator::configure(const numerics::EvaluatorParams& p) {
  numerics::reject_unknown_params(p, {"moment", "origin"}, "DipoleEvaluator");
  const Vec3 moment = numerics::param_vec3(p, "moment", moment_);
  const Vec3 origin = numerics::param_vec3(p, "origin", origin_);
  validate(moment, origin);
  moment_ = moment;
  origin_ = origin;
}

core::DeviceVectorField DipoleEvaluator::evaluate_B(
    const core::IFieldSource&, const core::DevicePointCloud& obs) const {
  // A zero moment has no field anywhere, including at the origin, so it short
  // circuits before the singularity check the kernel would otherwise apply.
  if (is_zero(moment_)) return core::DeviceVectorField(obs.size());

  const int M = checked_point_count(obs.size());
  core::DeviceVectorField out(obs.size(), backend::uninitialized);
  backend::DeviceBuffer<int> status(1);  // zero-initialized bit field

  ::launch_analytic_dipole_B(make_params(moment_, origin_), obs.x(), obs.y(),
                             obs.z(), M, out.x(), out.y(), out.z(),
                             status.device_ptr(), nullptr);

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  core::throw_on_evaluator_status(host_status, "DipoleEvaluator",
                                  "magnetic field");
  return out;
}

core::DeviceTensorField DipoleEvaluator::evaluate_grad_B(
    const core::IFieldSource&, const core::DevicePointCloud& obs) const {
  if (is_zero(moment_)) return core::DeviceTensorField(obs.size());

  const int M = checked_point_count(obs.size());
  core::DeviceTensorField out(obs.size(), backend::uninitialized);
  backend::DeviceBuffer<int> status(1);

  ::launch_analytic_dipole_gradB(make_params(moment_, origin_), obs.x(),
                                 obs.y(), obs.z(), M, out.data(),
                                 status.device_ptr(), nullptr);

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  core::throw_on_evaluator_status(host_status, "DipoleEvaluator",
                                  "magnetic-field gradient");
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("dipole", DipoleEvaluator)

}  // namespace quasar::analytic_fields
