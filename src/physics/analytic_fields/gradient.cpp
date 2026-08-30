#include "quasar/physics/analytic_fields/gradient.hpp"

#include "quasar/core/device_observations.hpp"
#include "quasar/physics/analytic_fields/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quasar::analytic_fields {

namespace {

bool finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void validate(Vec3 b0, const Mat3x3& grad, Vec3 origin) {
  if (!finite(b0) || !finite(origin)
      || !finite(grad.r0) || !finite(grad.r1) || !finite(grad.r2)) {
    throw std::invalid_argument{
        "GradientEvaluator: B0, gradient, and origin must have finite components"};
  }
  // Only diagonal entries participate in the trace. Basing this tolerance on
  // an unrelated, very large off-diagonal entry could otherwise admit a real
  // magnetic-monopole term.
  const Real scale = std::max({std::abs(grad.r0.x), std::abs(grad.r1.y),
                               std::abs(grad.r2.z)});
  const Real scaled_trace = scale == Real{0} ? Real{0}
      : grad.r0.x / scale + grad.r1.y / scale + grad.r2.z / scale;
  const Real tolerance = Real{64} * std::numeric_limits<Real>::epsilon();
  if (std::abs(scaled_trace) > tolerance) {
    throw std::invalid_argument{
        "GradientEvaluator: gradient trace must be zero (Maxwell div(B)=0)"};
  }
}

int checked_point_count(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "GradientEvaluator: observation count exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

}  // namespace

GradientEvaluator::GradientEvaluator(Vec3 b0, Mat3x3 grad, Vec3 origin)
    : b0_{b0}, grad_{grad}, origin_{origin} {
  validate(b0_, grad_, origin_);
}

void GradientEvaluator::configure(const numerics::EvaluatorParams& p) {
  numerics::reject_unknown_params(
      p, {"b0", "grad", "origin"}, "GradientEvaluator");
  const Vec3 b0 = numerics::param_vec3(p, "b0", b0_);
  const Mat3x3 grad = numerics::param_mat3x3(p, "grad", grad_);
  const Vec3 origin = numerics::param_vec3(p, "origin", origin_);
  validate(b0, grad, origin);
  b0_ = b0;
  grad_ = grad;
  origin_ = origin;
}

core::DeviceVectorField GradientEvaluator::evaluate_B(
    const core::IFieldSource&, const core::DevicePointCloud& obs) const {
  const int M = checked_point_count(obs.size());
  core::DeviceVectorField out(obs.size(), backend::uninitialized);
  backend::DeviceBuffer<int> status(1);

  QuasarAfGradientParams params{};
  params.b0[0] = b0_.x;
  params.b0[1] = b0_.y;
  params.b0[2] = b0_.z;
  const Vec3 rows[3] = {grad_.r0, grad_.r1, grad_.r2};
  for (int row = 0; row < 3; ++row) {
    params.grad[3 * row + 0] = rows[row].x;
    params.grad[3 * row + 1] = rows[row].y;
    params.grad[3 * row + 2] = rows[row].z;
  }
  params.origin[0] = origin_.x;
  params.origin[1] = origin_.y;
  params.origin[2] = origin_.z;

  ::launch_analytic_gradient_B(params, obs.x(), obs.y(), obs.z(), M, out.x(),
                               out.y(), out.z(), status.device_ptr(), nullptr);

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  core::throw_on_evaluator_status(host_status, "GradientEvaluator",
                                  "magnetic field");
  return out;
}

core::DeviceTensorField GradientEvaluator::evaluate_grad_B(
    const core::IFieldSource&, const core::DevicePointCloud& obs) const {
  // The Jacobian is the configured constant matrix at every point, so this is
  // three constant fills rather than a gradient kernel. DeviceTensorField is
  // component-major, so the three planes of matrix row `i` -- entries (i,0),
  // (i,1), (i,2) -- are the contiguous block starting at 3*i*n, which is
  // exactly the (ox, oy, oz) triple the uniform fill writes.
  const int M = checked_point_count(obs.size());
  core::DeviceTensorField out(obs.size(), backend::uninitialized);
  if (M == 0) return out;

  const std::size_t n = obs.size();
  const Vec3 rows[3] = {grad_.r0, grad_.r1, grad_.r2};
  for (int row = 0; row < 3; ++row) {
    Real* base = out.data() + static_cast<std::size_t>(3 * row) * n;
    ::launch_analytic_uniform_fill(rows[row].x, rows[row].y, rows[row].z, M,
                                   base, base + n, base + 2 * n, nullptr);
  }
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("gradient", GradientEvaluator)

}  // namespace quasar::analytic_fields
