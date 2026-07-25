#include "quasar/physics/analytic_fields/gradient.hpp"

#include "quasar/core/observations.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace quasar::analytic_fields {

namespace {

bool finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

Real checked_real(long double value) {
  constexpr long double max_real =
      static_cast<long double>(std::numeric_limits<Real>::max());
  if (!std::isfinite(value) || std::abs(value) > max_real) {
    throw std::overflow_error{
        "GradientEvaluator: magnetic field is not representable in host precision"};
  }
  return static_cast<Real>(value);
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

struct ScaledTerm {
  long double mantissa{};
  int exponent{};
};

ScaledTerm make_scaled_term(std::initializer_list<Real> factors) {
  ScaledTerm term{1.0L, 0};
  for (const Real factor : factors) {
    if (factor == Real{0}) return {};
    int factor_exponent = 0;
    const Real factor_mantissa = std::frexp(factor, &factor_exponent);
    term.mantissa *= static_cast<long double>(factor_mantissa);
    term.exponent += factor_exponent;
    int adjustment = 0;
    term.mantissa = std::frexp(term.mantissa, &adjustment);
    term.exponent += adjustment;
  }
  return term;
}

Real checked_scaled_sum(const std::array<ScaledTerm, 4>& terms) {
  std::array<ScaledTerm, 4> active{};
  std::size_t count = 0;
  for (const ScaledTerm term : terms) {
    if (term.mantissa != 0.0L) active[count++] = term;
  }
  if (count == 0) return Real{0};

  // Repeatedly combine the two largest-exponent terms, then reinsert the
  // normalized result.  Thus large cancelling terms meet before a much smaller
  // term is rescaled into their domain.  Unlike a single common-exponent sum,
  // this does not rely on long double having a wider exponent range than Real.
  while (count > 1) {
    std::size_t first = 0;
    for (std::size_t i = 1; i < count; ++i) {
      if (active[i].exponent > active[first].exponent) first = i;
    }
    std::size_t second = first == 0 ? 1 : 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (i != first && active[i].exponent > active[second].exponent) second = i;
    }
    const ScaledTerm a = active[first];
    const ScaledTerm b = active[second];
    const long double sum = a.mantissa
        + std::scalbn(b.mantissa, b.exponent - a.exponent);
    std::array<ScaledTerm, 4> remaining{};
    std::size_t remaining_count = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (i != first && i != second) remaining[remaining_count++] = active[i];
    }
    if (sum != 0.0L) {
      int adjustment = 0;
      const long double mantissa = std::frexp(sum, &adjustment);
      remaining[remaining_count++] =
          ScaledTerm{mantissa, a.exponent + adjustment};
    }
    active = remaining;
    count = remaining_count;
  }
  if (count == 0) return Real{0};
  return checked_real(std::scalbn(active[0].mantissa, active[0].exponent));
}

ScaledTerm displacement_term(Real gradient, Real point, Real origin) {
  if (gradient == Real{0}) return {};
  const Real delta = point - origin;
  if (std::isfinite(delta)) return make_scaled_term({gradient, delta});

  // Only divide first when the direct subtraction overflows.  For nearby large
  // coordinates, subtracting the two representable values first preserves the
  // exact small displacement; normalizing each coordinate separately does not.
  const Real coordinate_scale = std::max(std::abs(point), std::abs(origin));
  const Real scaled_delta = point / coordinate_scale - origin / coordinate_scale;
  return make_scaled_term({gradient, scaled_delta, coordinate_scale});
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

Field<Vec3> GradientEvaluator::evaluate_B(const core::IFieldSource&,
                                          const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  const auto& pts = obs.points();
  for (std::size_t i = 0; i < out.size(); ++i) {
    const Vec3 rows[] = {grad_.r0, grad_.r1, grad_.r2};
    Real values[3]{};
    for (int row = 0; row < 3; ++row) {
      const Real b = row == 0 ? b0_.x : row == 1 ? b0_.y : b0_.z;
      values[row] = checked_scaled_sum({
          make_scaled_term({b}),
          displacement_term(rows[row].x, pts[i].x, origin_.x),
          displacement_term(rows[row].y, pts[i].y, origin_.y),
          displacement_term(rows[row].z, pts[i].z, origin_.z)});
    }
    out[i] = Vec3{values[0], values[1], values[2]};
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
