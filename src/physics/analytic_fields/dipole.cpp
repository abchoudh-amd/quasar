#include "quasar/physics/analytic_fields/dipole.hpp"

#include "quasar/core/observations.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

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

Real checked_real(long double value, const char* quantity) {
  constexpr long double max_real =
      static_cast<long double>(std::numeric_limits<Real>::max());
  if (!std::isfinite(value) || std::abs(value) > max_real) {
    throw std::overflow_error{
        std::string{"DipoleEvaluator: "} + quantity
        + " is not representable in host precision"};
  }
  return static_cast<Real>(value);
}

Real checked_scaled_product(std::initializer_list<Real> factors,
                            long double angular_factor,
                            const char* quantity) {
  if (angular_factor == 0.0L) return Real{0};
  if (!std::isfinite(angular_factor)) {
    throw std::overflow_error{
        std::string{"DipoleEvaluator: "} + quantity
        + " is not representable in host precision"};
  }

  long double mantissa = std::copysign(1.0L, angular_factor);
  int exponent = 0;
  const auto accumulate = [&](long double factor) {
    if (factor == 0.0L) {
      mantissa = 0.0L;
      exponent = 0;
      return;
    }
    int factor_exponent = 0;
    const long double factor_mantissa =
        std::frexp(std::abs(factor), &factor_exponent);
    mantissa *= factor_mantissa;
    exponent += factor_exponent;
    int adjustment = 0;
    mantissa = std::frexp(mantissa, &adjustment);
    exponent += adjustment;
  };
  for (const Real factor : factors) {
    if (!std::isfinite(factor)) {
      throw std::overflow_error{
          std::string{"DipoleEvaluator: "} + quantity
          + " is not representable in host precision"};
    }
    accumulate(static_cast<long double>(factor));
    if (mantissa == 0.0L) return Real{0};
  }
  accumulate(std::abs(angular_factor));
  return checked_real(std::scalbn(mantissa, exponent), quantity);
}

[[noreturn]] void throw_singular() {
  throw std::domain_error{
      "DipoleEvaluator: magnetic field is singular at the ideal dipole origin"};
}

struct RadialDirection {
  Real nx{};
  Real ny{};
  Real nz{};
  Real inv_r{};
};

RadialDirection radial_direction(Vec3 point, Vec3 origin) {
  if (point.x == origin.x && point.y == origin.y && point.z == origin.z) {
    throw_singular();
  }

  Real rx = point.x - origin.x;
  Real ry = point.y - origin.y;
  Real rz = point.z - origin.z;
  Real distance_scale{};
  if (std::isfinite(rx) && std::isfinite(ry) && std::isfinite(rz)) {
    distance_scale = std::max({std::abs(rx), std::abs(ry), std::abs(rz)});
    rx /= distance_scale;
    ry /= distance_scale;
    rz /= distance_scale;
  } else {
    // Opposite-sign finite coordinates can have an unrepresentable direct
    // difference. Form the displacement in a common coordinate scale instead.
    distance_scale = std::max({std::abs(point.x), std::abs(point.y),
                               std::abs(point.z), std::abs(origin.x),
                               std::abs(origin.y), std::abs(origin.z)});
    rx = point.x / distance_scale - origin.x / distance_scale;
    ry = point.y / distance_scale - origin.y / distance_scale;
    rz = point.z / distance_scale - origin.z / distance_scale;
  }
  const Real scaled_length = std::hypot(rx, ry, rz);
  if (!(scaled_length > Real{0}) || !std::isfinite(scaled_length)) {
    throw std::overflow_error{
        "DipoleEvaluator: source displacement is not representable"};
  }
  return RadialDirection{rx / scaled_length, ry / scaled_length,
                         rz / scaled_length,
                         (Real{1} / distance_scale) / scaled_length};
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

Field<Vec3> DipoleEvaluator::evaluate_B(const core::IFieldSource&,
                                        const core::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  if (is_zero(moment_)) return out;

  const auto& pts = obs.points();
  for (std::size_t i = 0; i < out.size(); ++i) {
    const RadialDirection radial = radial_direction(pts[i], origin_);
    const Real moment_scale = std::max(
        {std::abs(moment_.x), std::abs(moment_.y), std::abs(moment_.z)});
    const Real mx = moment_.x / moment_scale;
    const Real my = moment_.y / moment_scale;
    const Real mz = moment_.z / moment_scale;
    const Real q = mx * radial.nx + my * radial.ny + mz * radial.nz;
    out[i] = Vec3{
        checked_scaled_product(
            {moment_scale, mu0_over_4pi, radial.inv_r, radial.inv_r,
             radial.inv_r},
            Real{3} * static_cast<long double>(q) * radial.nx - mx,
            "magnetic field"),
        checked_scaled_product(
            {moment_scale, mu0_over_4pi, radial.inv_r, radial.inv_r,
             radial.inv_r},
            Real{3} * static_cast<long double>(q) * radial.ny - my,
            "magnetic field"),
        checked_scaled_product(
            {moment_scale, mu0_over_4pi, radial.inv_r, radial.inv_r,
             radial.inv_r},
            Real{3} * static_cast<long double>(q) * radial.nz - mz,
            "magnetic field")};
  }
  return out;
}

Field<Mat3x3> DipoleEvaluator::evaluate_grad_B(
    const core::IFieldSource&, const core::PointCloud& obs) const {
  Field<Mat3x3> out(obs.size());
  if (is_zero(moment_)) return out;

  const auto& pts = obs.points();
  for (std::size_t k = 0; k < out.size(); ++k) {
    const RadialDirection radial = radial_direction(pts[k], origin_);
    const Real n[] = {radial.nx, radial.ny, radial.nz};
    const Real moment_scale = std::max(
        {std::abs(moment_.x), std::abs(moment_.y), std::abs(moment_.z)});
    const Real m[] = {moment_.x / moment_scale, moment_.y / moment_scale,
                      moment_.z / moment_scale};
    const Real q = m[0] * n[0] + m[1] * n[1] + m[2] * n[2];
    const auto entry = [&](int i, int j) {
      const long double angular =
          static_cast<long double>(m[j]) * n[i]
          + static_cast<long double>(m[i]) * n[j]
          + (i == j ? static_cast<long double>(q) : 0.0L)
          - 5.0L * q * n[i] * n[j];
      return checked_scaled_product(
          {Real{3}, mu0_over_4pi, moment_scale, radial.inv_r,
           radial.inv_r, radial.inv_r, radial.inv_r},
          angular, "magnetic-field gradient");
    };
    out[k] = Mat3x3{
        Vec3{entry(0, 0), entry(0, 1), entry(0, 2)},
        Vec3{entry(1, 0), entry(1, 1), entry(1, 2)},
        Vec3{entry(2, 0), entry(2, 1), entry(2, 2)}};
  }
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("dipole", DipoleEvaluator)

}  // namespace quasar::analytic_fields
