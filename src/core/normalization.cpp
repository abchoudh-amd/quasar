#include "quasar/core/normalization.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>

namespace quasar {

namespace {

struct ScaledMagnitude {
  Real mantissa{Real{0.5}};
  int exponent{1};
};

void multiply_scaled(ScaledMagnitude& value, Real factor, bool divide = false) {
  int factor_exponent = 0;
  const Real factor_mantissa = std::frexp(std::abs(factor), &factor_exponent);
  if (divide) {
    value.mantissa /= factor_mantissa;
    value.exponent -= factor_exponent;
  } else {
    value.mantissa *= factor_mantissa;
    value.exponent += factor_exponent;
  }
  int adjustment = 0;
  value.mantissa = std::frexp(value.mantissa, &adjustment);
  value.exponent += adjustment;
}

ScaledMagnitude scaled_ratio_parts(std::initializer_list<Real> numerator,
                                   std::initializer_list<Real> denominator) {
  ScaledMagnitude value;
  for (const Real factor : numerator) multiply_scaled(value, factor);
  for (const Real factor : denominator) multiply_scaled(value, factor, true);
  return value;
}

Real materialize(ScaledMagnitude value) noexcept {
  return std::scalbn(value.mantissa, value.exponent);
}

Real scaled_ratio(std::initializer_list<Real> numerator,
                  std::initializer_list<Real> denominator) noexcept {
  return materialize(scaled_ratio_parts(numerator, denominator));
}

Real scaled_sqrt_ratio(std::initializer_list<Real> numerator,
                       std::initializer_list<Real> denominator) noexcept {
  ScaledMagnitude value = scaled_ratio_parts(numerator, denominator);
  if (value.exponent % 2 != 0) {
    value.mantissa *= Real{2};
    --value.exponent;
  }
  value.mantissa = std::sqrt(value.mantissa);
  value.exponent /= 2;
  return materialize(value);
}

Real checked_positive_scale(Real value, const char* what) {
  if (!std::isfinite(value) || value <= Real{0}) {
    throw std::overflow_error{std::string{"Normalization::plasma: "} + what
                              + " is not representable"};
  }
  return value;
}

}  // namespace

Normalization Normalization::plasma(Real n, Real q, Real m) {
  if (!(std::isfinite(n) && std::isfinite(q) && std::isfinite(m))
      || n <= Real{0} || q == Real{0} || m <= Real{0}) {
    throw std::invalid_argument{"Normalization::plasma: invalid reference values"};
  }
  Normalization out;
  out.n_ref = n;
  out.q_ref = q;
  out.m_ref = m;
  // Keep a normalized mantissa plus a base-two exponent throughout.  This
  // avoids both q*q underflow and n/m overflow even when long double is merely
  // an alias for double.
  const Real q_magnitude = std::abs(q);
  out.omega_p_ref = checked_positive_scale(
      scaled_sqrt_ratio({n, q_magnitude, q_magnitude},
                        {constants::eps0, m}),
      "plasma frequency");

  // A normalization is only useful when every advertised conversion scale is
  // representable.  Validate them here rather than allowing a later conversion
  // to silently produce zero or infinity.
  const Real omega = out.omega_p_ref;
  (void)checked_positive_scale(
      scaled_ratio({constants::c0}, {omega}), "length scale");
  (void)checked_positive_scale(scaled_ratio({Real{1}}, {omega}), "time scale");
  (void)checked_positive_scale(
      scaled_ratio({m, constants::c0, omega}, {q_magnitude}),
      "electric-field scale");
  (void)checked_positive_scale(
      scaled_ratio({m, omega}, {q_magnitude}), "magnetic-field scale");
  (void)checked_positive_scale(
      scaled_ratio({m, constants::c0, constants::c0}, {constants::qe_abs}),
      "temperature scale");
  return out;
}

Real Normalization::length_scale() const noexcept {
  if (is_identity()) return Real{1};
  return scaled_ratio({constants::c0}, {omega_p_ref});
}

Real Normalization::time_scale() const noexcept {
  if (is_identity()) return Real{1};
  return scaled_ratio({Real{1}}, {omega_p_ref});
}

Real Normalization::e_field_scale() const noexcept {
  if (is_identity()) return Real{1};
  const Real magnitude = scaled_ratio(
      {m_ref, constants::c0, omega_p_ref}, {std::abs(q_ref)});
  return std::copysign(magnitude, q_ref);
}

Real Normalization::b_field_scale() const noexcept {
  if (is_identity()) return Real{1};
  const Real magnitude = scaled_ratio(
      {m_ref, omega_p_ref}, {std::abs(q_ref)});
  return std::copysign(magnitude, q_ref);
}

Real Normalization::temperature_eV_scale() const noexcept {
  if (is_identity()) return Real{1};
  // One internal energy unit is m_ref*c^2. Dividing joules by the exact
  // elementary charge yields electron-volts.
  return scaled_ratio(
      {m_ref, constants::c0, constants::c0}, {constants::qe_abs});
}

Real Normalization::to_internal(Real value, UnitTag tag) const {
  const bool identity = is_identity();
  switch (tag) {
    case UnitTag::time:          return identity ? value : value * omega_p_ref;
    case UnitTag::length:        return identity ? value : value / length_scale();
    case UnitTag::velocity:      return identity ? value : value / constants::c0;
    case UnitTag::e_field:       return identity ? value : value / e_field_scale();
    case UnitTag::b_field:       return identity ? value : value / b_field_scale();
    case UnitTag::density:       return identity ? value : value / n_ref;
    case UnitTag::charge:        return identity ? value : value / q_ref;
    case UnitTag::mass:          return identity ? value : value / m_ref;
    case UnitTag::temperature_eV:
      return identity ? value : value / temperature_eV_scale();
  }
  throw std::invalid_argument{"Normalization::to_internal: unknown unit tag"};
}

Real Normalization::to_si(Real value, UnitTag tag) const {
  const bool identity = is_identity();
  switch (tag) {
    case UnitTag::time:          return identity ? value : value / omega_p_ref;
    case UnitTag::length:        return identity ? value : value * length_scale();
    case UnitTag::velocity:      return identity ? value : value * constants::c0;
    case UnitTag::e_field:       return identity ? value : value * e_field_scale();
    case UnitTag::b_field:       return identity ? value : value * b_field_scale();
    case UnitTag::density:       return identity ? value : value * n_ref;
    case UnitTag::charge:        return identity ? value : value * q_ref;
    case UnitTag::mass:          return identity ? value : value * m_ref;
    case UnitTag::temperature_eV:
      return identity ? value : value * temperature_eV_scale();
  }
  throw std::invalid_argument{"Normalization::to_si: unknown unit tag"};
}

}  // namespace quasar
