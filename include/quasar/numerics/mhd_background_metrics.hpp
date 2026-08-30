#pragma once

// Scaled-arithmetic metrics for validating a prescribed MHD background field.
//
// These decide whether a supplied B0 is discretely divergence-free, discretely
// curl-free, and compatible with the configured homogeneous boundary closure.
// Every quantity is carried as a (mantissa, exponent) ScaledValue rather than a
// plain Real, because the interesting failures live where a plain difference
// cannot: a small represented slope sitting on a field near the ends of
// binary64's exponent range, or a thin annulus at a large radius. Forming the
// difference or the quotient directly there overflows or underflows and turns a
// real defect into a NaN or a zero.
//
// The header is QUASAR_HOST_DEVICE throughout because the sweeps that use it run
// as device reductions over the padded background buffers
// (src/backend/hip/mhd/mhd_background_validate.hip), while the host keeps only
// the final scalar comparison and the throw. It lives on the numerics axis for
// the same reason mhd_state.hpp does: it is shared between the MHD physics
// module and the HIP backend, and a backend-private header cannot be included
// from src/physics/.
//
// The tolerances are part of the contract, not tuning knobs. kDiscreteSolenoidal
// Tolerance is a scale-free O(machine-epsilon) criterion; the curl-free
// tolerance is deliberately looser because the fixed-boundary vacuum projection
// stops on a global algebraic residual, and it is defence in depth against a
// proof that plainly disagrees with its own samples rather than a proof itself.

#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <cmath>
#include <limits>

namespace quasar::numerics {

inline constexpr Real kDiscreteSolenoidalTolerance =
    Real{1024} * std::numeric_limits<Real>::epsilon();
inline constexpr int kDiscreteSolenoidalRoundoffUlpShift = 10;  // 2^10 ulps
// The fixed-boundary vacuum projection stops on a global algebraic residual,
// so its pointwise staggered curl is a few parts in 1e9 for the finest supplied
// example. This 1e-8 check is only defense in depth against a proof that plainly
// disagrees with its samples; it is not itself a proof or a force-error bound.
// Authorization to omit B0's self-stress must come from a trusted analytic or
// projection construction.
inline constexpr Real kDiscreteCurlFreeTolerance = Real{1e-8};

QUASAR_HOST_DEVICE inline int max_int(int a, int b) { return a > b ? a : b; }

// Form one signed directional derivative only after cancelling the local
// field offset. Retaining the difference and quotient in scaled form avoids
// both overflow and underflow when a small represented slope sits on a field
// near the ends of binary64's exponent range.
QUASAR_HOST_DEVICE inline ScaledValue scaled_directional_derivative(
    Real upper, Real lower, Real spacing) {
  const ScaledValue difference =
      scaled_difference_to_value(upper, lower);
  ScaledProductQuotientAccumulator<1> derivative;
  append_scaled_value_quotient(
      derivative, difference, Real{1}, spacing, Real{1});
  return finish_scaled_product_quotient_sum_to_value(derivative);
}

// B_phi is stored as an unweighted cell average.  For the only regular
// curl-free toroidal profile on an annulus, B_phi=C/r, that average is
//
//   Bbar_i = C log(r_hi/r_lo) / dr.
//
// Recover the represented invariant C in scaled form so the validation stays
// range-safe.  The axis cell has an infinite C/r average; only its regular
// value Bbar=0 can therefore represent a curl-free toroidal field.
QUASAR_HOST_DEVICE inline ScaledValue scaled_toroidal_flux_from_uniform_average(
    Real average, Real lower_radius, Real spacing) {
  if (!(std::isfinite(average) && std::isfinite(lower_radius) &&
        std::isfinite(spacing) && lower_radius >= Real{0} &&
        spacing > Real{0})) {
    return ScaledValue{
        std::numeric_limits<Real>::infinity(), 0};
  }
  if (lower_radius == Real{0}) {
    return average == Real{0}
        ? ScaledValue{}
        : ScaledValue{
              std::numeric_limits<Real>::infinity(), 0};
  }

  const Real radius_ratio = spacing / lower_radius;
  const Real log_ratio = std::isfinite(radius_ratio)
      ? std::log1p(radius_ratio)
      : std::log(spacing) - std::log(lower_radius);
  if (!(std::isfinite(log_ratio) && log_ratio > Real{0})) {
    return ScaledValue{
        std::numeric_limits<Real>::infinity(), 0};
  }

  ScaledProductQuotientAccumulator<1> toroidal_flux;
  append_scaled_product_quotient(
      toroidal_flux, average, spacing, log_ratio, Real{1});
  return finish_scaled_product_quotient_sum_to_value(
      toroidal_flux);
}

// One representational ulp as a scaled value. This is not a raw-field scale:
// only the last place that can be rounded when a stored face is updated enters
// the divergence uncertainty. The subnormal bin has one fixed ulp.
QUASAR_HOST_DEVICE inline ScaledValue scaled_ulp(Real value) {
  if (!std::isfinite(value)) {
    return ScaledValue{
        std::numeric_limits<Real>::infinity(), 0};
  }
  const Real magnitude = std::abs(value);
  if (magnitude < std::numeric_limits<Real>::min()) {
    int exponent = 0;
    const Real mantissa = std::frexp(
        std::numeric_limits<Real>::denorm_min(), &exponent);
    return ScaledValue{mantissa, exponent};
  }
  int exponent = 0;
  (void)std::frexp(magnitude, &exponent);
  return ScaledValue{
      Real{0.5}, exponent - std::numeric_limits<Real>::digits + 1};
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_directional_roundoff(
    Real upper, Real lower, Real spacing) {
  ScaledProductQuotientAccumulator<2> uncertainty;
  append_scaled_value_quotient(
      uncertainty, scaled_ulp(upper), Real{1}, spacing, Real{1});
  append_scaled_value_quotient(
      uncertainty, scaled_ulp(lower), Real{1}, spacing, Real{1});
  return finish_scaled_product_quotient_sum_to_value(uncertainty);
}

// The annular radial divergence is
//   (B_hi-B_lo)/dr + (B_hi+B_lo)/(2*r_c).
// Its first term cancels a local field offset, while the second retains the
// physical B_r/r curvature of a constant radial field. Keep both terms scaled
// so a thin, large-radius annulus neither loses curvature in rounded (1+-q)
// coefficients nor overflows an intermediate face difference.
QUASAR_HOST_DEVICE inline ScaledValue scaled_annular_radial_divergence(
    Real upper, Real lower, Real spacing, Real radius) {
  ScaledProductQuotientAccumulator<4> radial;
  append_scaled_product_quotient(
      radial, upper, Real{1}, spacing, Real{1});
  append_scaled_product_quotient(
      radial, lower, Real{-1}, spacing, Real{1});
  append_scaled_product_quotient(
      radial, upper, Real{1}, Real{2}, radius);
  append_scaled_product_quotient(
      radial, lower, Real{1}, Real{2}, radius);
  return finish_scaled_product_quotient_sum_to_value(radial);
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_annular_radial_roundoff(
    Real upper, Real lower, Real spacing, Real radius) {
  Real q = Real{0.5} * (spacing / radius);
  if (q > Real{1}) q = Real{1};
  ScaledProductQuotientAccumulator<2> uncertainty;
  append_scaled_value_quotient(
      uncertainty, scaled_ulp(upper), Real{1} + q, spacing, Real{1});
  append_scaled_value_quotient(
      uncertainty, scaled_ulp(lower), Real{1} - q, spacing, Real{1});
  return finish_scaled_product_quotient_sum_to_value(uncertainty);
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_directional_sum(
    const ScaledValue& lhs,
    const ScaledValue& rhs, Real rhs_sign) {
  if (!(std::isfinite(lhs.mantissa) && std::isfinite(rhs.mantissa) &&
        std::isfinite(rhs_sign))) {
    return ScaledValue{
        std::numeric_limits<Real>::infinity(), 0};
  }
  ScaledProductQuotientAccumulator<2> residual_sum;
  append_scaled_value_quotient(
      residual_sum, lhs, Real{1}, Real{1}, Real{1});
  append_scaled_value_quotient(
      residual_sum, rhs, rhs_sign, Real{1}, Real{1});
  return finish_scaled_product_quotient_sum_to_value(residual_sum);
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_directional_magnitude_sum(
    const ScaledValue& lhs,
    const ScaledValue& rhs) {
  const ScaledValue lhs_abs{std::abs(lhs.mantissa), lhs.exponent};
  const ScaledValue rhs_abs{std::abs(rhs.mantissa), rhs.exponent};
  return scaled_directional_sum(lhs_abs, rhs_abs, Real{1});
}

QUASAR_HOST_DEVICE inline bool scaled_abs_less_equal_power_of_two(
    const ScaledValue& lhs,
    const ScaledValue& rhs, int rhs_exponent_shift) {
  if (!(std::isfinite(lhs.mantissa) && std::isfinite(rhs.mantissa))) {
    return false;
  }
  if (lhs.mantissa == Real{0}) return true;
  if (rhs.mantissa == Real{0}) return false;
  const int shifted_rhs_exponent = rhs.exponent + rhs_exponent_shift;
  return lhs.exponent < shifted_rhs_exponent ||
      (lhs.exponent == shifted_rhs_exponent &&
       std::abs(lhs.mantissa) <= std::abs(rhs.mantissa));
}

// Solver-owned CT/RK updates round each face independently. At a genuine
// cross-direction cancellation, admit a residual no larger than 1024 ulps of
// the metric-weighted face storage. A lone one-ulp slope has no opposing
// directional term and therefore cannot use this allowance.
QUASAR_HOST_DEVICE inline bool residual_is_roundoff_explained(
    const ScaledValue& lhs,
    const ScaledValue& rhs,
    const ScaledValue& residual,
    const ScaledValue& uncertainty) {
  if (lhs.mantissa == Real{0} || rhs.mantissa == Real{0} ||
      std::signbit(lhs.mantissa) == std::signbit(rhs.mantissa)) {
    return false;
  }
  const ScaledValue scale =
      scaled_directional_magnitude_sum(lhs, rhs);
  if (!scaled_abs_less_equal_power_of_two(residual, scale, -1)) {
    return false;
  }
  return scaled_abs_less_equal_power_of_two(
      residual, uncertainty, kDiscreteSolenoidalRoundoffUlpShift);
}

QUASAR_HOST_DEVICE inline Real normalized_scaled_ratio(
    const ScaledValue& numerator,
    const ScaledValue& denominator) {
  if (!(std::isfinite(numerator.mantissa) &&
        std::isfinite(denominator.mantissa))) {
    return std::numeric_limits<Real>::infinity();
  }
  if (numerator.mantissa == Real{0}) return Real{0};
  if (!(denominator.mantissa > Real{0})) {
    return std::numeric_limits<Real>::infinity();
  }

  const int common_exponent = std::max(
      numerator.exponent, denominator.exponent);
  const Real scaled_numerator = std::abs(std::scalbn(
      numerator.mantissa, numerator.exponent - common_exponent));
  const Real scaled_denominator = std::abs(std::scalbn(
      denominator.mantissa, denominator.exponent - common_exponent));
  if (!(scaled_denominator > Real{0}) ||
      !std::isfinite(scaled_denominator) ||
      !std::isfinite(scaled_numerator)) {
    return std::numeric_limits<Real>::infinity();
  }
  return scaled_numerator / scaled_denominator;
}

QUASAR_HOST_DEVICE inline void retain_scaled_abs_max(const ScaledValue& candidate,
                           ScaledValue& maximum) {
  if (!std::isfinite(candidate.mantissa)) {
    maximum = ScaledValue{
        std::numeric_limits<Real>::infinity(), 0};
    return;
  }
  if (!std::isfinite(maximum.mantissa) || candidate.mantissa == Real{0}) {
    return;
  }
  if (maximum.mantissa == Real{0} ||
      candidate.exponent > maximum.exponent ||
      (candidate.exponent == maximum.exponent &&
       std::abs(candidate.mantissa) > std::abs(maximum.mantissa))) {
    maximum = ScaledValue{
        std::abs(candidate.mantissa), candidate.exponent};
  }
}

// Normalize cancellation between two independently computed directional
// contributions by their magnitudes, not by the magnitudes of the underlying
// field samples. A Cartesian DC offset therefore cannot hide a real derivative.
QUASAR_HOST_DEVICE inline Real normalized_directional_sum_defect(
    const ScaledValue& lhs,
    const ScaledValue& rhs, Real rhs_sign) {
  return normalized_scaled_ratio(
      scaled_directional_sum(lhs, rhs, rhs_sign),
      scaled_directional_magnitude_sum(lhs, rhs));
}

QUASAR_HOST_DEVICE inline Real normalized_pair_defect(Real lhs, Real rhs, Real rhs_sign) {
  if (!(std::isfinite(lhs) && std::isfinite(rhs))) {
    return std::numeric_limits<Real>::infinity();
  }
  if (lhs == Real{0} && rhs == Real{0}) return Real{0};
  int lhs_exponent = 0;
  int rhs_exponent = 0;
  (void)std::frexp(lhs, &lhs_exponent);
  (void)std::frexp(rhs, &rhs_exponent);
  const int exponent = (lhs_exponent > rhs_exponent ? lhs_exponent : rhs_exponent);
  const Real scaled_lhs = std::scalbn(lhs, -exponent);
  const Real scaled_rhs = std::scalbn(rhs, -exponent);
  const Real denominator = std::abs(scaled_lhs) + std::abs(scaled_rhs);
  if (!(denominator > Real{0}) || !std::isfinite(denominator)) {
    return std::numeric_limits<Real>::infinity();
  }
  return std::abs(scaled_lhs - rhs_sign * scaled_rhs) / denominator;
}
}  // namespace quasar::numerics
