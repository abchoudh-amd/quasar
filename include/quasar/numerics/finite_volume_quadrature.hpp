#pragma once

// Order-matched transverse quadrature for multidimensional finite-volume fluxes.
// The point-recovery rows map five/seven neighbouring face averages (offsets
// [-2,2] / [-3,3]) to the Gauss--Legendre nodes of the target face segment.

#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"

namespace quasar::numerics {

inline constexpr Real kMp5TransverseNodes[3] = {
    Real{-0.387298334620741702}, Real{0}, Real{0.387298334620741702}};
inline constexpr Real kMp5TransverseGaussWeights[3] = {
    Real{5} / Real{18}, Real{4} / Real{9}, Real{5} / Real{18}};
inline constexpr Real kMp5TransversePointWeights[3][5] = {
    {-0.0392523473402346598,  0.312987195324173573,
      0.910833333333333384, -0.216320528657506839,
      0.0317523473402346601},
    { 0.00468749999999999983, -0.0604166666666666671,
      1.11145833333333344,    -0.0604166666666666671,
      0.00468749999999999983},
    { 0.0317523473402346601, -0.216320528657506839,
      0.910833333333333384,  0.312987195324173573,
     -0.0392523473402346598}};
inline constexpr Real kMp7TransverseNodes[4] = {
    Real{-0.430568155797026286}, Real{-0.169990521792428156},
    Real{0.169990521792428156},  Real{0.430568155797026286}};
inline constexpr Real kMp7TransverseGaussWeights[4] = {
    Real{0.173927422568726925}, Real{0.326072577431273103},
    Real{0.326072577431273103}, Real{0.173927422568726925}};
inline constexpr Real kMp7TransversePointWeights[4][7] = {
    { 0.00860109764472105458, -0.0796691475531970394,
      0.415576982717296983,    0.849155569794037945,
     -0.245805403605616202,    0.0591434746546156687,
     -0.0070025736518584264},
    { 0.00327670662777140203, -0.0266359260617751309,
      0.0928282644451935562,   1.08046056237311472,
     -0.183384586025290314,    0.0375843394851378479,
     -0.00412936084415200903},
    {-0.00412936084415200903,  0.0375843394851378479,
     -0.183384586025290314,    1.08046056237311472,
      0.0928282644451935562,  -0.0266359260617751309,
      0.00327670662777140203},
    {-0.0070025736518584264,   0.0591434746546156687,
     -0.245805403605616202,    0.849155569794037945,
      0.415576982717296983,   -0.0796691475531970394,
      0.00860109764472105458}};

// Range-safe weighted reduction shared by face, cell, and cylindrical
// quadrature.  The exact constant shortcut is part of the numerical contract:
// a truly one-dimensional state must survive a transverse recovery bit for bit
// even when the stored floating-point weights do not sum to exactly one in the
// evaluation order.
template <int Count, class Sample>
QUASAR_HOST_DEVICE inline Real weighted_quadrature_value(
    const Real (&weights)[Count], const Sample& sample) {
  const Real reference = sample(Count / 2);
  bool constant = true;
  ScaledProductQuotientAccumulator<Count> sum;
  for (int q = 0; q < Count; ++q) {
    const Real value = sample(q);
    constant = constant && value == reference;
    append_scaled_product_quotient(
        sum, weights[q], value, Real{1}, Real{1});
  }
  return constant ? reference : finish_scaled_product_quotient_sum(sum);
}

// Runtime-width counterpart for radius-dependent rows.  All current radial
// reconstruction/collocation rows contain at most eight coefficients.
template <class Sample>
QUASAR_HOST_DEVICE inline Real weighted_quadrature_value(
    const Real* weights, int count, const Sample& sample) {
  const Real reference = sample(count / 2);
  bool constant = true;
  ScaledProductQuotientAccumulator<8> sum;
  for (int q = 0; q < count; ++q) {
    const Real value = sample(q);
    constant = constant && value == reference;
    append_scaled_product_quotient(
        sum, weights[q], value, Real{1}, Real{1});
  }
  return constant ? reference : finish_scaled_product_quotient_sum(sum);
}

// Return <a*b>-a_mean*b_mean without first forming any physical-exponent
// product.  The explicit constant-factor branch is both an optimization and an
// invariant: for a genuinely one-dimensional state/background the transverse
// correction must be bit-zero even if rounded quadrature weights do not sum to
// exactly one.  The expanded reduction also avoids an overflowing `a-a_mean`
// when finite samples lie near opposite ends of the binary64 range.
template <int Count, class ASample, class BSample>
QUASAR_HOST_DEVICE inline ScaledValue transverse_product_correction_scaled(
    const Real (&weights)[Count], Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  const Real a_reference = a(Count / 2);
  const Real b_reference = b(Count / 2);
  bool a_constant = true;
  bool b_constant = true;
  ScaledQuaternaryAccumulator sum;
  for (int q = 0; q < Count; ++q) {
    const Real aq = a(q);
    const Real bq = b(q);
    a_constant = a_constant && aq == a_reference;
    b_constant = b_constant && bq == b_reference;
    append_scaled_quaternary_product(sum, weights[q], aq, bq, Real{1});
  }
  if (a_constant || b_constant) return {};
  append_scaled_quaternary_product(
      sum, Real{-1}, a_mean, b_mean, Real{1});
  return finish_scaled_quaternary_sum_to_value(sum);
}

// Runtime-count counterpart for radius-dependent Gauss rows stored in a
// RadialTablesView.  MP5/MP7 use at most four nodes.  The constant-factor
// invariant is identical to the compile-time Cartesian overload above.
template <class ASample, class BSample>
QUASAR_HOST_DEVICE inline ScaledValue transverse_product_correction_scaled(
    const Real* weights, int count, Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  const Real a_reference = a(count / 2);
  const Real b_reference = b(count / 2);
  bool a_constant = true;
  bool b_constant = true;
  ScaledQuaternaryAccumulator sum;
  for (int q = 0; q < count; ++q) {
    const Real aq = a(q);
    const Real bq = b(q);
    a_constant = a_constant && aq == a_reference;
    b_constant = b_constant && bq == b_reference;
    append_scaled_quaternary_product(sum, weights[q], aq, bq, Real{1});
  }
  if (a_constant || b_constant) return {};
  append_scaled_quaternary_product(
      sum, Real{-1}, a_mean, b_mean, Real{1});
  return finish_scaled_quaternary_sum_to_value(sum);
}

// Return <a*b>_target-a_native*b_native when the factors' supplied means use a
// different finite-volume measure from `weights`.  This is a product-moment
// difference, not an ordinary covariance: one constant factor does not imply
// zero when the other factor has distinct target- and native-measure means.
// Preserve bit-zero only when a factor is identically zero or both sampled
// constants exactly equal their supplied native means.
template <class ASample, class BSample>
QUASAR_HOST_DEVICE inline ScaledValue transverse_product_difference_scaled(
    const Real* weights, int count, Real a_native_mean, Real b_native_mean,
    const ASample& a, const BSample& b) {
  const Real a_reference = a(count / 2);
  const Real b_reference = b(count / 2);
  bool a_constant = true;
  bool b_constant = true;
  bool a_zero = a_native_mean == Real{0};
  bool b_zero = b_native_mean == Real{0};
  ScaledQuaternaryAccumulator sum;
  for (int q = 0; q < count; ++q) {
    const Real aq = a(q);
    const Real bq = b(q);
    a_constant = a_constant && aq == a_reference;
    b_constant = b_constant && bq == b_reference;
    a_zero = a_zero && aq == Real{0};
    b_zero = b_zero && bq == Real{0};
    append_scaled_quaternary_product(sum, weights[q], aq, bq, Real{1});
  }
  if (a_zero || b_zero ||
      (a_constant && b_constant && a_native_mean == a_reference &&
       b_native_mean == b_reference)) {
    return {};
  }
  append_scaled_quaternary_product(
      sum, Real{-1}, a_native_mean, b_native_mean, Real{1});
  return finish_scaled_quaternary_sum_to_value(sum);
}

template <int Count, class ASample, class BSample>
QUASAR_HOST_DEVICE inline Real transverse_product_correction(
    const Real (&weights)[Count], Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  const ScaledValue value = transverse_product_correction_scaled(
      weights, a_mean, b_mean, a, b);
  return scalbn(value.mantissa, value.exponent);
}

}  // namespace quasar::numerics
