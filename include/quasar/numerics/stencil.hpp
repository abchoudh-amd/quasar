#pragma once

#include "quasar/core/grid.hpp"

namespace quasar::numerics {

namespace detail {

// A finite Real represented without committing its exponent back to Real's
// range.  Curl updates use this form until every signed term (old state,
// directional derivatives, and current) has been combined.  That permits a
// representable final value even when one mathematical term, or an intermediate
// partial sum, lies outside the normal/subnormal range of Real.
struct ScaledValue {
  Real mantissa{0};
  int exponent{0};
};

QUASAR_HOST_DEVICE inline ScaledValue scaled_value(Real value) noexcept {
  if (value == Real{0}) return {};
  int exponent = 0;
  return {frexp(value, &exponent), exponent};
}

QUASAR_HOST_DEVICE inline ScaledValue negate(ScaledValue value) noexcept {
  value.mantissa = -value.mantissa;
  return value;
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_product2_quotient_value(
    Real scale, Real normalized, Real multiplier, Real spacing) noexcept {
  if (scale == Real{0} || normalized == Real{0}
      || multiplier == Real{0}) return {};
  int es = 0, en = 0, em = 0, eh = 0;
  const Real ms = frexp(scale, &es);
  const Real mn = frexp(normalized, &en);
  const Real mm = frexp(multiplier, &em);
  const Real mh = frexp(spacing, &eh);
  Real mantissa = ((ms * mn) * mm) / mh;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return {mantissa, es + en + em - eh + adjustment};
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_product_value(
    Real a, Real b) noexcept {
  return scaled_product2_quotient_value(a, b, Real{1}, Real{1});
}

// Multiply an already exponent-scaled value by multiplier/spacing without
// first materialising the value in Real.  This is the bridge between a
// cancellation-safe stencil numerator and its derivative/increment: either
// the numerator or the bare derivative may lie outside Real's range even when
// the complete timestep-weighted update is representable.
QUASAR_HOST_DEVICE inline ScaledValue scaled_value_product_quotient(
    ScaledValue value, Real multiplier, Real spacing) noexcept {
  if (value.mantissa == Real{0} || multiplier == Real{0}) return {};
  int em = 0, eh = 0;
  const Real mm = frexp(multiplier, &em);
  const Real mh = frexp(spacing, &eh);
  Real mantissa = (value.mantissa * mm) / mh;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return {mantissa, value.exponent + em - eh + adjustment};
}

QUASAR_HOST_DEVICE inline ScaledValue add_scaled_pair(
    ScaledValue a, ScaledValue b) noexcept {
  if (a.mantissa == Real{0}) return b;
  if (b.mantissa == Real{0}) return a;
  if (b.exponent > a.exponent) {
    const ScaledValue tmp = a;
    a = b;
    b = tmp;
  }
  const Real aligned_b = scalbn(b.mantissa, b.exponent - a.exponent);
  const Real sum = a.mantissa + aligned_b;
  if (sum == Real{0}) return {};
  int adjustment = 0;
  const Real mantissa = frexp(sum, &adjustment);
  return {mantissa, a.exponent + adjustment};
}

QUASAR_HOST_DEVICE inline bool scaled_abs_less(
    ScaledValue a, ScaledValue b) noexcept {
  if (a.exponent != b.exponent) return a.exponent < b.exponent;
  return fabs(a.mantissa) < fabs(b.mantissa);
}

// Reduce a handful of signed exponent-scaled terms without ever materialising
// an out-of-range partial sum.  Opposite-sign terms with the closest exponents
// are paired first.  Besides preventing overflow, that order preserves a small
// old-state term in cases such as A - A + old where a common-exponent one-pass
// sum could underflow `old` before the large terms cancel.
template <int N>
QUASAR_HOST_DEVICE inline ScaledValue scaled_sum_value(
    ScaledValue (&input)[N]) noexcept {
  ScaledValue terms[N];
  int count = 0;
  for (int i = 0; i < N; ++i) {
    if (input[i].mantissa != Real{0}) terms[count++] = input[i];
  }
  while (count > 1) {
    int first = -1;
    int second = -1;
    int best_gap = 1 << 20;
    int best_high_exponent = -(1 << 20);
    for (int i = 0; i < count; ++i) {
      for (int j = i + 1; j < count; ++j) {
        const bool opposite = (terms[i].mantissa < Real{0})
                           != (terms[j].mantissa < Real{0});
        if (!opposite) continue;
        const int gap = terms[i].exponent > terms[j].exponent
                      ? terms[i].exponent - terms[j].exponent
                      : terms[j].exponent - terms[i].exponent;
        const int high = terms[i].exponent > terms[j].exponent
                       ? terms[i].exponent : terms[j].exponent;
        if (gap < best_gap || (gap == best_gap && high > best_high_exponent)) {
          first = i;
          second = j;
          best_gap = gap;
          best_high_exponent = high;
        }
      }
    }
    if (first < 0) {
      // No cancellation remains.  Add the two smallest magnitudes first so the
      // rounding error of an all-same-sign reduction is not needlessly enlarged.
      first = 0;
      second = 1;
      if (scaled_abs_less(terms[second], terms[first])) {
        const int tmp = first;
        first = second;
        second = tmp;
      }
      for (int i = 2; i < count; ++i) {
        if (scaled_abs_less(terms[i], terms[first])) {
          second = first;
          first = i;
        } else if (scaled_abs_less(terms[i], terms[second])) {
          second = i;
        }
      }
    }
    terms[first] = add_scaled_pair(terms[first], terms[second]);
    terms[second] = terms[count - 1];
    --count;
  }
  return count == 0 ? ScaledValue{} : terms[0];
}

// Prefer the direct subtraction whenever it stays finite.  For nearby
// same-sign values Sterbenz's lemma then preserves the local difference
// exactly; dividing both samples by an arbitrary (non-power-of-two) scale
// first can lose that difference.  Opposite-sign extremes take the scaled
// path so an out-of-range raw subtraction can still participate in a later
// cancellation or division.
QUASAR_HOST_DEVICE inline ScaledValue scaled_difference_value(
    Real high, Real low) noexcept {
  const Real direct = high - low;
  if (std::isfinite(direct)) return scaled_value(direct);
  ScaledValue terms[] = {scaled_value(high), negate(scaled_value(low))};
  return scaled_sum_value(terms);
}

template <class... Rest>
QUASAR_HOST_DEVICE inline Real scaled_signed_sum(
    ScaledValue first, Rest... rest) noexcept {
  ScaledValue terms[] = {first, rest...};
  const ScaledValue value = scaled_sum_value(terms);
  return scalbn(value.mantissa, value.exponent);
}

// Evaluate scale*normalized*multiplier/spacing without materialising a
// reciprocal spacing or any large intermediate product. This is the core of
// the timestep-weighted curl helpers below: even when 1/h overflows or the bare
// derivative underflows, dt*D(f) can remain finite and representable.
QUASAR_HOST_DEVICE inline Real scaled_product2_quotient(
    Real scale, Real normalized, Real multiplier, Real spacing) noexcept {
  const ScaledValue value = scaled_product2_quotient_value(
      scale, normalized, multiplier, spacing);
  return scalbn(value.mantissa, value.exponent);
}

QUASAR_HOST_DEVICE inline Real scaled_product_quotient(
    Real scale, Real normalized, Real spacing) noexcept {
  return scaled_product2_quotient(
      scale, normalized, Real{1}, spacing);
}

template <int Order>
QUASAR_HOST_DEVICE inline ScaledValue scaled_staggered_value(
    Real fm1, Real f0, Real f1, Real f2) noexcept {
  const ScaledValue inner = scaled_difference_value(f1, f0);
  if constexpr (Order == 4) {
    const ScaledValue outer = scaled_difference_value(f2, fm1);
    ScaledValue terms[] = {
        scaled_value_product_quotient(
            inner, Real{9} / Real{8}, Real{1}),
        scaled_value_product_quotient(
            outer, -Real{1} / Real{24}, Real{1})};
    return scaled_sum_value(terms);
  }
  return inner;
}

template <int Order>
QUASAR_HOST_DEVICE inline ScaledValue scaled_cylindrical_metric_value(
    Real fm1, Real f0, Real f1, Real f2) noexcept {
  if constexpr (Order == 4) {
    ScaledValue terms[] = {
        scaled_product_value(Real{9} / Real{16}, f1),
        scaled_product_value(Real{9} / Real{16}, f0),
        scaled_product_value(-Real{1} / Real{16}, f2),
        scaled_product_value(-Real{1} / Real{16}, fm1)};
    return scaled_sum_value(terms);
  }
  ScaledValue terms[] = {
      scaled_product_value(Real{0.5}, f1),
      scaled_product_value(Real{0.5}, f0)};
  return scaled_sum_value(terms);
}

template <int Order>
QUASAR_HOST_DEVICE inline ScaledValue cylindrical_radial_flux_scaled_impl(
    Real fm1, Real f0, Real f1, Real f2, Real r_c, Real dr,
    Real multiplier) noexcept {
  const ScaledValue derivative = scaled_staggered_value<Order>(
      fm1, f0, f1, f2);
  const ScaledValue metric = scaled_cylindrical_metric_value<Order>(
      fm1, f0, f1, f2);
  ScaledValue terms[] = {
      scaled_value_product_quotient(derivative, multiplier, dr),
      scaled_value_product_quotient(metric, multiplier, r_c)};
  return scaled_sum_value(terms);
}

QUASAR_HOST_DEVICE inline ScaledValue scaled_onesided_second_value(
    Real f0, Real f1, Real f2) noexcept {
  // (-3 f0 + 4 f1 - f2)/2, expressed as differences so a large
  // translational offset cancels before the small local slope is rounded.
  const ScaledValue near_difference = scaled_difference_value(f1, f0);
  const ScaledValue far_difference = scaled_difference_value(f2, f0);
  ScaledValue terms[] = {
      scaled_value_product_quotient(
          near_difference, Real{2}, Real{1}),
      scaled_value_product_quotient(
          far_difference, -Real{0.5}, Real{1})};
  return scaled_sum_value(terms);
}

}  // namespace detail

template <int Order>
QUASAR_HOST_DEVICE inline Real staggered_derivative_values(
    Real fm1, Real f0, Real f1, Real f2, Real spacing) noexcept {
  const auto numerator = detail::scaled_staggered_value<Order>(
      fm1, f0, f1, f2);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, spacing);
  return scalbn(value.mantissa, value.exponent);
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue staggered_derivative_scaled_values(
    Real fm1, Real f0, Real f1, Real f2, Real spacing) noexcept {
  const auto numerator = detail::scaled_staggered_value<Order>(
      fm1, f0, f1, f2);
  return detail::scaled_value_product_quotient(
      numerator, Real{1}, spacing);
}

// Staggered finite differences read neighbours through ghost cells via
// Grid2D::index. The per-side ghost fill (fill_field_ghosts in pic_solver.cpp::step)
// runs before every curl and populates the halo: periodic sides copy the opposite
// interior edge, PEC sides apply stagger/parity continuation, and outflow sides
// apply linear continuation. Order 4 reads i-1..i+2, so it needs nghost >= 2
// (enforced by the EmPic2D3V constructor). The Yee curl reads only along-axis
// neighbours, so corner ghosts are never touched by the separable stencil.
// The value form preserves finite local differences directly, then carries the
// stencil numerator and spacing in exponent-scaled form.  This avoids both a
// materialized reciprocal and an overflowing opposite-extreme difference.
template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered(const Real* f, const Grid2D& g,
                                             int i, int j) noexcept {
  if constexpr (Order == 4) {
    return staggered_derivative_values<Order>(
        f[g.index(i - 1, j)], f[g.index(i, j)], f[g.index(i + 1, j)],
        f[g.index(i + 2, j)], g.dx());
  }
  return staggered_derivative_values<Order>(
      Real{0}, f[g.index(i, j)], f[g.index(i + 1, j)], Real{0},
      g.dx());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered(const Real* f, const Grid2D& g,
                                             int i, int j) noexcept {
  if constexpr (Order == 4) {
    return staggered_derivative_values<Order>(
        f[g.index(i, j - 1)], f[g.index(i, j)], f[g.index(i, j + 1)],
        f[g.index(i, j + 2)], g.dy());
  }
  return staggered_derivative_values<Order>(
      Real{0}, f[g.index(i, j)], f[g.index(i, j + 1)], Real{0},
      g.dy());
}

// Backward-difference companions of ddx/ddy_staggered. The Yee scheme stays stable
// and energy-conserving (including at PEC walls) only when the two curls are
// adjoint: the E-update curl (curl_b) and Gauss/current divergence use the
// forward face-to-cell difference, while the B-update curl (curl_e) uses the
// backward difference below. With both curls forward the operator is
// non-adjoint and a hard wall drives an exponential instability.
template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered_bwd(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  if constexpr (Order == 4) {
    return staggered_derivative_values<Order>(
        f[g.index(i - 2, j)], f[g.index(i - 1, j)], f[g.index(i, j)],
        f[g.index(i + 1, j)], g.dx());
  }
  return staggered_derivative_values<Order>(
      Real{0}, f[g.index(i - 1, j)], f[g.index(i, j)], Real{0},
      g.dx());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered_bwd(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  if constexpr (Order == 4) {
    return staggered_derivative_values<Order>(
        f[g.index(i, j - 2)], f[g.index(i, j - 1)], f[g.index(i, j)],
        f[g.index(i, j + 1)], g.dy());
  }
  return staggered_derivative_values<Order>(
      Real{0}, f[g.index(i, j - 1)], f[g.index(i, j)], Real{0},
      g.dy());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real staggered_increment_values(
    Real fm1, Real f0, Real f1, Real f2, Real dt,
    Real spacing) noexcept {
  const auto numerator = detail::scaled_staggered_value<Order>(
      fm1, f0, f1, f2);
  const auto value = detail::scaled_value_product_quotient(
      numerator, dt, spacing);
  return scalbn(value.mantissa, value.exponent);
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue staggered_increment_scaled_values(
    Real fm1, Real f0, Real f1, Real f2, Real dt,
    Real spacing) noexcept {
  const auto numerator = detail::scaled_staggered_value<Order>(
      fm1, f0, f1, f2);
  return detail::scaled_value_product_quotient(
      numerator, dt, spacing);
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue ddx_staggered_increment_scaled(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_scaled_values<Order>(
        f[g.index(i - 1, j)], f[g.index(i, j)], f[g.index(i + 1, j)],
        f[g.index(i + 2, j)], dt, g.dx());
  }
  return staggered_increment_scaled_values<Order>(
      Real{0}, f[g.index(i, j)], f[g.index(i + 1, j)], Real{0}, dt,
      g.dx());
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue ddy_staggered_increment_scaled(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_scaled_values<Order>(
        f[g.index(i, j - 1)], f[g.index(i, j)], f[g.index(i, j + 1)],
        f[g.index(i, j + 2)], dt, g.dy());
  }
  return staggered_increment_scaled_values<Order>(
      Real{0}, f[g.index(i, j)], f[g.index(i, j + 1)], Real{0}, dt,
      g.dy());
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue ddx_staggered_bwd_increment_scaled(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_scaled_values<Order>(
        f[g.index(i - 2, j)], f[g.index(i - 1, j)], f[g.index(i, j)],
        f[g.index(i + 1, j)], dt, g.dx());
  }
  return staggered_increment_scaled_values<Order>(
      Real{0}, f[g.index(i - 1, j)], f[g.index(i, j)], Real{0}, dt,
      g.dx());
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue ddy_staggered_bwd_increment_scaled(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_scaled_values<Order>(
        f[g.index(i, j - 2)], f[g.index(i, j - 1)], f[g.index(i, j)],
        f[g.index(i, j + 1)], dt, g.dy());
  }
  return staggered_increment_scaled_values<Order>(
      Real{0}, f[g.index(i, j - 1)], f[g.index(i, j)], Real{0}, dt,
      g.dy());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered_increment(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_values<Order>(
        f[g.index(i - 1, j)], f[g.index(i, j)], f[g.index(i + 1, j)],
        f[g.index(i + 2, j)], dt, g.dx());
  }
  return staggered_increment_values<Order>(
      Real{0}, f[g.index(i, j)], f[g.index(i + 1, j)], Real{0}, dt,
      g.dx());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered_increment(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_values<Order>(
        f[g.index(i, j - 1)], f[g.index(i, j)], f[g.index(i, j + 1)],
        f[g.index(i, j + 2)], dt, g.dy());
  }
  return staggered_increment_values<Order>(
      Real{0}, f[g.index(i, j)], f[g.index(i, j + 1)], Real{0}, dt,
      g.dy());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered_bwd_increment(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_values<Order>(
        f[g.index(i - 2, j)], f[g.index(i - 1, j)], f[g.index(i, j)],
        f[g.index(i + 1, j)], dt, g.dx());
  }
  return staggered_increment_values<Order>(
      Real{0}, f[g.index(i - 1, j)], f[g.index(i, j)], Real{0}, dt,
      g.dx());
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered_bwd_increment(
    const Real* f, const Grid2D& g, int i, int j, Real dt) noexcept {
  if constexpr (Order == 4) {
    return staggered_increment_values<Order>(
        f[g.index(i, j - 2)], f[g.index(i, j - 1)], f[g.index(i, j)],
        f[g.index(i, j + 1)], dt, g.dy());
  }
  return staggered_increment_values<Order>(
      Real{0}, f[g.index(i, j - 1)], f[g.index(i, j)], Real{0}, dt,
      g.dy());
}

// Cylindrical face-to-cell divergence from the four face samples needed by the
// fourth-order stencil (order two ignores fm1/f2). Algebraically this is
//
//   D(r f)/(r dr) = D(f)/dr + M(f)/r,
//
// with D the ordinary staggered derivative and M its matching centred metric
// average. Expanding about r_c removes every r*f product.  The derivative and
// metric average are reduced as signed exponent-scaled terms and are combined
// before either is restored to Real, avoiding overflow near DBL_MAX while
// preserving local differences and cancellations at the axis.
template <int Order>
QUASAR_HOST_DEVICE inline Real cylindrical_radial_flux_values(
    Real fm1, Real f0, Real f1, Real f2, Real r_c, Real dr) noexcept {
  const auto value = detail::cylindrical_radial_flux_scaled_impl<Order>(
      fm1, f0, f1, f2, r_c, dr, Real{1});
  return scalbn(value.mantissa, value.exponent);
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue
cylindrical_radial_flux_scaled_values(
    Real fm1, Real f0, Real f1, Real f2, Real r_c, Real dr) noexcept {
  return detail::cylindrical_radial_flux_scaled_impl<Order>(
      fm1, f0, f1, f2, r_c, dr, Real{1});
}

template <int Order>
QUASAR_HOST_DEVICE inline Real cylindrical_radial_flux_increment_values(
    Real fm1, Real f0, Real f1, Real f2, Real r_c, Real dr,
    Real dt) noexcept {
  const auto value = detail::cylindrical_radial_flux_scaled_impl<Order>(
      fm1, f0, f1, f2, r_c, dr, dt);
  return scalbn(value.mantissa, value.exponent);
}

template <int Order>
QUASAR_HOST_DEVICE inline detail::ScaledValue
cylindrical_radial_flux_increment_scaled_values(
    Real fm1, Real f0, Real f1, Real f2, Real r_c, Real dr,
    Real dt) noexcept {
  return detail::cylindrical_radial_flux_scaled_impl<Order>(
      fm1, f0, f1, f2, r_c, dr, dt);
}

// One-sided first-derivative closures for a boundary node whose centered/staggered
// stencil would otherwise reach outside the domain. These read ONLY interior
// nodes (no ghost), so a non-periodic boundary can be closed without a ghost
// fill. `h` is the axis spacing (g.dx()/g.dy()).
//
// First order (two-node) variants — robust, used for the order-2 scheme and as
// the (order-reduced) closure for the outer two layers of the order-4 scheme:
QUASAR_HOST_DEVICE inline Real ddx_onesided_fwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_difference_value(
      f[g.index(i + 1, j)], f[g.index(i, j)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dx());
  return scalbn(value.mantissa, value.exponent);
}
QUASAR_HOST_DEVICE inline Real ddx_onesided_bwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_difference_value(
      f[g.index(i, j)], f[g.index(i - 1, j)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dx());
  return scalbn(value.mantissa, value.exponent);
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_fwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_difference_value(
      f[g.index(i, j + 1)], f[g.index(i, j)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dy());
  return scalbn(value.mantissa, value.exponent);
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_bwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_difference_value(
      f[g.index(i, j)], f[g.index(i, j - 1)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dy());
  return scalbn(value.mantissa, value.exponent);
}

// Second order (three-node) variants — preserve the interior order at the
// boundary node where stability allows:
//   forward : (-3 f0 + 4 f1 - f2) / (2h)
//   backward: ( 3 f0 - 4 f-1 + f-2) / (2h)
QUASAR_HOST_DEVICE inline Real ddx_onesided_fwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_onesided_second_value(
      f[g.index(i, j)], f[g.index(i + 1, j)], f[g.index(i + 2, j)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dx());
  return scalbn(value.mantissa, value.exponent);
}
QUASAR_HOST_DEVICE inline Real ddx_onesided_bwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_onesided_second_value(
      f[g.index(i, j)], f[g.index(i - 1, j)], f[g.index(i - 2, j)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dx());
  return -scalbn(value.mantissa, value.exponent);
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_fwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_onesided_second_value(
      f[g.index(i, j)], f[g.index(i, j + 1)], f[g.index(i, j + 2)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dy());
  return scalbn(value.mantissa, value.exponent);
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_bwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  const auto numerator = detail::scaled_onesided_second_value(
      f[g.index(i, j)], f[g.index(i, j - 1)], f[g.index(i, j - 2)]);
  const auto value = detail::scaled_value_product_quotient(
      numerator, Real{1}, g.dy());
  return -scalbn(value.mantissa, value.exponent);
}

template <int Order>
QUASAR_HOST_DEVICE inline Vec3 curl_e_at(const Real* ex, const Real* ey, const Real* ez,
                                         const Grid2D& g, int i, int j) noexcept {
  const Real d_ez_dy = ddy_staggered_bwd<Order>(ez, g, i, j);
  const Real d_ez_dx = ddx_staggered_bwd<Order>(ez, g, i, j);
  const Real d_ey_dx = ddx_staggered_bwd<Order>(ey, g, i, j);
  const Real d_ex_dy = ddy_staggered_bwd<Order>(ex, g, i, j);
  return Vec3{d_ez_dy, -d_ez_dx, d_ey_dx - d_ex_dy};
}

template <int Order>
QUASAR_HOST_DEVICE inline Vec3 curl_b_at(const Real* bx, const Real* by, const Real* bz,
                                         const Grid2D& g, int i, int j) noexcept {
  const Real d_bz_dy = ddy_staggered<Order>(bz, g, i, j);
  const Real d_bz_dx = ddx_staggered<Order>(bz, g, i, j);
  const Real d_by_dx = ddx_staggered<Order>(by, g, i, j);
  const Real d_bx_dy = ddy_staggered<Order>(bx, g, i, j);
  return Vec3{d_bz_dy, -d_bz_dx, d_by_dx - d_bx_dy};
}

}  // namespace quasar::numerics
