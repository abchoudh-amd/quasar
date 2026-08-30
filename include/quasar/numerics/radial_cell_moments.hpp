#pragma once

// Closed-form finite-volume moments of a unit-width cylindrical cell, callable
// from host and device.
//
// These are the integrals behind every radial interpolation row: the m-th
// moment of a unit cell centred at `cell_rho`, taken about `origin`, under one
// of the three equation-native measures (dr, |r| dr, or r^2 dr) and normalized
// so the constant mode is exactly one. Coordinates are dimensionless -- radii
// and offsets are in units of the uniform radial cell width.
//
// It lives on the numerics axis, shared between the host scalar accessors in
// radial_moments.hpp and the assembly kernels in
// src/backend/hip/numerics/radial_moments_hip.hip, for the same reason
// mhd_background_metrics.hpp does: a backend-private header cannot be included
// from a public one, and duplicating the closed form in both places would be
// two implementations of the same integral.
//
// Precision note. The host implementation this replaces carried these integrals
// in `long double`. A device has no such type, so they are binary64 here. The
// expansion below is what makes that acceptable: both the monomial and the
// radial weight are expanded about the cell centre, so nothing ever forms the
// difference of two nearly equal global powers of a large radius. The
// downstream stencil solve remains the conditioning-sensitive step, and it is
// gated on a measured residual rather than on trust.

#include "quasar/core/types.hpp"

#include <cmath>
#include <limits>

namespace quasar::numerics {

// Cell-average measure associated with a conserved cylindrical component.
// Most state variables use the annular |r| dr measure.  Azimuthal momentum
// evolves angular momentum under r^2 dr, while toroidal induction is the
// metric-free point equation and therefore uses the uniform dr measure.
enum class RadialCellMeasure {
  uniform,
  annular,
  angular_momentum,
};

QUASAR_HOST_DEVICE inline int radial_measure_power(RadialCellMeasure measure) {
  switch (measure) {
    case RadialCellMeasure::uniform: return 0;
    case RadialCellMeasure::annular: return 1;
    case RadialCellMeasure::angular_momentum: return 2;
  }
  return -1;  // Unreachable for a valid enumerator; the host validates.
}

QUASAR_HOST_DEVICE inline Real radial_integer_power(Real x, int exponent) {
  Real result = Real{1};
  while (exponent > 0) {
    if ((exponent & 1) != 0) result *= x;
    x *= x;
    exponent >>= 1;
  }
  return result;
}

QUASAR_HOST_DEVICE inline Real radial_binomial(int n, int k) {
  if (k < 0 || k > n) return Real{0};
  if (k > n - k) k = n - k;
  Real value = Real{1};
  for (int j = 1; j <= k; ++j) {
    value *= static_cast<Real>(n - k + j);
    value /= static_cast<Real>(j);
  }
  return value;
}

QUASAR_HOST_DEVICE inline Real radial_power_integral(int exponent, Real lo,
                                                     Real hi) {
  const int antiderivative_exponent = exponent + 1;
  return (radial_integer_power(hi, antiderivative_exponent)
          - radial_integer_power(lo, antiderivative_exponent))
         / static_cast<Real>(antiderivative_exponent);
}

// Integrate (x-origin)^m w(x) over a unit-width cell centred at cell_rho, where
// w is 1, |x|, or x^2. Expanding both factors about the cell centre keeps the
// calculation well scaled when the global radius is large, unlike subtracting
// two nearly equal global powers of x.
QUASAR_HOST_DEVICE inline Real shifted_weighted_cell_integral(
    Real cell_rho, Real origin, int m, RadialCellMeasure measure) {
  const int radial_power = radial_measure_power(measure);
  const Real displacement = cell_rho - origin;

  // |x| changes sign inside the cell only when the axis falls within it, which
  // is why the segment integral takes an explicit sign and the caller may split
  // the cell at the axis.
  const auto integrate_segment = [&](Real lo, Real hi, Real radial_sign) {
    Real value = Real{0};
    for (int p = 0; p <= m; ++p) {
      const Real coefficient =
          radial_binomial(m, p) * radial_integer_power(displacement, m - p);
      Real weighted_power = Real{0};
      for (int q = 0; q <= radial_power; ++q) {
        weighted_power +=
            radial_binomial(radial_power, q) *
            radial_integer_power(cell_rho, radial_power - q) *
            radial_power_integral(p + q, lo, hi);
      }
      value += radial_sign * coefficient * weighted_power;
    }
    return value;
  };

  constexpr Real lo = Real{-0.5};
  constexpr Real hi = Real{0.5};
  if (measure != RadialCellMeasure::annular) {
    return integrate_segment(lo, hi, Real{1});
  }
  if (cell_rho >= Real{0.5}) return integrate_segment(lo, hi, Real{1});
  if (cell_rho <= Real{-0.5}) return integrate_segment(lo, hi, Real{-1});

  const Real axis = -cell_rho;
  return integrate_segment(lo, axis, Real{-1})
         + integrate_segment(axis, hi, Real{1});
}

// The normalized moment. Returns a non-finite value when the cell's weighted
// volume is not positive and finite; the caller decides whether that is an
// exception (host) or a status bit (device), because a kernel cannot throw.
QUASAR_HOST_DEVICE inline Real shifted_normalized_cell_moment(
    Real cell_rho, Real origin, int m, RadialCellMeasure measure) {
  const Real volume =
      shifted_weighted_cell_integral(cell_rho, origin, 0, measure);
  if (!(volume > Real{0}) || !std::isfinite(volume)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  return shifted_weighted_cell_integral(cell_rho, origin, m, measure) / volume;
}

}  // namespace quasar::numerics
