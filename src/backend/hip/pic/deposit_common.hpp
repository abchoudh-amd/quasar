// Device-inline deposit helpers shared by the Cartesian and cylindrical
// charge-conserving (Esirkepov) current-deposition kernels (deposit_hip.hip and
// deposit_cyl_hip.hip). These were verbatim-identical in both translation units;
// extracting them here keeps the per-axis node indexing, the local stencil-window
// sizing, the 1-D shape weight, and the sum-to-1 deposition-window overflow guard
// in one place. The cylindrical r-weighting / V_cell current normalisation stays
// in deposit_cyl_hip.hip; each deposit kernel forms its node index inline from
// deposit_axis_x/y (colx[a] + rowy[b]).
//
// Header-only `__device__ inline` (no ODR/link issue). Included only by the .hip
// definitions under src/backend/hip/pic/; never include from a non-HIP TU.
#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"

#include <hip/hip_runtime.h>

#include <climits>

namespace quasar::backend::pic {

__device__ inline void mark_deposit_error(unsigned int* error) {
  atomicExch(error, 1u);
}

__device__ inline bool reduced_coordinate_representable(double value) {
  constexpr double margin = 16.0;
  return isfinite(value)
      && value >= static_cast<double>(INT_MIN) + margin
      && value <= static_cast<double>(INT_MAX) - margin;
}

// Exponent-scaled product/ratio accumulator. Kernel formulas such as
// q*w/(dx*dy) are often finite even when q*w, 1/dx, or dx*dy is not. Keeping a
// normalized mantissa plus a base-two exponent avoids those false range errors.
struct ScaledProduct {
  double mantissa{0.5};
  int exponent{1};
  bool zero{false};
};

__device__ inline void scaled_multiply(ScaledProduct& value, double factor) {
  if (value.zero) return;
  if (factor == 0.0) {
    value.zero = true;
    return;
  }
  int factor_exponent = 0;
  const double factor_mantissa = frexp(factor, &factor_exponent);
  value.mantissa *= factor_mantissa;
  value.exponent += factor_exponent;
  int adjustment = 0;
  value.mantissa = frexp(value.mantissa, &adjustment);
  value.exponent += adjustment;
}

__device__ inline void scaled_divide(ScaledProduct& value, double factor) {
  int factor_exponent = 0;
  const double factor_mantissa = frexp(factor, &factor_exponent);
  value.mantissa /= factor_mantissa;
  value.exponent -= factor_exponent;
  int adjustment = 0;
  value.mantissa = frexp(value.mantissa, &adjustment);
  value.exponent += adjustment;
}

__device__ inline double scaled_value(const ScaledProduct& value) {
  return value.zero ? 0.0 : scalbn(value.mantissa, value.exponent);
}

// A nonzero exact product that converts to zero has truly underflowed; silently
// treating it as an exact zero would drop charge/current and break continuity.
// Call this after materialising the complete node contribution (not a partial
// prefactor that a later small shape delta could bring back into range).
__device__ inline bool scaled_value_representable(const ScaledProduct& scaled,
                                                   double value) {
  return isfinite(value) && (value != 0.0 || scaled.zero);
}

// Local stencil window sized per shape order. CFL keeps |displacement| < ~1 cell,
// so the window must span the lowest support node of the min endpoint to the
// highest support node of the max endpoint, plus a one-node margin each side.
// CIC (support 2) needs a 5-node window with base floor(min)-1; TSC (support 3)
// needs a 6-node window with base floor(min)-2. Both follow from:
//   window = ShapeOrder + 4,  base margin = ShapeOrder.
// (The TSC values match the historical fixed 6 / -2, so that path is unchanged.)
template <int ShapeOrder>
inline constexpr int kWindowFor = ShapeOrder + 4;

// Per-axis deposit indexing. On a periodic axis (both sides periodic) the node is
// wrapped exactly as before (bit-identical to the historical behaviour). On a
// non-periodic (wall) axis the node is NOT wrapped: out-of-domain shape tails land
// in the ghost layer via Grid2D::index, and are later reflected into the interior
// by the specular fold-back kernel. The clamp keeps a rare deep overshoot inside
// allocated ghost storage instead of wrapping it to the far edge.
__device__ inline int deposit_clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline int deposit_axis_x(const quasar::Grid2D& g, int i, bool periodic_x) {
  return periodic_x ? g.wrap_i(i)
                    : deposit_clamp_axis(i, -g.nghost, g.nx - 1 + g.nghost);
}

__device__ inline int deposit_axis_y(const quasar::Grid2D& g, int j, bool periodic_y) {
  return periodic_y ? g.wrap_j(j)
                    : deposit_clamp_axis(j, -g.nghost, g.ny - 1 + g.nghost);
}

template <int ShapeOrder>
__device__ inline double weight_1d(double r) {
  if constexpr (ShapeOrder == 1) {
    const double a = r < 0.0 ? -r : r;
    return a < 1.0 ? 1.0 - a : 0.0;
  } else {
    return quasar::numerics::tsc_weight_1d(r);
  }
}

// Sum-to-1 deposition-window guard for one shape strip. kWindow is sized for
// sub-CFL particle motion; if an endpoint shape spills outside the window the
// running sum departs from 1 and continuing would silently truncate the current
// path and violate discrete continuity. Returns true iff the strip sums to 1
// within tolerance. The sequential accumulation order matches the original inline
// loops, so the result is bit-identical.
__device__ inline bool shape_strip_sums_to_one(const double* S, int n) {
  constexpr double kWeightTol = 1.0e-10;
  double sum = 0.0;
  for (int a = 0; a < n; ++a) sum += S[a];
  return fabs(sum - 1.0) <= kWeightTol;
}

}  // namespace quasar::backend::pic
