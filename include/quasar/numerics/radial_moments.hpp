#pragma once

// Host-side construction of finite-volume interpolation rows for cylindrical
// equation-native measures (dr, |r| dr, or r^2 dr). Coordinates are
// dimensionless: radii and offsets are expressed in units of the uniform
// radial cell width.

#include "quasar/core/types.hpp"

namespace quasar::numerics {

inline constexpr int kMaxRadialStencilWidth = 8;

// Select which side of the interpolation map is represented by cell averages.
enum class RadialMomentTarget {
  point_value,
  cell_average,
};

// Cell-average measure associated with a conserved cylindrical component.
// Most state variables use the annular |r| dr measure.  Azimuthal momentum
// evolves angular momentum under r^2 dr, while toroidal induction is the
// metric-free point equation and therefore uses the uniform dr measure.
enum class RadialCellMeasure {
  uniform,
  annular,
  angular_momentum,
};

struct RadialStencilRow {
  int width{};
  Real c[kMaxRadialStencilWidth]{};
  Real residual{};
};

// Return the normalized moment of one unit-width cell centered at rho:
//
//   integral_[rho-1/2,rho+1/2] x^m |x| dx
//   ------------------------------------------------- .
//          integral_[rho-1/2,rho+1/2] |x| dx
//
// The signed monomial and |x| measure make the same function valid for
// physical cells, reflected axis ghosts, and the (occasionally useful) cell
// centered exactly on the axis.
long double normalized_cell_moment(long double rho, int m);

// Measure-selectable counterpart used to build component-specific
// reconstruction rows.  The returned moment is normalized by the selected
// cell measure, so the constant mode is exactly one for every family.
long double normalized_cell_moment(
    long double rho, int m, RadialCellMeasure measure);

// Construct a polynomial-exact row of the requested width.
//
// point_value:
//   input k is the ring average in the cell centered at
//   rho_anchor + offset + k, and the target point is
//   rho_anchor + node_xi.
//
// cell_average:
//   input k is a point sample at
//   rho_anchor + offset + k + node_xi, and the target is the ring average in
//   the unit-width cell centered at rho_anchor.  For example, the centered
//   four-face-to-cell row is (width=4, offset=-2, node_xi=0.5).
//
// The solve is performed in long double with one iterative-refinement step.
// Returned coefficients are binary64, sum to exactly 1.0 under an in-order
// binary64 reduction, and residual is the largest defect for monomials
// m=0..width-1 in units of dr^m.
RadialStencilRow solve_radial_row(
    long double rho_anchor, int width, int offset,
    RadialMomentTarget target, long double node_xi = 0.0L);

// Construct a row for cell averages stored under an explicit cylindrical
// measure.  Point targets remain ordinary point values; cell-average targets
// use the same selected measure as their inputs.
RadialStencilRow solve_radial_row(
    long double rho_anchor, int width, int offset,
    RadialMomentTarget target, RadialCellMeasure measure,
    long double node_xi = 0.0L);

// Fold the cylindrical radius into an existing Gauss--Legendre rule on the
// unit cell.  The input Cartesian weights are expected to integrate the unit
// interval (sum to one).  `residual` measures local-monomial exactness through
// degree 2*count-2, the maximum degree retained after multiplying the
// integrand by the linear radial measure.
RadialStencilRow radial_gauss_weights(
    long double rho_anchor, int count, const Real* node_xi,
    const Real* cartesian_weights);

RadialStencilRow radial_gauss_weights(
    long double rho_anchor, int count, const Real* node_xi,
    const Real* cartesian_weights, RadialCellMeasure measure);

}  // namespace quasar::numerics
