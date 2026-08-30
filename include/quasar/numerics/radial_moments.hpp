#pragma once

// Finite-volume interpolation rows for cylindrical equation-native measures
// (dr, |r| dr, or r^2 dr). Coordinates are dimensionless: radii and offsets are
// expressed in units of the uniform radial cell width.
//
// The rows are solved on the device, in batch. A radial table needs roughly
// twenty rows per radial index across the padded grid, each an independent
// general (non-symmetric, Vandermonde-like) system of width at most eight, so
// the batched entry points below are the primary interface and the
// single-row convenience wrappers are one-element batches. Building a table one
// row at a time would spend all its time on launch overhead.
//
// Precision. The host implementation this replaces carried the whole
// calculation in `long double` with one iterative-refinement step. A device has
// no `long double`, so this is binary64 -- but it keeps the refinement step,
// which is what makes the result backward stable: the reported residual tracks
// working precision times the matrix norm, not the condition number. Measured
// over the padded grid for every width and measure the tables use, the worst
// binary64 residual is 3.4e-12 against a 2.0e-12 `long double` reference and an
// acceptance threshold of 1e-11. That margin is the reason the port is safe,
// and `residual` on every returned row is what enforces it.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/radial_cell_moments.hpp"

#include <vector>

namespace quasar::numerics {

inline constexpr int kMaxRadialStencilWidth = 8;

// Select which side of the interpolation map is represented by cell averages.
enum class RadialMomentTarget {
  point_value,
  cell_average,
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
// centered exactly on the axis. This is a scalar query and stays on the host;
// the closed form is shared with the assembly kernels through
// radial_cell_moments.hpp.
Real normalized_cell_moment(Real rho, int m);

// Measure-selectable counterpart used to build component-specific
// reconstruction rows.  The returned moment is normalized by the selected
// cell measure, so the constant mode is exactly one for every family.
Real normalized_cell_moment(Real rho, int m, RadialCellMeasure measure);

// One row of the batch.
//
// point_value:
//   input k is the ring average in the cell centered at
//   rho_anchor + offset + k, and the target point is rho_anchor + node_xi.
//
// cell_average:
//   input k is a point sample at rho_anchor + offset + k + node_xi, and the
//   target is the ring average in the unit-width cell centered at rho_anchor.
//   For example, the centered four-face-to-cell row is
//   (width=4, offset=-2, node_xi=0.5).
struct RadialRowSpec {
  Real rho_anchor{};
  Real node_xi{};
  int width{};
  int offset{};
  RadialMomentTarget target{RadialMomentTarget::point_value};
  RadialCellMeasure measure{RadialCellMeasure::annular};
};

// Solves every row in one batched factorization. Returned coefficients are
// binary64, sum to exactly 1.0 under an in-order binary64 reduction, and
// `residual` is the largest defect for monomials m=0..width-1 in units of dr^m.
//
// Throws std::invalid_argument for a malformed spec and std::runtime_error when
// a row's system is singular or its constant mode cannot be normalized; both
// name the offending row index, because a batch that fails at one row is not
// usable and the caller needs to know which.
std::vector<RadialStencilRow> solve_radial_rows(
    const std::vector<RadialRowSpec>& specs,
    backend::stream_t stream = nullptr);

// One-element batches, retained for tests and for the few scalar call sites.
RadialStencilRow solve_radial_row(
    Real rho_anchor, int width, int offset,
    RadialMomentTarget target, Real node_xi = Real{0});

RadialStencilRow solve_radial_row(
    Real rho_anchor, int width, int offset,
    RadialMomentTarget target, RadialCellMeasure measure,
    Real node_xi = Real{0});

// Fold the cylindrical radius into an existing Gauss--Legendre rule on the
// unit cell.  The input Cartesian weights are expected to integrate the unit
// interval (sum to one).  `residual` measures local-monomial exactness through
// degree 2*count-2, the maximum degree retained after multiplying the
// integrand by the linear radial measure.
//
// This needs no linear solve -- it is a weighted normalization -- but it is
// still per-row arithmetic over the padded grid, so it batches the same way.
struct RadialGaussSpec {
  Real rho_anchor{};
  int count{};
  RadialCellMeasure measure{RadialCellMeasure::annular};
  Real node_xi[kMaxRadialStencilWidth]{};
  Real cartesian_weights[kMaxRadialStencilWidth]{};
};

std::vector<RadialStencilRow> radial_gauss_weight_rows(
    const std::vector<RadialGaussSpec>& specs,
    backend::stream_t stream = nullptr);

// Slope factor for extrapolating a cell's linear profile to one of its faces
// from the neighbour on the given side:
//
//   beta = (rho_center + face_offset - c) / (c - n),
//
// with c and n the first normalized moments of this cell and that neighbour.
// The MP limiter uses it to bound a reconstructed face value, and each
// conserved measure needs its own factor because the moments differ.
//
// Batched for the same reason the stencil rows are: there are six per radial
// index. Note that the result is one ulp from the exact small rationals the
// `long double` host version produced for simple configurations -- the
// difference is in the moment integral itself, not the division, and it is
// inert in a limiter bound.
struct RadialExtrapolationSpec {
  Real rho_center{};
  Real face_offset{};
  int neighbor_offset{};
  RadialCellMeasure measure{RadialCellMeasure::annular};
};

std::vector<Real> radial_face_extrapolation_factors(
    const std::vector<RadialExtrapolationSpec>& specs,
    backend::stream_t stream = nullptr);

RadialStencilRow radial_gauss_weights(
    Real rho_anchor, int count, const Real* node_xi,
    const Real* cartesian_weights);

RadialStencilRow radial_gauss_weights(
    Real rho_anchor, int count, const Real* node_xi,
    const Real* cartesian_weights, RadialCellMeasure measure);

}  // namespace quasar::numerics
