// Device-inline cylindrical (m=0) FDTD finite-difference operators shared by the
// cylindrical E-update and B-update kernels (fdtd_e_cyl_hip.hip and
// fdtd_b_cyl_hip.hip). These were verbatim-identical in both translation units;
// extracting them here keeps the timestep-weighted adjoint operator pair (the
// forward face-to-node radial flux and the backward radial difference, plus the
// forward/backward axial differences) in one place so the Ampere and Faraday
// curls cannot drift apart -- which would silently break the discrete adjointness
// the cylindrical Yee scheme relies on.
//
// Header-only `__device__ inline` (no ODR/link issue). Included only by the .hip
// definitions under src/backend/hip/pic/; never include from a non-HIP TU.
#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/numerics/stencil.hpp"

#include <hip/hip_runtime.h>

namespace quasar::backend::pic {

// Forward radial finite-volume divergence (1/r) d(r f)/dr at cell centre
// (i,j).  The fourth-order form is the ordinary fourth-order staggered
// derivative applied to q=r*f, followed by division by r_c. The implementation
// uses the algebraically equivalent D(f)/dr + M(f)/r form from stencil.hpp, so
// it never materialises r*f. At the axis q is
// even (r and the face-centred vector component are both odd), so q(-dr)=q(dr);
// the axis ghost fill supplies exactly that parity.  This gives, at i=0,
//   [(7/6) q_1 - (1/24) q_2] / (r_{1/2} dr),
// the regular fourth-order limit rather than a singular 1/r evaluation.
template <int Order>
__device__ inline double radial_flux_fwd_increment(
    const double* __restrict__ f, const quasar::Grid2D& g, int i, int j,
    double dt) {
  const double r_c = g.r_at_cell_center(i);
  if constexpr (Order == 4) {
    return quasar::numerics::cylindrical_radial_flux_increment_values<Order>(
        f[g.index(i - 1, j)], f[g.index(i, j)],
        f[g.index(i + 1, j)], f[g.index(i + 2, j)], r_c, g.dx(), dt);
  }
  return quasar::numerics::cylindrical_radial_flux_increment_values<Order>(
      0.0, f[g.index(i, j)], f[g.index(i + 1, j)], 0.0, r_c,
      g.dx(), dt);
}

template <int Order>
__device__ inline quasar::numerics::detail::ScaledValue
radial_flux_fwd_increment_scaled(
    const double* __restrict__ f, const quasar::Grid2D& g, int i, int j,
    double dt) {
  const double r_c = g.r_at_cell_center(i);
  if constexpr (Order == 4) {
    return quasar::numerics::cylindrical_radial_flux_increment_scaled_values<Order>(
        f[g.index(i - 1, j)], f[g.index(i, j)],
        f[g.index(i + 1, j)], f[g.index(i + 2, j)], r_c, g.dx(), dt);
  }
  return quasar::numerics::cylindrical_radial_flux_increment_scaled_values<Order>(
      0.0, f[g.index(i, j)], f[g.index(i + 1, j)], 0.0, r_c,
      g.dx(), dt);
}

// Plain staggered forward/backward axial increments.
template <int Order>
__device__ inline double ddz_fwd_increment(const double* __restrict__ f,
                                           const quasar::Grid2D& g,
                                           int i, int j, double dt) {
  return quasar::numerics::ddy_staggered_increment<Order>(f, g, i, j, dt);
}

template <int Order>
__device__ inline quasar::numerics::detail::ScaledValue ddz_fwd_increment_scaled(
    const double* __restrict__ f, const quasar::Grid2D& g,
    int i, int j, double dt) {
  return quasar::numerics::ddy_staggered_increment_scaled<Order>(
      f, g, i, j, dt);
}

template <int Order>
__device__ inline double ddz_bwd_increment(const double* __restrict__ f,
                                           const quasar::Grid2D& g,
                                           int i, int j, double dt) {
  return quasar::numerics::ddy_staggered_bwd_increment<Order>(
      f, g, i, j, dt);
}

template <int Order>
__device__ inline quasar::numerics::detail::ScaledValue ddz_bwd_increment_scaled(
    const double* __restrict__ f, const quasar::Grid2D& g,
    int i, int j, double dt) {
  return quasar::numerics::ddy_staggered_bwd_increment_scaled<Order>(
      f, g, i, j, dt);
}

// Plain backward radial increment. This face-located derivative of a node
// quantity is the discrete adjoint of the forward face-to-node flux, so the
// Bz/Ephi (and Ez/Bphi) polarisations stay adjoint.
template <int Order>
__device__ inline double ddr_bwd_increment(const double* __restrict__ f,
                                           const quasar::Grid2D& g,
                                           int i, int j, double dt) {
  return quasar::numerics::ddx_staggered_bwd_increment<Order>(
      f, g, i, j, dt);
}

template <int Order>
__device__ inline quasar::numerics::detail::ScaledValue ddr_bwd_increment_scaled(
    const double* __restrict__ f, const quasar::Grid2D& g,
    int i, int j, double dt) {
  return quasar::numerics::ddx_staggered_bwd_increment_scaled<Order>(
      f, g, i, j, dt);
}

}  // namespace quasar::backend::pic
