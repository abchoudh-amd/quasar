// Device-inline cylindrical (m=0) FDTD finite-difference operators shared by the
// cylindrical E-update and B-update kernels (fdtd_e_cyl_hip.hip and
// fdtd_b_cyl_hip.hip). These were verbatim-identical in both translation units;
// extracting them here keeps the adjoint operator pair (the forward face-to-node
// radial flux radial_flux_fwd and the backward radial difference ddr_bwd, plus
// the forward/backward axial differences) in one place so the Ampere and Faraday
// curls cannot drift apart -- which would silently break the discrete adjointness
// the cylindrical Yee scheme relies on.
//
// Header-only `__device__ inline` (no ODR/link issue). Included only by the .hip
// definitions under src/backend/hip/pic/; never include from a non-HIP TU.
#pragma once

#include "quasar/core/grid.hpp"

#include <hip/hip_runtime.h>

namespace quasar::backend::pic {

// Forward radial finite-volume divergence (1/r) d(r f)/dr at node (i,j), valid
// for ALL i including i=0 (where r_e(0)=0 zeroes the inner-face term, giving the
// natural axis closure for the even Ez/Bz components). r_c is the node radius;
// r_e(i)/r_e(i+1) are the lower/upper face radii of the radial cell straddling
// the node. This is the forward face-to-node flux; the Faraday update of the even
// component reads the odd face component through it, the adjoint of ddr_bwd.
__device__ inline double radial_flux_fwd(const double* __restrict__ f,
                                         const quasar::Grid2D& g, int i, int j) {
  const double inv_dr = 1.0 / g.dx();
  const double r_c = g.r_at_cell_center(i);
  const double r_lo = g.r_at_edge(i);
  const double r_hi = g.r_at_edge(i + 1);
  const double flux = r_hi * f[g.index(i + 1, j)] - r_lo * f[g.index(i, j)];
  return flux * inv_dr / r_c;
}

// Plain forward axial difference df/dz.
__device__ inline double ddz_fwd(const double* __restrict__ f, const quasar::Grid2D& g,
                                 int i, int j) {
  return (f[g.index(i, j + 1)] - f[g.index(i, j)]) / g.dy();
}

// Plain backward axial difference df/dz.
__device__ inline double ddz_bwd(const double* __restrict__ f, const quasar::Grid2D& g,
                                 int i, int j) {
  return (f[g.index(i, j)] - f[g.index(i, j - 1)]) / g.dy();
}

// Plain backward radial difference df/dr. The face-located radial derivative of a
// node quantity (Bz(i)-Bz(i-1)); the discrete adjoint of the forward face-to-node
// flux radial_flux_fwd, so the Bz/Ephi (and Ez/Bphi) polarisations stay adjoint.
__device__ inline double ddr_bwd(const double* __restrict__ f, const quasar::Grid2D& g,
                                 int i, int j) {
  return (f[g.index(i, j)] - f[g.index(i - 1, j)]) / g.dx();
}

}  // namespace quasar::backend::pic
