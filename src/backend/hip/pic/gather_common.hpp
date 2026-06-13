// Device-inline gather/push helpers shared by the Cartesian and cylindrical
// particle gather+push kernels (gather_push_hip.hip and gather_push_cyl_hip.hip).
// These were verbatim-identical in both translation units; extracting them here
// keeps the field-gather indexing and the Boris half-rotation in one place so the
// two kernels cannot drift. The cylindrical-specific position-advance coordinate
// rotation stays in gather_push_cyl_hip.hip.
//
// Header-only `__device__ inline` (no ODR/link issue). Included only by the .hip
// definitions under src/backend/hip/pic/ (it pulls in <hip/hip_runtime.h>); never
// include from a non-HIP TU.
#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::backend::pic {

// Per-axis gather indexing. A periodic axis wraps (historical behaviour); a
// non-periodic (wall) axis clamps the shape node into the ghost layer via
// Grid2D::index, reading the field values the field-boundary layer maintains
// there (mirror/one-sided closure for the self field, edge-replicated external
// field) instead of wrapping to the far edge. Mirrors the deposit node indexing
// (deposit_axis_x/y in deposit_common.hpp).
__device__ inline int clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline std::size_t gather_index(const quasar::Grid2D& g, int i, int j,
                                           bool periodic_x, bool periodic_y) {
  const int ii = periodic_x ? g.wrap_i(i) : clamp_axis(i, -g.nghost, g.nx - 1 + g.nghost);
  const int jj = periodic_y ? g.wrap_j(j) : clamp_axis(j, -g.nghost, g.ny - 1 + g.nghost);
  return g.index(ii, jj);
}

// Gathers the self field and the external field of one (vector) quantity in a
// single sweep over the shape stencil, reusing the precomputed weights `w` and
// computing each node's wrapped/clamped index exactly once. `out_self` and
// `out_ext` accumulate the two contributions.
template <class Weights>
__device__ inline void gather_pair(const quasar::Grid2D& g, const Weights& w,
                                   const double* __restrict__ sx, const double* __restrict__ sy,
                                   const double* __restrict__ sz,
                                   const double* __restrict__ ex, const double* __restrict__ ey,
                                   const double* __restrict__ ez,
                                   bool periodic_x, bool periodic_y,
                                   quasar::Vec3& out_self, quasar::Vec3& out_ext) {
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const double ww = w.wx[ii] * w.wy[jj];
      const std::size_t k = gather_index(g, w.ix[ii], w.iy[jj], periodic_x, periodic_y);
      out_self.x += ww * sx[k];
      out_self.y += ww * sy[k];
      out_self.z += ww * sz[k];
      out_ext.x += ww * ex[k];
      out_ext.y += ww * ey[k];
      out_ext.z += ww * ez[k];
    }
  }
}

__device__ inline quasar::Vec3 boris(double qm, quasar::Vec3 v, quasar::Vec3 e,
                                     quasar::Vec3 b, double dt) {
  const quasar::Vec3 v_minus = v + (qm * dt * 0.5) * e;
  const quasar::Vec3 t = (qm * dt * 0.5) * b;
  const double t2 = quasar::dot(t, t);
  const quasar::Vec3 s = (2.0 / (1.0 + t2)) * t;
  const quasar::Vec3 v_prime = v_minus + quasar::cross(v_minus, t);
  const quasar::Vec3 v_plus = v_minus + quasar::cross(v_prime, s);
  return v_plus + (qm * dt * 0.5) * e;
}

}  // namespace quasar::backend::pic
