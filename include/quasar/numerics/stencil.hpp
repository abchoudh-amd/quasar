#pragma once

#include "quasar/core/grid.hpp"

namespace quasar::numerics {

template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered(const Real* f, const Grid2D& g,
                                             int i, int j) noexcept {
  if constexpr (Order == 4) {
    return (Real{9} / Real{8}) * (f[g.periodic_index(i + 1, j)] - f[g.periodic_index(i, j)]) / g.dx()
         - (Real{1} / Real{24}) * (f[g.periodic_index(i + 2, j)] - f[g.periodic_index(i - 1, j)]) / g.dx();
  } else {
    return (f[g.periodic_index(i + 1, j)] - f[g.periodic_index(i, j)]) / g.dx();
  }
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered(const Real* f, const Grid2D& g,
                                             int i, int j) noexcept {
  if constexpr (Order == 4) {
    return (Real{9} / Real{8}) * (f[g.periodic_index(i, j + 1)] - f[g.periodic_index(i, j)]) / g.dy()
         - (Real{1} / Real{24}) * (f[g.periodic_index(i, j + 2)] - f[g.periodic_index(i, j - 1)]) / g.dy();
  } else {
    return (f[g.periodic_index(i, j + 1)] - f[g.periodic_index(i, j)]) / g.dy();
  }
}

template <int Order>
QUASAR_HOST_DEVICE inline Vec3 curl_e_at(const Real* ex, const Real* ey, const Real* ez,
                                         const Grid2D& g, int i, int j) noexcept {
  const Real d_ez_dy = ddy_staggered<Order>(ez, g, i, j);
  const Real d_ez_dx = ddx_staggered<Order>(ez, g, i, j);
  const Real d_ey_dx = ddx_staggered<Order>(ey, g, i, j);
  const Real d_ex_dy = ddy_staggered<Order>(ex, g, i, j);
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
