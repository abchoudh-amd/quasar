#pragma once

#include "quasar/core/grid.hpp"

namespace quasar::numerics {

// Staggered finite differences read neighbours through ghost cells via
// Grid2D::index. The per-side ghost fill (fill_field_ghosts in pic_solver.cpp::step)
// runs before every curl and populates the halo: a periodic side copies the
// opposite interior edge (so the stencil is bit-for-bit identical to the old
// implicit Grid2D::periodic_index wrap), while a PEC side writes the mirror image
// (enabling reflecting field walls). Order 4 reads i-1..i+2, so it needs
// nghost >= 2 (enforced by the EmPic2D3V constructor). The Yee curl reads only
// along-axis neighbours, so corner ghosts are never touched and need no fill.
template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered(const Real* f, const Grid2D& g,
                                             int i, int j) noexcept {
  if constexpr (Order == 4) {
    return (Real{9} / Real{8}) * (f[g.index(i + 1, j)] - f[g.index(i, j)]) / g.dx()
         - (Real{1} / Real{24}) * (f[g.index(i + 2, j)] - f[g.index(i - 1, j)]) / g.dx();
  } else {
    return (f[g.index(i + 1, j)] - f[g.index(i, j)]) / g.dx();
  }
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered(const Real* f, const Grid2D& g,
                                             int i, int j) noexcept {
  if constexpr (Order == 4) {
    return (Real{9} / Real{8}) * (f[g.index(i, j + 1)] - f[g.index(i, j)]) / g.dy()
         - (Real{1} / Real{24}) * (f[g.index(i, j + 2)] - f[g.index(i, j - 1)]) / g.dy();
  } else {
    return (f[g.index(i, j + 1)] - f[g.index(i, j)]) / g.dy();
  }
}

// Backward-difference companions of ddx/ddy_staggered. The Yee scheme stays stable
// and energy-conserving (including at PEC walls) only when the two curls are
// adjoint: the E-update curl (curl_b) uses the forward difference matched to the
// charge-conserving deposit's backward-difference divergence, and the B-update
// curl (curl_e) uses the backward difference below. With both curls forward the
// operator is non-adjoint and a hard wall drives an exponential instability.
template <int Order>
QUASAR_HOST_DEVICE inline Real ddx_staggered_bwd(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  if constexpr (Order == 4) {
    return (Real{9} / Real{8}) * (f[g.index(i, j)] - f[g.index(i - 1, j)]) / g.dx()
         - (Real{1} / Real{24}) * (f[g.index(i + 1, j)] - f[g.index(i - 2, j)]) / g.dx();
  } else {
    return (f[g.index(i, j)] - f[g.index(i - 1, j)]) / g.dx();
  }
}

template <int Order>
QUASAR_HOST_DEVICE inline Real ddy_staggered_bwd(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  if constexpr (Order == 4) {
    return (Real{9} / Real{8}) * (f[g.index(i, j)] - f[g.index(i, j - 1)]) / g.dy()
         - (Real{1} / Real{24}) * (f[g.index(i, j + 1)] - f[g.index(i, j - 2)]) / g.dy();
  } else {
    return (f[g.index(i, j)] - f[g.index(i, j - 1)]) / g.dy();
  }
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
  return (f[g.index(i + 1, j)] - f[g.index(i, j)]) / g.dx();
}
QUASAR_HOST_DEVICE inline Real ddx_onesided_bwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (f[g.index(i, j)] - f[g.index(i - 1, j)]) / g.dx();
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_fwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (f[g.index(i, j + 1)] - f[g.index(i, j)]) / g.dy();
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_bwd1(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (f[g.index(i, j)] - f[g.index(i, j - 1)]) / g.dy();
}

// Second order (three-node) variants — preserve the interior order at the
// boundary node where stability allows:
//   forward : (-3 f0 + 4 f1 - f2) / (2h)
//   backward: ( 3 f0 - 4 f-1 + f-2) / (2h)
QUASAR_HOST_DEVICE inline Real ddx_onesided_fwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (Real{-3} * f[g.index(i, j)] + Real{4} * f[g.index(i + 1, j)]
          - f[g.index(i + 2, j)]) / (Real{2} * g.dx());
}
QUASAR_HOST_DEVICE inline Real ddx_onesided_bwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (Real{3} * f[g.index(i, j)] - Real{4} * f[g.index(i - 1, j)]
          + f[g.index(i - 2, j)]) / (Real{2} * g.dx());
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_fwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (Real{-3} * f[g.index(i, j)] + Real{4} * f[g.index(i, j + 1)]
          - f[g.index(i, j + 2)]) / (Real{2} * g.dy());
}
QUASAR_HOST_DEVICE inline Real ddy_onesided_bwd2(const Real* f, const Grid2D& g,
                                                 int i, int j) noexcept {
  return (Real{3} * f[g.index(i, j)] - Real{4} * f[g.index(i, j - 1)]
          + f[g.index(i, j - 2)]) / (Real{2} * g.dy());
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
