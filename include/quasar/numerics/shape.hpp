#pragma once

#include "quasar/core/grid.hpp"

#include <cmath>

namespace quasar::numerics {

template <int MaxN>
struct ShapeWeights2D {
  int  ix[MaxN]{};
  int  iy[MaxN]{};
  Real wx[MaxN]{};
  Real wy[MaxN]{};
  int  nx{0};
  int  ny{0};
};

// Shape weights on an arbitrary Yee sub-lattice. `offset_{x,y}` are measured in
// cells from the domain origin: 0 denotes a face/node and 1/2 a cell centre.
// Keeping the offset in the shape primitive avoids the physically incorrect
// practice of gathering every component from the charge (cell-centred) lattice.
QUASAR_HOST_DEVICE inline ShapeWeights2D<2> cic_weights_2d_at_offset(
    Real x, Real y, const Grid2D& g, Real offset_x, Real offset_y) noexcept {
  const Real gx = quasar::detail::scaled_difference_quotient(
      x, g.origin_x, g.dx()) - offset_x;
  const Real gy = quasar::detail::scaled_difference_quotient(
      y, g.origin_y, g.dy()) - offset_y;
  const int i0 = static_cast<int>(std::floor(gx));
  const int j0 = static_cast<int>(std::floor(gy));
  const Real fx = gx - static_cast<Real>(i0);
  const Real fy = gy - static_cast<Real>(j0);

  ShapeWeights2D<2> w;
  w.nx = 2;
  w.ny = 2;
  w.ix[0] = i0;
  w.ix[1] = i0 + 1;
  w.iy[0] = j0;
  w.iy[1] = j0 + 1;
  w.wx[0] = Real{1} - fx;
  w.wx[1] = fx;
  w.wy[0] = Real{1} - fy;
  w.wy[1] = fy;
  return w;
}

QUASAR_HOST_DEVICE inline ShapeWeights2D<2> cic_weights_2d(Real x, Real y,
                                                           const Grid2D& g) noexcept {
  return cic_weights_2d_at_offset(x, y, g, Real{0.5}, Real{0.5});
}

QUASAR_HOST_DEVICE inline Real tsc_weight_1d(Real r) noexcept {
  const Real ar = r < Real{0} ? -r : r;
  if (ar < Real{0.5}) {
    return Real{0.75} - ar * ar;
  }
  if (ar < Real{1.5}) {
    const Real t = Real{1.5} - ar;
    return Real{0.5} * t * t;
  }
  return Real{0};
}

QUASAR_HOST_DEVICE inline ShapeWeights2D<3> tsc_weights_2d_at_offset(
    Real x, Real y, const Grid2D& g, Real offset_x, Real offset_y) noexcept {
  const Real gx = quasar::detail::scaled_difference_quotient(
      x, g.origin_x, g.dx()) - offset_x;
  const Real gy = quasar::detail::scaled_difference_quotient(
      y, g.origin_y, g.dy()) - offset_y;
  const int ic = static_cast<int>(std::floor(gx + Real{0.5}));
  const int jc = static_cast<int>(std::floor(gy + Real{0.5}));

  ShapeWeights2D<3> w;
  w.nx = 3;
  w.ny = 3;
  for (int n = 0; n < 3; ++n) {
    w.ix[n] = ic + n - 1;
    w.iy[n] = jc + n - 1;
    w.wx[n] = tsc_weight_1d(gx - static_cast<Real>(w.ix[n]));
    w.wy[n] = tsc_weight_1d(gy - static_cast<Real>(w.iy[n]));
  }
  return w;
}

QUASAR_HOST_DEVICE inline ShapeWeights2D<3> tsc_weights_2d(Real x, Real y,
                                                           const Grid2D& g) noexcept {
  return tsc_weights_2d_at_offset(x, y, g, Real{0.5}, Real{0.5});
}

template <int ShapeOrder>
QUASAR_HOST_DEVICE inline auto shape_weights_2d(Real x, Real y, const Grid2D& g) noexcept {
  if constexpr (ShapeOrder == 1) {
    return cic_weights_2d(x, y, g);
  } else {
    return tsc_weights_2d(x, y, g);
  }
}


template <int ShapeOrder>
QUASAR_HOST_DEVICE inline auto shape_weights_2d_at_offset(
    Real x, Real y, const Grid2D& g, Real offset_x, Real offset_y) noexcept {
  if constexpr (ShapeOrder == 1) {
    return cic_weights_2d_at_offset(x, y, g, offset_x, offset_y);
  } else {
    return tsc_weights_2d_at_offset(x, y, g, offset_x, offset_y);
  }
}

}  // namespace quasar::numerics
