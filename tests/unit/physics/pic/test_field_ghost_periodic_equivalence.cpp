// The FDTD stencil reads field neighbours through ghost cells (Grid2D::index)
// rather than the old implicit periodic wrap (Grid2D::periodic_index). For a
// periodic domain the per-side ghost fill copies the opposite interior edge, so
// reading the (filled) ghost cell must be bit-for-bit identical to wrapping.
//
// This is verified at the stencil level on the host, where the arithmetic is
// identical to the device: fill the halo with the periodic copy, then assert that
// ddx/ddy_staggered<Order> read through g.index equals the same expression
// evaluated through g.periodic_index for every interior node. (A GPU-vs-host
// re-implementation would differ by an FMA ULP and is the wrong oracle.)

#include "quasar/core/grid.hpp"
#include "quasar/numerics/stencil.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// Fill nghost layers on all four sides with the periodic copy (matches
// boundary_periodic_hip.hip::periodic_fill_component).
void periodic_fill(std::vector<double>& f, const quasar::Grid2D& g) {
  for (int gh = 1; gh <= g.nghost; ++gh) {
    for (int j = 0; j < g.ny; ++j) {
      f[g.index(-gh, j)] = f[g.index(g.nx - gh, j)];
      f[g.index(g.nx - 1 + gh, j)] = f[g.index(gh - 1, j)];
    }
    for (int i = 0; i < g.nx; ++i) {
      f[g.index(i, -gh)] = f[g.index(i, g.ny - gh)];
      f[g.index(i, g.ny - 1 + gh)] = f[g.index(i, gh - 1)];
    }
  }
}

// The same range-safe arithmetic as the production stencil, but with every
// sample read through periodic_index. This isolates the ghost-copy/indexing
// equivalence without making a historical, differently rounded formula the
// floating-point oracle.
template <int Order>
double wrap_ddx(const double* f, const quasar::Grid2D& g, int i, int j) {
  if constexpr (Order == 4) {
    return quasar::numerics::staggered_derivative_values<Order>(
        f[g.periodic_index(i - 1, j)], f[g.periodic_index(i, j)],
        f[g.periodic_index(i + 1, j)], f[g.periodic_index(i + 2, j)],
        g.dx());
  }
  return quasar::numerics::staggered_derivative_values<Order>(
      0.0, f[g.periodic_index(i, j)], f[g.periodic_index(i + 1, j)],
      0.0, g.dx());
}

template <int Order>
double wrap_ddy(const double* f, const quasar::Grid2D& g, int i, int j) {
  if constexpr (Order == 4) {
    return quasar::numerics::staggered_derivative_values<Order>(
        f[g.periodic_index(i, j - 1)], f[g.periodic_index(i, j)],
        f[g.periodic_index(i, j + 1)], f[g.periodic_index(i, j + 2)],
        g.dy());
  }
  return quasar::numerics::staggered_derivative_values<Order>(
      0.0, f[g.periodic_index(i, j)], f[g.periodic_index(i, j + 1)],
      0.0, g.dy());
}

template <int Order>
void run_equivalence(int nghost) {
  const int nx = 16, ny = 16;
  quasar::Grid2D g{nx, ny, 1.0, 1.0, 0.0, 0.0, nghost};
  std::vector<double> f(g.storage_size(), 0.0);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double xx = (i + 0.5) / nx, yy = (j + 0.5) / ny;
      f[g.index(i, j)] = std::sin(2 * M_PI * xx) * std::cos(4 * M_PI * yy)
                       + 0.3 * std::cos(2 * M_PI * (xx + yy));
    }
  }
  periodic_fill(f, g);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double ix = quasar::numerics::ddx_staggered<Order>(f.data(), g, i, j);
      const double iy = quasar::numerics::ddy_staggered<Order>(f.data(), g, i, j);
      EXPECT_EQ(ix, (wrap_ddx<Order>(f.data(), g, i, j)))
          << "ddx mismatch at (" << i << "," << j << ")";
      EXPECT_EQ(iy, (wrap_ddy<Order>(f.data(), g, i, j)))
          << "ddy mismatch at (" << i << "," << j << ")";
    }
  }
}

}  // namespace

TEST(PicFieldGhostPeriodicEquivalence, Order2StencilMatchesWrapBitForBit) {
  run_equivalence<2>(1);
}

TEST(PicFieldGhostPeriodicEquivalence, Order4StencilMatchesWrapBitForBit) {
  run_equivalence<4>(2);
}
