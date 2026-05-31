#include "quasar/numerics/stencil.hpp"

#include <gtest/gtest.h>

#include <vector>

// One-sided boundary closures must read only interior nodes and reproduce the
// derivative to their stated order: the *_fwd1/*_bwd1 pair is first-order exact
// (exact on a linear field), the *_fwd2/*_bwd2 pair is second-order exact (exact
// on a quadratic field). These back the non-periodic field-boundary closures.

namespace {

constexpr int kNx = 8;
constexpr int kNy = 8;

quasar::Grid2D make_grid() {
  // lx so that dx = lx/nx = 0.5; nghost = 2 so order-4 windows stay in storage.
  return quasar::Grid2D{kNx, kNy, 4.0, 4.0, 0.0, 0.0, 2};
}

// Fill f(i,j) = a + b*x + c*x^2 along x (x = i*dx), independent of j.
std::vector<double> fill_x(const quasar::Grid2D& g, double a, double b, double c) {
  std::vector<double> f(g.storage_size(), 0.0);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const double x = i * g.dx();
      f[g.index(i, j)] = a + b * x + c * x * x;
    }
  }
  return f;
}

}  // namespace

TEST(OneSidedStencil, FirstOrderExactOnLinearField) {
  const auto g = make_grid();
  const double b = 2.5;
  const auto f = fill_x(g, 1.0, b, 0.0);  // df/dx = b everywhere
  const int j = 3;
  // Low edge i=0 uses forward; high edge i=nx-1 uses backward.
  EXPECT_NEAR(quasar::numerics::ddx_onesided_fwd1(f.data(), g, 0, j), b, 1e-12);
  EXPECT_NEAR(quasar::numerics::ddx_onesided_bwd1(f.data(), g, kNx - 1, j), b, 1e-12);
}

TEST(OneSidedStencil, SecondOrderExactOnQuadraticField) {
  const auto g = make_grid();
  const double b = 1.5, c = 0.75;
  const auto f = fill_x(g, 0.3, b, c);  // df/dx = b + 2c*x
  const int j = 4;
  // Forward at i=0: x=0 -> derivative b.
  EXPECT_NEAR(quasar::numerics::ddx_onesided_fwd2(f.data(), g, 0, j), b, 1e-10);
  // Backward at i=nx-1: x=(nx-1)*dx.
  const double x_hi = (kNx - 1) * g.dx();
  EXPECT_NEAR(quasar::numerics::ddx_onesided_bwd2(f.data(), g, kNx - 1, j),
              b + 2.0 * c * x_hi, 1e-10);
}

TEST(OneSidedStencil, FirstOrderNotExactOnQuadratic) {
  // Sanity: the first-order closure carries O(h) error on a quadratic, so it is
  // distinguishable from the second-order one (guards against an accidental dup).
  const auto g = make_grid();
  const auto f = fill_x(g, 0.0, 0.0, 1.0);  // f = x^2, df/dx = 2x; at i=0 -> 0
  const double approx = quasar::numerics::ddx_onesided_fwd1(f.data(), g, 0, 2);
  EXPECT_GT(approx, 0.1);  // (x1^2 - 0)/dx = dx > 0, not the true 0
}

TEST(OneSidedStencil, YAxisMatchesXAxis) {
  const auto g = make_grid();
  std::vector<double> f(g.storage_size(), 0.0);
  const double b = -1.25;
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      f[g.index(i, j)] = 3.0 + b * (j * g.dy());
    }
  }
  EXPECT_NEAR(quasar::numerics::ddy_onesided_fwd1(f.data(), g, 2, 0), b, 1e-12);
  EXPECT_NEAR(quasar::numerics::ddy_onesided_bwd1(f.data(), g, 2, kNy - 1), b, 1e-12);
  EXPECT_NEAR(quasar::numerics::ddy_onesided_fwd2(f.data(), g, 2, 0), b, 1e-10);
  EXPECT_NEAR(quasar::numerics::ddy_onesided_bwd2(f.data(), g, 2, kNy - 1), b, 1e-10);
}
