#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

using quasar::Grid2D;
using quasar::Real;

namespace {

// Tight relative tolerance for closed-form double-precision comparisons.
constexpr Real kTol = Real{1e-12};

// ---------------------------------------------------------------------------
// r_at_edge: r = origin_x + i*dr. On an axis-touching grid (origin_x == 0)
// the i=0 edge sits exactly on the symmetry axis (r = 0).
// ---------------------------------------------------------------------------
TEST(GridCylindrical, EdgeOnAxisGrid) {
  // nx=4, lx=2 -> dr = 0.5; origin_x = 0 (domain starts on the axis).
  const Grid2D g{4, 8, Real{2}, Real{4}, Real{0}, Real{0}, 1};
  const Real dr = g.dx();
  ASSERT_DOUBLE_EQ(dr, Real{0.5});

  // The r=0 axis edge.
  EXPECT_DOUBLE_EQ(g.r_at_edge(0), g.origin_x);
  EXPECT_DOUBLE_EQ(g.r_at_edge(0), Real{0});

  // Linear in i with spacing dr.
  for (int i = 0; i <= g.nx; ++i) {
    EXPECT_NEAR(g.r_at_edge(i), g.origin_x + static_cast<Real>(i) * dr,
                kTol * (Real{1} + std::abs(g.r_at_edge(i))));
  }
  // Constant-spacing check between successive edges.
  EXPECT_DOUBLE_EQ(g.r_at_edge(1) - g.r_at_edge(0), dr);
  EXPECT_DOUBLE_EQ(g.r_at_edge(3) - g.r_at_edge(2), dr);
}

TEST(GridCylindrical, EdgeOffAxisGrid) {
  // origin_x = r_min = 1.0, nx=5, lx=2.5 -> dr = 0.5.
  const Grid2D g{5, 4, Real{2.5}, Real{2}, Real{1}, Real{0}, 1};
  const Real dr = g.dx();
  ASSERT_DOUBLE_EQ(dr, Real{0.5});

  // i=0 edge sits at origin_x (r_min), not on the axis.
  EXPECT_DOUBLE_EQ(g.r_at_edge(0), Real{1});
  for (int i = 0; i <= g.nx; ++i) {
    EXPECT_NEAR(g.r_at_edge(i), Real{1} + static_cast<Real>(i) * dr,
                kTol * (Real{1} + std::abs(g.r_at_edge(i))));
  }
}

// ---------------------------------------------------------------------------
// r_at_cell_center: r = origin_x + (i+0.5)*dr. Centers sit half a cell
// outboard of the left edge of the same column.
// ---------------------------------------------------------------------------
TEST(GridCylindrical, CellCenterFormula) {
  const Grid2D g{4, 8, Real{2}, Real{4}, Real{0}, Real{0}, 1};
  const Real dr = g.dx();
  for (int i = 0; i < g.nx; ++i) {
    EXPECT_NEAR(g.r_at_cell_center(i),
                g.origin_x + (static_cast<Real>(i) + Real{0.5}) * dr,
                kTol * (Real{1} + std::abs(g.r_at_cell_center(i))));
    // Center is exactly half a cell outboard of the left edge.
    EXPECT_NEAR(g.r_at_cell_center(i) - g.r_at_edge(i), Real{0.5} * dr, kTol);
  }
}

TEST(GridCylindrical, CellCenterOffAxis) {
  const Grid2D g{5, 4, Real{2.5}, Real{2}, Real{1}, Real{0}, 1};
  const Real dr = g.dx();
  // First cell center: r_min + 0.5*dr = 1.0 + 0.25 = 1.25.
  EXPECT_DOUBLE_EQ(g.r_at_cell_center(0), Real{1.25});
  for (int i = 0; i < g.nx; ++i) {
    EXPECT_NEAR(g.r_at_cell_center(i) - g.r_at_edge(i), Real{0.5} * dr, kTol);
  }
}

// ---------------------------------------------------------------------------
// cell_volume: V_i = 2*pi * r_at_cell_center(i) * dr * dz  (azimuthal 2*pi
// m=0 ring of one cell).
// ---------------------------------------------------------------------------
TEST(GridCylindrical, CellVolumeFormula) {
  const Grid2D g{4, 8, Real{2}, Real{4}, Real{0}, Real{0}, 1};
  const Real dr = g.dx();
  const Real dz = g.dy();
  for (int i = 0; i < g.nx; ++i) {
    const Real expected =
        Real{2} * quasar::pi * g.r_at_cell_center(i) * dr * dz;
    EXPECT_NEAR(g.cell_volume(i), expected,
                kTol * (Real{1} + std::abs(expected)));
  }
}

TEST(GridCylindrical, CellVolumeNearAxisIsSmallButNonZero) {
  // origin_x == 0: i=0 cell center is at 0.5*dr, so its volume is the small
  // near-axis ring value, NOT zero.
  const Grid2D g{4, 8, Real{2}, Real{4}, Real{0}, Real{0}, 1};
  const Real dr = g.dx();
  const Real dz = g.dy();
  const Real v0 = g.cell_volume(0);
  EXPECT_GT(v0, Real{0});
  // Exactly 2*pi * (0.5*dr) * dr * dz.
  const Real expected = Real{2} * quasar::pi * (Real{0.5} * dr) * dr * dz;
  EXPECT_NEAR(v0, expected, kTol * (Real{1} + std::abs(expected)));
}

TEST(GridCylindrical, CellVolumeAvoidsIntermediateRangeFailure) {
  const Grid2D underflow_order{1, 1, Real{1e-200}, Real{1e100},
                               Real{1e-200}, Real{0}, 0};
  const long double expected_small =
      2.0L * static_cast<long double>(quasar::pi)
      * static_cast<long double>(underflow_order.r_at_cell_center(0))
      * static_cast<long double>(underflow_order.dx())
      * static_cast<long double>(underflow_order.dy());
  const Real small = underflow_order.cell_volume(0);
  ASSERT_TRUE(std::isfinite(small));
  ASSERT_GT(small, Real{0});
  EXPECT_NEAR(small / static_cast<Real>(expected_small), Real{1},
              Real{8} * std::numeric_limits<Real>::epsilon());

  const Grid2D overflow_order{1, 1, Real{1e200}, Real{1e-100},
                              Real{1e200}, Real{0}, 0};
  const long double expected_large =
      2.0L * static_cast<long double>(quasar::pi)
      * static_cast<long double>(overflow_order.r_at_cell_center(0))
      * static_cast<long double>(overflow_order.dx())
      * static_cast<long double>(overflow_order.dy());
  const Real large = overflow_order.cell_volume(0);
  ASSERT_TRUE(std::isfinite(large));
  ASSERT_GT(large, Real{0});
  EXPECT_NEAR(large / static_cast<Real>(expected_large), Real{1},
              Real{8} * std::numeric_limits<Real>::epsilon());
}

TEST(GridCylindrical, CellVolumeGrowsWithRadius) {
  const Grid2D g{6, 4, Real{3}, Real{2}, Real{0}, Real{0}, 1};
  // Strictly increasing in i (radius grows outboard).
  for (int i = 0; i + 1 < g.nx; ++i) {
    EXPECT_GT(g.cell_volume(i + 1), g.cell_volume(i));
    // Ratio equals r_center(i+1)/r_center(i) since dr, dz, 2*pi cancel.
    const Real ratio = g.cell_volume(i + 1) / g.cell_volume(i);
    const Real expected_ratio =
        g.r_at_cell_center(i + 1) / g.r_at_cell_center(i);
    EXPECT_NEAR(ratio, expected_ratio, kTol * (Real{1} + expected_ratio));
    // The ratio itself shrinks toward 1 as r grows (radii get closer
    // proportionally), so cell volume grows but the growth factor decreases.
  }
  // Confirm the growth-factor (ratio) decreases with r.
  const Real ratio_lo = g.cell_volume(1) / g.cell_volume(0);
  const Real ratio_hi = g.cell_volume(g.nx - 1) / g.cell_volume(g.nx - 2);
  EXPECT_GT(ratio_lo, ratio_hi);
}

// ---------------------------------------------------------------------------
// cyl_cfl_dt: 2nd-order Cartesian Courant limit using (dr, dz):
//   dt = 1 / ( c * sqrt(1/dr^2 + 1/dz^2) ).
// ---------------------------------------------------------------------------
TEST(GridCylindrical, CflClosedForm) {
  const Grid2D g{8, 16, Real{2}, Real{4}, Real{0}, Real{0}, 1};
  const Real dr = g.dx();   // 0.25
  const Real dz = g.dy();   // 0.25
  const Real c = Real{3};
  const Real sr = Real{1} / (dr * dr);
  const Real sz = Real{1} / (dz * dz);
  const Real expected = Real{1} / (c * std::sqrt(sr + sz));
  EXPECT_NEAR(quasar::cyl_cfl_dt(g, c), expected,
              kTol * (Real{1} + expected));
}

TEST(GridCylindrical, CflMatchesSecondOrderCartesian) {
  // cyl_cfl_dt is defined as the 2nd-order cfl_dt on the same grid.
  const Grid2D g{10, 7, Real{2.5}, Real{1.4}, Real{1}, Real{0}, 1};
  EXPECT_DOUBLE_EQ(quasar::cyl_cfl_dt(g, Real{2}),
                   quasar::cfl_dt(g, 2, Real{2}));
  // Preserve the historical two-argument API even when the wave speed is an
  // integer literal; it must not be reinterpreted as an FDTD order.
  EXPECT_DOUBLE_EQ(quasar::cyl_cfl_dt(g, 2),
                   quasar::cfl_dt(g, 2, Real{2}));
}

TEST(GridCylindrical, CflInverselyProportionalToWaveSpeed) {
  const Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, 1};
  const Real dt1 = quasar::cyl_cfl_dt(g, Real{1});
  const Real dt2 = quasar::cyl_cfl_dt(g, Real{2});
  // Doubling c halves dt.
  EXPECT_NEAR(dt2, Real{0.5} * dt1, kTol * (Real{1} + dt1));
}

TEST(GridCylindrical, CflFinerDrGivesSmallerDt) {
  // Same dz, finer dr (more radial cells over the same lr) -> smaller dt.
  const Grid2D coarse{8, 8, Real{2}, Real{2}, Real{0}, Real{0}, 1};
  const Grid2D fine{16, 8, Real{2}, Real{2}, Real{0}, Real{0}, 1};
  EXPECT_GT(quasar::cyl_cfl_dt(coarse, Real{1}),
            quasar::cyl_cfl_dt(fine, Real{1}));
}

TEST(GridCylindrical, CflFinerDzGivesSmallerDt) {
  // Same dr, finer dz -> smaller dt.
  const Grid2D coarse{8, 8, Real{2}, Real{2}, Real{0}, Real{0}, 1};
  const Grid2D fine{8, 16, Real{2}, Real{2}, Real{0}, Real{0}, 1};
  EXPECT_GT(quasar::cyl_cfl_dt(coarse, Real{1}),
            quasar::cyl_cfl_dt(fine, Real{1}));
}

}  // namespace
