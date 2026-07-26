#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

using quasar::Grid2D;
using quasar::Real;

namespace {

// Tight relative tolerance for closed-form double-precision comparisons.
constexpr Real kTol = Real{1e-12};

// Assemble the order-four radial wave operator -A4*B4 on an axis-touching
// unit-spaced grid with an outer PEC wall. This independent test construction
// spells out the same staggered derivative and parity closures used in the HIP
// kernels; nx>=2 keeps the two wall ghost sources distinct.
std::vector<std::vector<long double>> radial_wave_matrix(int nx) {
  using Matrix = std::vector<std::vector<long double>>;
  Matrix a(nx, std::vector<long double>(nx + 1, 0.0L));
  Matrix b(nx + 1, std::vector<long double>(nx, 0.0L));

  const auto face_value = [nx](int k, const std::vector<long double>& values) {
    if (k < 0) return -values[static_cast<std::size_t>(-k)];
    if (k > nx) return values[static_cast<std::size_t>(2 * nx - k)];
    return values[static_cast<std::size_t>(k)];
  };
  const auto cell_value = [nx](int k, const std::vector<long double>& values) {
    if (k < 0) return values[static_cast<std::size_t>(-1 - k)];
    if (k >= nx) return -values[static_cast<std::size_t>(2 * nx - 1 - k)];
    return values[static_cast<std::size_t>(k)];
  };

  for (int i = 0; i < nx; ++i) {
    const long double radius = static_cast<long double>(i) + 0.5L;
    for (int basis = 0; basis <= nx; ++basis) {
      std::vector<long double> values(nx + 1, 0.0L);
      values[static_cast<std::size_t>(basis)] = 1.0L;
      const auto flux = [&](int k) {
        return static_cast<long double>(k) * face_value(k, values);
      };
      a[static_cast<std::size_t>(i)][static_cast<std::size_t>(basis)] =
          ((9.0L / 8.0L) * (flux(i + 1) - flux(i))
           - (1.0L / 24.0L) * (flux(i + 2) - flux(i - 1))) / radius;
    }
  }
  for (int face = 0; face <= nx; ++face) {
    for (int basis = 0; basis < nx; ++basis) {
      std::vector<long double> values(nx, 0.0L);
      values[static_cast<std::size_t>(basis)] = 1.0L;
      b[static_cast<std::size_t>(face)][static_cast<std::size_t>(basis)] =
          (9.0L / 8.0L)
              * (cell_value(face, values) - cell_value(face - 1, values))
          - (1.0L / 24.0L)
              * (cell_value(face + 1, values)
                 - cell_value(face - 2, values));
    }
  }

  Matrix wave(nx, std::vector<long double>(nx, 0.0L));
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < nx; ++j) {
      for (int k = 0; k <= nx; ++k) {
        wave[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] -=
            a[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
            * b[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
      }
    }
  }
  return wave;
}

long double dominant_eigenvalue(const std::vector<std::vector<long double>>& m) {
  std::vector<long double> x(m.size(), 1.0L);
  for (int iteration = 0; iteration < 256; ++iteration) {
    std::vector<long double> y(m.size(), 0.0L);
    for (std::size_t i = 0; i < m.size(); ++i) {
      for (std::size_t j = 0; j < m.size(); ++j) y[i] += m[i][j] * x[j];
    }
    long double scale = 0.0L;
    for (long double value : y) scale = std::max(scale, std::abs(value));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = y[i] / scale;
  }
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (std::size_t i = 0; i < m.size(); ++i) {
    long double mx = 0.0L;
    for (std::size_t j = 0; j < m.size(); ++j) mx += m[i][j] * x[j];
    numerator += x[i] * mx;
    denominator += x[i] * x[i];
  }
  return numerator / denominator;
}

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

TEST(GridCylindrical, FourthOrderCflUsesProvenAxisBound) {
  const Grid2D g{8, 16, Real{2}, Real{8}, Real{0}, Real{0}, 2};
  const Real dr = g.dx();
  const Real dz = g.dy();
  const Real c = Real{3};
  const Real expected = Real{1} /
      (c * std::sqrt(Real{35} / (Real{24} * dr * dr)
                     + Real{49} / (Real{36} * dz * dz)));
  EXPECT_NEAR(quasar::cyl_cfl_dt(g, 4, c), expected,
              Real{8} * std::numeric_limits<Real>::epsilon() * expected);
  EXPECT_LT(quasar::cyl_cfl_dt(g, 4, c), quasar::cfl_dt(g, 4, c));
}

TEST(GridCylindrical, FourthOrderCflPreservesExtremeAspectRatios) {
  const Grid2D radial_dominated{
      2, 2, Real{2e-300}, Real{2e300}, Real{0}, Real{0}, 2};
  const Grid2D axial_dominated{
      2, 2, Real{2e300}, Real{2e-300}, Real{0}, Real{0}, 2};
  const Real radial_dt = quasar::cyl_cfl_dt(radial_dominated, 4, Real{1});
  const Real axial_dt = quasar::cyl_cfl_dt(axial_dominated, 4, Real{1});
  ASSERT_TRUE(std::isfinite(radial_dt));
  ASSERT_TRUE(std::isfinite(axial_dt));
  ASSERT_GT(radial_dt, Real{0});
  ASSERT_GT(axial_dt, Real{0});
  EXPECT_NEAR(
      radial_dt / (Real{1e-300} / std::sqrt(Real{35} / Real{24})),
      Real{1}, Real{8} * std::numeric_limits<Real>::epsilon());
  EXPECT_NEAR(
      axial_dt / (Real{1e-300} / std::sqrt(Real{49} / Real{36})),
      Real{1}, Real{8} * std::numeric_limits<Real>::epsilon());
}

TEST(GridCylindrical, FourthOrderAxisBoundCoversSmallValidRadialGrids) {
  struct Reference {
    int nx;
    long double spectral_radius;
  };
  constexpr Reference references[] = {
      {2, 5.444010369561416L},
      {3, 5.4444536388582545L},
      {4, 5.444444234189444L},
  };
  constexpr long double proved_bound = 35.0L / 6.0L;
  for (const auto& reference : references) {
    const long double actual = dominant_eigenvalue(
        radial_wave_matrix(reference.nx));
    EXPECT_NEAR(static_cast<double>(actual),
                static_cast<double>(reference.spectral_radius), 2.0e-14)
        << "nx=" << reference.nx;
    EXPECT_LE(actual, proved_bound) << "nx=" << reference.nx;
  }
}

TEST(GridCylindrical, ThreeCellAxisOperatorExceedsCartesianOrderFourSymbol) {
  const auto wave = radial_wave_matrix(3);
  const long double old_bound = 49.0L / 9.0L;
  EXPECT_GT(dominant_eigenvalue(wave), old_bound);

  // For this exact matrix, p(lambda)=det(lambda I-(-A4 B4)) satisfies
  // p(49/9)=-7/69120. The negative sign (with the other two eigenvalues below
  // 49/9) proves that the largest eigenvalue is strictly above the Cartesian
  // radial symbol; this is not a tolerance-driven numerical observation.
  const long double a00 = old_bound - wave[0][0];
  const long double a01 = -wave[0][1];
  const long double a02 = -wave[0][2];
  const long double a10 = -wave[1][0];
  const long double a11 = old_bound - wave[1][1];
  const long double a12 = -wave[1][2];
  const long double a20 = -wave[2][0];
  const long double a21 = -wave[2][1];
  const long double a22 = old_bound - wave[2][2];
  const long double determinant =
      a00 * (a11 * a22 - a12 * a21)
      - a01 * (a10 * a22 - a12 * a20)
      + a02 * (a10 * a21 - a11 * a20);
  EXPECT_NEAR(static_cast<double>(determinant),
              static_cast<double>(-7.0L / 69120.0L), 5.0e-15);
}

}  // namespace
