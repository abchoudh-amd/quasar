#include "quasar/core/grid.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using quasar::Grid2D;

TEST(Grid2D, ExplicitSpacingPreservesParentGridSpacing) {
  const Grid2D parent{5, 7, 1.0, 1.0};
  const Grid2D tile = Grid2D::from_cell_spacing(
      3, 4, parent.dx(), parent.dy(), parent.origin_x, parent.origin_y, 2);

  EXPECT_EQ(tile.dx(), parent.dx());
  EXPECT_EQ(tile.dy(), parent.dy());
  EXPECT_EQ(tile.lx, parent.dx() * 3.0);
  EXPECT_EQ(tile.ly, parent.dy() * 4.0);
  EXPECT_NO_THROW(tile.validate());
}
using quasar::Real;

TEST(Grid2D, IndexingAndPeriodicWrap) {
  const Grid2D g{8, 4, Real{2}, Real{1}, Real{0}, Real{0}, 2};
  EXPECT_EQ(g.pitch(), 12);
  EXPECT_EQ(g.height(), 8);
  EXPECT_EQ(g.wrap_i(-1), 7);
  EXPECT_EQ(g.wrap_i(8), 0);
  EXPECT_EQ(g.wrap_j(-1), 3);
  EXPECT_EQ(g.periodic_index(-1, -1), g.index(7, 3));
}

TEST(Grid2D, RejectsUnrepresentableGeometryAndHalo) {
  EXPECT_THROW((quasar::Grid2D{2, 1, std::numeric_limits<double>::denorm_min(),
                               1.0}), std::overflow_error);
  EXPECT_THROW((quasar::Grid2D{1, 1, std::numeric_limits<double>::max(), 1.0,
                               std::numeric_limits<double>::max(), 0.0}),
               std::overflow_error);
  EXPECT_THROW((quasar::Grid2D{1, 1, 1.0, 1.0, 0.0, 0.0,
                               std::numeric_limits<int>::max()}),
               std::overflow_error);
  EXPECT_THROW((quasar::Grid2D{
                   std::numeric_limits<int>::max(), 1,
                   static_cast<double>(std::numeric_limits<int>::max()), 1.0,
                   0.0, 0.0, 0}),
               std::overflow_error);
}

TEST(Grid2D, RejectsActualHighCellCenterCollapsedOntoHighFace) {
  // With these exact binary values, upper - dx/2 is distinct from upper, but
  // the expression used by x_at_cell_center(1), origin + 1.5*dx, rounds onto
  // upper.  Validation must protect the coordinates consumers actually read.
  constexpr double origin = 0x1.58be77ab34af0p-1002;
  constexpr double length = 0x0.0000000373a5dp-1022;
  EXPECT_THROW((Grid2D{2, 1, length, 1.0, origin, 0.0, 0}),
               std::overflow_error);
}

TEST(Grid2D, ExtremeTranslatedAffineCoordinatesRemainRepresentable) {
  const double maximum = std::numeric_limits<double>::max();
  EXPECT_DOUBLE_EQ(
      quasar::detail::scaled_difference_quotient(
          0.5 * maximum, -maximum, maximum),
      1.5);

  // Evaluating origin + i*dx as two rounded operations would overflow at
  // i=2 even though the complete affine coordinate is exactly DBL_MAX.
  const Grid2D g{1, 1, maximum, 1.0, -maximum, 0.0, 0};
  EXPECT_DOUBLE_EQ(g.r_at_edge(2), maximum);
}

TEST(Grid2D, NearbyHugeCoordinatesRetainTheirExactLocalOffset) {
  constexpr double origin = 0x1.8000000000000p+500;
  constexpr double spacing = 0x1.0000000000000p+449;
  const double first_center =
      std::nextafter(origin, std::numeric_limits<double>::infinity());

  ASSERT_DOUBLE_EQ(first_center - origin, 0.5 * spacing);
  EXPECT_DOUBLE_EQ(
      quasar::detail::scaled_difference_quotient(
          first_center, origin, spacing),
      0.5);

  const Grid2D g{1, 1, spacing, 1.0, origin, 0.0, 1};
  EXPECT_DOUBLE_EQ(g.x_at_cell_center(0), first_center);
}

TEST(Grid2D, CflEvaluationIsStableForLargeAspectRatio) {
  const quasar::Grid2D g{4, 4, 4.0e-200, 4.0e100};
  const double dt = quasar::cfl_dt(g, 2);
  EXPECT_TRUE(std::isfinite(dt));
  EXPECT_GT(dt, 0.0);
  EXPECT_NEAR(dt, 1.0e-200, 1.0e-214);
}

TEST(Grid2D, CflOrderFactor) {
  const Grid2D g{16, 16, Real{1}, Real{1}};
  EXPECT_GT(quasar::cfl_dt(g, 2), quasar::cfl_dt(g, 4));
}

TEST(Grid2D, CflLargeEqualSpacingAndSubunitWaveSpeedRemainFinite) {
  const double maximum = std::numeric_limits<double>::max();
  const Grid2D g{1, 1, maximum, maximum, 0.0, 0.0, 0};
  const double dt = quasar::cfl_dt(g, 2, 0.75);
  const double expected = (maximum / std::sqrt(2.0)) / 0.75;
  EXPECT_TRUE(std::isfinite(dt));
  EXPECT_GT(dt, 0.0);
  EXPECT_NEAR(dt / expected, 1.0, 4.0 * std::numeric_limits<double>::epsilon());
}

TEST(Grid2D, CflSubnormalSpacingAndWaveSpeedKeepRepresentableRatio) {
  const double scale = 2.0 * std::numeric_limits<double>::denorm_min();
  const Grid2D g{1, 1, scale, scale, 0.0, 0.0, 0};
  const double dt = quasar::cfl_dt(g, 2, scale);
  const double expected = 1.0 / std::sqrt(2.0);
  EXPECT_TRUE(std::isfinite(dt));
  EXPECT_NEAR(dt, expected, 4.0 * std::numeric_limits<double>::epsilon());
}
