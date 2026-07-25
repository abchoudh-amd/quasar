#include "quasar/numerics/shape.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

TEST(CicShape, PartitionOfUnity) {
  const quasar::Grid2D g{16, 16, 1.0, 1.0};
  const auto w = quasar::numerics::cic_weights_2d(0.3, 0.4, g);
  double sx = 0.0;
  double sy = 0.0;
  for (int i = 0; i < w.nx; ++i) sx += w.wx[i];
  for (int j = 0; j < w.ny; ++j) sy += w.wy[j];
  EXPECT_NEAR(sx, 1.0, 1e-14);
  EXPECT_NEAR(sy, 1.0, 1e-14);
}

TEST(CicShape, ExtremeTranslatedCoordinateHasFiniteLogicalStencil) {
  const double maximum = std::numeric_limits<double>::max();
  const quasar::Grid2D g{1, 1, maximum, 1.0, -maximum, 0.0, 0};

  // x-origin would overflow if it were materialised before division.  This is
  // the one-cell periodic image range used by charge-conserving deposition.
  const auto w = quasar::numerics::cic_weights_2d(
      0.5 * maximum, 0.5, g);
  ASSERT_EQ(w.nx, 2);
  ASSERT_EQ(w.ny, 2);
  EXPECT_EQ(w.ix[0], 1);
  EXPECT_EQ(w.ix[1], 2);
  EXPECT_EQ(w.iy[0], 0);
  EXPECT_EQ(w.iy[1], 1);
  for (int n = 0; n < 2; ++n) {
    EXPECT_TRUE(std::isfinite(w.wx[n]));
    EXPECT_TRUE(std::isfinite(w.wy[n]));
  }
  EXPECT_DOUBLE_EQ(w.wx[0] + w.wx[1], 1.0);
  EXPECT_DOUBLE_EQ(w.wy[0] + w.wy[1], 1.0);
}

TEST(CicShape, NearbyHugeTranslationKeepsCellCenterWeights) {
  constexpr double origin = 0x1.8000000000000p+500;
  constexpr double spacing = 0x1.0000000000000p+449;
  const quasar::Grid2D g{1, 1, spacing, 1.0, origin, 0.0, 1};
  const double first_center =
      std::nextafter(origin, std::numeric_limits<double>::infinity());
  const double x = origin + spacing;

  // At this magnitude one representable step is half a cell: first_center is
  // exactly the first cell centre and would correctly have weights {1, 0}.
  // One complete spacing lies halfway between the first two cell centres,
  // where the cell-centred CIC weights are {1/2, 1/2}.
  ASSERT_DOUBLE_EQ(first_center - origin, 0.5 * spacing);
  ASSERT_DOUBLE_EQ(x - origin, spacing);

  const auto w = quasar::numerics::cic_weights_2d(x, 0.5, g);
  ASSERT_EQ(w.nx, 2);
  EXPECT_EQ(w.ix[0], 0);
  EXPECT_EQ(w.ix[1], 1);
  EXPECT_DOUBLE_EQ(w.wx[0], 0.5);
  EXPECT_DOUBLE_EQ(w.wx[1], 0.5);
}
