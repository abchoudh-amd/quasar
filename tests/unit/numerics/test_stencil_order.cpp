#include "quasar/numerics/stencil.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

TEST(Stencils, ConstantFieldCurlIsZero) {
  const quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 2};
  std::vector<double> f(g.storage_size(), 2.0);
  const auto c2 = quasar::numerics::curl_e_at<2>(f.data(), f.data(), f.data(), g, 3, 3);
  const auto c4 = quasar::numerics::curl_e_at<4>(f.data(), f.data(), f.data(), g, 3, 3);
  EXPECT_DOUBLE_EQ(c2.x, 0.0);
  EXPECT_DOUBLE_EQ(c2.y, 0.0);
  EXPECT_DOUBLE_EQ(c2.z, 0.0);
  EXPECT_DOUBLE_EQ(c4.x, 0.0);
  EXPECT_DOUBLE_EQ(c4.y, 0.0);
  EXPECT_DOUBLE_EQ(c4.z, 0.0);
}

TEST(Stencils, NearbyHugeStaggeredDifferencesKeepTheirLocalSlope) {
  // Dividing these samples separately by their non-power-of-two maximum
  // rounds away one quarter of the one-ulp separation.  The direct finite
  // difference is exact and both supported stencils must retain it.
  const double f0 = 0x1.8000000000000p+500;
  const double h = 0x1.0000000000000p+448;
  const double fm1 = f0 - h;
  const double f1 = std::nextafter(f0, std::numeric_limits<double>::infinity());
  const double f2 = f1 + h;
  ASSERT_DOUBLE_EQ(f1 - f0, h);

  EXPECT_DOUBLE_EQ(
      quasar::numerics::staggered_derivative_values<2>(0.0, f0, f1, 0.0, h),
      1.0);
  EXPECT_DOUBLE_EQ(
      quasar::numerics::staggered_derivative_values<4>(fm1, f0, f1, f2, h),
      1.0);
}

TEST(Stencils, OppositeExtremeStaggeredDifferenceDividesBeforeRestoringRange) {
  const double largest = std::numeric_limits<double>::max();
  ASSERT_FALSE(std::isfinite(largest - (-largest)));
  EXPECT_DOUBLE_EQ(
      quasar::numerics::staggered_derivative_values<2>(
          0.0, -largest, largest, 0.0, largest),
      2.0);
}

TEST(Stencils, CylindricalFluxPreservesNearbyHugeDifferenceAndMetric) {
  const double f0 = 0x1.8000000000000p+500;
  const double h = 0x1.0000000000000p+448;
  const double f1 = std::nextafter(f0, std::numeric_limits<double>::infinity());

  // D(f)/dr = 1 and the centered metric term rounds to one at r=f0.
  // The old normalized subtraction produced 0.75 + 1 instead.
  EXPECT_DOUBLE_EQ(
      quasar::numerics::cylindrical_radial_flux_values<2>(
          0.0, f0, f1, 0.0, f0, h),
      2.0);

  const double largest = std::numeric_limits<double>::max();
  EXPECT_DOUBLE_EQ(
      quasar::numerics::cylindrical_radial_flux_values<2>(
          0.0, -largest, largest, 0.0, largest / 2.0, largest),
      2.0);
}
