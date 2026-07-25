#include "quasar/core/types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

TEST(Vec3Math, LengthAvoidsFalseOverflowAndUnderflow) {
  const double large = std::numeric_limits<double>::max() / 4.0;
  const quasar::Vec3 large_vector{large, large, 0.0};
  const double large_norm = quasar::length(large_vector);
  ASSERT_TRUE(std::isfinite(large_norm));
  EXPECT_NEAR(large_norm / large, std::sqrt(2.0), 4.0e-16);

  const double tiny = std::numeric_limits<double>::denorm_min() * 4.0;
  const quasar::Vec3 tiny_vector{tiny, tiny, 0.0};
  const double tiny_norm = quasar::length(tiny_vector);
  EXPECT_GT(tiny_norm, 0.0);
  EXPECT_NEAR(tiny_norm / tiny, std::sqrt(2.0), 0.15);
}

TEST(Vec3Math, NormalizedExtremeFiniteVectorRetainsDirection) {
  const double largest = std::numeric_limits<double>::max();
  const quasar::Vec3 unit = quasar::normalized(
      quasar::Vec3{largest, -largest, largest});
  const double expected = 1.0 / std::sqrt(3.0);
  EXPECT_NEAR(unit.x, expected, 2.0e-16);
  EXPECT_NEAR(unit.y, -expected, 2.0e-16);
  EXPECT_NEAR(unit.z, expected, 2.0e-16);
  EXPECT_NEAR(quasar::length(unit), 1.0, 2.0e-16);
}

TEST(Vec3Math, LengthOfAnInfiniteVectorIsInfinite) {
  const double infinity = std::numeric_limits<double>::infinity();
  EXPECT_EQ(quasar::length(quasar::Vec3{infinity, 0.0, 0.0}), infinity);
  EXPECT_EQ(quasar::length(quasar::Vec3{infinity,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       0.0}), infinity);
  EXPECT_TRUE(std::isnan(quasar::length(quasar::Vec3{
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0})));
}
