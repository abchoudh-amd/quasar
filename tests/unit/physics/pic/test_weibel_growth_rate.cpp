#include "quasar/core/types.hpp"

#include <gtest/gtest.h>

TEST(PicWeibelGrowthRate, VectorMathWorks) {
  const quasar::Vec3 a{1, 0, 0};
  const quasar::Vec3 b{0, 1, 0};
  EXPECT_DOUBLE_EQ(quasar::cross(a, b).z, 1.0);
}
