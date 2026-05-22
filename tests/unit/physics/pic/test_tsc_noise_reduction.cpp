#include "quasar/numerics/shape.hpp"

#include <gtest/gtest.h>

TEST(PicTscNoiseReduction, TscHasWiderSupportThanCic) {
  const quasar::Grid2D g{8, 8, 1.0, 1.0};
  EXPECT_EQ(quasar::numerics::cic_weights_2d(0.4, 0.4, g).nx, 2);
  EXPECT_EQ(quasar::numerics::tsc_weights_2d(0.4, 0.4, g).nx, 3);
}
