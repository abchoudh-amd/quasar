#include "quasar/numerics/shape.hpp"

#include <gtest/gtest.h>

TEST(TscShape, PartitionOfUnity) {
  const quasar::Grid2D g{16, 16, 1.0, 1.0};
  const auto w = quasar::numerics::tsc_weights_2d(0.3, 0.4, g);
  double sx = 0.0;
  double sy = 0.0;
  for (int i = 0; i < w.nx; ++i) sx += w.wx[i];
  for (int j = 0; j < w.ny; ++j) sy += w.wy[j];
  EXPECT_NEAR(sx, 1.0, 1e-14);
  EXPECT_NEAR(sy, 1.0, 1e-14);
}
