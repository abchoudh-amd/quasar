#include "quasar/numerics/stencil.hpp"

#include <gtest/gtest.h>

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
