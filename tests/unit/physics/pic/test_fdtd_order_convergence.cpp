#include "quasar/core/grid.hpp"

#include <gtest/gtest.h>

TEST(PicFdtdOrderConvergence, FourthOrderCflIsStricter) {
  const quasar::Grid2D g{16, 16, 1.0, 1.0};
  EXPECT_LT(quasar::cfl_dt(g, 4), quasar::cfl_dt(g, 2));
}
