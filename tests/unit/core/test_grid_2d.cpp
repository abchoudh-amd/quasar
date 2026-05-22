#include "quasar/core/grid.hpp"

#include <gtest/gtest.h>

using quasar::Grid2D;
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

TEST(Grid2D, CflOrderFactor) {
  const Grid2D g{16, 16, Real{1}, Real{1}};
  EXPECT_GT(quasar::cfl_dt(g, 2), quasar::cfl_dt(g, 4));
}
