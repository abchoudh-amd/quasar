#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

TEST(PicEmWaveDispersion, SolverConstructs) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::pic::EmPic2D3V solver{{quasar::Grid2D{8, 8, 1.0, 1.0}, 2, 1}};
  EXPECT_EQ(solver.grid().nx, 8);
}
