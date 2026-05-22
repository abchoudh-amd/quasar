#include "quasar/backend/device.hpp"
#include "quasar/boundary/wall.hpp"

#include <gtest/gtest.h>

TEST(PecGhosts, LauncherIsCallable) {
  if (!quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  quasar::YeeField2D<double> field{quasar::Grid2D{4, 4, 1.0, 1.0, 0.0, 0.0, 2}};
  quasar::boundary::PecFieldBC bc;
  EXPECT_NO_THROW(bc.fill_ghosts(field, quasar::Side::y_lo));
}
