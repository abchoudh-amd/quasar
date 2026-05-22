#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

TEST(PicSingleParticle, ExBConfigurationAccepted) {
  quasar::pic::EmPicConfig cfg{quasar::Grid2D{4, 4, 1.0, 1.0}, 2, 1};
  EXPECT_EQ(cfg.shape_order, 1);
}
