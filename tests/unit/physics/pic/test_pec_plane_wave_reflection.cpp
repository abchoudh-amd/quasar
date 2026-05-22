#include "quasar/boundary/wall.hpp"

#include <gtest/gtest.h>

TEST(PicPecPlaneWaveReflection, BoundaryTypeConstructs) {
  quasar::boundary::PecFieldBC bc;
  (void)bc;
  SUCCEED();
}
