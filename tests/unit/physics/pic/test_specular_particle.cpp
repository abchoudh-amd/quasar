#include "quasar/boundary/wall.hpp"

#include <gtest/gtest.h>

TEST(PicSpecularParticle, BoundaryTypeConstructs) {
  quasar::boundary::SpecularParticleBC bc;
  (void)bc;
  SUCCEED();
}
