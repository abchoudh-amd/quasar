#include "quasar/boundary/wall.hpp"

#include <gtest/gtest.h>

TEST(PicAbsorbingParticleDecay, BoundaryTypeConstructs) {
  quasar::boundary::AbsorbingParticleBC bc;
  (void)bc;
  SUCCEED();
}
