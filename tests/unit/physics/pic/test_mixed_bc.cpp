#include "quasar/boundary/boundary_condition.hpp"

#include <gtest/gtest.h>

TEST(PicMixedBc, SpecCanMixPeriodicAndWalls) {
  quasar::boundary::BoundarySpec spec;
  spec.field[2] = quasar::boundary::FieldBoundaryKind::pec;
  spec.particle[3] = quasar::boundary::ParticleBoundaryKind::absorbing;
  EXPECT_EQ(spec.field[2], quasar::boundary::FieldBoundaryKind::pec);
  EXPECT_EQ(spec.particle[3], quasar::boundary::ParticleBoundaryKind::absorbing);
}
