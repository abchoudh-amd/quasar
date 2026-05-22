#include "quasar/physics/pic/diagnostics.hpp"

#include <gtest/gtest.h>

TEST(PicEnergyConservation, DiagnosticsStubReturnsFinite) {
  EXPECT_DOUBLE_EQ(quasar::pic::total_em_energy(quasar::YeeField2D<double>{}, quasar::Grid2D{}), 0.0);
}
