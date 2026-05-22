#include "quasar/core/normalization.hpp"

#include <gtest/gtest.h>

TEST(Normalization, PlasmaReferenceRoundTrips) {
  const auto norm = quasar::Normalization::plasma(
      1.0e18, -quasar::constants::qe_abs, quasar::constants::me);
  EXPECT_GT(norm.omega_p_ref, 0.0);

  for (auto tag : {quasar::UnitTag::time, quasar::UnitTag::length,
                   quasar::UnitTag::velocity, quasar::UnitTag::e_field,
                   quasar::UnitTag::b_field, quasar::UnitTag::density,
                   quasar::UnitTag::charge, quasar::UnitTag::mass,
                   quasar::UnitTag::temperature_eV}) {
    const double si = 3.25;
    EXPECT_NEAR(norm.to_si(norm.to_internal(si, tag), tag), si, 1e-12 * si);
  }
}
