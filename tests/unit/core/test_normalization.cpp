#include "quasar/core/normalization.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

TEST(Normalization, PlasmaRejectsInvalidReferences) {
  using quasar::Normalization;
  using quasar::constants::me;
  using quasar::constants::qe_abs;
  // n_ref must be > 0.
  EXPECT_THROW(Normalization::plasma(0.0, qe_abs, me), std::invalid_argument);
  EXPECT_THROW(Normalization::plasma(-1.0e18, qe_abs, me), std::invalid_argument);
  // q_ref must be non-zero.
  EXPECT_THROW(Normalization::plasma(1.0e18, 0.0, me), std::invalid_argument);
  // m_ref must be > 0.
  EXPECT_THROW(Normalization::plasma(1.0e18, qe_abs, 0.0), std::invalid_argument);
  EXPECT_THROW(Normalization::plasma(1.0e18, qe_abs, -me), std::invalid_argument);
  // Non-finite inputs are rejected.
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(Normalization::plasma(inf, qe_abs, me), std::invalid_argument);
}

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
