#include "quasar/core/normalization.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

TEST(Normalization, DefaultAndFactoryAreIdentityForEveryUnitTag) {
  const auto default_norm = quasar::Normalization{};
  const auto identity = quasar::Normalization::identity();
  const auto helper_identity = quasar::identity_normalization();
  EXPECT_TRUE(default_norm.is_identity());
  EXPECT_TRUE(identity.is_identity());
  EXPECT_TRUE(helper_identity.is_identity());
  EXPECT_DOUBLE_EQ(identity.length_scale(), 1.0);
  EXPECT_DOUBLE_EQ(identity.time_scale(), 1.0);
  EXPECT_DOUBLE_EQ(identity.e_field_scale(), 1.0);
  EXPECT_DOUBLE_EQ(identity.b_field_scale(), 1.0);
  EXPECT_DOUBLE_EQ(identity.temperature_eV_scale(), 1.0);

  for (auto tag : {quasar::UnitTag::time, quasar::UnitTag::length,
                   quasar::UnitTag::velocity, quasar::UnitTag::e_field,
                   quasar::UnitTag::b_field, quasar::UnitTag::density,
                   quasar::UnitTag::charge, quasar::UnitTag::mass,
                   quasar::UnitTag::temperature_eV}) {
    constexpr double value = -3.25;
    EXPECT_DOUBLE_EQ(default_norm.to_internal(value, tag), value);
    EXPECT_DOUBLE_EQ(default_norm.to_si(value, tag), value);
    EXPECT_DOUBLE_EQ(identity.to_internal(value, tag), value);
    EXPECT_DOUBLE_EQ(identity.to_si(value, tag), value);
    EXPECT_DOUBLE_EQ(helper_identity.to_internal(value, tag), value);
    EXPECT_DOUBLE_EQ(helper_identity.to_si(value, tag), value);
  }
}

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
  EXPECT_FALSE(norm.is_identity());
  EXPECT_FALSE(quasar::Normalization::plasma(1.0, 1.0, 1.0).is_identity());
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

TEST(Normalization, TemperatureEvUsesRestEnergyScale) {
  const auto norm = quasar::Normalization::plasma(
      1.0e18, quasar::constants::qe_abs, quasar::constants::me);
  const double electron_rest_energy_eV =
      quasar::constants::me * quasar::constants::c0 * quasar::constants::c0
      / quasar::constants::qe_abs;
  EXPECT_NEAR(norm.temperature_eV_scale(), electron_rest_energy_eV,
              1e-14 * electron_rest_energy_eV);
  EXPECT_NEAR(norm.to_internal(1.0, quasar::UnitTag::temperature_eV),
              1.0 / electron_rest_energy_eV,
              1e-14 / electron_rest_energy_eV);
}

TEST(Normalization, PlasmaRejectsUnrepresentableDerivedScales) {
  using quasar::Normalization;
  const double denorm = std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(Normalization::plasma(denorm, denorm, 1.0),
               std::overflow_error);
  EXPECT_THROW(Normalization::plasma(std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max(), denorm),
               std::overflow_error);
}

TEST(Normalization, ExponentScalingAvoidsFalseIntermediateRangeFailures) {
  // Both cases have well-represented final plasma and conversion scales, but a
  // double-only implementation respectively underflows q*q and overflows
  // m*c before the compensating division.  This regression is intentionally
  // independent of whether the host gives long double a wider exponent range.
  for (const auto& norm : {
           quasar::Normalization::plasma(1.0e200, 1.0e-200, 1.0e-200),
           quasar::Normalization::plasma(1.0e-200, 1.0e200, 1.0e200)}) {
    EXPECT_TRUE(std::isfinite(norm.omega_p_ref));
    EXPECT_GT(norm.omega_p_ref, 0.0);
    EXPECT_TRUE(std::isfinite(norm.length_scale()));
    EXPECT_TRUE(std::isfinite(norm.time_scale()));
    EXPECT_TRUE(std::isfinite(norm.e_field_scale()));
    EXPECT_TRUE(std::isfinite(norm.b_field_scale()));
    EXPECT_TRUE(std::isfinite(norm.temperature_eV_scale()));
  }
}

TEST(Normalization, SignedChargeKeepsSignedFieldConvention) {
  const auto positive = quasar::Normalization::plasma(1.0e18, 2.0, 3.0);
  const auto negative = quasar::Normalization::plasma(1.0e18, -2.0, 3.0);
  EXPECT_DOUBLE_EQ(negative.omega_p_ref, positive.omega_p_ref);
  EXPECT_DOUBLE_EQ(negative.e_field_scale(), -positive.e_field_scale());
  EXPECT_DOUBLE_EQ(negative.b_field_scale(), -positive.b_field_scale());
  EXPECT_DOUBLE_EQ(negative.temperature_eV_scale(),
                   positive.temperature_eV_scale());
}
