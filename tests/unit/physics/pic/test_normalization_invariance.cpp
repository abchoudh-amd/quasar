#include "quasar/core/normalization.hpp"

#include <gtest/gtest.h>

TEST(PicNormalizationInvariance, SiAndInternalRoundTrip) {
  const auto norm = quasar::Normalization::plasma(
      1e18, -quasar::constants::qe_abs, quasar::constants::me);
  const double length = 2.0e-3;
  EXPECT_NEAR(norm.to_si(norm.to_internal(length, quasar::UnitTag::length),
                         quasar::UnitTag::length),
              length, 1e-15);
}
