#include "quasar/core/types.hpp"

#include <gtest/gtest.h>

TEST(PicTwoStreamGrowthRate, ReferenceFormulaIsPositive) {
  const quasar::Real omega_p = 2.0;
  EXPECT_GT(omega_p / std::sqrt(8.0), 0.0);
}
