#include "quasar/numerics/deposit.hpp"

#include <gtest/gtest.h>

TEST(PicChargeConservation, DepositTypesExist) {
  quasar::numerics::Esirkepov2D<1> cic;
  quasar::numerics::Esirkepov2D<2> tsc;
  (void)cic;
  (void)tsc;
  SUCCEED();
}
