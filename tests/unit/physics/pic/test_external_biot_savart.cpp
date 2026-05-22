#include "quasar/physics/magnetostatics/biot_savart.hpp"

#include <gtest/gtest.h>

#include <type_traits>

TEST(PicExternalField, BiotSavartImplementsFieldEvaluator) {
  EXPECT_TRUE((std::is_base_of_v<quasar::numerics::IFieldEvaluator,
                                quasar::magnetostatics::BiotSavartEvaluator>));
}
