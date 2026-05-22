#include "quasar/numerics/filter.hpp"

#include <gtest/gtest.h>

TEST(PicFilterTwoStream, FilterTypesConstruct) {
  quasar::numerics::BinomialFilter b{4};
  quasar::numerics::CompensatedBinomialFilter c{1};
  EXPECT_EQ(b.passes(), 4);
  EXPECT_EQ(c.passes(), 1);
}
