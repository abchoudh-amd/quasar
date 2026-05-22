#include "quasar/core/types.hpp"

#include <gtest/gtest.h>

TEST(PicFp32VsFp64, TypesExist) {
  quasar::Vec3 a{1, 2, 3};
  quasar::Vec3f b{1.0f, 2.0f, 3.0f};
  EXPECT_DOUBLE_EQ(a.x, static_cast<double>(b.x));
}
