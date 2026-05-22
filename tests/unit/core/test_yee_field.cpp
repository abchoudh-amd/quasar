#include "quasar/backend/device.hpp"
#include "quasar/core/yee_field.hpp"

#include <gtest/gtest.h>

TEST(YeeField2D, AllocatesSixComponents) {
  if (!quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const quasar::Grid2D g{4, 3, 1.0, 2.0, 0.0, 0.0, 2};
  quasar::YeeField2D<double> fields{g};
  EXPECT_EQ(fields.component_size(), g.storage_size());
  EXPECT_EQ(fields.total_values(), 6 * g.storage_size());
  quasar::JField2D<double> current{g};
  EXPECT_EQ(current.total_values(), 3 * g.storage_size());
}
