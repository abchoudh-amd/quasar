#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/numerics/filter.hpp"

#include <gtest/gtest.h>

TEST(CurrentFilter, EmptyPipelineIsNoop) {
  if (!quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const quasar::Grid2D g{4, 4, 1.0, 1.0};
  quasar::JField2D<double> j{g};
  quasar::boundary::BoundarySpec bc{};
  quasar::numerics::FilterPipeline pipeline;
  EXPECT_TRUE(pipeline.empty());
  pipeline.apply(j, bc);
}
