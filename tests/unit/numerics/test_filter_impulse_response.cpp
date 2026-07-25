#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/numerics/filter.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

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

TEST(CurrentFilter, CylindricalJphiIncludesPhysicalHighRadialFace) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const quasar::Grid2D g{8, 6, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::JField2D<double> j{g};
  std::vector<double> jphi(g.storage_size(), 0.0);
  const int row = 3;
  jphi[g.index(g.nx, row)] = 1.0;
  j.jz.copy_from_host(jphi.data(), jphi.size());

  quasar::boundary::BoundarySpec bc{};
  bc.field[0] = "axis";
  bc.field[1] = "pec";
  bc.particle[0] = "axis";
  bc.particle[1] = "specular";
  quasar::numerics::FilterPipeline pipeline;
  pipeline.add(std::make_unique<quasar::numerics::BinomialFilter>(1));
  pipeline.apply(j, bc, /*cylindrical=*/true);
  j.jz.copy_to_host(jphi.data(), jphi.size());

  EXPECT_GT(jphi[g.index(g.nx, row)], 0.0);
  EXPECT_LT(jphi[g.index(g.nx, row)], 1.0);
  EXPECT_GT(jphi[g.index(g.nx - 1, row)], 0.0)
      << "high-r Jphi face was omitted from the cylindrical filter extent";
  EXPECT_DOUBLE_EQ(jphi[g.index(0, row)], 0.0)
      << "the odd r=0 Jphi regularity condition must survive smoothing";
}
