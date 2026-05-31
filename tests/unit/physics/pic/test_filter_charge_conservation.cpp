#include "quasar/numerics/filter.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/grid.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(PicFilterChargeConservation, PipelineCanOwnFilters) {
  quasar::numerics::FilterPipeline pipeline;
  pipeline.add(std::make_unique<quasar::numerics::BinomialFilter>(1));
  EXPECT_EQ(pipeline.size(), 1u);
}

TEST(PicFilterChargeConservation, RejectsNonPositivePassCount) {
  EXPECT_THROW(quasar::numerics::BinomialFilter{0}, std::invalid_argument);
  EXPECT_THROW(quasar::numerics::CompensatedBinomialFilter{0}, std::invalid_argument);
}

TEST(PicFilterChargeConservation, NonPeriodicAxisDoesNotWrapCurrentAcrossEdge) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::JField2D<quasar::Real> current{g};
  std::vector<quasar::Real> host(g.storage_size(), 0.0);
  host[g.index(0, 4)] = 1.0;
  current.jz.copy_from_host(host.data(), host.size());

  quasar::boundary::BoundarySpec bc;
  bc.particle[0] = "specular";
  bc.particle[1] = "specular";
  bc.field[0] = "pec";
  bc.field[1] = "pec";

  quasar::numerics::BinomialFilter filter{1};
  filter.apply(current, bc);

  std::vector<quasar::Real> filtered(g.storage_size(), 0.0);
  current.jz.copy_to_host(filtered.data(), filtered.size());
  EXPECT_NEAR(filtered[g.index(g.nx - 1, 4)], 0.0, 1.0e-14)
      << "non-periodic x filter wrapped current to the far edge";
  EXPECT_GT(filtered[g.index(1, 4)], 0.0);
}
