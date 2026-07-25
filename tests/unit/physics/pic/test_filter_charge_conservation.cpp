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

TEST(PicFilterChargeConservation, InPlaneContinuityCurrentIsBitwiseUnchanged) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::JField2D<quasar::Real> current{g};
  std::vector<quasar::Real> jx(g.storage_size(), 0.0);
  std::vector<quasar::Real> jy(g.storage_size(), 0.0);
  std::vector<quasar::Real> jz(g.storage_size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      jx[g.index(i, j)] = 0.13 * i - 0.07 * j;
      jy[g.index(i, j)] = 0.11 * i + 0.05 * j;
      jz[g.index(i, j)] = (i == 3 && j == 4) ? 1.0 : 0.0;
    }
  }
  current.jx.copy_from_host(jx.data(), jx.size());
  current.jy.copy_from_host(jy.data(), jy.size());
  current.jz.copy_from_host(jz.data(), jz.size());

  quasar::boundary::BoundarySpec bc;
  quasar::numerics::CompensatedBinomialFilter filter{3};
  filter.apply(current, bc);

  std::vector<quasar::Real> after_x(g.storage_size());
  std::vector<quasar::Real> after_y(g.storage_size());
  current.jx.copy_to_host(after_x.data(), after_x.size());
  current.jy.copy_to_host(after_y.data(), after_y.size());
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const auto k = g.index(i, j);
      EXPECT_DOUBLE_EQ(after_x[k], jx[k]);
      EXPECT_DOUBLE_EQ(after_y[k], jy[k]);
    }
  }
}

TEST(PicFilterChargeConservation, OnePassCompensatorHasCorrectLowKCoefficients) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{9, 9, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::JField2D<quasar::Real> current{g};
  std::vector<quasar::Real> impulse(g.storage_size(), 0.0);
  impulse[g.index(4, 4)] = 1.0;
  current.jz.copy_from_host(impulse.data(), impulse.size());
  quasar::boundary::BoundarySpec bc;
  quasar::numerics::CompensatedBinomialFilter filter{1};
  filter.apply(current, bc);

  std::vector<quasar::Real> result(g.storage_size());
  current.jz.copy_to_host(result.data(), result.size());
  EXPECT_NEAR(result[g.index(4, 4)], 0.625 * 0.625, 1.0e-14);
  EXPECT_NEAR(result[g.index(5, 4)], 0.25 * 0.625, 1.0e-14);
  EXPECT_NEAR(result[g.index(6, 4)], -0.0625 * 0.625, 1.0e-14);
}
