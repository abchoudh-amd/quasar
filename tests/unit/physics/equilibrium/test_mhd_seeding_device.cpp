// Device-backed end-to-end coverage for equilibrium-to-MHD projection. The
// host-only validation and topology tests live in test_mhd_seeding.cpp so they
// continue to run on machines without a visible HIP device.

#include "quasar/backend/device.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/mhd_seeding.hpp"

#include <gtest/gtest.h>

#include <memory>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::EllipticGrid;
using namespace quasar::equilibrium;

EllipticGrid gs_grid() {
  return EllipticGrid{33, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
}

Grid2D mhd_grid() {
  constexpr int n = 64;
  return Grid2D::from_cell_spacing(n, n, Real{1.6} / static_cast<Real>(n),
                                   Real{1.6} / static_cast<Real>(n),
                                   Real{0.3}, Real{-0.8}, 4);
}

}  // namespace

TEST(MhdSeedingDevice, EndToEndFromASolvedEquilibrium) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  GsConfig cfg;
  cfg.grid = gs_grid();
  cfg.coils = {
      {Real{2.4}, Real{0.9},  Real{-3.0e5}},
      {Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      {Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1.0e6};
  cfg.max_iterations = 400;
  cfg.tolerance = Real{1e-9};

  auto prof = std::make_shared<PolynomialProfile>();
  GsSolver solver{cfg, prof};
  const GsResult r = solver.solve();
  ASSERT_EQ(r.status, GsStatus::converged);

  const Grid2D mg = mhd_grid();
  const auto sb =
      project_to_mhd(cfg.grid, r.psi, r.critical, mg, [](Real) { return Real{5}; });

  const Real scale = field_scale(sb);
  ASSERT_GT(scale, Real{0});
  EXPECT_LT(max_divergence(sb) / scale, 1e-11)
      << "a solved equilibrium must project to a solenoidal MHD background";

  // The projected poloidal field must be non-trivial: a silently zero field
  // would pass every divergence check.
  EXPECT_GT(scale, Real{1e-3}) << "projected field is implausibly weak";
}
