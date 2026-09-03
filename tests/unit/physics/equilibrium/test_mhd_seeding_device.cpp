// Device-backed end-to-end coverage for equilibrium-to-MHD projection. The
// host-only validation and topology tests live in test_mhd_seeding.cpp so they
// continue to run on machines without a visible HIP device.

#include "quasar/backend/device.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"
#include "quasar/physics/equilibrium/mhd_seeding.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

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

// The device projection must agree with the host reference it replaces.
//
// The relationship here is the same one the rest of this module already has
// with free_boundary.hpp and flux_surfaces.hpp: the host form stays as the
// oracle, and the kernel is checked against it. The module is compiled
// -ffp-contract=off, and the projection is a chain of differences, divisions and
// a Horner evaluation with no reassociation, so this is held to an EQUALITY
// rather than a tolerance -- the same standard launch_gs_apply_l6 is held to.
TEST(MhdSeedingDevice, ProjectionMatchesTheHostReference) {
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

  // Lower the SAME profile the host lambda will evaluate, so the comparison is
  // of the projection and not of two different F(psi_N).
  const ProfileCoefficients coefficients = to_coefficients(*prof);
  const auto f_of_psi_n = [&](Real pn) {
    Real acc = Real{0};
    for (int k = coefficients.n_f; k-- > 0;) {
      acc = acc * pn + coefficients.f_coeffs[k];
    }
    return acc;
  };

  const Grid2D mg = mhd_grid();
  const StaggeredBackground host = project_to_mhd(cfg.grid, r.psi, r.critical,
                                                  mg, f_of_psi_n);

  // Device side: upload psi, build the mask with the existing device flood
  // fill, then project.
  const std::size_t source_size = cfg.grid.size();
  quasar::backend::DeviceBuffer<Real> d_psi{source_size};
  d_psi.copy_from_host(r.psi.data(), source_size);

  GsOperatorScratch operator_scratch{cfg.grid};
  GsDerivativeFields derivatives{cfg.grid};
  launch_gs_compute_derivatives(cfg.grid, d_psi.device_ptr(), derivatives,
                                operator_scratch, nullptr);
  GsPlasmaMaskScratch mask_scratch{cfg.grid};
  launch_gs_build_plasma_mask(cfg.grid, d_psi.device_ptr(), derivatives,
                              r.critical.axis.r, r.critical.axis.z,
                              r.critical.psi_axis, r.critical.psi_boundary,
                              mask_scratch, nullptr);

  GsProjectionSpec spec;
  spec.source = cfg.grid;
  spec.target = mg;
  spec.psi_axis = r.critical.psi_axis;
  spec.psi_boundary = r.critical.psi_boundary;
  spec.profile = coefficients;

  const std::size_t target_size = mg.storage_size();
  quasar::backend::DeviceBuffer<Real> b0x{target_size};
  quasar::backend::DeviceBuffer<Real> b0y{target_size};
  quasar::backend::DeviceBuffer<Real> b0z{target_size};
  quasar::backend::DeviceBuffer<int> status{1};
  const int zero = 0;
  status.copy_from_host(&zero, 1);

  launch_gs_project_to_mhd(spec, d_psi.device_ptr(),
                           mask_scratch.mask.device_ptr(), b0x, b0y, b0z,
                           status.device_ptr(), nullptr);
  int host_status = 1;
  status.copy_to_host(&host_status, 1);
  quasar::backend::device_synchronize(nullptr);
  ASSERT_EQ(host_status, 0) << "device projection reported a failure status";

  std::vector<Real> got_x(target_size), got_y(target_size), got_z(target_size);
  b0x.copy_to_host(got_x.data(), target_size);
  b0y.copy_to_host(got_y.data(), target_size);
  b0z.copy_to_host(got_z.data(), target_size);
  quasar::backend::device_synchronize(nullptr);

  for (std::size_t k = 0; k < target_size; ++k) {
    ASSERT_EQ(got_x[k], host.b0x_face[k]) << "b0x differs at " << k;
    ASSERT_EQ(got_y[k], host.b0y_face[k]) << "b0y differs at " << k;
    ASSERT_EQ(got_z[k], host.b0z_cell[k]) << "b0z differs at " << k;
  }

  // The acceptance criterion must agree too: max is associative and rounds
  // nowhere, so a tree and a sequential scan give the same value exactly.
  const Real device_divergence =
      launch_gs_projection_max_divergence(mg, b0x, b0y, nullptr);
  EXPECT_EQ(device_divergence, max_divergence(host));

  // The fluid projection shares the profile-coordinate path, so it is checked
  // against its own host form here rather than in a second test that would pay
  // for another equilibrium solve.
  constexpr Real kRhoAxis = Real{2.5};
  constexpr Real kRhoEdge = Real{0.25};
  const auto p_of_psi_n = [&](Real pn) {
    Real acc = Real{0};
    for (int k = coefficients.n_p; k-- > 0;) {
      acc = acc * pn + coefficients.p_coeffs[k];
    }
    return acc;
  };
  const FluidSeed host_fluid = project_fluid(cfg.grid, r.psi, r.critical, mg,
                                             p_of_psi_n, kRhoEdge, kRhoAxis);

  quasar::backend::DeviceBuffer<Real> rho{target_size};
  quasar::backend::DeviceBuffer<Real> pressure{target_size};
  status.copy_from_host(&zero, 1);
  launch_gs_project_fluid(spec, d_psi.device_ptr(),
                          mask_scratch.mask.device_ptr(), kRhoAxis, kRhoEdge,
                          rho, pressure, status.device_ptr(), nullptr);
  host_status = 1;
  status.copy_to_host(&host_status, 1);
  quasar::backend::device_synchronize(nullptr);
  ASSERT_EQ(host_status, 0) << "device fluid projection reported a failure";

  std::vector<Real> got_rho(target_size), got_pressure(target_size);
  rho.copy_to_host(got_rho.data(), target_size);
  pressure.copy_to_host(got_pressure.data(), target_size);
  quasar::backend::device_synchronize(nullptr);
  for (std::size_t k = 0; k < target_size; ++k) {
    ASSERT_EQ(got_rho[k], host_fluid.rho[k]) << "rho differs at " << k;
    ASSERT_EQ(got_pressure[k], host_fluid.pressure[k])
        << "pressure differs at " << k;
  }
}
