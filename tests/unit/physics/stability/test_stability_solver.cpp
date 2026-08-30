#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/stability/stability_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using quasar::Real;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::CriticalKind;
using quasar::equilibrium::CriticalPoint;
using quasar::equilibrium::GsConfig;
using quasar::equilibrium::GsDeviceResult;
using quasar::equilibrium::GsFluxSurfaces;
using quasar::equilibrium::GsMagneticField;
using quasar::equilibrium::GsSolver;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;
using quasar::stability::ModeSolveStatus;
using quasar::stability::ChebyshevBasis;
using quasar::stability::FluxCoordinateGrid;
using quasar::stability::StabilityClassification;
using quasar::stability::StabilityConfig;
using quasar::stability::StabilityProfiles;
using quasar::stability::StabilitySolver;
using quasar::stability::StabilityStatus;

StabilityProfiles constant_profiles() {
  StabilityProfiles profiles;
  profiles.f_vacuum = Real{2};
  profiles.density.count = 1;
  profiles.density.coefficients[0] = Real{1};
  return profiles;
}

StabilityConfig small_config() {
  StabilityConfig config;
  config.toroidal_modes = {1, -1};
  config.lambda_inner = Real{0.1};
  config.lambda_outer = Real{0.6};
  config.q_probe_count = 9;
  config.contour_points = 512;
  config.minimum_radial_domains = 1;
  config.chebyshev_order = 3;
  config.m_max = 0;
  config.n_theta = 16;
  config.f_profile_samples = 65;
  config.minimum_domain_width = Real{1e-3};
  // This smoke fixture deliberately uses a tiny contour/Fourier grid.  Its
  // independent signed-geometry residual is about 9e-2; a reversed
  // orientation remains O(2) and is still decisively rejected.
  config.assembly.field_pitch_tolerance = Real{1e-1};
  config.assembly.flux_scale_tolerance = Real{1e-1};
  config.assembly.pest_tolerance = Real{1e-1};
  config.eigenvalue_relative_tolerance = Real{1e-9};
  config.maximum_mass_digits_lost = Real{14};
  return config;
}

GsDeviceResult circular_device_equilibrium() {
  GsDeviceResult result;
  result.status = GsStatus::converged;
  result.grid = EllipticGrid{33, 33, Real{1}, Real{3}, Real{-1}, Real{1}};
  result.psi = quasar::backend::DeviceBuffer<Real>{result.grid.size()};
  result.j_phi = quasar::backend::DeviceBuffer<Real>{result.grid.size()};

  std::vector<Real> psi(result.grid.size());
  for (int j = 0; j < result.grid.nz; ++j) {
    const Real z = result.grid.z(j);
    for (int i = 0; i < result.grid.nr; ++i) {
      const Real r = result.grid.r(i);
      const Real minor_squared =
          (r - Real{2}) * (r - Real{2}) + z * z;
      psi[result.grid.index(i, j)] = Real{1} - minor_squared;
    }
  }
  result.psi.copy_from_host(psi.data(), psi.size());
  result.critical.axis = CriticalPoint{
      CriticalKind::o_point, Real{2}, Real{0}, Real{1}, true};
  result.critical.psi_axis = Real{1};
  result.critical.psi_boundary = Real{0};
  result.critical.has_closed_surface = true;
  result.profile_scale = Real{1};
  result.profile_coefficients.n_p = 1;
  result.profile_coefficients.p_coeffs[0] = Real{0};
  result.profile_coefficients.n_f = 1;
  result.profile_coefficients.f_coeffs[0] = Real{0};
  return result;
}

GsDeviceResult metadata_only_device_equilibrium() {
  GsDeviceResult result;
  result.status = GsStatus::converged;
  result.grid = EllipticGrid{3, 3, Real{1}, Real{3}, Real{-1}, Real{1}};
  result.critical.axis = CriticalPoint{
      CriticalKind::o_point, Real{2}, Real{0}, Real{1}, true};
  result.critical.psi_axis = Real{1};
  result.critical.psi_boundary = Real{0};
  result.critical.has_closed_surface = true;
  result.profile_scale = Real{1};
  result.profile_coefficients.n_p = 1;
  result.profile_coefficients.p_coeffs[0] = Real{0};
  result.profile_coefficients.n_f = 1;
  result.profile_coefficients.f_coeffs[0] = Real{0};
  return result;
}

GsDeviceResult solved_gs_device_equilibrium() {
  GsConfig config;
  config.grid = EllipticGrid{33, 33, Real{0.3}, Real{1.9},
                             Real{-0.8}, Real{0.8}};
  config.coils = {
      CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
      CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  config.plasma_current = Real{1e6};
  config.max_iterations = 400;
  config.tolerance = Real{1e-9};
  auto profile = std::make_shared<PolynomialProfile>(
      std::vector<Real>{Real{1}, Real{-1}},
      std::vector<Real>{Real{1}, Real{-1}});
  return GsSolver{config, std::move(profile)}.solve_device();
}

TEST(StabilitySolver, RejectsMalformedConfiguration) {
  StabilityConfig config = small_config();
  config.toroidal_modes = {0};
  EXPECT_THROW((void)StabilitySolver(config, constant_profiles()),
               std::invalid_argument);

  config = small_config();
  config.lambda_inner = Real{0};
  EXPECT_THROW((void)StabilitySolver(config, constant_profiles()),
               std::invalid_argument);

  config = small_config();
  config.n_theta = 0;
  EXPECT_THROW((void)StabilitySolver(config, constant_profiles()),
               std::invalid_argument);

  StabilityProfiles profiles = constant_profiles();
  profiles.density.count = 0;
  EXPECT_THROW((void)StabilitySolver(small_config(), profiles),
               std::invalid_argument);
}

TEST(StabilitySolver, RejectsExtremeResolutionWithoutSignedOverflow) {
  StabilityConfig config = small_config();
  config.m_max = std::numeric_limits<int>::max();
  config.n_theta = std::numeric_limits<int>::max();
  EXPECT_THROW((void)StabilitySolver(config, constant_profiles()),
               std::invalid_argument);

  config = small_config();
  config.minimum_radial_domains = 2;
  config.chebyshev_order = std::numeric_limits<int>::max() / 2;
  EXPECT_THROW((void)StabilitySolver(config, constant_profiles()),
               std::length_error);
}

TEST(StabilityStorage, RejectsInvalidShapesBeforeDeviceAllocation) {
  ChebyshevBasis basis;
  EXPECT_THROW(basis.resize(-1, 1), std::invalid_argument);
  EXPECT_THROW(basis.resize(1, -1), std::invalid_argument);
  EXPECT_THROW(basis.resize(0, 1), std::invalid_argument);
  EXPECT_THROW(basis.resize(1, 0), std::invalid_argument);
  EXPECT_EQ(basis.order, 0);
  EXPECT_EQ(basis.n_nodes, 0);
  EXPECT_EQ(basis.n_domains, 0);

  EXPECT_THROW(basis.resize(std::numeric_limits<int>::max(), 1),
               std::length_error);
  EXPECT_THROW(basis.resize(std::numeric_limits<int>::max() - 1,
                            std::numeric_limits<int>::max()),
               std::length_error);

  FluxCoordinateGrid coordinates;
  EXPECT_THROW(coordinates.resize(-1, 1), std::invalid_argument);
  EXPECT_THROW(coordinates.resize(1, -1), std::invalid_argument);
  EXPECT_EQ(coordinates.n_psi, 0);
  EXPECT_EQ(coordinates.n_theta, 0);
  EXPECT_THROW(coordinates.resize(std::numeric_limits<int>::max(),
                                  std::numeric_limits<int>::max()),
               std::length_error);
}

TEST(StabilityStorage, RejectsUnrepresentableSpectralBasisMetadataOnHost) {
  ChebyshevBasis basis;
  basis.order = std::numeric_limits<int>::max();
  basis.n_nodes = std::numeric_limits<int>::min();
  basis.n_domains = 2;

  const EllipticGrid grid{4, 4, Real{1}, Real{2}, Real{-1}, Real{1}};
  GsFluxSurfaces surfaces;
  GsMagneticField field;
  FluxCoordinateGrid coordinates;
  EXPECT_THROW(quasar::stability::launch_build_spectral_flux_coordinates(
                   grid, surfaces, field, basis, coordinates, nullptr),
               std::invalid_argument);
}

TEST(StabilitySolver, PropagatesFailedEquilibriumWithoutLaunchingKernels) {
  GsDeviceResult equilibrium;
  equilibrium.status = GsStatus::residual_stalled;
  const StabilitySolver solver{small_config(), constant_profiles()};

  const auto scan = solver.solve(equilibrium);
  EXPECT_EQ(scan.status, StabilityStatus::equilibrium_not_converged);
  EXPECT_EQ(scan.equilibrium_status, GsStatus::residual_stalled);
  EXPECT_TRUE(scan.scanned_n.empty());
  EXPECT_TRUE(scan.modes.empty());
  EXPECT_EQ(scan.classification, StabilityClassification::unresolved);
  EXPECT_FALSE(scan.ok());

  const auto mode = solver.solve_mode(equilibrium, 1);
  EXPECT_EQ(mode.summary.status,
            ModeSolveStatus::equilibrium_not_converged);
  EXPECT_FALSE(mode.summary.assembly.has_value());
  EXPECT_FALSE(mode.summary.stiffness_condition.has_value());
  EXPECT_FALSE(mode.summary.inertia_condition.has_value());
  EXPECT_FALSE(mode.summary.eigen_status.has_value());
}

TEST(StabilitySolver,
     RejectsMalformedEquilibriumMetadataBeforeLaunchingKernels) {
  const StabilitySolver solver{small_config(), constant_profiles()};
  const auto expect_rejected = [&](auto mutate, std::string_view reason) {
    GsDeviceResult equilibrium = metadata_only_device_equilibrium();
    mutate(equilibrium);
    try {
      (void)solver.solve_mode(equilibrium, 1);
      ADD_FAILURE() << "malformed equilibrium metadata was accepted";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string_view{error.what()}.find(reason),
                std::string_view::npos)
          << error.what();
    }
  };

  expect_rejected([](GsDeviceResult& value) { value.critical.axis.valid = false; },
                  "magnetic axis");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.axis.kind = CriticalKind::x_point;
      },
      "magnetic axis");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.axis.r = std::numeric_limits<Real>::quiet_NaN();
      },
      "magnetic axis");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.axis.z = std::numeric_limits<Real>::infinity();
      },
      "magnetic axis");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.axis.psi =
            std::numeric_limits<Real>::quiet_NaN();
      },
      "magnetic axis");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.axis.r = value.grid.r_min;
      },
      "magnetic axis");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.has_closed_surface = false;
      },
      "closed flux surface");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.critical_point_overflow = true;
      },
      "closed flux surface");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.psi_axis =
            std::numeric_limits<Real>::quiet_NaN();
      },
      "flux normalization");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.psi_boundary =
            std::numeric_limits<Real>::infinity();
      },
      "flux normalization");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.axis.psi = value.critical.psi_axis + Real{1};
      },
      "magnetic-axis flux");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.psi_boundary = value.critical.psi_axis;
      },
      "normalization span");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.critical.psi_axis = std::numeric_limits<Real>::max();
        value.critical.axis.psi = value.critical.psi_axis;
        value.critical.psi_boundary = -std::numeric_limits<Real>::max();
      },
      "normalization span");
  expect_rejected(
      [](GsDeviceResult& value) {
        value.profile_scale = std::numeric_limits<Real>::quiet_NaN();
      },
      "profile scale");
}

TEST(StabilitySolver, SolvesAndScansAnAxisExcludedCircularDeviceEquilibrium) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  GsDeviceResult equilibrium = circular_device_equilibrium();
  const StabilityConfig config = small_config();
  const StabilitySolver solver{config, constant_profiles()};

  auto one = solver.solve_mode(equilibrium, 1);
  ASSERT_EQ(one.summary.status, ModeSolveStatus::solved)
      << "q=" << one.summary.geometry.maximum_q_relative_deviation
      << " S="
      << one.summary.geometry.maximum_flux_scale_relative_deviation
      << " pest="
      << one.summary.geometry.maximum_pest_j_over_r2_relative_deviation
      << " invalid=" << one.summary.geometry.invalid_surface_count;
  EXPECT_TRUE(one.summary.geometry.ok());
  ASSERT_TRUE(one.summary.assembly.has_value());
  EXPECT_TRUE(one.summary.assembly->ok());
  EXPECT_TRUE(one.summary.stiffness_condition.has_value());
  EXPECT_TRUE(one.summary.inertia_condition.has_value());
  EXPECT_TRUE(one.summary.eigen_status.has_value());
  EXPECT_GT(one.summary.real_order, 0);
  EXPECT_EQ(one.summary.lambda_inner, config.lambda_inner);
  EXPECT_EQ(one.summary.lambda_outer, config.lambda_outer);
  EXPECT_EQ(one.eigensystem.order, one.summary.real_order);
  EXPECT_EQ(one.radial_nodes.size(),
            static_cast<std::size_t>(config.chebyshev_order + 1)
                * one.summary.radial_domains.n_domains);
  EXPECT_TRUE(std::isfinite(one.summary.minimum_omega_squared));
  EXPECT_TRUE(std::isfinite(one.summary.maximum_absolute_omega_squared));
  EXPECT_TRUE(std::isfinite(one.summary.growth_rate));
  EXPECT_TRUE(std::isfinite(one.summary.eigenvalue_resolution_threshold));

  const auto scan = solver.solve(equilibrium);
  ASSERT_EQ(scan.status, StabilityStatus::complete);
  ASSERT_EQ(scan.modes.size(), 2u);
  EXPECT_EQ(scan.scanned_n, config.toroidal_modes);
  EXPECT_EQ(scan.lambda_inner, config.lambda_inner);
  EXPECT_EQ(scan.lambda_outer, config.lambda_outer);
  EXPECT_EQ(scan.modes[0].n_toroidal, 1);
  EXPECT_EQ(scan.modes[1].n_toroidal, -1);
  ASSERT_TRUE(scan.modes[0].solved());
  ASSERT_TRUE(scan.modes[1].solved());
  EXPECT_NEAR(scan.modes[0].minimum_omega_squared,
              scan.modes[1].minimum_omega_squared,
              Real{1e-8}
                  * (Real{1}
                     + std::abs(scan.modes[0].minimum_omega_squared)));
  EXPECT_TRUE(std::isfinite(scan.aggregate_margin));
  EXPECT_TRUE(std::isfinite(scan.maximum_growth_rate));
  EXPECT_TRUE(scan.worst_n == 1 || scan.worst_n == -1);
}

TEST(StabilitySolver, MapsPhysicalEquilibriumValidationFailureToModeStatus) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  GsDeviceResult equilibrium = circular_device_equilibrium();
  StabilityProfiles profiles = constant_profiles();
  profiles.density.coefficients[0] = Real{-1};
  const StabilitySolver solver{small_config(), profiles};

  const auto result = solver.solve_mode(equilibrium, 1);
  EXPECT_EQ(result.summary.status,
            ModeSolveStatus::equilibrium_fields_failed);
  EXPECT_FALSE(result.summary.assembly.has_value());
  EXPECT_FALSE(result.summary.stiffness_condition.has_value());
  EXPECT_FALSE(result.summary.inertia_condition.has_value());
  EXPECT_FALSE(result.summary.eigen_status.has_value());
}

TEST(StabilitySolver, PropagatesMalformedEquilibriumContracts) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  GsDeviceResult equilibrium = circular_device_equilibrium();
  equilibrium.j_phi = quasar::backend::DeviceBuffer<Real>{1};
  const StabilitySolver solver{small_config(), constant_profiles()};

  EXPECT_THROW((void)solver.solve_mode(equilibrium, 1), std::invalid_argument);
}

TEST(StabilitySolver, SolvesFromAConvergedGradShafranovDeviceResult) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  GsDeviceResult equilibrium = solved_gs_device_equilibrium();
  ASSERT_EQ(equilibrium.status, GsStatus::converged);
  ASSERT_EQ(equilibrium.profile_coefficients.n_p, 2);
  ASSERT_EQ(equilibrium.profile_coefficients.n_f, 2);
  EXPECT_EQ(equilibrium.profile_coefficients.p_coeffs[0], Real{1});
  EXPECT_EQ(equilibrium.profile_coefficients.p_coeffs[1], Real{-1});
  EXPECT_EQ(equilibrium.profile_coefficients.f_coeffs[0], Real{1});
  EXPECT_EQ(equilibrium.profile_coefficients.f_coeffs[1], Real{-1});

  StabilityConfig config = small_config();
  config.toroidal_modes = {1};
  config.lambda_inner = Real{0.2};
  config.lambda_outer = Real{0.6};
  config.m_max = 1;
  config.n_theta = 16;

  StabilityProfiles profiles = constant_profiles();
  profiles.f_vacuum = Real{5};
  const StabilitySolver solver{config, profiles};
  const auto result = solver.solve_mode(equilibrium, 1);

  ASSERT_EQ(result.summary.status, ModeSolveStatus::solved)
      << "q=" << result.summary.geometry.maximum_q_relative_deviation
      << " S="
      << result.summary.geometry.maximum_flux_scale_relative_deviation
      << " pest="
      << result.summary.geometry.maximum_pest_j_over_r2_relative_deviation
      << " invalid=" << result.summary.geometry.invalid_surface_count;
  EXPECT_EQ(result.summary.rational_surfaces.count, 0);
  EXPECT_GT(result.summary.real_order, 0);
  EXPECT_NE(result.summary.classification,
            StabilityClassification::unresolved);
  EXPECT_TRUE(result.summary.assembly.has_value());
  ASSERT_TRUE(result.summary.stiffness_condition.has_value());
  ASSERT_TRUE(result.summary.inertia_condition.has_value());
  EXPECT_TRUE(result.summary.eigen_status.has_value());
  EXPECT_TRUE(result.summary.stiffness_condition->ok());
  EXPECT_TRUE(result.summary.inertia_condition->ok());
  EXPECT_TRUE(std::isfinite(result.summary.minimum_omega_squared));
  EXPECT_TRUE(std::isfinite(result.summary.maximum_absolute_omega_squared));
  EXPECT_TRUE(std::isfinite(result.summary.eigenvalue_resolution_threshold));
  EXPECT_EQ(result.eigensystem.order, result.summary.real_order);
}

TEST(StabilitySolver, ReportsDenseStoragePolicyFailure) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  GsDeviceResult equilibrium = circular_device_equilibrium();
  StabilityConfig config = small_config();
  config.toroidal_modes = {1};
  config.assembly.max_dense_storage_bytes = 1;
  const StabilitySolver solver{config, constant_profiles()};
  const auto result = solver.solve_mode(equilibrium, 1);
  EXPECT_EQ(result.summary.status, ModeSolveStatus::assembly_failed);
  ASSERT_TRUE(result.summary.assembly.has_value());
  EXPECT_TRUE(quasar::stability::has_status(
      result.summary.assembly->status,
      quasar::stability::ToroidalAssemblyStatus::storage_limit_exceeded));
}

}  // namespace
