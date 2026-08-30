#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceStream;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::CriticalKind;
using quasar::equilibrium::CriticalPoint;
using quasar::equilibrium::GsConfig;
using quasar::equilibrium::GsDeviceResult;
using quasar::equilibrium::GsResult;
using quasar::equilibrium::GsSolver;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;

static_assert(!std::is_copy_constructible_v<GsDeviceResult>);
static_assert(!std::is_copy_assignable_v<GsDeviceResult>);
static_assert(std::is_nothrow_move_constructible_v<GsDeviceResult>);
static_assert(std::is_nothrow_move_assignable_v<GsDeviceResult>);

GsConfig converging_config() {
  GsConfig cfg;
  cfg.grid = EllipticGrid{33, 33, Real{0.3}, Real{1.9},
                         Real{-0.8}, Real{0.8}};
  cfg.coils = {
      CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
      CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1e6};
  cfg.max_iterations = 400;
  cfg.tolerance = Real{1e-9};
  return cfg;
}

std::shared_ptr<quasar::equilibrium::IEquilibriumProfile> make_profile() {
  return std::make_shared<PolynomialProfile>(
      std::vector<Real>{Real{1}, Real{-1}},
      std::vector<Real>{Real{1}, Real{-1}});
}

std::size_t bitwise_mismatches(const std::vector<Real>& a,
                               const std::vector<Real>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(Real)) != 0) ++mismatches;
  }
  return mismatches;
}

Real peak_magnitude(const std::vector<Real>& field) {
  Real peak = Real{0};
  for (const Real value : field) peak = std::max(peak, std::abs(value));
  return peak;
}

void expect_same_metadata(const GsResult& a, const GsResult& b) {
  EXPECT_EQ(a.status, b.status);
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(a.residual, b.residual);
  EXPECT_EQ(a.residual_history, b.residual_history);
  EXPECT_EQ(a.critical.axis.kind, b.critical.axis.kind);
  EXPECT_EQ(a.critical.axis.r, b.critical.axis.r);
  EXPECT_EQ(a.critical.axis.z, b.critical.axis.z);
  EXPECT_EQ(a.critical.axis.psi, b.critical.axis.psi);
  EXPECT_EQ(a.critical.axis.valid, b.critical.axis.valid);
  EXPECT_EQ(a.critical.psi_axis, b.critical.psi_axis);
  EXPECT_EQ(a.critical.psi_boundary, b.critical.psi_boundary);
  EXPECT_EQ(a.critical.has_closed_surface, b.critical.has_closed_surface);
  EXPECT_EQ(a.critical.critical_point_overflow,
            b.critical.critical_point_overflow);
  ASSERT_EQ(a.critical.x_points.size(), b.critical.x_points.size());
  for (std::size_t i = 0; i < a.critical.x_points.size(); ++i) {
    EXPECT_EQ(a.critical.x_points[i].kind, b.critical.x_points[i].kind);
    EXPECT_EQ(a.critical.x_points[i].r, b.critical.x_points[i].r);
    EXPECT_EQ(a.critical.x_points[i].z, b.critical.x_points[i].z);
    EXPECT_EQ(a.critical.x_points[i].psi, b.critical.x_points[i].psi);
    EXPECT_EQ(a.critical.x_points[i].valid, b.critical.x_points[i].valid);
  }
  EXPECT_EQ(a.achieved_current, b.achieved_current);
  EXPECT_EQ(a.profile_scale, b.profile_scale);
  EXPECT_EQ(a.profile_coefficients.n_p, b.profile_coefficients.n_p);
  EXPECT_EQ(a.profile_coefficients.n_f, b.profile_coefficients.n_f);
  for (int k = 0; k < a.profile_coefficients.n_p; ++k) {
    EXPECT_EQ(a.profile_coefficients.p_coeffs[k],
              b.profile_coefficients.p_coeffs[k]);
  }
  for (int k = 0; k < a.profile_coefficients.n_f; ++k) {
    EXPECT_EQ(a.profile_coefficients.f_coeffs[k],
              b.profile_coefficients.f_coeffs[k]);
  }
  EXPECT_EQ(a.newton_steps, b.newton_steps);
}

TEST(GsDeviceResult, CopiesKnownDeviceFieldsAndMetadataWithoutASecondSolve) {
  GsDeviceResult device;
  device.status = GsStatus::residual_stalled;
  device.iterations = 17;
  device.residual = Real{0.0125};
  device.residual_history = {Real{1}, Real{0.25}, Real{0.0125}};
  device.grid = EllipticGrid{3, 3, Real{0.4}, Real{1.6},
                            Real{-0.2}, Real{0.2}};
  const std::vector<Real> expected_psi{
      Real{1.25}, Real{-2.5}, Real{3.75},
      Real{4.5}, Real{-5.25}, Real{6.0},
      Real{-6.75}, Real{7.5}, Real{-8.25}};
  const std::vector<Real> expected_j_phi{
      Real{-7}, Real{8}, Real{-9}, Real{10}, Real{-11}, Real{12},
      Real{-13}, Real{14}, Real{-15}};
  device.psi = quasar::backend::DeviceBuffer<Real>{expected_psi.size()};
  device.j_phi =
      quasar::backend::DeviceBuffer<Real>{expected_j_phi.size()};
  device.psi.copy_from_host(expected_psi.data(), expected_psi.size());
  device.j_phi.copy_from_host(expected_j_phi.data(), expected_j_phi.size());
  device.critical.axis = CriticalPoint{
      CriticalKind::o_point, Real{1.1}, Real{-0.03}, Real{-0.7}, true};
  device.critical.x_points = {
      CriticalPoint{CriticalKind::x_point, Real{1.5}, Real{0.18},
                    Real{-0.2}, true}};
  device.critical.psi_axis = Real{-0.7};
  device.critical.psi_boundary = Real{-0.2};
  device.critical.has_closed_surface = true;
  device.critical.critical_point_overflow = true;
  device.achieved_current = Real{9.5e5};
  device.profile_scale = Real{1.75};
  device.profile_coefficients.n_p = 2;
  device.profile_coefficients.p_coeffs[0] = Real{1.25};
  device.profile_coefficients.p_coeffs[1] = Real{-0.5};
  device.profile_coefficients.n_f = 1;
  device.profile_coefficients.f_coeffs[0] = Real{0.75};
  device.newton_steps = 3;

  DeviceStream stream;
  const GsResult host = device.copy_to_host(stream.get());
  EXPECT_EQ(host.status, device.status);
  EXPECT_EQ(host.iterations, device.iterations);
  EXPECT_EQ(host.residual, device.residual);
  EXPECT_EQ(host.residual_history, device.residual_history);
  EXPECT_EQ(host.psi, expected_psi);
  EXPECT_EQ(host.j_phi, expected_j_phi);
  EXPECT_EQ(host.critical.axis.kind, CriticalKind::o_point);
  EXPECT_EQ(host.critical.axis.r, Real{1.1});
  EXPECT_EQ(host.critical.axis.z, Real{-0.03});
  EXPECT_EQ(host.critical.axis.psi, Real{-0.7});
  ASSERT_EQ(host.critical.x_points.size(), 1u);
  EXPECT_EQ(host.critical.x_points[0].kind, CriticalKind::x_point);
  EXPECT_EQ(host.critical.x_points[0].r, Real{1.5});
  EXPECT_EQ(host.critical.x_points[0].z, Real{0.18});
  EXPECT_EQ(host.critical.x_points[0].psi, Real{-0.2});
  EXPECT_EQ(host.critical.psi_axis, Real{-0.7});
  EXPECT_EQ(host.critical.psi_boundary, Real{-0.2});
  EXPECT_TRUE(host.critical.has_closed_surface);
  EXPECT_TRUE(host.critical.critical_point_overflow);
  EXPECT_EQ(host.achieved_current, Real{9.5e5});
  EXPECT_EQ(host.profile_scale, Real{1.75});
  EXPECT_EQ(host.profile_coefficients.n_p, 2);
  EXPECT_EQ(host.profile_coefficients.p_coeffs[0], Real{1.25});
  EXPECT_EQ(host.profile_coefficients.p_coeffs[1], Real{-0.5});
  EXPECT_EQ(host.profile_coefficients.n_f, 1);
  EXPECT_EQ(host.profile_coefficients.f_coeffs[0], Real{0.75});
  EXPECT_EQ(host.newton_steps, 3);
}

TEST(GsDeviceResult, RejectsFieldsOwnedByDifferentDevices) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  if (quasar::backend::device_count() < 2) {
    GTEST_SKIP() << "requires two HIP devices";
  }

  GsDeviceResult device;
  device.psi = quasar::backend::DeviceBuffer<Real>{
      1, quasar::backend::on_device(0)};
  device.j_phi = quasar::backend::DeviceBuffer<Real>{
      1, quasar::backend::on_device(1)};

  EXPECT_THROW((void)device.copy_to_host(), std::invalid_argument);
}

TEST(GsDeviceResult, OwnsFullFieldsAndCopiesBitwiseToTheHostContract) {
  const GsConfig cfg = converging_config();
  DeviceStream stream;

  GsDeviceResult device =
      GsSolver{cfg, make_profile()}.solve_device(stream.get());

  ASSERT_EQ(device.status, GsStatus::converged);
  EXPECT_TRUE(device.ok());
  EXPECT_EQ(device.grid.nr, cfg.grid.nr);
  EXPECT_EQ(device.grid.nz, cfg.grid.nz);
  EXPECT_EQ(device.grid.r_min, cfg.grid.r_min);
  EXPECT_EQ(device.grid.r_max, cfg.grid.r_max);
  EXPECT_EQ(device.grid.z_min, cfg.grid.z_min);
  EXPECT_EQ(device.grid.z_max, cfg.grid.z_max);
  EXPECT_EQ(device.psi.size(), cfg.grid.size());
  EXPECT_EQ(device.j_phi.size(), cfg.grid.size());
  EXPECT_NE(device.psi.device_ptr(), nullptr);
  EXPECT_NE(device.j_phi.device_ptr(), nullptr);
  EXPECT_EQ(device.profile_coefficients.n_p, 2);
  EXPECT_EQ(device.profile_coefficients.p_coeffs[0], Real{1});
  EXPECT_EQ(device.profile_coefficients.p_coeffs[1], Real{-1});
  EXPECT_EQ(device.profile_coefficients.n_f, 2);
  EXPECT_EQ(device.profile_coefficients.f_coeffs[0], Real{1});
  EXPECT_EQ(device.profile_coefficients.f_coeffs[1], Real{-1});

  const GsResult copied = device.copy_to_host(stream.get());
  const GsResult legacy = GsSolver{cfg, make_profile()}.solve();

  expect_same_metadata(copied, legacy);
  EXPECT_EQ(bitwise_mismatches(copied.psi, legacy.psi), 0u);
  EXPECT_EQ(bitwise_mismatches(copied.j_phi, legacy.j_phi), 0u);
}

TEST(GsDeviceResult, FailureRetainsNonzeroDeviceFields) {
  GsConfig cfg = converging_config();
  cfg.max_iterations = 10;
  cfg.coils = {
      CoilFilament{Real{1.75}, Real{0.60}, Real{3.6e5}},
      CoilFilament{Real{1.75}, Real{-0.60}, Real{3.6e5}},
      CoilFilament{Real{0.42}, Real{0.0}, Real{-1.2e5}},
  };

  GsDeviceResult device = GsSolver{cfg, make_profile()}.solve_device();

  ASSERT_NE(device.status, GsStatus::converged)
      << "this deck must fail for the retention check to be meaningful";
  EXPECT_FALSE(device.ok());
  EXPECT_EQ(device.psi.size(), cfg.grid.size());
  EXPECT_EQ(device.j_phi.size(), cfg.grid.size());

  const GsResult host = device.copy_to_host();
  EXPECT_EQ(host.status, device.status);
  EXPECT_GT(peak_magnitude(host.psi), Real{0});
  EXPECT_GT(peak_magnitude(host.j_phi), Real{1});
  for (const Real value : host.psi) EXPECT_TRUE(std::isfinite(value));
  for (const Real value : host.j_phi) EXPECT_TRUE(std::isfinite(value));
}

}  // namespace
