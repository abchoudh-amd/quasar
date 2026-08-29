// Source-convention equilibrium fields used by the toroidal delta-W operator.
//
// These tests keep the profile oracle analytic and independent of the device
// implementation.  The geometry test uses a circular mapping whose poloidal
// phase twists with lambda, making the stored/source mixed-metric sign visible.
// The validation test uses a constructed PEST patch with exactly constant
// J/R^2 and grid-node radial samples, so cubic interpolation introduces no
// approximation into the signed q and S checks.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"
#include "quasar/physics/stability/kernels.hpp"
#include "quasar/physics/stability/toroidal_equilibrium.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::GsFluxSurfaces;
using quasar::equilibrium::GsMagneticField;
using quasar::equilibrium::ProfileCoefficients;
using quasar::numerics::EllipticGrid;
using quasar::stability::ChebyshevBasis;
using quasar::stability::DensityProfileCoefficients;
using quasar::stability::FluxCoordinateGrid;
using quasar::stability::RadialDomains;
using quasar::stability::ToroidalEquilibriumFields;
using quasar::stability::ToroidalGeometryValidation;

constexpr Real kPi = Real{3.14159265358979323846};

template <class T>
std::vector<T> download(const DeviceBuffer<T>& buffer) {
  std::vector<T> host(buffer.size());
  buffer.copy_to_host(host.data(), host.size());
  return host;
}

template <class T>
void upload(DeviceBuffer<T>& buffer, const std::vector<T>& host) {
  ASSERT_EQ(buffer.size(), host.size());
  buffer.copy_from_host(host.data(), host.size());
}

Real polynomial(const Real* coefficients, int count, Real x) {
  Real value = Real{0};
  for (int k = count; k-- > 0;) value = value * x + coefficients[k];
  return value;
}

Real integral_to_one(const Real* coefficients, int count, Real x) {
  Real result = Real{0};
  Real power = x;
  for (int k = 0; k < count; ++k) {
    result += coefficients[k]
            * (Real{1} - power) / static_cast<Real>(k + 1);
    power *= x;
  }
  return result;
}

struct CoordinateHostData {
  int n_lambda{0};
  int n_theta{0};
  std::vector<Real> lambda;
  std::vector<Real> q;
  std::vector<int> valid;
  std::vector<Real> r;
  std::vector<Real> z;
  std::vector<Real> r_lambda;
  std::vector<Real> z_lambda;
  std::vector<Real> r_theta;
  std::vector<Real> z_theta;
  std::vector<Real> jacobian;
  std::vector<Real> g_ll_contravariant;
  std::vector<Real> g_lt_contravariant;
  std::vector<Real> g_tt_contravariant;
};

FluxCoordinateGrid to_device(const CoordinateHostData& host) {
  FluxCoordinateGrid coords{host.n_lambda, host.n_theta};
  upload(coords.psi_n, host.lambda);
  upload(coords.q, host.q);
  upload(coords.valid, host.valid);
  upload(coords.r, host.r);
  upload(coords.z, host.z);
  upload(coords.dr_dpsi, host.r_lambda);
  upload(coords.dz_dpsi, host.z_lambda);
  upload(coords.dr_dtheta, host.r_theta);
  upload(coords.dz_dtheta, host.z_theta);
  upload(coords.jacobian, host.jacobian);
  upload(coords.g_psipsi, host.g_ll_contravariant);
  upload(coords.g_psitheta, host.g_lt_contravariant);
  upload(coords.g_thetatheta, host.g_tt_contravariant);
  return coords;
}

void append_metric(CoordinateHostData& host, Real r, Real z, Real rl,
                   Real zl, Real rt, Real zt) {
  const Real det = rl * zt - rt * zl;
  const Real cov_ll = rl * rl + zl * zl;
  const Real cov_lt = rl * rt + zl * zt;
  const Real cov_tt = rt * rt + zt * zt;
  const Real det2 = det * det;
  host.r.push_back(r);
  host.z.push_back(z);
  host.r_lambda.push_back(rl);
  host.z_lambda.push_back(zl);
  host.r_theta.push_back(rt);
  host.z_theta.push_back(zt);
  host.jacobian.push_back(r * det);
  host.g_ll_contravariant.push_back(cov_tt / det2);
  host.g_lt_contravariant.push_back(-cov_lt / det2);
  host.g_tt_contravariant.push_back(cov_ll / det2);
}

CoordinateHostData circular_twisted_coordinates() {
  CoordinateHostData host;
  host.n_lambda = 3;
  host.n_theta = 7;
  host.lambda = {Real{0.1}, Real{0.55}, Real{1}};
  host.q = {Real{1.15}, Real{1.4}, Real{1.75}};
  host.valid.assign(static_cast<std::size_t>(host.n_lambda), 1);

  constexpr Real major_radius = Real{2.1};
  constexpr Real axis_z = Real{-0.08};
  constexpr Real radius_intercept = Real{0.31};
  constexpr Real radius_slope = Real{0.12};
  constexpr Real radial_twist = Real{0.27};
  for (int s = 0; s < host.n_lambda; ++s) {
    const Real lambda = host.lambda[static_cast<std::size_t>(s)];
    const Real a = radius_intercept + radius_slope * lambda;
    for (int j = 0; j < host.n_theta; ++j) {
      const Real stored_theta = Real{2} * kPi * static_cast<Real>(j)
                              / static_cast<Real>(host.n_theta);
      const Real alpha = stored_theta + radial_twist * lambda;
      const Real c = std::cos(alpha);
      const Real sn = std::sin(alpha);
      append_metric(host, major_radius + a * c, axis_z + a * sn,
                    radius_slope * c - a * radial_twist * sn,
                    radius_slope * sn + a * radial_twist * c, -a * sn,
                    a * c);
    }
  }
  return host;
}

ProfileCoefficients analytic_profile() {
  ProfileCoefficients profile;
  profile.n_p = 3;
  profile.p_coeffs[0] = Real{1.25};
  profile.p_coeffs[1] = Real{-0.4};
  profile.p_coeffs[2] = Real{0.3};
  profile.n_f = 3;
  profile.f_coeffs[0] = Real{0.2};
  profile.f_coeffs[1] = Real{-0.15};
  profile.f_coeffs[2] = Real{0.05};
  return profile;
}

DensityProfileCoefficients positive_density() {
  DensityProfileCoefficients density;
  density.count = 3;
  density.coefficients[0] = Real{0.9};
  density.coefficients[1] = Real{0.25};
  density.coefficients[2] = Real{0.1};
  return density;
}

TEST(ToroidalEquilibrium,
     EvaluatesAnalyticProfilesSourceMetricsAndPhysicalCurrents) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const CoordinateHostData host = circular_twisted_coordinates();
  FluxCoordinateGrid coords = to_device(host);
  const ProfileCoefficients profile = analytic_profile();
  const DensityProfileCoefficients density_profile = positive_density();
  constexpr Real psi_axis = Real{2.1};
  constexpr Real psi_boundary = Real{-0.3};
  constexpr Real profile_scale = Real{1.7};
  constexpr Real f_vacuum = Real{-3.2};
  constexpr Real signed_flux_scale = psi_axis - psi_boundary;
  constexpr Real delta_psi = psi_boundary - psi_axis;

  ToroidalEquilibriumFields fields{host.n_lambda, host.n_theta};
  ASSERT_NO_THROW(quasar::stability::launch_build_toroidal_equilibrium(
      coords, profile, psi_axis, psi_boundary, profile_scale, f_vacuum,
      density_profile, fields, nullptr));
  EXPECT_DOUBLE_EQ(fields.signed_flux_scale, signed_flux_scale);

  const auto pressure = download(fields.pressure);
  const auto pressure_lambda = download(fields.pressure_lambda);
  const auto f = download(fields.f);
  const auto f_squared = download(fields.f_squared);
  const auto f_lambda = download(fields.f_lambda);
  const auto ff_lambda = download(fields.ff_lambda);
  const auto density = download(fields.density);
  const auto g_ll = download(fields.g_lambda_lambda);
  const auto g_lt = download(fields.g_lambda_theta);
  const auto g_tt = download(fields.g_theta_theta);
  const auto g_pp = download(fields.g_phi_phi);
  const auto jacobian = download(fields.jacobian);
  const auto b_theta = download(fields.b_theta);
  const auto b_phi = download(fields.b_phi);
  const auto j_theta = download(fields.j_theta);
  const auto j_phi = download(fields.j_phi);

  for (int s = 0; s < host.n_lambda; ++s) {
    const Real lambda = host.lambda[static_cast<std::size_t>(s)];
    const Real p_prime =
        polynomial(profile.p_coeffs, profile.n_p, lambda);
    const Real ff_prime =
        polynomial(profile.f_coeffs, profile.n_f, lambda);
    const Real want_pressure = -delta_psi * profile_scale
                             * integral_to_one(profile.p_coeffs, profile.n_p,
                                               lambda);
    const Real want_pressure_lambda = delta_psi * profile_scale * p_prime;
    const Real want_f2 = f_vacuum * f_vacuum
                       - Real{2} * delta_psi * profile_scale
                         * integral_to_one(profile.f_coeffs, profile.n_f,
                                           lambda);
    const Real want_f = std::copysign(std::sqrt(want_f2), f_vacuum);
    const Real want_ff_lambda = delta_psi * profile_scale * ff_prime;
    const Real want_f_lambda = want_ff_lambda / want_f;
    const Real want_density = polynomial(
        density_profile.coefficients, density_profile.count, lambda);

    EXPECT_NEAR(pressure[static_cast<std::size_t>(s)], want_pressure,
                Real{2e-14} * (Real{1} + std::abs(want_pressure)));
    EXPECT_NEAR(pressure_lambda[static_cast<std::size_t>(s)],
                want_pressure_lambda,
                Real{2e-14} * (Real{1} + std::abs(want_pressure_lambda)));
    EXPECT_NEAR(f_squared[static_cast<std::size_t>(s)], want_f2,
                Real{2e-14} * want_f2);
    EXPECT_NEAR(f[static_cast<std::size_t>(s)], want_f,
                Real{2e-14} * std::abs(want_f));
    EXPECT_NEAR(ff_lambda[static_cast<std::size_t>(s)], want_ff_lambda,
                Real{2e-14} * (Real{1} + std::abs(want_ff_lambda)));
    EXPECT_NEAR(f_lambda[static_cast<std::size_t>(s)], want_f_lambda,
                Real{2e-14} * (Real{1} + std::abs(want_f_lambda)));
    EXPECT_NEAR(density[static_cast<std::size_t>(s)], want_density,
                Real{2e-14} * want_density);

    for (int theta = 0; theta < host.n_theta; ++theta) {
      const std::size_t k = static_cast<std::size_t>(s) * host.n_theta
                          + theta;
      const Real rl = host.r_lambda[k];
      const Real zl = host.z_lambda[k];
      const Real rt = host.r_theta[k];
      const Real zt = host.z_theta[k];
      const Real stored_mixed = rl * rt + zl * zt;
      const Real want_jacobian = host.r[k] * (rl * zt - rt * zl);
      const Real want_b_theta = signed_flux_scale / want_jacobian;
      const Real want_j_theta =
          -want_f_lambda / (quasar::mu0 * want_jacobian);
      const Real want_j_phi = host.q[static_cast<std::size_t>(s)]
                            * want_j_theta
                            - want_pressure_lambda / signed_flux_scale;

      EXPECT_NEAR(g_ll[k], rl * rl + zl * zl, Real{2e-15});
      // This is the convention pin: theta_source = -theta_stored.
      EXPECT_NEAR(g_lt[k], -stored_mixed, Real{2e-15});
      EXPECT_NEAR(g_tt[k], rt * rt + zt * zt, Real{2e-15});
      EXPECT_NEAR(g_pp[k], host.r[k] * host.r[k], Real{2e-15});
      EXPECT_NEAR(jacobian[k], want_jacobian, Real{2e-15});
      EXPECT_NEAR(b_theta[k], want_b_theta,
                  Real{2e-14} * std::abs(want_b_theta));
      EXPECT_NEAR(b_phi[k], host.q[static_cast<std::size_t>(s)]
                                * want_b_theta,
                  Real{2e-14} * std::abs(b_phi[k]));
      EXPECT_NEAR(j_theta[k], want_j_theta,
                  Real{3e-14} * std::abs(want_j_theta));
      EXPECT_NEAR(j_phi[k], want_j_phi,
                  Real{3e-14} * (Real{1} + std::abs(want_j_phi)));
    }
  }

  // The analytic boundary datums are exact at lambda=1.
  EXPECT_DOUBLE_EQ(pressure.back(), Real{0});
  EXPECT_DOUBLE_EQ(f.back(), f_vacuum);
}

TEST(ToroidalEquilibrium, RejectsNonpositiveDensityAndToroidalFieldSquared) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const CoordinateHostData host = circular_twisted_coordinates();
  FluxCoordinateGrid coords = to_device(host);
  ToroidalEquilibriumFields fields{host.n_lambda, host.n_theta};

  DensityProfileCoefficients bad_density;
  bad_density.count = 1;
  bad_density.coefficients[0] = Real{-0.25};
  try {
    quasar::stability::launch_build_toroidal_equilibrium(
        coords, analytic_profile(), Real{2}, Real{0}, Real{1}, Real{2},
        bad_density, fields, nullptr);
    FAIL() << "negative density was accepted";
  } catch (const quasar::stability::ToroidalEquilibriumValidationError& error) {
    EXPECT_EQ(error.status(),
              quasar::stability::ToroidalEquilibriumValidationStatus::
                  nonpositive_density);
  }

  ProfileCoefficients bad_f;
  bad_f.n_p = 1;
  bad_f.p_coeffs[0] = Real{0};
  bad_f.n_f = 1;
  bad_f.f_coeffs[0] = Real{-1};
  DensityProfileCoefficients good_density;
  good_density.count = 1;
  good_density.coefficients[0] = Real{1};
  try {
    quasar::stability::launch_build_toroidal_equilibrium(
        coords, bad_f, Real{2}, Real{0}, Real{1}, Real{0.5}, good_density,
        fields, nullptr);
    FAIL() << "nonpositive F squared was accepted";
  } catch (const quasar::stability::ToroidalEquilibriumValidationError& error) {
    EXPECT_EQ(error.status(),
              quasar::stability::ToroidalEquilibriumValidationStatus::
                  nonpositive_f_squared);
  }

  DensityProfileCoefficients nonfinite_density = good_density;
  nonfinite_density.coefficients[0] =
      std::numeric_limits<Real>::quiet_NaN();
  try {
    quasar::stability::launch_build_toroidal_equilibrium(
        coords, analytic_profile(), Real{2}, Real{0}, Real{1}, Real{2},
        nonfinite_density, fields, nullptr);
    FAIL() << "non-finite density was accepted";
  } catch (const quasar::stability::ToroidalEquilibriumValidationError& error) {
    EXPECT_EQ(error.status(),
              quasar::stability::ToroidalEquilibriumValidationStatus::
                  nonfinite_profile);
  }
}

CoordinateHostData constructed_pest_coordinates(const EllipticGrid& grid) {
  CoordinateHostData host;
  host.n_lambda = 2;
  host.n_theta = 6;
  host.lambda = {Real{0.3}, Real{0.7}};
  host.q = {Real{1.5}, Real{1.7}};
  host.valid = {1, 1};
  const Real radii[] = {grid.r(1), grid.r(3)};
  for (int s = 0; s < host.n_lambda; ++s) {
    for (int theta = 0; theta < host.n_theta; ++theta) {
      const Real angle = Real{2} * kPi * static_cast<Real>(theta)
                       / static_cast<Real>(host.n_theta);
      // A local analytic PEST patch: R is constant on each surface,
      // R_lambda=0.2, and Z_thetaStored=1.  Thus J/R^2=0.2/R is exactly a
      // flux function.  Point locations remain inside the cubic field grid.
      append_metric(host, radii[s], Real{0.2} * std::sin(angle), Real{0.2},
                    Real{0}, Real{0}, Real{1});
    }
  }
  return host;
}

TEST(ToroidalGeometryValidation,
     ChecksSignedQFluxScaleAndPestJacobianWithoutAbsoluteValues) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const EllipticGrid grid{5, 4, Real{1.8}, Real{2.6}, Real{-0.6}, Real{0.6}};
  const CoordinateHostData host = constructed_pest_coordinates(grid);
  FluxCoordinateGrid coords = to_device(host);

  ProfileCoefficients profile;
  profile.n_p = 1;
  profile.p_coeffs[0] = Real{0};
  profile.n_f = 1;
  profile.f_coeffs[0] = Real{0};
  DensityProfileCoefficients density;
  density.count = 1;
  density.coefficients[0] = Real{1};
  constexpr Real psi_axis = Real{2.5};
  constexpr Real psi_boundary = Real{0.5};
  constexpr Real signed_flux_scale = psi_axis - psi_boundary;
  ToroidalEquilibriumFields equilibrium{host.n_lambda, host.n_theta};
  quasar::stability::launch_build_toroidal_equilibrium(
      coords, profile, psi_axis, psi_boundary, Real{1}, Real{2}, density,
      equilibrium, nullptr);

  GsMagneticField field{grid};
  std::vector<Real> br(grid.size(), Real{0});
  std::vector<Real> bz(grid.size());
  std::vector<Real> bphi(grid.size());
  for (int j = 0; j < grid.nz; ++j) {
    for (int i = 0; i < grid.nr; ++i) {
      const Real radius = grid.r(i);
      const Real q_of_radius = Real{0.5} + Real{0.5} * radius;
      const std::size_t k = grid.index(i, j);
      bz[k] = -signed_flux_scale / (Real{0.2} * radius);
      bphi[k] = q_of_radius * signed_flux_scale / Real{0.2};
    }
  }
  upload(field.b_r, br);
  upload(field.b_z, bz);
  upload(field.b_phi, bphi);

  ToroidalGeometryValidation validation{host.n_lambda};
  quasar::stability::launch_validate_toroidal_geometry(
      coords, grid, field, equilibrium, validation, nullptr);
  quasar::backend::device_synchronize(nullptr);
  const auto summary =
      quasar::stability::summarize_toroidal_geometry_validation(validation,
                                                                nullptr);
  EXPECT_TRUE(summary.ok());
  EXPECT_EQ(summary.invalid_surface_count, 0);
  EXPECT_EQ(summary.first_invalid_surface, -1);
  const auto q_deviation = download(validation.q_relative_deviation);
  const auto s_deviation = download(validation.flux_scale_relative_deviation);
  const auto pest_deviation =
      download(validation.pest_j_over_r2_relative_deviation);
  for (int s = 0; s < host.n_lambda; ++s) {
    EXPECT_TRUE(std::isfinite(q_deviation[static_cast<std::size_t>(s)]));
    EXPECT_TRUE(std::isfinite(s_deviation[static_cast<std::size_t>(s)]));
    EXPECT_TRUE(std::isfinite(pest_deviation[static_cast<std::size_t>(s)]));
    EXPECT_LT(q_deviation[static_cast<std::size_t>(s)], Real{2e-14});
    EXPECT_LT(s_deviation[static_cast<std::size_t>(s)], Real{2e-14});
    EXPECT_LT(pest_deviation[static_cast<std::size_t>(s)], Real{2e-14});
  }
  EXPECT_LT(summary.maximum_q_relative_deviation, Real{2e-14});
  EXPECT_LT(summary.maximum_flux_scale_relative_deviation, Real{2e-14});
  EXPECT_LT(summary.maximum_pest_j_over_r2_relative_deviation, Real{2e-14});

  // Reverse only the poloidal field.  Signed checks must report q_field=-q
  // and J*B^theta=-S; taking abs(q) or abs(S) would incorrectly pass.
  for (Real& value : bz) value = -value;
  upload(field.b_z, bz);
  quasar::stability::launch_validate_toroidal_geometry(
      coords, grid, field, equilibrium, validation, nullptr);
  quasar::backend::device_synchronize(nullptr);
  const auto reversed_summary =
      quasar::stability::summarize_toroidal_geometry_validation(validation,
                                                                nullptr);
  EXPECT_TRUE(reversed_summary.ok());
  EXPECT_NEAR(reversed_summary.maximum_q_relative_deviation, Real{2},
              Real{2e-14});
  EXPECT_NEAR(reversed_summary.maximum_flux_scale_relative_deviation, Real{2},
              Real{2e-14});
  const auto reversed_q = download(validation.q_relative_deviation);
  const auto reversed_s = download(validation.flux_scale_relative_deviation);
  for (int s = 0; s < host.n_lambda; ++s) {
    EXPECT_TRUE(std::isfinite(reversed_q[static_cast<std::size_t>(s)]));
    EXPECT_TRUE(std::isfinite(reversed_s[static_cast<std::size_t>(s)]));
    EXPECT_NEAR(reversed_q[static_cast<std::size_t>(s)], Real{2}, Real{2e-14});
    EXPECT_NEAR(reversed_s[static_cast<std::size_t>(s)], Real{2}, Real{2e-14});
  }
}

TEST(ToroidalGeometryValidationSummary,
     ReportsMaximaAndInvalidSurfacesWithStableFirstIndex) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  ToroidalGeometryValidation validation{5};
  const Real maximum = std::numeric_limits<Real>::max();
  upload(validation.q_relative_deviation,
         {Real{0.1}, std::numeric_limits<Real>::quiet_NaN(), Real{0.4},
          maximum, Real{0.25}});
  upload(validation.flux_scale_relative_deviation,
         {Real{0.3}, Real{0.2}, std::numeric_limits<Real>::infinity(), maximum,
          Real{0.35}});
  upload(validation.pest_j_over_r2_relative_deviation,
         {Real{0.5}, Real{0.6}, Real{0.8}, maximum, Real{-0.1}});

  const auto summary =
      quasar::stability::summarize_toroidal_geometry_validation(validation,
                                                                nullptr);
  EXPECT_FALSE(summary.ok());
  EXPECT_EQ(summary.invalid_surface_count, 4);
  EXPECT_EQ(summary.first_invalid_surface, 1);
  EXPECT_DOUBLE_EQ(summary.maximum_q_relative_deviation, Real{0.4});
  EXPECT_DOUBLE_EQ(summary.maximum_flux_scale_relative_deviation, Real{0.35});
  EXPECT_DOUBLE_EQ(summary.maximum_pest_j_over_r2_relative_deviation,
                   Real{0.8});
}

TEST(ToroidalGeometryValidation,
     AcceptsCurrentSpectralOrientationForAnAnalyticCircularEquilibrium) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr int order = 14;
  constexpr int n_nodes = order + 1;
  constexpr int n_contour = 4096;
  constexpr int n_theta = 256;
  constexpr Real major_radius = Real{2};
  constexpr Real radius_intercept = Real{0.3};
  constexpr Real radius_slope = Real{0.2};
  constexpr Real signed_flux_scale = Real{1};
  constexpr Real toroidal_field_function = Real{2};

  RadialDomains domains;
  domains.n_domains = 1;
  domains.breakpoints[0] = Real{0.1};
  domains.breakpoints[1] = Real{0.9};
  DeviceBuffer<RadialDomains> device_domains{1};
  device_domains.copy_from_host(&domains, 1);
  ChebyshevBasis basis{order, 1};
  quasar::stability::launch_build_chebyshev_basis(
      device_domains.device_ptr(), basis, nullptr);
  quasar::backend::device_synchronize(nullptr);
  const std::vector<Real> nodes = download(basis.nodes);

  const EllipticGrid grid{257, 257, Real{1.15}, Real{2.85}, Real{-0.85},
                          Real{0.85}};
  GsMagneticField field{grid};
  std::vector<Real> br(grid.size());
  std::vector<Real> bz(grid.size());
  std::vector<Real> bphi(grid.size());
  std::vector<Real> bpol(grid.size());
  for (int j = 0; j < grid.nz; ++j) {
    const Real z = grid.z(j);
    for (int i = 0; i < grid.nr; ++i) {
      const Real r = grid.r(i);
      const Real x = r - major_radius;
      const Real minor_radius = std::sqrt(x * x + z * z);
      const std::size_t k = grid.index(i, j);
      if (minor_radius > Real{0}) {
        const Real magnitude =
            signed_flux_scale / (r * radius_slope);
        br[k] = magnitude * z / minor_radius;
        bz[k] = -magnitude * x / minor_radius;
        bpol[k] = magnitude;
      } else {
        br[k] = bz[k] = bpol[k] = Real{0};
      }
      bphi[k] = toroidal_field_function / r;
    }
  }
  upload(field.b_r, br);
  upload(field.b_z, bz);
  upload(field.b_phi, bphi);
  upload(field.b_poloidal, bpol);

  GsFluxSurfaces surfaces{n_nodes, n_contour};
  std::vector<Real> surface_r(static_cast<std::size_t>(n_nodes) * n_contour);
  std::vector<Real> surface_z(surface_r.size());
  std::vector<int> counts(n_nodes, n_contour);
  std::vector<int> closed(n_nodes, 1);
  for (int s = 0; s < n_nodes; ++s) {
    const Real a = radius_intercept
                 + radius_slope * nodes[static_cast<std::size_t>(s)];
    const std::size_t base = static_cast<std::size_t>(s) * n_contour;
    for (int theta = 0; theta < n_contour; ++theta) {
      const Real angle = Real{2} * kPi * static_cast<Real>(theta)
                       / static_cast<Real>(n_contour);
      surface_r[base + theta] = major_radius + a * std::cos(angle);
      surface_z[base + theta] = a * std::sin(angle);
    }
  }
  upload(surfaces.r, surface_r);
  upload(surfaces.z, surface_z);
  upload(surfaces.count, counts);
  upload(surfaces.closed, closed);
  upload(surfaces.psi_n, nodes);

  FluxCoordinateGrid coords{n_nodes, n_theta};
  quasar::stability::launch_build_spectral_flux_coordinates(
      grid, surfaces, field, basis, coords, nullptr);

  ProfileCoefficients profile;
  profile.n_p = 1;
  profile.p_coeffs[0] = Real{0};
  profile.n_f = 1;
  profile.f_coeffs[0] = Real{0};
  DensityProfileCoefficients density;
  density.count = 1;
  density.coefficients[0] = Real{1};
  ToroidalEquilibriumFields equilibrium{n_nodes, n_theta};
  quasar::stability::launch_build_toroidal_equilibrium(
      coords, profile, Real{1}, Real{0}, Real{1},
      toroidal_field_function, density, equilibrium, nullptr);

  ToroidalGeometryValidation validation{n_nodes};
  quasar::stability::launch_validate_toroidal_geometry(
      coords, grid, field, equilibrium, validation, nullptr);
  quasar::backend::device_synchronize(nullptr);
  const auto q_deviation = download(validation.q_relative_deviation);
  const auto s_deviation = download(validation.flux_scale_relative_deviation);
  const auto pest_deviation =
      download(validation.pest_j_over_r2_relative_deviation);

  Real worst_q = Real{0};
  Real worst_s = Real{0};
  Real worst_pest = Real{0};
  for (int s = 0; s < n_nodes; ++s) {
    ASSERT_TRUE(std::isfinite(q_deviation[static_cast<std::size_t>(s)]));
    ASSERT_TRUE(std::isfinite(s_deviation[static_cast<std::size_t>(s)]));
    ASSERT_TRUE(std::isfinite(pest_deviation[static_cast<std::size_t>(s)]));
    worst_q = std::max(worst_q, q_deviation[static_cast<std::size_t>(s)]);
    worst_s = std::max(worst_s, s_deviation[static_cast<std::size_t>(s)]);
    worst_pest =
        std::max(worst_pest, pest_deviation[static_cast<std::size_t>(s)]);
  }
  // At this 4096-ray/256-mode resolution the signed S and PEST residuals are
  // 1.27e-4 and 1.34e-4, respectively; the remaining error converges with the
  // piecewise-linear contour resampling.  A theta-sign error is O(2), four
  // orders of magnitude larger, so this tolerance distinguishes the contract
  // without pretending the traced polygon is an exact analytic circle.
  EXPECT_LT(worst_q, Real{2e-4}) << "q orientation/straightness residual";
  EXPECT_LT(worst_s, Real{2e-4}) << "signed J B^theta residual";
  EXPECT_LT(worst_pest, Real{2e-4}) << "PEST J/R^2 residual";
}

}  // namespace
