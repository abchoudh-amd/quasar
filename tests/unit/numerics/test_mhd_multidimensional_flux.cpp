#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::MhdInterfaceStates;
using quasar::numerics::MhdState;

Real monomial_cell_average(int cell, int degree) {
  const Real lo = static_cast<Real>(cell) - Real{0.5};
  const Real hi = static_cast<Real>(cell) + Real{0.5};
  return (std::pow(hi, degree + 1) - std::pow(lo, degree + 1)) /
         static_cast<Real>(degree + 1);
}

template <int Points, int Cells>
void expect_point_recovery_exact(
    const Real (&nodes)[Points], const Real (&weights)[Points][Cells],
    int half, int maximum_degree) {
  for (int degree = 0; degree <= maximum_degree; ++degree) {
    for (int q = 0; q < Points; ++q) {
      Real recovered = Real{0};
      for (int k = 0; k < Cells; ++k) {
        recovered += weights[q][k] *
                     monomial_cell_average(k - half, degree);
      }
      const Real exact = std::pow(nodes[q], degree);
      EXPECT_NEAR(recovered, exact, Real{3e-13})
          << "degree=" << degree << " node=" << q;
    }
  }
}

template <int Points>
void expect_gauss_exact(const Real (&nodes)[Points],
                        const Real (&weights)[Points], int maximum_degree) {
  for (int degree = 0; degree <= maximum_degree; ++degree) {
    Real numerical = Real{0};
    for (int q = 0; q < Points; ++q) {
      numerical += weights[q] * std::pow(nodes[q], degree);
    }
    const Real exact = degree % 2 == 0
        ? Real{2} * std::pow(Real{0.5}, degree + 1) /
              static_cast<Real>(degree + 1)
        : Real{0};
    EXPECT_NEAR(numerical, exact, Real{3e-15}) << "degree=" << degree;
  }
}

void expect_hlld_skips_non_owned_boundary(int dir, int skipped_side) {
  Grid2D g{6, 5, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  const std::size_t n = g.storage_size();
  MhdInterfaceStates<Real> iface{g, dir};

  const auto seed_interface = [n](auto& left, auto& right, Real value) {
    const std::vector<Real> values(n, value);
    left.copy_from_host(values.data(), n);
    right.copy_from_host(values.data(), n);
  };
  seed_interface(iface.Lrho, iface.Rrho, Real{1});
  seed_interface(iface.Lmx, iface.Rmx, Real{0.2});
  seed_interface(iface.Lmy, iface.Rmy, Real{-0.1});
  seed_interface(iface.Lmz, iface.Rmz, Real{0.05});
  seed_interface(iface.Lenergy, iface.Renergy, Real{4});
  seed_interface(iface.Lbx, iface.Rbx, Real{0.3});
  seed_interface(iface.Lby, iface.Rby, Real{-0.2});
  seed_interface(iface.Lbz, iface.Rbz, Real{0.1});

  quasar::mhd::MhdBackgroundField<Real> background{g};
  background.active = true;
  const std::vector<Real> b0x(n, Real{0.03});
  const std::vector<Real> b0y(n, Real{-0.02});
  const std::vector<Real> b0z(n, Real{0.01});
  background.b0x_face.copy_from_host(b0x.data(), n);
  background.b0y_face.copy_from_host(b0y.data(), n);
  background.b0z_cell.copy_from_host(b0z.data(), n);

  quasar::mhd::MhdField2D<Real> flux{g};
  quasar::mhd::MhdMomentumFluxParts2D<Real> parts{g};
  constexpr Real real_sentinel = Real{-7654.25};
  const std::vector<Real> real_sentinels(n, real_sentinel);
  const std::array<quasar::backend::DeviceBuffer<Real>*, 11> scalar_records{
      &flux.rho, &flux.mx, &flux.my, &flux.mz, &flux.energy,
      &flux.bx_face, &flux.by_face, &flux.bz_cell,
      &parts.wave_x, &parts.wave_y, &parts.wave_z};
  for (auto* record : scalar_records) {
    record->copy_from_host(real_sentinels.data(), n);
  }
  for (auto& point : parts.cross_b_point) {
    point.x.copy_from_host(real_sentinels.data(), n);
    point.y.copy_from_host(real_sentinels.data(), n);
    point.z.copy_from_host(real_sentinels.data(), n);
  }
  constexpr int valid_sentinel = -17;
  const std::vector<int> valid_sentinels(n, valid_sentinel);
  parts.quadrature_valid.copy_from_host(valid_sentinels.data(), n);
  const quasar::numerics::ScaledValue covariance_sentinel{Real{0.625}, 123};
  const std::vector<quasar::numerics::ScaledValue> covariance_sentinels(
      n, covariance_sentinel);
  parts.b0_induction_covariance.copy_from_host(
      covariance_sentinels.data(), n);

  quasar::mhd::FaceOwnershipFlags4 ownership{};
  ownership.side[skipped_side] = 0;
  const quasar::mhd::BoundaryFlags4 boundaries{{4, 4, 4, 4}};
  quasar::mhd::launch_mhd_hlld_flux(
      iface, background, dir, flux, boundaries, gamma, /*stream=*/nullptr,
      /*hll_only=*/false, &parts, /*scheme_order=*/2, ownership);
  quasar::backend::device_synchronize(nullptr);

  int skipped_i = 2;
  int skipped_j = 2;
  if (skipped_side == 0) skipped_i = 0;
  if (skipped_side == 1) skipped_i = g.nx;
  if (skipped_side == 2) skipped_j = 0;
  if (skipped_side == 3) skipped_j = g.ny;
  const std::size_t skipped = g.index(skipped_i, skipped_j);
  const std::size_t owned = g.index(2, 2);

  const auto expect_real_record = [&](const auto& record) {
    std::vector<Real> values(n);
    record.copy_to_host(values.data(), n);
    EXPECT_EQ(values[skipped], real_sentinel);
    EXPECT_NE(values[owned], real_sentinel);
    EXPECT_TRUE(std::isfinite(values[owned]));
  };
  for (std::size_t record = 0; record < scalar_records.size(); ++record) {
    SCOPED_TRACE(::testing::Message{} << "scalar record " << record);
    expect_real_record(*scalar_records[record]);
  }
  for (std::size_t q = 0; q < parts.cross_b_point.size(); ++q) {
    SCOPED_TRACE(::testing::Message{} << "cross-B point " << q);
    expect_real_record(parts.cross_b_point[q].x);
    expect_real_record(parts.cross_b_point[q].y);
    expect_real_record(parts.cross_b_point[q].z);
  }

  std::vector<int> valid(n);
  parts.quadrature_valid.copy_to_host(valid.data(), n);
  EXPECT_EQ(valid[skipped], valid_sentinel);
  EXPECT_NE(valid[owned], valid_sentinel);
  std::vector<quasar::numerics::ScaledValue> covariance(n);
  parts.b0_induction_covariance.copy_to_host(covariance.data(), n);
  EXPECT_EQ(covariance[skipped].mantissa, covariance_sentinel.mantissa);
  EXPECT_EQ(covariance[skipped].exponent, covariance_sentinel.exponent);
  EXPECT_FALSE(covariance[owned].mantissa == covariance_sentinel.mantissa &&
               covariance[owned].exponent == covariance_sentinel.exponent);
}

TEST(MhdMultidimensionalFlux, TransverseQuadratureIsPolynomialExact) {
  expect_point_recovery_exact(
      quasar::numerics::kMp5TransverseNodes,
      quasar::numerics::kMp5TransversePointWeights,
      /*half=*/2, /*maximum_degree=*/4);
  expect_gauss_exact(
      quasar::numerics::kMp5TransverseNodes,
      quasar::numerics::kMp5TransverseGaussWeights,
      /*maximum_degree=*/5);
  expect_point_recovery_exact(
      quasar::numerics::kMp7TransverseNodes,
      quasar::numerics::kMp7TransversePointWeights,
      /*half=*/3, /*maximum_degree=*/6);
  expect_gauss_exact(
      quasar::numerics::kMp7TransverseNodes,
      quasar::numerics::kMp7TransverseGaussWeights,
      /*maximum_degree=*/7);
}

TEST(MhdMultidimensionalFlux,
     PointProductExpansionRetainsSurvivorAcrossExtremeCancellation) {
  const Real a[3] = {std::scalbn(Real{1}, 600),
                     std::scalbn(Real{1}, 600), Real{1}};
  const Real b[3] = {std::scalbn(Real{1}, 400),
                     -Real{5} / Real{8} * std::scalbn(Real{1}, 400),
                     Real{18} / Real{5}};
  const auto& weights = quasar::numerics::kMp5TransverseGaussWeights;
  quasar::numerics::ScaledProductQuotientAccumulator<3> sum;
  for (int q = 0; q < 3; ++q) {
    quasar::numerics::append_scaled_triple_product_quotient(
        sum, weights[q], a[q], b[q], Real{1}, Real{1});
  }
  EXPECT_EQ(quasar::numerics::finish_scaled_product_quotient_sum(sum),
            Real{1});
}

TEST(MhdMultidimensionalFlux,
     HlldComputesOnlyCanonicalSharedBoundaryRecords) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  expect_hlld_skips_non_owned_boundary(/*dir=*/0, /*skipped_side=*/0);
  expect_hlld_skips_non_owned_boundary(/*dir=*/1, /*skipped_side=*/3);
}

TEST(MhdMultidimensionalFlux,
     ScaledReducerRetainsSurvivorAfterDistributedCancellation) {
  const Real unit = std::scalbn(Real{1}, 900);
  const Real ulp = std::scalbn(Real{1}, -54);
  const Real terms[4] = {
      Real{1}, -(Real{31} / Real{64} + ulp),
      -Real{28} / Real{64}, -(Real{5} / Real{64} - ulp)};
  quasar::numerics::ScaledProductQuotientAccumulator<5> sum;
  for (const Real term : terms) {
    quasar::numerics::append_scaled_product_quotient(
        sum, term, unit, Real{1}, Real{1});
  }
  quasar::numerics::append_scaled_product_quotient(
      sum, Real{1}, Real{1}, Real{1}, Real{1});
  EXPECT_EQ(quasar::numerics::finish_scaled_product_quotient_sum(sum),
            Real{1});
}

TEST(MhdMultidimensionalFlux,
     TransverseProductCorrectionIsRangeSafeAndConstantExact) {
  const Real large = Real{0.75} * std::numeric_limits<Real>::max();
  const Real small = Real{1} / large;
  const Real a[3] = {large, -large, -large};
  const Real b[3] = {small, Real{-2} * small, Real{4} * small};
  const auto& weights = quasar::numerics::kMp5TransverseGaussWeights;
  const Real a_mean = quasar::numerics::product_sum3(
      weights[0], a[0], weights[1], a[1], weights[2], a[2]);
  const Real b_mean = quasar::numerics::product_sum3(
      weights[0], b[0], weights[1], b[1], weights[2], b[2]);
  ASSERT_TRUE(std::isfinite(a_mean));
  // The first centered difference overflows in ordinary binary64 even though
  // the required product correction is finite.
  EXPECT_FALSE(std::isfinite(a[0] - a_mean));

  const Real correction = quasar::numerics::transverse_product_correction(
      weights, a_mean, b_mean, [&](int q) { return a[q]; },
      [&](int q) { return b[q]; });
  long double expected = 0;
  for (int q = 0; q < 3; ++q) {
    expected += static_cast<long double>(weights[q]) *
                static_cast<long double>(a[q]) *
                static_cast<long double>(b[q]);
  }
  expected -= static_cast<long double>(a_mean) *
              static_cast<long double>(b_mean);
  EXPECT_TRUE(std::isfinite(correction));
  EXPECT_NEAR(correction, static_cast<Real>(expected), Real{3e-15});

  const Real constant[3] = {large, large, large};
  EXPECT_EQ(quasar::numerics::transverse_product_correction(
                weights, a_mean, large, [&](int q) { return a[q]; },
                [&](int q) { return constant[q]; }),
            Real{0});

  // The correction itself may exceed binary64 even though recombining it with
  // the factorized mean gives a finite (here cancellation-sized) face moment.
  // Carrying mantissa/exponent separately must defer that cancellation to the
  // final accumulator instead of materializing +/-Inf in a face buffer.
  constexpr Real background_scale = Real{1e200};
  constexpr Real perturbation_scale = Real{1e110};
  const Real wide_a[3] = {
      background_scale, background_scale, -background_scale};
  const Real wide_b[3] = {
      perturbation_scale, perturbation_scale,
      (weights[0] + weights[1]) / weights[2] * perturbation_scale};
  const Real wide_a_mean = quasar::numerics::product_sum3(
      weights[0], wide_a[0], weights[1], wide_a[1],
      weights[2], wide_a[2]);
  const Real wide_b_mean = quasar::numerics::product_sum3(
      weights[0], wide_b[0], weights[1], wide_b[1],
      weights[2], wide_b[2]);
  const auto wide_correction =
      quasar::numerics::transverse_product_correction_scaled(
          weights, wide_a_mean, wide_b_mean,
          [&](int q) { return wide_a[q]; },
          [&](int q) { return wide_b[q]; });
  EXPECT_FALSE(std::isfinite(scalbn(
      wide_correction.mantissa, wide_correction.exponent)));

  quasar::numerics::ScaledProductQuotientAccumulator<2> recomposed;
  quasar::numerics::append_scaled_product_quotient(
      recomposed, wide_a_mean, wide_b_mean, Real{1}, Real{1});
  quasar::numerics::append_scaled_value_quotient(
      recomposed, wide_correction, Real{1}, Real{1}, Real{1});
  const Real face_moment =
      quasar::numerics::finish_scaled_product_quotient_sum(recomposed);
  EXPECT_TRUE(std::isfinite(face_moment));
}

TEST(MhdMultidimensionalFlux,
     InadmissibleTransverseRecoveryUsesExactBaseFaceFallback) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real magnetic = Real{10};
  constexpr Real pressure = Real{1};
  const Real energy = pressure / (gamma - Real{1}) +
                      Real{0.5} * magnetic * magnetic;
  const std::size_t n = g.storage_size();
  std::vector<MhdState> state(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    const Real bz = (j & 1) == 0 ? magnetic : -magnetic;
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      state[g.index(i, j)] = MhdState{
          Real{1}, Real{0}, Real{0}, Real{0}, energy,
          Real{0}, Real{0}, bz};
    }
  }
  Real recovered_bz = Real{0};
  for (int k = 0; k < 5; ++k) {
    const Real sample = (k & 1) == 0 ? magnetic : -magnetic;
    recovered_bz +=
        quasar::numerics::kMp5TransversePointWeights[1][k] * sample;
  }
  const MhdState recovered{
      Real{1}, Real{0}, Real{0}, Real{0}, energy,
      Real{0}, Real{0}, recovered_bz};
  ASSERT_LT(quasar::numerics::pressure(recovered, gamma), Real{0});

  MhdInterfaceStates<Real> iface{g, /*dir=*/0};
  std::vector<Real> values(n);
  const auto copy = [&](auto& left, auto& right, Real MhdState::*member) {
    for (std::size_t k = 0; k < n; ++k) values[k] = state[k].*member;
    left.copy_from_host(values.data(), n);
    right.copy_from_host(values.data(), n);
  };
  copy(iface.Lrho, iface.Rrho, &MhdState::rho);
  copy(iface.Lmx, iface.Rmx, &MhdState::mx);
  copy(iface.Lmy, iface.Rmy, &MhdState::my);
  copy(iface.Lmz, iface.Rmz, &MhdState::mz);
  copy(iface.Lenergy, iface.Renergy, &MhdState::energy);
  copy(iface.Lbx, iface.Rbx, &MhdState::bx);
  copy(iface.Lby, iface.Rby, &MhdState::by);
  copy(iface.Lbz, iface.Rbz, &MhdState::bz);

  quasar::mhd::MhdBackgroundField<Real> background{g};
  background.active = true;
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> b0z(n, Real{0.5});
  background.b0x_face.copy_from_host(zero.data(), n);
  background.b0y_face.copy_from_host(zero.data(), n);
  background.b0z_cell.copy_from_host(b0z.data(), n);
  quasar::mhd::MhdField2D<Real> high_flux{g}, base_flux{g};
  quasar::mhd::MhdMomentumFluxParts2D<Real> high_parts{g}, base_parts{g};
  quasar::mhd::launch_mhd_hlld_flux(
      iface, background, /*dir=*/0, high_flux,
      quasar::mhd::BoundaryFlags4{}, gamma, /*stream=*/nullptr,
      /*hll_only=*/false, &high_parts, /*scheme_order=*/5);
  quasar::mhd::launch_mhd_hlld_flux(
      iface, background, /*dir=*/0, base_flux,
      quasar::mhd::BoundaryFlags4{}, gamma, /*stream=*/nullptr,
      /*hll_only=*/false, &base_parts, /*scheme_order=*/2);
  quasar::backend::device_synchronize(nullptr);

  const int target_i = 3;
  const int target_j = 4;
  const std::size_t target = g.index(target_i, target_j);
  std::vector<Real> high_mx(n), base_mx(n), high_wave(n), base_wave(n);
  std::vector<int> high_valid(n), base_valid(n);
  high_flux.mx.copy_to_host(high_mx.data(), n);
  base_flux.mx.copy_to_host(base_mx.data(), n);
  high_parts.wave_x.copy_to_host(high_wave.data(), n);
  base_parts.wave_x.copy_to_host(base_wave.data(), n);
  high_parts.quadrature_valid.copy_to_host(high_valid.data(), n);
  base_parts.quadrature_valid.copy_to_host(base_valid.data(), n);
  EXPECT_EQ(high_valid[target], 0);
  EXPECT_EQ(base_valid[target], 0);
  EXPECT_EQ(high_mx[target], base_mx[target]);
  EXPECT_EQ(high_wave[target], base_wave[target]);

  for (int q = 0; q < 4; ++q) {
    std::vector<Real> high_cross(n), base_cross(n);
    high_parts.cross_b_point[q].z.copy_to_host(high_cross.data(), n);
    base_parts.cross_b_point[q].z.copy_to_host(base_cross.data(), n);
    EXPECT_EQ(high_cross[target], base_cross[target]) << "point=" << q;
    EXPECT_EQ(high_cross[target], q == 0 ? magnetic : Real{0})
        << "point=" << q;
  }
}

constexpr Real kGamma = Real{5} / Real{3};
constexpr Real kPressure = Real{1.1};
constexpr Real kVelocity = Real{0.7};
constexpr Real kXAmplitude = Real{0.11};
constexpr Real kYAmplitude = Real{0.23};

Real average_cos(Real lo, Real hi) {
  const Real wave = Real{2} * quasar::pi;
  return (std::sin(wave * hi) - std::sin(wave * lo)) /
         (wave * (hi - lo));
}

Real average_cos_squared(Real lo, Real hi) {
  const Real wave = Real{2} * quasar::pi;
  return Real{0.5} +
         (std::sin(Real{2} * wave * hi) -
          std::sin(Real{2} * wave * lo)) /
             (Real{4} * wave * (hi - lo));
}

void seed_nonlinear_face_averages(MhdInterfaceStates<Real>& iface) {
  const Grid2D& g = iface.grid;
  const std::size_t n = g.storage_size();
  std::vector<MhdState> state(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    const Real y_lo = g.origin_y + static_cast<Real>(j) * g.dy();
    const Real y_hi = y_lo + g.dy();
    const Real cy = average_cos(y_lo, y_hi);
    const Real cy2 = average_cos_squared(y_lo, y_hi);
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      // iface(dir=0,i,j) is an x-face average.  Vary the mean in both x and y;
      // the nonlinear momentum flux then cannot commute with the y average.
      const Real x = g.origin_x + static_cast<Real>(i) * g.dx();
      const Real base = kVelocity + kXAmplitude *
          std::sin(Real{2} * quasar::pi * x);
      const Real vx_average = base + kYAmplitude * cy;
      const Real vx2_average = base * base +
          Real{2} * base * kYAmplitude * cy +
          kYAmplitude * kYAmplitude * cy2;
      state[g.index(i, j)] = MhdState{
          Real{1}, vx_average, Real{0}, Real{0},
          kPressure / (kGamma - Real{1}) + Real{0.5} * vx2_average,
          Real{0}, Real{0}, Real{0}};
    }
  }

  std::vector<Real> values(n);
  const auto copy = [&](auto& left, auto& right, Real MhdState::*member) {
    for (std::size_t k = 0; k < n; ++k) values[k] = state[k].*member;
    left.copy_from_host(values.data(), n);
    right.copy_from_host(values.data(), n);
  };
  copy(iface.Lrho, iface.Rrho, &MhdState::rho);
  copy(iface.Lmx, iface.Rmx, &MhdState::mx);
  copy(iface.Lmy, iface.Rmy, &MhdState::my);
  copy(iface.Lmz, iface.Rmz, &MhdState::mz);
  copy(iface.Lenergy, iface.Renergy, &MhdState::energy);
  copy(iface.Lbx, iface.Rbx, &MhdState::bx);
  copy(iface.Lby, iface.Rby, &MhdState::by);
  copy(iface.Lbz, iface.Rbz, &MhdState::bz);
}

Real nonlinear_flux_error(int order, int resolution) {
  Grid2D g{resolution, resolution, Real{1}, Real{1}, Real{0}, Real{0},
           /*nghost=*/4};
  MhdInterfaceStates<Real> iface{g, /*dir=*/0};
  seed_nonlinear_face_averages(iface);
  quasar::mhd::MhdField2D<Real> flux{g};
  const quasar::mhd::MhdBackgroundField<Real> background{};
  const quasar::mhd::BoundaryFlags4 periodic{};
  quasar::mhd::launch_mhd_hlld_flux(
      iface, background, /*dir=*/0, flux, periodic, kGamma,
      /*stream=*/nullptr, /*hll_only=*/false, /*momentum_parts=*/nullptr,
      order);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> momentum_flux(g.storage_size());
  flux.mx.copy_to_host(momentum_flux.data(), momentum_flux.size());
  Real error = Real{0};
  for (int j = 0; j < g.ny; ++j) {
    const Real y_lo = g.origin_y + static_cast<Real>(j) * g.dy();
    const Real y_hi = y_lo + g.dy();
    const Real cy = average_cos(y_lo, y_hi);
    const Real cy2 = average_cos_squared(y_lo, y_hi);
    for (int i = 0; i < g.nx; ++i) {
      const Real x = g.origin_x + static_cast<Real>(i) * g.dx();
      const Real base = kVelocity + kXAmplitude *
          std::sin(Real{2} * quasar::pi * x);
      const Real vx2_average = base * base +
          Real{2} * base * kYAmplitude * cy +
          kYAmplitude * kYAmplitude * cy2;
      const Real exact = kPressure + vx2_average;
      error += std::abs(momentum_flux[g.index(i, j)] - exact);
    }
  }
  return error / static_cast<Real>(g.nx * g.ny);
}

TEST(MhdMultidimensionalFlux, SmoothNonlinearFluxKeepsMpOrderInTwoDimensions) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real mp5_coarse = nonlinear_flux_error(/*order=*/5, /*resolution=*/12);
  const Real mp5_fine = nonlinear_flux_error(/*order=*/5, /*resolution=*/24);
  const Real mp7_coarse = nonlinear_flux_error(/*order=*/7, /*resolution=*/10);
  const Real mp7_fine = nonlinear_flux_error(/*order=*/7, /*resolution=*/20);
  ASSERT_GT(mp5_coarse, Real{0});
  ASSERT_GT(mp5_fine, Real{0});
  ASSERT_GT(mp7_coarse, Real{0});
  ASSERT_GT(mp7_fine, Real{0});
  const Real mp5_rate = std::log2(mp5_coarse / mp5_fine);
  const Real mp7_rate = std::log2(mp7_coarse / mp7_fine);
  EXPECT_GT(mp5_rate, Real{4.3})
      << "errors " << mp5_coarse << " -> " << mp5_fine;
  EXPECT_GT(mp7_rate, Real{6.0})
      << "errors " << mp7_coarse << " -> " << mp7_fine;
}

long double radial_exponential_average(long double lo, long double hi,
                                       long double rate) {
  const auto antiderivative = [rate](long double radius) {
    return std::exp(rate * (radius - 1.0L)) *
           (radius / rate - 1.0L / (rate * rate));
  };
  const long double volume = (hi * hi - lo * lo) / 2.0L;
  return (antiderivative(hi) - antiderivative(lo)) / volume;
}

Real cylindrical_face_flux_error(int order, int resolution,
                                 bool radial_quadrature) {
  // A z-normal face is averaged in the radial direction.  Supplying exact
  // annular averages here isolates the production transverse recovery and
  // HLLD face integration from the normal reconstruction kernel.
  Grid2D g{resolution, 2, Real{1}, Real{1}, Real{1}, Real{0},
           /*nghost=*/4};
  MhdInterfaceStates<Real> iface{g, /*dir=*/1};
  constexpr long double velocity_mean = 0.7L;
  constexpr long double velocity_amplitude = 0.2L;
  constexpr long double exponential_rate = 2.5L;
  const std::size_t n = g.storage_size();
  std::vector<MhdState> state(n);
  for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
    const long double lo = static_cast<long double>(g.origin_x) +
                           static_cast<long double>(i) *
                               static_cast<long double>(g.dx());
    const long double hi = lo + static_cast<long double>(g.dx());
    const long double exponential_average =
        radial_exponential_average(lo, hi, exponential_rate);
    const long double exponential_squared_average =
        radial_exponential_average(lo, hi, 2.0L * exponential_rate);
    const Real velocity_average = static_cast<Real>(
        velocity_mean + velocity_amplitude * exponential_average);
    const Real velocity_squared_average = static_cast<Real>(
        velocity_mean * velocity_mean +
        2.0L * velocity_mean * velocity_amplitude * exponential_average +
        velocity_amplitude * velocity_amplitude *
            exponential_squared_average);
    const MhdState face_average{
        Real{1}, Real{0}, velocity_average, Real{0},
        kPressure / (kGamma - Real{1}) +
            Real{0.5} * velocity_squared_average,
        Real{0}, Real{0}, Real{0}};
    for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
      state[g.index(i, j)] = face_average;
    }
  }

  std::vector<Real> values(n);
  const auto copy = [&](auto& left, auto& right, Real MhdState::*member) {
    for (std::size_t k = 0; k < n; ++k) values[k] = state[k].*member;
    left.copy_from_host(values.data(), n);
    right.copy_from_host(values.data(), n);
  };
  copy(iface.Lrho, iface.Rrho, &MhdState::rho);
  copy(iface.Lmx, iface.Rmx, &MhdState::mx);
  copy(iface.Lmy, iface.Rmy, &MhdState::my);
  copy(iface.Lmz, iface.Rmz, &MhdState::mz);
  copy(iface.Lenergy, iface.Renergy, &MhdState::energy);
  copy(iface.Lbx, iface.Rbx, &MhdState::bx);
  copy(iface.Lby, iface.Rby, &MhdState::by);
  copy(iface.Lbz, iface.Rbz, &MhdState::bz);

  quasar::numerics::RadialTables table_owner{g, order};
  auto radial_tables = table_owner.view();
  if (!radial_quadrature) radial_tables.active = 0;
  quasar::mhd::MhdField2D<Real> flux{g};
  const quasar::mhd::BoundaryFlags4 outflow{{1, 1, 1, 1}};
  quasar::mhd::launch_mhd_hlld_flux(
      iface, quasar::mhd::MhdBackgroundField<Real>{}, /*dir=*/1, flux,
      outflow, kGamma, /*stream=*/nullptr, /*hll_only=*/false,
      /*momentum_parts=*/nullptr, order,
      quasar::mhd::FaceOwnershipFlags4{}, radial_tables);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> momentum_flux(n);
  flux.my.copy_to_host(momentum_flux.data(), n);
  Real error = Real{0};
  for (int i = 0; i < g.nx; ++i) {
    const long double lo = static_cast<long double>(g.origin_x) +
                           static_cast<long double>(i) *
                               static_cast<long double>(g.dx());
    const long double hi = lo + static_cast<long double>(g.dx());
    const long double exponential_average =
        radial_exponential_average(lo, hi, exponential_rate);
    const long double exponential_squared_average =
        radial_exponential_average(lo, hi, 2.0L * exponential_rate);
    const Real exact = kPressure + static_cast<Real>(
        velocity_mean * velocity_mean +
        2.0L * velocity_mean * velocity_amplitude * exponential_average +
        velocity_amplitude * velocity_amplitude *
            exponential_squared_average);
    error += std::abs(momentum_flux[g.index(i, 0)] - exact);
  }
  return error / static_cast<Real>(g.nx);
}

TEST(MhdMultidimensionalFlux,
     CylindricalTransverseQuadratureUsesRadialMeasure) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Real mp5_coarse = cylindrical_face_flux_error(
      /*order=*/5, /*resolution=*/10, /*radial_quadrature=*/true);
  const Real mp5_fine = cylindrical_face_flux_error(
      /*order=*/5, /*resolution=*/20, /*radial_quadrature=*/true);
  const Real mp5_cartesian_coarse = cylindrical_face_flux_error(
      /*order=*/5, /*resolution=*/10, /*radial_quadrature=*/false);
  const Real mp5_cartesian_fine = cylindrical_face_flux_error(
      /*order=*/5, /*resolution=*/20, /*radial_quadrature=*/false);
  const Real mp7_coarse = cylindrical_face_flux_error(
      /*order=*/7, /*resolution=*/8, /*radial_quadrature=*/true);
  const Real mp7_fine = cylindrical_face_flux_error(
      /*order=*/7, /*resolution=*/16, /*radial_quadrature=*/true);
  const Real mp7_cartesian_coarse = cylindrical_face_flux_error(
      /*order=*/7, /*resolution=*/8, /*radial_quadrature=*/false);
  const Real mp7_cartesian_fine = cylindrical_face_flux_error(
      /*order=*/7, /*resolution=*/16, /*radial_quadrature=*/false);

  const auto slope = [](Real coarse, Real fine) {
    EXPECT_GT(coarse, Real{0});
    EXPECT_GT(fine, Real{0});
    return std::log2(coarse / fine);
  };
  const Real mp5_rate = slope(mp5_coarse, mp5_fine);
  const Real mp5_cartesian_rate =
      slope(mp5_cartesian_coarse, mp5_cartesian_fine);
  const Real mp7_rate = slope(mp7_coarse, mp7_fine);
  const Real mp7_cartesian_rate =
      slope(mp7_cartesian_coarse, mp7_cartesian_fine);
  ::testing::Test::RecordProperty("mp5_radial_coarse_error", mp5_coarse);
  ::testing::Test::RecordProperty("mp5_radial_fine_error", mp5_fine);
  ::testing::Test::RecordProperty("mp5_radial_observed_order", mp5_rate);
  ::testing::Test::RecordProperty(
      "mp5_cartesian_observed_order", mp5_cartesian_rate);
  ::testing::Test::RecordProperty("mp7_radial_coarse_error", mp7_coarse);
  ::testing::Test::RecordProperty("mp7_radial_fine_error", mp7_fine);
  ::testing::Test::RecordProperty("mp7_radial_observed_order", mp7_rate);
  ::testing::Test::RecordProperty(
      "mp7_cartesian_observed_order", mp7_cartesian_rate);
  EXPECT_GT(mp5_rate, Real{4.6})
      << "radial errors " << mp5_coarse << " -> " << mp5_fine;
  EXPECT_LT(mp5_cartesian_rate, Real{4.2})
      << "Cartesian negative-control errors " << mp5_cartesian_coarse
      << " -> " << mp5_cartesian_fine;
  EXPECT_GT(mp7_rate, Real{6.4})
      << "radial errors " << mp7_coarse << " -> " << mp7_fine;
  EXPECT_LT(mp7_cartesian_rate, Real{4.5})
      << "Cartesian negative-control errors " << mp7_cartesian_coarse
      << " -> " << mp7_cartesian_fine;
}

Real active_background_flux_error(int order, int resolution) {
  Grid2D g{resolution, resolution, Real{1}, Real{1}, Real{0}, Real{0},
           /*nghost=*/4};
  constexpr Real b_mean = Real{0.31};
  constexpr Real b_amplitude = Real{0.19};
  constexpr Real b0_mean = Real{-0.27};
  constexpr Real b0_amplitude = Real{0.23};
  const std::size_t n = g.storage_size();
  std::vector<MhdState> state(n);
  std::vector<Real> b0z(n), zero(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    const Real y_lo = g.origin_y + static_cast<Real>(j) * g.dy();
    const Real y_hi = y_lo + g.dy();
    const Real cy = average_cos(y_lo, y_hi);
    const Real cy2 = average_cos_squared(y_lo, y_hi);
    const Real bz_average = b_mean + b_amplitude * cy;
    const Real bz2_average = b_mean * b_mean +
        Real{2} * b_mean * b_amplitude * cy +
        b_amplitude * b_amplitude * cy2;
    const Real b0z_average = b0_mean + b0_amplitude * cy;
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      state[k] = MhdState{
          Real{1}, Real{0}, Real{0}, Real{0},
          kPressure / (kGamma - Real{1}) + Real{0.5} * bz2_average,
          Real{0}, Real{0}, bz_average};
      b0z[k] = b0z_average;
    }
  }

  MhdInterfaceStates<Real> iface{g, /*dir=*/0};
  std::vector<Real> values(n);
  const auto copy = [&](auto& left, auto& right, Real MhdState::*member) {
    for (std::size_t k = 0; k < n; ++k) values[k] = state[k].*member;
    left.copy_from_host(values.data(), n);
    right.copy_from_host(values.data(), n);
  };
  copy(iface.Lrho, iface.Rrho, &MhdState::rho);
  copy(iface.Lmx, iface.Rmx, &MhdState::mx);
  copy(iface.Lmy, iface.Rmy, &MhdState::my);
  copy(iface.Lmz, iface.Rmz, &MhdState::mz);
  copy(iface.Lenergy, iface.Renergy, &MhdState::energy);
  copy(iface.Lbx, iface.Rbx, &MhdState::bx);
  copy(iface.Lby, iface.Rby, &MhdState::by);
  copy(iface.Lbz, iface.Rbz, &MhdState::bz);

  quasar::mhd::MhdBackgroundField<Real> background{g};
  background.active = true;
  background.b0x_face.copy_from_host(zero.data(), n);
  background.b0y_face.copy_from_host(zero.data(), n);
  background.b0z_cell.copy_from_host(b0z.data(), n);
  quasar::mhd::MhdField2D<Real> flux{g};
  quasar::mhd::launch_mhd_hlld_flux(
      iface, background, /*dir=*/0, flux,
      quasar::mhd::BoundaryFlags4{}, kGamma, /*stream=*/nullptr,
      /*hll_only=*/false, /*momentum_parts=*/nullptr, order);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> momentum_flux(n);
  flux.mx.copy_to_host(momentum_flux.data(), n);
  Real error = Real{0};
  for (int j = 0; j < g.ny; ++j) {
    const Real y_lo = g.origin_y + static_cast<Real>(j) * g.dy();
    const Real y_hi = y_lo + g.dy();
    const Real cy = average_cos(y_lo, y_hi);
    const Real cy2 = average_cos_squared(y_lo, y_hi);
    const Real bz2_average = b_mean * b_mean +
        Real{2} * b_mean * b_amplitude * cy +
        b_amplitude * b_amplitude * cy2;
    const Real cross_average = b0_mean * b_mean +
        (b0_mean * b_amplitude + b0_amplitude * b_mean) * cy +
        b0_amplitude * b_amplitude * cy2;
    const Real exact = kPressure + Real{0.5} * bz2_average + cross_average;
    for (int i = 0; i < g.nx; ++i) {
      error += std::abs(momentum_flux[g.index(i, j)] - exact);
    }
  }
  return error / static_cast<Real>(g.nx * g.ny);
}

TEST(MhdMultidimensionalFlux,
     ActiveBackgroundTransverseProductsKeepMpOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real mp5_coarse =
      active_background_flux_error(/*order=*/5, /*resolution=*/12);
  const Real mp5_fine =
      active_background_flux_error(/*order=*/5, /*resolution=*/24);
  const Real mp7_coarse =
      active_background_flux_error(/*order=*/7, /*resolution=*/10);
  const Real mp7_fine =
      active_background_flux_error(/*order=*/7, /*resolution=*/20);
  ASSERT_GT(mp5_coarse, Real{0});
  ASSERT_GT(mp5_fine, Real{0});
  ASSERT_GT(mp7_coarse, Real{0});
  ASSERT_GT(mp7_fine, Real{0});
  EXPECT_GT(std::log2(mp5_coarse / mp5_fine), Real{4.3})
      << "errors " << mp5_coarse << " -> " << mp5_fine;
  EXPECT_GT(std::log2(mp7_coarse / mp7_fine), Real{6.0})
      << "errors " << mp7_coarse << " -> " << mp7_fine;
}

}  // namespace
