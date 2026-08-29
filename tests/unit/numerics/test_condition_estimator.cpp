#include "quasar/numerics/condition_estimator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::numerics::MatrixTriangle;
using quasar::numerics::SymmetricConditionStatus;

DeviceBuffer<Real> upload(const std::vector<Real>& host) {
  DeviceBuffer<Real> device{host.size()};
  device.copy_from_host(host.data(), host.size());
  return device;
}

std::vector<Real> download(const DeviceBuffer<Real>& device) {
  std::vector<Real> host(device.size());
  device.copy_to_host(host.data(), host.size());
  return host;
}

TEST(SymmetricConditionEstimator, UsesAbsoluteSpectrumAndPreservesInput) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  // Upper triangle is diag(-8, 2, 0.5); the unused lower triangle is poison.
  const std::vector<Real> host{
      Real{-8}, Real{91}, Real{-72},
      Real{0}, Real{2}, Real{43},
      Real{0}, Real{0}, Real{0.5},
  };
  auto matrix = upload(host);
  const auto result = quasar::numerics::estimate_symmetric_condition(
      matrix, 3, MatrixTriangle::upper);

  ASSERT_EQ(result.status, SymmetricConditionStatus::success);
  EXPECT_EQ(result.solver_info, 0);
  EXPECT_NEAR(result.smallest_absolute_eigenvalue, Real{0.5}, Real{1e-14});
  EXPECT_NEAR(result.largest_absolute_eigenvalue, Real{8}, Real{1e-14});
  EXPECT_NEAR(result.condition_estimate, Real{16}, Real{2e-13});
  EXPECT_NEAR(result.digits_lost, std::log10(Real{16}), Real{2e-14});
  EXPECT_EQ(download(matrix), host);
}

TEST(SymmetricConditionEstimator, ReportsNumericalSingularity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  auto matrix = upload({Real{1}, Real{0}, Real{0}, Real{1e-18}});
  const auto result =
      quasar::numerics::estimate_symmetric_condition(matrix, 2);
  EXPECT_EQ(result.status, SymmetricConditionStatus::numerically_singular);
  EXPECT_TRUE(std::isinf(result.condition_estimate));
  EXPECT_TRUE(std::isinf(result.digits_lost));
  EXPECT_EQ(result.smallest_absolute_eigenvalue, Real{1e-18});
  EXPECT_EQ(result.largest_absolute_eigenvalue, Real{1});
}

TEST(SymmetricConditionEstimator, RejectsNonFiniteSpectrum) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  auto matrix = upload({Real{1}, Real{0}, Real{0}, nan});
  const auto result =
      quasar::numerics::estimate_symmetric_condition(matrix, 2);
  EXPECT_EQ(result.status, SymmetricConditionStatus::numerically_singular);
  EXPECT_TRUE(std::isinf(result.condition_estimate));
  EXPECT_TRUE(std::isinf(result.digits_lost));
}

TEST(SymmetricConditionEstimator, HonorsCallerFloorAndRejectsBadShapes) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  auto matrix = upload({Real{1}, Real{0}, Real{0}, Real{1e-4}});
  const auto resolved = quasar::numerics::estimate_symmetric_condition(
      matrix, 2, MatrixTriangle::lower, Real{1e-5});
  EXPECT_TRUE(resolved.ok());
  const auto unresolved = quasar::numerics::estimate_symmetric_condition(
      matrix, 2, MatrixTriangle::lower, Real{1e-3});
  EXPECT_EQ(unresolved.status,
            SymmetricConditionStatus::numerically_singular);

  auto short_matrix = upload({Real{1}, Real{0}, Real{0}});
  EXPECT_THROW(
      (void)quasar::numerics::estimate_symmetric_condition(short_matrix, 2),
      std::invalid_argument);
  EXPECT_THROW(
      (void)quasar::numerics::estimate_symmetric_condition(matrix, 0),
      std::invalid_argument);
  EXPECT_THROW(
      (void)quasar::numerics::estimate_symmetric_condition(
          matrix, 2, MatrixTriangle::lower, Real{-1}),
      std::invalid_argument);
}

}  // namespace
