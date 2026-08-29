#include "quasar/numerics/generalized_eigensolver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::numerics::GeneralizedEigenStatus;

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

Real element(const std::vector<Real>& matrix, int n, int row, int column) {
  return matrix[static_cast<std::size_t>(row)
                + static_cast<std::size_t>(column) * n];
}

std::vector<Real> matvec(const std::vector<Real>& matrix,
                         const std::vector<Real>& vector,
                         int n) {
  std::vector<Real> result(static_cast<std::size_t>(n), Real{0});
  for (int row = 0; row < n; ++row) {
    for (int column = 0; column < n; ++column) {
      result[static_cast<std::size_t>(row)] +=
          element(matrix, n, row, column)
          * vector[static_cast<std::size_t>(column)];
    }
  }
  return result;
}

TEST(GeneralizedEigensolver, ReproducesNonDiagonalPairAndBOrthogonality) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  // A and B share the ordinary eigenvectors (1,1) and (1,-1).  Their
  // eigenvalue ratios are exactly 9/3=3 and 4/1=4, respectively.
  constexpr int n = 2;
  const std::vector<Real> a{
      Real{6.5}, Real{2.5},
      Real{2.5}, Real{6.5},
  };
  const std::vector<Real> b{
      Real{2}, Real{1},
      Real{1}, Real{2},
  };
  auto device_a = upload(a);
  auto device_b = upload(b);

  const auto result =
      quasar::numerics::solve_generalized_symmetric_eigenproblem(
          device_a, device_b, n);
  ASSERT_EQ(result.status, GeneralizedEigenStatus::success);
  EXPECT_EQ(result.solver_info, 0);

  const auto eigenvalues = download(result.eigenvalues);
  const auto eigenvectors = download(result.eigenvectors);
  ASSERT_EQ(eigenvalues.size(), 2U);
  EXPECT_NEAR(eigenvalues[0], Real{3}, Real{2e-13});
  EXPECT_NEAR(eigenvalues[1], Real{4}, Real{2e-13});

  for (int column = 0; column < n; ++column) {
    std::vector<Real> v(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
      v[static_cast<std::size_t>(row)] =
          element(eigenvectors, n, row, column);
    }
    const auto av = matvec(a, v, n);
    const auto bv = matvec(b, v, n);
    for (int row = 0; row < n; ++row) {
      EXPECT_NEAR(av[static_cast<std::size_t>(row)],
                  eigenvalues[static_cast<std::size_t>(column)]
                      * bv[static_cast<std::size_t>(row)],
                  Real{2e-13});
    }
  }

  // hipSOLVER normalizes generalized eigenvectors in the B inner product.
  for (int left = 0; left < n; ++left) {
    for (int right = 0; right < n; ++right) {
      Real product = Real{0};
      for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
          product += element(eigenvectors, n, row, left)
                   * element(b, n, row, column)
                   * element(eigenvectors, n, column, right);
        }
      }
      EXPECT_NEAR(product, left == right ? Real{1} : Real{0}, Real{3e-13});
    }
  }

  // The wrapper promises const inputs despite sygvd's destructive ABI.
  EXPECT_EQ(download(device_a), a);
  EXPECT_EQ(download(device_b), b);
}

TEST(GeneralizedEigensolver, ReproducesDirichletToeplitzSpectrum) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  constexpr int n = 9;
  std::vector<Real> a(static_cast<std::size_t>(n * n), Real{0});
  std::vector<Real> b(static_cast<std::size_t>(n * n), Real{0});
  for (int i = 0; i < n; ++i) {
    a[static_cast<std::size_t>(i + i * n)] = Real{2};
    b[static_cast<std::size_t>(i + i * n)] = Real{1};
    if (i + 1 < n) {
      a[static_cast<std::size_t>((i + 1) + i * n)] = Real{-1};
      a[static_cast<std::size_t>(i + (i + 1) * n)] = Real{-1};
    }
  }

  auto device_a = upload(a);
  auto device_b = upload(b);
  const auto result =
      quasar::numerics::solve_generalized_symmetric_eigenproblem(
          device_a, device_b, n);
  ASSERT_TRUE(result.ok());

  const auto eigenvalues = download(result.eigenvalues);
  for (int k = 1; k <= n; ++k) {
    const Real expected = Real{2} - Real{2}
        * std::cos(static_cast<Real>(k) * quasar::pi
                   / static_cast<Real>(n + 1));
    EXPECT_NEAR(eigenvalues[static_cast<std::size_t>(k - 1)], expected,
                Real{3e-13}) << "mode " << k;
  }
}

TEST(GeneralizedEigensolver, ReportsIndefiniteMassLeadingMinor) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  constexpr int n = 2;
  auto device_a = upload({Real{1}, Real{0}, Real{0}, Real{1}});
  auto device_b = upload({Real{1}, Real{0}, Real{0}, Real{-1}});

  const auto result =
      quasar::numerics::solve_generalized_symmetric_eigenproblem(
          device_a, device_b, n);
  EXPECT_EQ(result.status,
            GeneralizedEigenStatus::mass_not_positive_definite);
  EXPECT_EQ(result.solver_info, n + 2);
  EXPECT_EQ(result.failed_leading_minor_order, 2);
  EXPECT_EQ(result.invalid_argument_position, 0);
}

TEST(GeneralizedEigensolver, SupportsUpperTriangleAndRejectsMalformedShapes) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  constexpr int n = 2;
  // Deliberately poison the unused lower triangle.  The upper-triangle solve
  // must see diag(2, 8) against diag(1, 2), hence eigenvalues 2 and 4.
  auto a = upload({Real{2}, Real{-99}, Real{0}, Real{8}});
  auto b = upload({Real{1}, Real{77}, Real{0}, Real{2}});
  const auto result =
      quasar::numerics::solve_generalized_symmetric_eigenproblem(
          a, b, n, quasar::numerics::MatrixTriangle::upper);
  ASSERT_TRUE(result.ok());
  const auto eigenvalues = download(result.eigenvalues);
  EXPECT_NEAR(eigenvalues[0], Real{2}, Real{2e-13});
  EXPECT_NEAR(eigenvalues[1], Real{4}, Real{2e-13});

  auto short_a = upload({Real{1}, Real{0}, Real{0}});
  EXPECT_THROW(
      (void)quasar::numerics::solve_generalized_symmetric_eigenproblem(
          short_a, b, n),
      std::invalid_argument);
  EXPECT_THROW(
      (void)quasar::numerics::solve_generalized_symmetric_eigenproblem(
          a, b, 0),
      std::invalid_argument);
}

}  // namespace
