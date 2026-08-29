#include "quasar/numerics/block_tridiagonal.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::numerics::BlockTridiagonalStatus;

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

std::size_t block_matrix_index(int block, int order, int row, int column) {
  return static_cast<std::size_t>(block) * order * order
       + static_cast<std::size_t>(row)
       + static_cast<std::size_t>(column) * order;
}

std::size_t block_rhs_index(int block, int order, int rhs_count,
                            int row, int rhs) {
  return static_cast<std::size_t>(block) * order * rhs_count
       + static_cast<std::size_t>(row)
       + static_cast<std::size_t>(rhs) * order;
}

std::vector<Real> apply_block_tridiagonal(
    const std::vector<Real>& lower,
    const std::vector<Real>& diagonal,
    const std::vector<Real>& upper,
    const std::vector<Real>& x,
    int block_count,
    int block_order,
    int rhs_count) {
  std::vector<Real> result(x.size(), Real{0});
  for (int block = 0; block < block_count; ++block) {
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
      for (int row = 0; row < block_order; ++row) {
        Real sum = Real{0};
        for (int column = 0; column < block_order; ++column) {
          sum += diagonal[block_matrix_index(block, block_order, row, column)]
               * x[block_rhs_index(block, block_order, rhs_count,
                                   column, rhs)];
          if (block > 0) {
            sum += lower[block_matrix_index(block - 1, block_order,
                                            row, column)]
                 * x[block_rhs_index(block - 1, block_order, rhs_count,
                                     column, rhs)];
          }
          if (block + 1 < block_count) {
            sum += upper[block_matrix_index(block, block_order, row, column)]
                 * x[block_rhs_index(block + 1, block_order, rhs_count,
                                     column, rhs)];
          }
        }
        result[block_rhs_index(block, block_order, rhs_count, row, rhs)] = sum;
      }
    }
  }
  return result;
}

std::vector<Real> assemble_global_matrix(const std::vector<Real>& lower,
                                         const std::vector<Real>& diagonal,
                                         const std::vector<Real>& upper,
                                         int block_count,
                                         int block_order) {
  const int n = block_count * block_order;
  std::vector<Real> matrix(static_cast<std::size_t>(n * n), Real{0});
  const auto put_block = [&](const std::vector<Real>& blocks,
                             int stored_block, int block_row,
                             int block_column) {
    for (int row = 0; row < block_order; ++row) {
      for (int column = 0; column < block_order; ++column) {
        const int global_row = block_row * block_order + row;
        const int global_column = block_column * block_order + column;
        matrix[static_cast<std::size_t>(global_row * n + global_column)] =
            blocks[block_matrix_index(stored_block, block_order, row, column)];
      }
    }
  };
  for (int block = 0; block < block_count; ++block) {
    put_block(diagonal, block, block, block);
    if (block + 1 < block_count) {
      put_block(upper, block, block, block + 1);
      put_block(lower, block, block + 1, block);
    }
  }
  return matrix;
}

// Small independent dense partial-pivoting oracle, operating in row-major
// storage.  It deliberately shares no factorization code with the GPU path.
std::vector<Real> dense_lu_solve(std::vector<Real> matrix,
                                 std::vector<Real> rhs,
                                 int n,
                                 int rhs_count) {
  for (int pivot_column = 0; pivot_column < n; ++pivot_column) {
    int pivot_row = pivot_column;
    for (int row = pivot_column + 1; row < n; ++row) {
      if (std::abs(matrix[static_cast<std::size_t>(row * n + pivot_column)])
          > std::abs(matrix[static_cast<std::size_t>(pivot_row * n
                                                     + pivot_column)])) {
        pivot_row = row;
      }
    }
    if (matrix[static_cast<std::size_t>(pivot_row * n + pivot_column)]
        == Real{0}) {
      throw std::runtime_error{"dense oracle encountered a singular matrix"};
    }
    if (pivot_row != pivot_column) {
      for (int column = 0; column < n; ++column) {
        std::swap(matrix[static_cast<std::size_t>(pivot_column * n + column)],
                  matrix[static_cast<std::size_t>(pivot_row * n + column)]);
      }
      for (int rhs_column = 0; rhs_column < rhs_count; ++rhs_column) {
        std::swap(rhs[static_cast<std::size_t>(pivot_column * rhs_count
                                              + rhs_column)],
                  rhs[static_cast<std::size_t>(pivot_row * rhs_count
                                              + rhs_column)]);
      }
    }
    for (int row = pivot_column + 1; row < n; ++row) {
      const Real factor =
          matrix[static_cast<std::size_t>(row * n + pivot_column)]
          / matrix[static_cast<std::size_t>(pivot_column * n + pivot_column)];
      matrix[static_cast<std::size_t>(row * n + pivot_column)] = Real{0};
      for (int column = pivot_column + 1; column < n; ++column) {
        matrix[static_cast<std::size_t>(row * n + column)] -=
            factor
            * matrix[static_cast<std::size_t>(pivot_column * n + column)];
      }
      for (int rhs_column = 0; rhs_column < rhs_count; ++rhs_column) {
        rhs[static_cast<std::size_t>(row * rhs_count + rhs_column)] -=
            factor * rhs[static_cast<std::size_t>(pivot_column * rhs_count
                                                  + rhs_column)];
      }
    }
  }

  for (int row = n; row-- > 0;) {
    for (int rhs_column = 0; rhs_column < rhs_count; ++rhs_column) {
      Real value = rhs[static_cast<std::size_t>(row * rhs_count + rhs_column)];
      for (int column = row + 1; column < n; ++column) {
        value -= matrix[static_cast<std::size_t>(row * n + column)]
               * rhs[static_cast<std::size_t>(column * rhs_count
                                              + rhs_column)];
      }
      rhs[static_cast<std::size_t>(row * rhs_count + rhs_column)] =
          value / matrix[static_cast<std::size_t>(row * n + row)];
    }
  }
  return rhs;
}

std::vector<Real> block_rhs_to_row_major(const std::vector<Real>& block_rhs,
                                         int block_count,
                                         int block_order,
                                         int rhs_count) {
  const int n = block_count * block_order;
  std::vector<Real> result(static_cast<std::size_t>(n * rhs_count));
  for (int block = 0; block < block_count; ++block) {
    for (int row = 0; row < block_order; ++row) {
      for (int rhs = 0; rhs < rhs_count; ++rhs) {
        result[static_cast<std::size_t>((block * block_order + row)
                                        * rhs_count + rhs)] =
            block_rhs[block_rhs_index(block, block_order, rhs_count,
                                      row, rhs)];
      }
    }
  }
  return result;
}

TEST(BlockTridiagonal, NonsymmetricPivotedMultiRhsMatchesGlobalDenseLu) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  constexpr int blocks = 3;
  constexpr int order = 2;
  constexpr int rhs_count = 2;
  const std::vector<Real> diagonal{
      // D0 has a zero leading entry and therefore requires an intra-block row
      // interchange.  The complete block remains nonsingular.
      Real{0}, Real{1}, Real{2}, Real{3},
      Real{4}, Real{-2}, Real{1}, Real{3},
      Real{3}, Real{2}, Real{-1}, Real{5},
  };
  const std::vector<Real> lower{
      Real{-0.5}, Real{1}, Real{0.25}, Real{0.5},
      Real{1}, Real{0.5}, Real{-0.25}, Real{1},
  };
  const std::vector<Real> upper{
      Real{1}, Real{0.5}, Real{-1}, Real{2},
      Real{0.5}, Real{-1}, Real{1.5}, Real{0.75},
  };
  const std::vector<Real> exact{
      Real{1}, Real{-2}, Real{0.5}, Real{1.5},
      Real{3}, Real{0.25}, Real{-1}, Real{2},
      Real{-0.75}, Real{4}, Real{2.5}, Real{-3},
  };
  const auto rhs = apply_block_tridiagonal(
      lower, diagonal, upper, exact, blocks, order, rhs_count);

  auto device_lower = upload(lower);
  auto device_diagonal = upload(diagonal);
  auto device_upper = upload(upper);
  auto device_rhs = upload(rhs);
  const auto result = quasar::numerics::solve_block_tridiagonal(
      device_lower, device_diagonal, device_upper, device_rhs,
      blocks, order, rhs_count);
  ASSERT_EQ(result.status, BlockTridiagonalStatus::success);
  EXPECT_EQ(result.factorization_info, 0);
  EXPECT_EQ(result.solve_info, 0);

  const auto got = download(result.solution);
  ASSERT_EQ(got.size(), exact.size());
  for (std::size_t i = 0; i < exact.size(); ++i) {
    EXPECT_NEAR(got[i], exact[i], Real{2e-12}) << "flat index " << i;
  }

  const auto dense_matrix =
      assemble_global_matrix(lower, diagonal, upper, blocks, order);
  const auto dense_rhs =
      block_rhs_to_row_major(rhs, blocks, order, rhs_count);
  const auto dense_solution =
      dense_lu_solve(dense_matrix, dense_rhs, blocks * order, rhs_count);
  for (int block = 0; block < blocks; ++block) {
    for (int row = 0; row < order; ++row) {
      for (int rhs_column = 0; rhs_column < rhs_count; ++rhs_column) {
        EXPECT_NEAR(
            got[block_rhs_index(block, order, rhs_count, row, rhs_column)],
            dense_solution[static_cast<std::size_t>(
                (block * order + row) * rhs_count + rhs_column)],
            Real{2e-12});
      }
    }
  }

  // Factorization and solve both promise not to overwrite their const inputs.
  EXPECT_EQ(download(device_lower), lower);
  EXPECT_EQ(download(device_diagonal), diagonal);
  EXPECT_EQ(download(device_upper), upper);
  EXPECT_EQ(download(device_rhs), rhs);
}

TEST(BlockTridiagonal, ScalarBlocksSolveToeplitzSystem) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  constexpr int blocks = 8;
  constexpr int order = 1;
  constexpr int rhs_count = 1;
  const std::vector<Real> diagonal(static_cast<std::size_t>(blocks), Real{2});
  const std::vector<Real> lower(static_cast<std::size_t>(blocks - 1), Real{-1});
  const std::vector<Real> upper(static_cast<std::size_t>(blocks - 1), Real{-1});
  std::vector<Real> exact(static_cast<std::size_t>(blocks));
  for (int i = 0; i < blocks; ++i) {
    exact[static_cast<std::size_t>(i)] = static_cast<Real>((i + 1) * (i + 1));
  }
  const auto rhs = apply_block_tridiagonal(
      lower, diagonal, upper, exact, blocks, order, rhs_count);

  auto device_lower = upload(lower);
  auto device_diagonal = upload(diagonal);
  auto device_upper = upload(upper);
  auto device_rhs = upload(rhs);
  const auto result = quasar::numerics::solve_block_tridiagonal(
      device_lower, device_diagonal, device_upper, device_rhs,
      blocks, order, rhs_count);
  ASSERT_TRUE(result.ok());
  const auto got = download(result.solution);
  for (int i = 0; i < blocks; ++i) {
    EXPECT_NEAR(got[static_cast<std::size_t>(i)],
                exact[static_cast<std::size_t>(i)], Real{2e-12});
  }
}

TEST(BlockTridiagonal, ReusesOneFactorizationForIndependentRhsBatches) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  constexpr int blocks = 3;
  constexpr int order = 2;
  const std::vector<Real> diagonal{
      Real{4}, Real{1}, Real{2}, Real{5},
      Real{6}, Real{-1}, Real{1}, Real{4},
      Real{5}, Real{2}, Real{-1}, Real{3},
  };
  const std::vector<Real> lower{
      Real{0.5}, Real{-0.25}, Real{1}, Real{0.75},
      Real{-0.5}, Real{0.25}, Real{0.5}, Real{-1},
  };
  const std::vector<Real> upper{
      Real{0.25}, Real{0.5}, Real{-0.75}, Real{1},
      Real{1}, Real{-0.5}, Real{0.25}, Real{0.5},
  };
  const std::vector<Real> exact_one{
      Real{1}, Real{-2}, Real{3}, Real{0.5}, Real{-1}, Real{4},
  };
  const std::vector<Real> exact_two{
      Real{-3}, Real{1.5}, Real{0.25}, Real{2}, Real{5}, Real{-0.75},
  };

  auto device_lower = upload(lower);
  auto device_diagonal = upload(diagonal);
  auto device_upper = upload(upper);
  const auto factors = quasar::numerics::factor_block_tridiagonal(
      device_lower, device_diagonal, device_upper, blocks, order);
  ASSERT_TRUE(factors.ok());

  const auto rhs_one = apply_block_tridiagonal(
      lower, diagonal, upper, exact_one, blocks, order, 1);
  const auto rhs_two = apply_block_tridiagonal(
      lower, diagonal, upper, exact_two, blocks, order, 1);
  auto device_rhs_one = upload(rhs_one);
  auto device_rhs_two = upload(rhs_two);

  const auto first =
      quasar::numerics::solve_block_tridiagonal(factors, device_rhs_one, 1);
  ASSERT_TRUE(first.ok());
  const auto second =
      quasar::numerics::solve_block_tridiagonal(factors, device_rhs_two, 1);
  ASSERT_TRUE(second.ok());

  const auto got_one = download(first.solution);
  const auto got_two = download(second.solution);
  for (std::size_t i = 0; i < exact_one.size(); ++i) {
    EXPECT_NEAR(got_one[i], exact_one[i], Real{3e-12}) << "first rhs " << i;
    EXPECT_NEAR(got_two[i], exact_two[i], Real{3e-12}) << "second rhs " << i;
  }

  // Reuse must not mutate either the factors or either caller-owned RHS.
  EXPECT_EQ(download(device_rhs_one), rhs_one);
  EXPECT_EQ(download(device_rhs_two), rhs_two);
  EXPECT_EQ(factors.factorization_info, 0);
  EXPECT_EQ(factors.solve_info, 0);
}

TEST(BlockTridiagonal, RejectsMalformedReusableSolveStorage) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  auto lower = upload({Real{-1}});
  auto diagonal = upload({Real{2}, Real{2}});
  auto upper = upload({Real{-1}});
  const auto factors = quasar::numerics::factor_block_tridiagonal(
      lower, diagonal, upper, 2, 1);
  ASSERT_TRUE(factors.ok());

  auto short_rhs = upload({Real{1}});
  EXPECT_THROW(
      (void)quasar::numerics::solve_block_tridiagonal(factors, short_rhs, 1),
      std::invalid_argument);
  auto rhs = upload({Real{1}, Real{1}});
  EXPECT_THROW(
      (void)quasar::numerics::solve_block_tridiagonal(factors, rhs, 0),
      std::invalid_argument);
}

TEST(BlockTridiagonal, ReportsSingularSchurBlockAndPivot) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  // Eliminating block 0 leaves D1 - L0 D0^-1 U0 = 1 - 1 = 0.
  auto lower = upload({Real{1}});
  auto diagonal = upload({Real{1}, Real{1}});
  auto upper = upload({Real{1}});
  auto rhs = upload({Real{2}, Real{2}});
  const auto result = quasar::numerics::solve_block_tridiagonal(
      lower, diagonal, upper, rhs, 2, 1, 1);

  EXPECT_EQ(result.status,
            BlockTridiagonalStatus::singular_diagonal_block);
  EXPECT_EQ(result.failure_block, 1);
  EXPECT_EQ(result.failure_pivot, 0);
  EXPECT_EQ(result.factorization_info, 1);
  EXPECT_EQ(result.solve_info, 0);
}

TEST(BlockTridiagonal, GloballyNonsingularSystemCanNeedInterBlockPivoting) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();

  // [[0,1],[1,0]] has determinant -1, but scalar block 0 is singular.  This is
  // the documented boundary of block Thomas with intra-block pivots only.
  auto lower = upload({Real{1}});
  auto diagonal = upload({Real{0}, Real{0}});
  auto upper = upload({Real{1}});
  auto rhs = upload({Real{2}, Real{3}});
  const auto result = quasar::numerics::solve_block_tridiagonal(
      lower, diagonal, upper, rhs, 2, 1, 1);

  EXPECT_EQ(result.status,
            BlockTridiagonalStatus::singular_diagonal_block);
  EXPECT_EQ(result.failure_block, 0);
  EXPECT_EQ(result.failure_pivot, 0);
  EXPECT_EQ(result.factorization_info, 1);
  EXPECT_EQ(result.solve_info, 0);
}

}  // namespace
