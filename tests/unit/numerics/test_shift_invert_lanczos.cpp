// Shift-invert Lanczos against a dense reference.
//
// Every case here builds a pencil whose exact eigenvalues are either known in
// closed form or computed by the dense symmetric-definite eigensolver, so the
// assertions compare the iteration to an independent answer rather than to
// itself.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/block_tridiagonal.hpp"
#include "quasar/numerics/generalized_eigensolver.hpp"
#include "quasar/numerics/shift_invert_lanczos.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::numerics::BlockPartition;
using quasar::numerics::ShiftInvertLanczosConfig;
using quasar::numerics::ShiftInvertStatus;

DeviceBuffer<Real> upload(const std::vector<Real>& host) {
  DeviceBuffer<Real> device{host.size(), quasar::backend::uninitialized};
  device.copy_from_host_async(host.data(), host.size(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  return device;
}

// Dense (order x order) column-major index.
std::size_t at(int order, int row, int column) {
  return static_cast<std::size_t>(row)
       + static_cast<std::size_t>(column) * static_cast<std::size_t>(order);
}

// Scatters a dense block-tridiagonal matrix into the per-block storage the
// factorization expects, for an arbitrary partition.
struct BlockStorage {
  std::vector<Real> lower;
  std::vector<Real> diagonal;
  std::vector<Real> upper;
};

BlockStorage scatter(const std::vector<Real>& dense, int order,
                     const BlockPartition& partition) {
  BlockStorage storage;
  storage.lower.assign(partition.off_diagonal_size(), Real{0});
  storage.diagonal.assign(partition.diagonal_size(), Real{0});
  storage.upper.assign(partition.off_diagonal_size(), Real{0});

  std::vector<int> first_row(
      static_cast<std::size_t>(partition.block_count()) + 1, 0);
  for (int block = 0; block < partition.block_count(); ++block) {
    first_row[static_cast<std::size_t>(block) + 1] =
        first_row[static_cast<std::size_t>(block)]
        + partition.block_order(block);
  }

  for (int block = 0; block < partition.block_count(); ++block) {
    const int n = partition.block_order(block);
    const int base = first_row[static_cast<std::size_t>(block)];
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < n; ++row) {
        storage.diagonal[partition.diagonal_offset(block)
                         + static_cast<std::size_t>(row)
                         + static_cast<std::size_t>(column) * n] =
            dense[at(order, base + row, base + column)];
      }
    }
    if (block + 1 >= partition.block_count()) continue;

    const int next = partition.block_order(block + 1);
    const int next_base = first_row[static_cast<std::size_t>(block) + 1];
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < next; ++row) {
        storage.lower[partition.off_diagonal_offset(block)
                      + static_cast<std::size_t>(row)
                      + static_cast<std::size_t>(column) * next] =
            dense[at(order, next_base + row, base + column)];
      }
    }
    for (int column = 0; column < next; ++column) {
      for (int row = 0; row < n; ++row) {
        storage.upper[partition.off_diagonal_offset(block)
                      + static_cast<std::size_t>(row)
                      + static_cast<std::size_t>(column) * n] =
            dense[at(order, base + row, next_base + column)];
      }
    }
  }
  return storage;
}

// K is the 1-D Laplacian, M the identity: eigenvalues 4 sin^2(k pi / 2(n+1)).
struct LaplacePencil {
  int order;
  std::vector<Real> stiffness;
  std::vector<Real> mass;
};

LaplacePencil make_laplace_pencil(int order, Real mass_scale) {
  LaplacePencil pencil;
  pencil.order = order;
  pencil.stiffness.assign(
      static_cast<std::size_t>(order) * static_cast<std::size_t>(order),
      Real{0});
  pencil.mass = pencil.stiffness;
  for (int index = 0; index < order; ++index) {
    pencil.stiffness[at(order, index, index)] = Real{2};
    if (index + 1 < order) {
      pencil.stiffness[at(order, index, index + 1)] = Real{-1};
      pencil.stiffness[at(order, index + 1, index)] = Real{-1};
    }
    pencil.mass[at(order, index, index)] = mass_scale;
  }
  return pencil;
}

std::vector<Real> dense_reference(const LaplacePencil& pencil) {
  auto stiffness = upload(pencil.stiffness);
  auto mass = upload(pencil.mass);
  const auto solved = quasar::numerics::solve_generalized_symmetric_eigenproblem(
      stiffness, mass, pencil.order);
  EXPECT_TRUE(solved.ok());
  std::vector<Real> values(static_cast<std::size_t>(pencil.order));
  solved.eigenvalues.copy_to_host_async(values.data(), values.size(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  return values;
}

std::vector<Real> shifted_dense(const LaplacePencil& pencil, Real shift) {
  std::vector<Real> shifted = pencil.stiffness;
  for (std::size_t index = 0; index < shifted.size(); ++index) {
    shifted[index] -= shift * pencil.mass[index];
  }
  return shifted;
}

TEST(ShiftInvertLanczos, MatchesTheDenseSpectrumWithUniformBlocks) {
  constexpr int kBlockCount = 8;
  constexpr int kBlockOrder = 3;
  const int order = kBlockCount * kBlockOrder;
  const LaplacePencil pencil = make_laplace_pencil(order, Real{1});
  const std::vector<Real> reference = dense_reference(pencil);

  const BlockPartition partition =
      BlockPartition::uniform(kBlockCount, kBlockOrder);
  const Real shift = Real{-0.05};
  const BlockStorage storage =
      scatter(shifted_dense(pencil, shift), order, partition);

  auto factors = quasar::numerics::factor_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      partition);
  ASSERT_TRUE(factors.ok());

  ShiftInvertLanczosConfig config;
  config.wanted = 4;
  config.shift = shift;
  config.residual_tolerance = Real{1e-11};
  const auto result = quasar::numerics::solve_shift_invert_lanczos(
      factors, upload(pencil.mass), config);

  ASSERT_EQ(result.status, ShiftInvertStatus::success) << result.iterations;
  ASSERT_EQ(result.eigenvalues.size(), 4u);
  for (std::size_t index = 0; index < result.eigenvalues.size(); ++index) {
    EXPECT_NEAR(result.eigenvalues[index], reference[index], 1e-9)
        << "eigenvalue " << index;
  }
}

// The Chebyshev spectral-element case: block sizes are not all equal, which is
// the structural reason the non-uniform partition exists at all.
TEST(ShiftInvertLanczos, MatchesTheDenseSpectrumWithNonUniformBlocks) {
  const std::vector<int> orders{4, 4, 4, 1};
  const int total = 13;
  const LaplacePencil pencil = make_laplace_pencil(total, Real{2});
  const std::vector<Real> reference = dense_reference(pencil);

  const BlockPartition partition{orders};
  ASSERT_FALSE(partition.is_uniform());
  ASSERT_EQ(partition.total_rows(), static_cast<std::size_t>(total));

  const Real shift = Real{-0.25};
  const BlockStorage storage =
      scatter(shifted_dense(pencil, shift), total, partition);

  auto factors = quasar::numerics::factor_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      partition);
  ASSERT_TRUE(factors.ok());

  ShiftInvertLanczosConfig config;
  config.wanted = 3;
  config.shift = shift;
  const auto result = quasar::numerics::solve_shift_invert_lanczos(
      factors, upload(pencil.mass), config);

  ASSERT_EQ(result.status, ShiftInvertStatus::success);
  ASSERT_EQ(result.eigenvalues.size(), 3u);
  for (std::size_t index = 0; index < result.eigenvalues.size(); ++index) {
    EXPECT_NEAR(result.eigenvalues[index], reference[index], 1e-9);
  }
}

TEST(ShiftInvertLanczos, TargetsTheEigenvaluesNearestTheShift) {
  const int order = 12;
  const LaplacePencil pencil = make_laplace_pencil(order, Real{1});
  const std::vector<Real> reference = dense_reference(pencil);

  // A shift in the middle of the spectrum must return the interior
  // eigenvalues around it, not the smallest ones.
  const Real shift = reference[6] - Real{1e-3};
  const BlockPartition partition = BlockPartition::uniform(6, 2);
  const BlockStorage storage =
      scatter(shifted_dense(pencil, shift), order, partition);

  auto factors = quasar::numerics::factor_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      partition);
  ASSERT_TRUE(factors.ok());

  ShiftInvertLanczosConfig config;
  config.wanted = 2;
  config.shift = shift;
  const auto result = quasar::numerics::solve_shift_invert_lanczos(
      factors, upload(pencil.mass), config);

  ASSERT_EQ(result.status, ShiftInvertStatus::success);
  ASSERT_EQ(result.eigenvalues.size(), 2u);
  for (const Real value : result.eigenvalues) {
    const auto nearest = std::min_element(
        reference.begin(), reference.end(),
        [value](Real a, Real b) {
          return std::abs(a - value) < std::abs(b - value);
        });
    EXPECT_NEAR(value, *nearest, 1e-9);
    // The returned pair must be closer to the shift than the extremes are.
    EXPECT_LT(std::abs(value - shift), std::abs(reference.back() - shift));
  }
}

TEST(ShiftInvertLanczos, IsReproducibleForAFixedSeed) {
  const int order = 10;
  const LaplacePencil pencil = make_laplace_pencil(order, Real{1});
  const BlockPartition partition = BlockPartition::uniform(5, 2);
  const Real shift = Real{-0.1};
  const BlockStorage storage =
      scatter(shifted_dense(pencil, shift), order, partition);

  auto factors = quasar::numerics::factor_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      partition);
  ASSERT_TRUE(factors.ok());

  ShiftInvertLanczosConfig config;
  config.wanted = 3;
  config.shift = shift;
  config.random_seed = 20260830;
  const auto first = quasar::numerics::solve_shift_invert_lanczos(
      factors, upload(pencil.mass), config);
  const auto second = quasar::numerics::solve_shift_invert_lanczos(
      factors, upload(pencil.mass), config);

  ASSERT_TRUE(first.ok());
  ASSERT_EQ(first.eigenvalues.size(), second.eigenvalues.size());
  for (std::size_t index = 0; index < first.eigenvalues.size(); ++index) {
    EXPECT_EQ(first.eigenvalues[index], second.eigenvalues[index]);
  }
  EXPECT_EQ(first.iterations, second.iterations);
}

TEST(ShiftInvertLanczos, ReportsASingularShiftedMatrix) {
  const int order = 6;
  const LaplacePencil pencil = make_laplace_pencil(order, Real{1});
  const std::vector<Real> reference = dense_reference(pencil);

  // Shifting exactly onto an eigenvalue makes (K - sigma M) singular.
  const BlockPartition partition = BlockPartition::uniform(3, 2);
  const BlockStorage storage =
      scatter(shifted_dense(pencil, reference[0]), order, partition);
  auto factors = quasar::numerics::factor_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      partition);

  if (factors.ok()) {
    // Round-off can leave the shifted matrix numerically nonsingular; the
    // iteration must then still return the shifted eigenvalue, not a NaN.
    ShiftInvertLanczosConfig config;
    config.wanted = 1;
    config.shift = reference[0];
    const auto result = quasar::numerics::solve_shift_invert_lanczos(
        factors, upload(pencil.mass), config);
    if (result.ok()) {
      ASSERT_EQ(result.eigenvalues.size(), 1u);
      EXPECT_TRUE(std::isfinite(result.eigenvalues.front()));
      EXPECT_NEAR(result.eigenvalues.front(), reference[0], 1e-6);
    }
    return;
  }

  ShiftInvertLanczosConfig config;
  config.wanted = 1;
  config.shift = reference[0];
  const auto result = quasar::numerics::solve_shift_invert_lanczos(
      factors, upload(pencil.mass), config);
  EXPECT_EQ(result.status, ShiftInvertStatus::shifted_matrix_singular);
}

TEST(ShiftInvertLanczos, RejectsInvalidArguments) {
  const int order = 6;
  const LaplacePencil pencil = make_laplace_pencil(order, Real{1});
  const BlockPartition partition = BlockPartition::uniform(3, 2);
  const BlockStorage storage =
      scatter(shifted_dense(pencil, Real{-1}), order, partition);
  auto factors = quasar::numerics::factor_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      partition);
  ASSERT_TRUE(factors.ok());
  auto mass = upload(pencil.mass);

  ShiftInvertLanczosConfig config;
  config.shift = Real{-1};

  config.wanted = 0;
  EXPECT_THROW((void)quasar::numerics::solve_shift_invert_lanczos(
                   factors, mass, config),
               std::invalid_argument);

  config.wanted = order + 1;
  EXPECT_THROW((void)quasar::numerics::solve_shift_invert_lanczos(
                   factors, mass, config),
               std::invalid_argument);

  config.wanted = 1;
  config.residual_tolerance = Real{-1};
  EXPECT_THROW((void)quasar::numerics::solve_shift_invert_lanczos(
                   factors, mass, config),
               std::invalid_argument);

  config.residual_tolerance = Real{1e-10};
  std::vector<Real> short_mass(pencil.mass.begin(),
                               pencil.mass.begin() + 4);
  auto small = upload(short_mass);
  EXPECT_THROW((void)quasar::numerics::solve_shift_invert_lanczos(
                   factors, small, config),
               std::invalid_argument);
}

TEST(BlockPartition, RejectsMalformedOrders) {
  // The braced initializers below hold commas, so each construction is wrapped
  // in a helper rather than passed straight to the two-argument macro.
  const auto construct = [](std::vector<int> orders) {
    return BlockPartition{std::move(orders)};
  };
  EXPECT_THROW((void)construct(std::vector<int>{}), std::invalid_argument);
  EXPECT_THROW((void)construct(std::vector<int>{2, 0, 3}),
               std::invalid_argument);
  EXPECT_THROW((void)BlockPartition::uniform(0, 3), std::invalid_argument);
  EXPECT_THROW((void)BlockPartition::uniform(3, -1), std::invalid_argument);
}

TEST(BlockPartition, ComputesOffsetsForNonUniformBlocks) {
  const BlockPartition partition{std::vector<int>{3, 2, 4}};
  EXPECT_EQ(partition.block_count(), 3);
  EXPECT_EQ(partition.total_rows(), 9u);
  EXPECT_EQ(partition.maximum_block_order(), 4);
  EXPECT_FALSE(partition.is_uniform());

  EXPECT_EQ(partition.diagonal_offset(0), 0u);
  EXPECT_EQ(partition.diagonal_offset(1), 9u);
  EXPECT_EQ(partition.diagonal_offset(2), 13u);
  EXPECT_EQ(partition.diagonal_size(), 29u);

  EXPECT_EQ(partition.off_diagonal_offset(0), 0u);
  EXPECT_EQ(partition.off_diagonal_offset(1), 6u);
  EXPECT_EQ(partition.off_diagonal_size(), 14u);

  EXPECT_EQ(partition.pivot_offset(0), 0u);
  EXPECT_EQ(partition.pivot_offset(1), 3u);
  EXPECT_EQ(partition.pivot_offset(2), 5u);

  EXPECT_EQ(partition.rhs_offset(1, 2), 6u);
  EXPECT_EQ(partition.rhs_size(2), 18u);

  EXPECT_THROW((void)partition.block_order(3), std::out_of_range);
  EXPECT_THROW((void)partition.off_diagonal_offset(2), std::out_of_range);
}

TEST(BlockTridiagonal, SolvesWithNonUniformBlocks) {
  const std::vector<int> orders{2, 3, 1};
  const BlockPartition partition{orders};
  const int total = 6;

  // A diagonally dominant symmetric block-tridiagonal system.
  std::vector<Real> dense(
      static_cast<std::size_t>(total) * static_cast<std::size_t>(total),
      Real{0});
  for (int index = 0; index < total; ++index) {
    dense[at(total, index, index)] = Real{4} + Real{index};
    if (index + 1 < total) {
      dense[at(total, index, index + 1)] = Real{-1};
      dense[at(total, index + 1, index)] = Real{-1};
    }
  }
  const BlockStorage storage = scatter(dense, total, partition);

  std::vector<Real> expected{1, -2, 3, -4, 5, -6};
  std::vector<Real> rhs(static_cast<std::size_t>(total), Real{0});
  for (int row = 0; row < total; ++row) {
    Real sum = 0;
    for (int column = 0; column < total; ++column) {
      sum += dense[at(total, row, column)]
           * expected[static_cast<std::size_t>(column)];
    }
    rhs[static_cast<std::size_t>(row)] = sum;
  }

  const auto result = quasar::numerics::solve_block_tridiagonal(
      upload(storage.lower), upload(storage.diagonal), upload(storage.upper),
      upload(rhs), partition, 1);
  ASSERT_TRUE(result.ok());

  std::vector<Real> solution(static_cast<std::size_t>(total));
  result.solution.copy_to_host_async(solution.data(), solution.size(),
                                     nullptr);
  quasar::backend::device_synchronize(nullptr);
  for (std::size_t index = 0; index < solution.size(); ++index) {
    EXPECT_NEAR(solution[index], expected[index], 1e-12);
  }
}

}  // namespace
