// The radial-major block structure of the assembled stability pencil, and the
// shift-invert path built on it.
//
// The load-bearing claim these tests defend is structural: reordered
// radial-major, the assembled matrices are block-tridiagonal EXACTLY, with no
// nonzeros discarded. If that ever stops holding, the extraction silently drops
// couplings and the shift-invert eigenvalues quietly stop matching the dense
// ones, so it is checked directly rather than inferred from the eigenvalues.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/block_tridiagonal.hpp"
#include "quasar/numerics/generalized_eigensolver.hpp"
#include "quasar/numerics/shift_invert_lanczos.hpp"
#include "quasar/physics/stability/spectral_blocks.hpp"
#include "quasar/physics/stability/spectral_layout.hpp"
#include "quasar/physics/stability/toroidal_energy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <vector>

namespace {

using quasar::Real;
using quasar::stability::DisplacementComponent;
using quasar::stability::FixedBoundaryDofMap;
using quasar::stability::SpectralBlockStatus;
using quasar::stability::SpectralDofLayout;
using quasar::stability::build_spectral_block_structure;

SpectralDofLayout make_layout(int n_domains, int order, int m_max) {
  return SpectralDofLayout{n_domains, order, m_max, 4 * m_max + 1};
}

std::vector<std::size_t> free_map(const SpectralDofLayout& layout) {
  const FixedBoundaryDofMap map{layout};
  return map.free_to_full_complex();
}

TEST(SpectralBlockStructure, PartitionsRadialNodesByDomain) {
  constexpr int kDomains = 3;
  constexpr int kOrder = 4;
  constexpr int kMMax = 2;
  const SpectralDofLayout layout = make_layout(kDomains, kOrder, kMMax);
  const auto structure =
      build_spectral_block_structure(layout, free_map(layout), kOrder);

  ASSERT_TRUE(structure.ok());
  // n_domains blocks of `order` radial nodes, plus the final shared endpoint.
  EXPECT_EQ(structure.partition.block_count(), kDomains + 1);
  EXPECT_FALSE(structure.partition.is_uniform())
      << "the trailing single-node block is what makes this non-uniform";

  const int harmonics = 2 * kMMax + 1;
  const int per_node = harmonics * SpectralDofLayout::kComponentCount * 2;
  for (int block = 0; block < kDomains; ++block) {
    EXPECT_EQ(structure.partition.block_order(block), kOrder * per_node);
  }
  // The last radial node has xi^lambda eliminated for every harmonic.
  EXPECT_EQ(structure.partition.block_order(kDomains),
            per_node - harmonics * 2);
}

TEST(SpectralBlockStructure, PermutationIsABijectionOntoFreeRealDofs) {
  const SpectralDofLayout layout = make_layout(2, 3, 1);
  const auto map = free_map(layout);
  const auto structure = build_spectral_block_structure(layout, map, 3);
  ASSERT_TRUE(structure.ok());

  EXPECT_EQ(structure.order, static_cast<int>(2 * map.size()));
  EXPECT_EQ(structure.permutation.size(),
            static_cast<std::size_t>(structure.order));
  const std::set<int> unique(structure.permutation.begin(),
                             structure.permutation.end());
  EXPECT_EQ(unique.size(), structure.permutation.size())
      << "every free real DOF must appear exactly once";
  EXPECT_EQ(*unique.begin(), 0);
  EXPECT_EQ(*unique.rbegin(), structure.order - 1);
}

TEST(SpectralBlockStructure, RejectsAMismatchedChebyshevOrder) {
  const SpectralDofLayout layout = make_layout(2, 3, 1);
  const auto map = free_map(layout);
  // The layout was built for order 3; claiming order 4 makes the radial count
  // inconsistent and must be refused rather than mis-partitioned.
  const auto structure = build_spectral_block_structure(layout, map, 4);
  EXPECT_FALSE(structure.ok());
  EXPECT_EQ(structure.status,
            SpectralBlockStatus::nonuniform_harmonic_radial_counts);
}

// Builds a synthetic pencil with exactly the sparsity the assembly produces:
// two DOFs couple only when their radial nodes share a Chebyshev domain.
struct SyntheticPencil {
  int order{0};
  std::vector<Real> stiffness{};
  std::vector<Real> inertia{};
};

SyntheticPencil make_synthetic_pencil(
    const quasar::stability::SpectralBlockStructure& structure,
    const SpectralDofLayout& layout, int chebyshev_order, bool leak) {
  const int order = structure.order;
  SyntheticPencil pencil;
  pencil.order = order;
  pencil.stiffness.assign(
      static_cast<std::size_t>(order) * static_cast<std::size_t>(order),
      Real{0});
  pencil.inertia = pencil.stiffness;

  // radial_of[p] is the radial node of permuted position p.
  std::vector<int> radial_of(static_cast<std::size_t>(order), 0);
  int position = 0;
  const auto radial_count =
      static_cast<int>(layout.n_radial(-layout.m_max()));
  for (int radial = 0; radial < radial_count; ++radial) {
    for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
      for (int component = 0;
           component < SpectralDofLayout::kComponentCount; ++component) {
        const std::size_t full = layout.complex_dof(
            static_cast<std::size_t>(radial), m,
            static_cast<DisplacementComponent>(component));
        const bool eliminated =
            radial == radial_count - 1
            && component == static_cast<int>(DisplacementComponent::psi);
        (void)full;
        if (eliminated) continue;
        radial_of[static_cast<std::size_t>(position)] = radial;
        radial_of[static_cast<std::size_t>(position) + 1] = radial;
        position += 2;
      }
    }
  }

  const auto share_domain = [chebyshev_order](int a, int b) {
    const int domain_a_low = (a - 1) / chebyshev_order;
    const int domain_a_high = a / chebyshev_order;
    const int domain_b_low = (b - 1) / chebyshev_order;
    const int domain_b_high = b / chebyshev_order;
    return std::max(domain_a_low, domain_b_low)
        <= std::min(domain_a_high, domain_b_high);
  };

  for (int row = 0; row < order; ++row) {
    for (int column = row; column < order; ++column) {
      const int radial_row = radial_of[static_cast<std::size_t>(row)];
      const int radial_column = radial_of[static_cast<std::size_t>(column)];
      if (!share_domain(radial_row, radial_column)) continue;
      const Real value =
          Real{1} / (Real{1} + std::abs(Real(row) - Real(column)));
      const std::size_t forward =
          static_cast<std::size_t>(structure.permutation[
              static_cast<std::size_t>(row)])
          + static_cast<std::size_t>(structure.permutation[
                static_cast<std::size_t>(column)])
                * static_cast<std::size_t>(order);
      const std::size_t backward =
          static_cast<std::size_t>(structure.permutation[
              static_cast<std::size_t>(column)])
          + static_cast<std::size_t>(structure.permutation[
                static_cast<std::size_t>(row)])
                * static_cast<std::size_t>(order);
      pencil.stiffness[forward] = value;
      pencil.stiffness[backward] = value;
      if (row == column) {
        pencil.stiffness[forward] = Real{8} + Real(row) * Real{0.01};
        pencil.inertia[forward] = Real{1} + Real(row) * Real{0.001};
      }
    }
  }

  if (leak) {
    // A coupling between the first and last radial node, which share no domain.
    const std::size_t first = static_cast<std::size_t>(
        structure.permutation.front());
    const std::size_t last = static_cast<std::size_t>(
        structure.permutation.back());
    pencil.stiffness[first + last * static_cast<std::size_t>(order)] =
        Real{0.5};
    pencil.stiffness[last + first * static_cast<std::size_t>(order)] =
        Real{0.5};
  }
  return pencil;
}

quasar::backend::DeviceBuffer<Real> upload(const std::vector<Real>& host) {
  quasar::backend::DeviceBuffer<Real> device{host.size(),
                                             quasar::backend::uninitialized};
  device.copy_from_host_async(host.data(), host.size(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  return device;
}

TEST(SpectralBlockExtraction, DiscardsNothingWhenTheStructureHolds) {
  constexpr int kOrder = 3;
  const SpectralDofLayout layout = make_layout(3, kOrder, 1);
  const auto structure =
      build_spectral_block_structure(layout, free_map(layout), kOrder);
  ASSERT_TRUE(structure.ok());

  const SyntheticPencil pencil =
      make_synthetic_pencil(structure, layout, kOrder, /*leak=*/false);
  quasar::stability::SpectralBlockMatrix blocks;
  quasar::stability::launch_extract_shifted_block_tridiagonal(
      upload(pencil.stiffness), upload(pencil.inertia), Real{-0.5}, structure,
      blocks, nullptr);

  EXPECT_EQ(blocks.maximum_outside_pattern, Real{0})
      << "the domain-local sparsity must be exactly block-tridiagonal";
}

TEST(SpectralBlockExtraction, ReportsCouplingsOutsideThePattern) {
  constexpr int kOrder = 3;
  const SpectralDofLayout layout = make_layout(3, kOrder, 1);
  const auto structure =
      build_spectral_block_structure(layout, free_map(layout), kOrder);
  ASSERT_TRUE(structure.ok());

  const SyntheticPencil pencil =
      make_synthetic_pencil(structure, layout, kOrder, /*leak=*/true);
  quasar::stability::SpectralBlockMatrix blocks;
  quasar::stability::launch_extract_shifted_block_tridiagonal(
      upload(pencil.stiffness), upload(pencil.inertia), Real{0}, structure,
      blocks, nullptr);

  EXPECT_NEAR(blocks.maximum_outside_pattern, Real{0.5}, 1e-15)
      << "a discarded coupling must be reported, not silently dropped";
}

TEST(SpectralBlockExtraction, PermutedMassMatchesAHostPermutation) {
  constexpr int kOrder = 2;
  const SpectralDofLayout layout = make_layout(2, kOrder, 1);
  const auto structure =
      build_spectral_block_structure(layout, free_map(layout), kOrder);
  ASSERT_TRUE(structure.ok());
  const SyntheticPencil pencil =
      make_synthetic_pencil(structure, layout, kOrder, /*leak=*/false);

  quasar::backend::DeviceBuffer<Real> permuted;
  quasar::stability::launch_permute_dense_symmetric(
      upload(pencil.inertia), structure, permuted, nullptr);

  const auto entries = static_cast<std::size_t>(structure.order)
                     * static_cast<std::size_t>(structure.order);
  std::vector<Real> host(entries);
  permuted.copy_to_host_async(host.data(), host.size(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  for (int column = 0; column < structure.order; ++column) {
    for (int row = 0; row < structure.order; ++row) {
      const std::size_t source =
          static_cast<std::size_t>(structure.permutation[
              static_cast<std::size_t>(row)])
          + static_cast<std::size_t>(structure.permutation[
                static_cast<std::size_t>(column)])
                * static_cast<std::size_t>(structure.order);
      const std::size_t destination =
          static_cast<std::size_t>(row)
          + static_cast<std::size_t>(column)
                * static_cast<std::size_t>(structure.order);
      EXPECT_EQ(host[destination], pencil.inertia[source]);
    }
  }
}

TEST(SpectralBlockExtraction, ShiftInvertReproducesTheDenseSpectrum) {
  constexpr int kOrder = 3;
  const SpectralDofLayout layout = make_layout(3, kOrder, 1);
  const auto structure =
      build_spectral_block_structure(layout, free_map(layout), kOrder);
  ASSERT_TRUE(structure.ok());
  const SyntheticPencil pencil =
      make_synthetic_pencil(structure, layout, kOrder, /*leak=*/false);

  auto stiffness = upload(pencil.stiffness);
  auto inertia = upload(pencil.inertia);
  const auto dense =
      quasar::numerics::solve_generalized_symmetric_eigenproblem(
          stiffness, inertia, structure.order);
  ASSERT_TRUE(dense.ok());
  std::vector<Real> reference(static_cast<std::size_t>(structure.order));
  dense.eigenvalues.copy_to_host_async(reference.data(), reference.size(),
                                       nullptr);
  quasar::backend::device_synchronize(nullptr);

  const Real shift = reference.front() - Real{0.5};
  quasar::stability::SpectralBlockMatrix blocks;
  quasar::stability::launch_extract_shifted_block_tridiagonal(
      stiffness, inertia, shift, structure, blocks, nullptr);
  ASSERT_EQ(blocks.maximum_outside_pattern, Real{0});

  auto factors = quasar::numerics::factor_block_tridiagonal(
      blocks.lower, blocks.diagonal, blocks.upper, structure.partition);
  ASSERT_TRUE(factors.ok());

  quasar::backend::DeviceBuffer<Real> permuted_mass;
  quasar::stability::launch_permute_dense_symmetric(inertia, structure,
                                                    permuted_mass, nullptr);

  quasar::numerics::ShiftInvertLanczosConfig config;
  config.wanted = 3;
  config.shift = shift;
  // The clustered spectrum of this assembled-pencil surrogate needs more
  // Krylov vectors than the small-problem default before all three Ritz
  // residuals reach 1e-10. This test checks the converged spectrum rather than
  // the default iteration-budget heuristic.
  config.maximum_iterations = 80;
  const auto result = quasar::numerics::solve_shift_invert_lanczos(
      factors, permuted_mass, config);

  ASSERT_EQ(result.status, quasar::numerics::ShiftInvertStatus::success)
      << "iterations=" << result.iterations
      << ", converged=" << result.converged
      << ", residuals="
      << (result.residuals.empty() ? Real{-1} : result.residuals.front())
      << ","
      << (result.residuals.size() < 2 ? Real{-1} : result.residuals[1])
      << ","
      << (result.residuals.size() < 3 ? Real{-1} : result.residuals[2]);
  ASSERT_EQ(result.eigenvalues.size(), 3u);
  for (std::size_t index = 0; index < result.eigenvalues.size(); ++index) {
    EXPECT_NEAR(result.eigenvalues[index], reference[index], 1e-8)
        << "shift-invert eigenvalue " << index;
  }
}

}  // namespace
