#include "quasar/physics/stability/spectral_layout.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <set>

namespace {

using quasar::stability::DisplacementComponent;
using quasar::stability::FourierQuadrature;
using quasar::stability::RadialDomains;
using quasar::stability::SpectralDofLayout;

RadialDomains harmonic_topology() {
  RadialDomains domains{};
  domains.n_domains = 3;
  domains.breakpoints[0] = 0.1;
  domains.breakpoints[1] = 0.4;
  domains.breakpoints[2] = 0.7;
  domains.breakpoints[3] = 0.9;

  // Breakpoint 1 is rational for m=1 and m=2; breakpoint 2 is rational only
  // for m=2.  All other harmonics retain ordinary interface continuity.
  domains.resonance_offsets[0] = 0;
  domains.resonance_offsets[1] = 0;
  domains.resonance_offsets[2] = 2;
  domains.resonance_offsets[3] = 3;
  domains.resonance_offsets[4] = 3;
  domains.resonant_m[0] = 1;
  domains.resonant_m[1] = 2;
  domains.resonant_m[2] = 2;
  domains.resonant_psi_n[0] = 0.4;
  domains.resonant_psi_n[1] = 0.40001;
  domains.resonant_psi_n[2] = 0.7;
  domains.resonance_count = 3;
  return domains;
}

TEST(SpectralDofLayout, MergesDuplicatedChebyshevInterfaces) {
  const SpectralDofLayout layout{/*n_domains=*/3, /*order=*/4,
                                 /*m_max=*/2, /*n_theta=*/9};

  EXPECT_EQ(layout.n_radial(), 13u);
  EXPECT_EQ(layout.global_radial(0, 4), 0u);
  EXPECT_EQ(layout.global_radial(0, 0), 4u);
  EXPECT_EQ(layout.global_radial(1, 4), 4u);
  EXPECT_EQ(layout.global_radial(1, 0), 8u);
  EXPECT_EQ(layout.global_radial(2, 4), 8u);
  EXPECT_EQ(layout.global_radial(2, 0), 12u);
}

TEST(SpectralDofLayout, PacksEveryRealDegreeOfFreedomExactlyOnce) {
  const SpectralDofLayout layout{/*n_domains=*/2, /*order=*/3,
                                 /*m_max=*/2, /*n_theta=*/12};
  ASSERT_EQ(layout.n_harmonics(), 5u);
  ASSERT_EQ(layout.complex_dof_count(), 7u * 5u * 3u);
  ASSERT_EQ(layout.dof_count(), 7u * 5u * 3u * 2u);

  std::set<std::size_t> seen;
  for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
    for (std::size_t radial = 0; radial < layout.n_radial(m); ++radial) {
      for (int component = 0;
           component < SpectralDofLayout::kComponentCount; ++component) {
        for (int quadrature = 0;
             quadrature < SpectralDofLayout::kQuadratureCount; ++quadrature) {
          const auto index = layout.dof(
              radial, m, static_cast<DisplacementComponent>(component),
              static_cast<FourierQuadrature>(quadrature));
          EXPECT_LT(index, layout.dof_count());
          EXPECT_TRUE(seen.insert(index).second) << "duplicate index " << index;
        }
      }
    }
  }
  EXPECT_EQ(seen.size(), layout.dof_count());
  EXPECT_EQ(*seen.begin(), 0u);
  EXPECT_EQ(*seen.rbegin(), layout.dof_count() - 1u);
}

TEST(SpectralDofLayout, UsesDocumentedFastToSlowOrdering) {
  const SpectralDofLayout layout{/*n_domains=*/1, /*order=*/2,
                                 /*m_max=*/1, /*n_theta=*/5};

  const auto cos_psi = layout.dof(0, -1, DisplacementComponent::psi,
                                  FourierQuadrature::cosine);
  const auto sin_psi = layout.dof(0, -1, DisplacementComponent::psi,
                                  FourierQuadrature::sine);
  const auto cos_theta = layout.dof(0, -1, DisplacementComponent::theta,
                                    FourierQuadrature::cosine);
  const auto next_harmonic = layout.dof(0, 0, DisplacementComponent::psi,
                                        FourierQuadrature::cosine);
  const auto next_radial = layout.dof(1, -1, DisplacementComponent::psi,
                                      FourierQuadrature::cosine);

  EXPECT_EQ(sin_psi, cos_psi + 1u);
  EXPECT_EQ(cos_theta, cos_psi + 2u);
  EXPECT_EQ(next_radial, cos_psi + 6u);
  EXPECT_EQ(next_harmonic, cos_psi + 18u);
}

TEST(SpectralDofLayout, SplitsOnlyTheTaggedHarmonicAndUsesPrefixOffsets) {
  const RadialDomains domains = harmonic_topology();
  const SpectralDofLayout layout{domains, /*order=*/2, /*m_max=*/2,
                                 /*n_theta=*/9};

  EXPECT_EQ(layout.n_radial(-2), 7u);
  EXPECT_EQ(layout.n_radial(0), 7u);
  EXPECT_EQ(layout.n_radial(1), 8u);
  EXPECT_EQ(layout.n_radial(2), 9u);
  EXPECT_FALSE(layout.interface_is_split(1, 0));
  EXPECT_TRUE(layout.interface_is_split(1, 1));
  EXPECT_TRUE(layout.interface_is_split(1, 2));
  EXPECT_FALSE(layout.interface_is_split(2, 1));
  EXPECT_TRUE(layout.interface_is_split(2, 2));

  // Interface 1 merges for m=0 but has independent left/right indices for
  // m=1. Interface 2 remains merged for m=1 and splits for m=2.
  EXPECT_EQ(layout.global_radial(0, 0, 0), 2u);
  EXPECT_EQ(layout.global_radial(1, 2, 0), 2u);
  EXPECT_EQ(layout.global_radial(0, 0, 1), 2u);
  EXPECT_EQ(layout.global_radial(1, 2, 1), 3u);
  EXPECT_EQ(layout.global_radial(1, 0, 1), 5u);
  EXPECT_EQ(layout.global_radial(2, 2, 1), 5u);
  EXPECT_EQ(layout.global_radial(1, 0, 2), 5u);
  EXPECT_EQ(layout.global_radial(2, 2, 2), 6u);

  // Prefixes for m=-2,-1,0,1,2 are 0,7,14,21,29 radial slots.
  EXPECT_EQ(layout.complex_dof(0, 1, DisplacementComponent::psi), 63u);
  EXPECT_EQ(layout.complex_dof(0, 2, DisplacementComponent::psi), 87u);
  EXPECT_EQ(layout.complex_dof(8, 2, DisplacementComponent::phi), 113u);
  EXPECT_EQ(layout.complex_dof_count(), (7u + 7u + 7u + 8u + 9u) * 3u);
  EXPECT_EQ(layout.dof_count(), layout.complex_dof_count() * 2u);

  EXPECT_THROW((void)layout.global_radial(0, 0), std::logic_error);
  EXPECT_THROW((void)layout.dof(layout.n_radial(1), 1,
                                DisplacementComponent::psi,
                                FourierQuadrature::cosine),
               std::out_of_range);
}

TEST(SpectralDofLayout, DenseMatricesAreColumnMajor) {
  const SpectralDofLayout layout{/*n_domains=*/1, /*order=*/1,
                                 /*m_max=*/0, /*n_theta=*/1};
  const std::size_t n = layout.dof_count();
  EXPECT_EQ(layout.matrix_index(0, 0), 0u);
  EXPECT_EQ(layout.matrix_index(1, 0), 1u);
  EXPECT_EQ(layout.matrix_index(0, 1), n);
  EXPECT_EQ(layout.matrix_index(n - 1, n - 1), n * n - 1u);

  const auto base = layout.dof(1, 0, DisplacementComponent::phi,
                               FourierQuadrature::sine);
  EXPECT_EQ(layout.eigenfunction_index(3, 1, 0,
                                       DisplacementComponent::phi,
                                       FourierQuadrature::sine),
            base + 3u * n);
  EXPECT_THROW((void)layout.eigenfunction_index(
                   n, 1, 0, DisplacementComponent::phi,
                   FourierQuadrature::sine),
               std::out_of_range);
}

TEST(SpectralDofLayout, RejectsInvalidOrAliasedConfigurations) {
  EXPECT_THROW((SpectralDofLayout{0, 4, 2, 9}), std::invalid_argument);
  EXPECT_THROW((SpectralDofLayout{2, 0, 2, 9}), std::invalid_argument);
  EXPECT_THROW((SpectralDofLayout{2, 4, -1, 9}), std::invalid_argument);
  EXPECT_THROW((SpectralDofLayout{2, 4, 2, 8}), std::invalid_argument);

  const SpectralDofLayout layout{2, 4, 2, 9};
  EXPECT_THROW((void)layout.global_radial(2, 0), std::out_of_range);
  EXPECT_THROW((void)layout.global_radial(0, 5), std::out_of_range);
  EXPECT_THROW((void)layout.harmonic_slot(3), std::out_of_range);
  EXPECT_THROW((void)layout.dof(layout.n_radial(), 0,
                                DisplacementComponent::psi,
                                FourierQuadrature::cosine),
               std::out_of_range);
  EXPECT_THROW((void)layout.matrix_index(layout.dof_count(), 0),
               std::out_of_range);

  RadialDomains invalid = harmonic_topology();
  invalid.resonance_offsets[2] = 4;
  EXPECT_THROW((SpectralDofLayout{invalid, 2, 2, 9}),
               std::invalid_argument);
}

}  // namespace
