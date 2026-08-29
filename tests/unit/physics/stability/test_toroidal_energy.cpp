#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/numerics/generalized_eigensolver.hpp"
#include "quasar/physics/stability/toroidal_energy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::stability::ChebyshevBasis;
using quasar::stability::DisplacementComponent;
using quasar::stability::FixedBoundaryDofMap;
using quasar::stability::FluxCoordinateGrid;
using quasar::stability::FourierQuadrature;
using quasar::stability::RadialDomains;
using quasar::stability::SpectralDofLayout;
using quasar::stability::ToroidalAssemblyConfig;
using quasar::stability::ToroidalAssemblyStatus;
using quasar::stability::ToroidalEquilibriumFields;
using quasar::stability::ToroidalMatrixPair;

constexpr int kNoResonantHarmonic = std::numeric_limits<int>::max();

RadialDomains make_toroidal_domains(int n_domains,
                                    int resonant_harmonic) {
  if (n_domains <= 0 || n_domains > RadialDomains::kMaxDomains) {
    throw std::invalid_argument{"invalid toroidal test domain count"};
  }
  if (resonant_harmonic != kNoResonantHarmonic && n_domains < 2) {
    throw std::invalid_argument{
        "a toroidal test resonance requires an interior interface"};
  }

  RadialDomains domains{};
  domains.n_domains = n_domains;
  for (int breakpoint = 0; breakpoint <= n_domains; ++breakpoint) {
    domains.breakpoints[breakpoint] =
        Real{0.2} + Real{0.6} * static_cast<Real>(breakpoint)
                        / static_cast<Real>(n_domains);
    domains.resonance_offsets[breakpoint] = domains.resonance_count;
    if (breakpoint == 1 && resonant_harmonic != kNoResonantHarmonic) {
      const int tag = domains.resonance_count++;
      domains.resonant_m[tag] = resonant_harmonic;
      domains.resonant_psi_n[tag] = domains.breakpoints[breakpoint];
    }
  }
  domains.resonance_offsets[n_domains + 1] = domains.resonance_count;
  return domains;
}

template <class T>
std::vector<T> download(const DeviceBuffer<T>& device) {
  std::vector<T> host(device.size());
  device.copy_to_host(host.data(), host.size());
  return host;
}

template <class T>
void upload(DeviceBuffer<T>& device, const std::vector<T>& host) {
  ASSERT_EQ(device.size(), host.size());
  device.copy_from_host(host.data(), host.size());
}

struct ConstantToroidalProblem {
  RadialDomains domains{};
  ChebyshevBasis basis{};
  SpectralDofLayout layout;
  FluxCoordinateGrid coordinates{};
  ToroidalEquilibriumFields equilibrium{};
  std::vector<Real> nodes{};

  int order;
  int n_theta;
  Real q_value;
  Real density_value;
  Real pressure_value;
  Real jacobian_value;
  Real phi_metric_value;

  ConstantToroidalProblem(int order_in = 2, int m_max = 1,
                          Real density_in = Real{3},
                          bool angular_metric = false,
                          int n_domains = 1,
                          int resonant_harmonic = kNoResonantHarmonic)
      : domains{make_toroidal_domains(n_domains, resonant_harmonic)},
        layout{domains, order_in, m_max, 4 * m_max + 1}, order{order_in},
        n_theta{4 * m_max + 1}, q_value{Real{1}},
        density_value{density_in}, pressure_value{Real{5}},
        jacobian_value{Real{2}}, phi_metric_value{Real{4}} {
    DeviceBuffer<RadialDomains> device_domains{1};
    device_domains.copy_from_host(&domains, 1);
    basis.resize(order, domains.n_domains);
    quasar::stability::launch_build_chebyshev_basis(
        device_domains.device_ptr(), basis, nullptr);
    quasar::backend::device_synchronize(nullptr);
    nodes = download(basis.nodes);

    const int n_lambda = static_cast<int>(nodes.size());
    const std::size_t geometry =
        static_cast<std::size_t>(n_lambda) * n_theta;
    coordinates.resize(n_lambda, n_theta);
    upload(coordinates.psi_n, nodes);
    upload(coordinates.q, std::vector<Real>(n_lambda, q_value));
    upload(coordinates.valid, std::vector<int>(n_lambda, 1));
    upload(coordinates.r, std::vector<Real>(geometry, Real{2}));

    equilibrium.resize(n_lambda, n_theta);
    equilibrium.signed_flux_scale = Real{1};
    upload(equilibrium.pressure,
           std::vector<Real>(n_lambda, pressure_value));
    upload(equilibrium.pressure_lambda,
           std::vector<Real>(n_lambda, Real{0}));
    upload(equilibrium.density,
           std::vector<Real>(n_lambda, density_value));

    std::vector<Real> gll(geometry, Real{1.5});
    std::vector<Real> glt(geometry, Real{0.2});
    std::vector<Real> gtt(geometry, Real{2});
    std::vector<Real> gpp(geometry, phi_metric_value);
    std::vector<Real> jacobian(geometry, jacobian_value);
    std::vector<Real> btheta(geometry,
                             equilibrium.signed_flux_scale / jacobian_value);
    std::vector<Real> bphi(geometry, q_value * btheta.front());
    if (angular_metric) {
      for (int radial = 0; radial < n_lambda; ++radial) {
        for (int theta = 0; theta < n_theta; ++theta) {
          const Real angle = Real{2} * quasar::pi * theta / n_theta;
          const std::size_t index =
              static_cast<std::size_t>(radial) * n_theta + theta;
          gtt[index] += Real{0.25} * std::sin(angle);
          glt[index] += Real{0.08} * std::cos(angle);
        }
      }
    }
    upload(equilibrium.g_lambda_lambda, gll);
    upload(equilibrium.g_lambda_theta, glt);
    upload(equilibrium.g_theta_theta, gtt);
    upload(equilibrium.g_phi_phi, gpp);
    upload(equilibrium.jacobian, jacobian);
    upload(equilibrium.b_theta, btheta);
    upload(equilibrium.b_phi, bphi);
    upload(equilibrium.j_theta, std::vector<Real>(geometry, Real{0}));
    upload(equilibrium.j_phi, std::vector<Real>(geometry, Real{0}));
  }

  ToroidalAssemblyConfig assembly_config(
      Real gamma = Real{5} / Real{3}, int n_toroidal = 1) const {
    ToroidalAssemblyConfig config;
    config.n_toroidal = n_toroidal;
    config.gamma = gamma;
    return config;
  }

  ToroidalMatrixPair assemble(Real gamma = Real{5} / Real{3},
                              int n_toroidal = 1) const {
    const ToroidalAssemblyConfig config =
        assembly_config(gamma, n_toroidal);
    ToroidalMatrixPair matrices;
    quasar::stability::launch_assemble_fixed_boundary_toroidal_matrices(
        basis, domains, coordinates, equilibrium, layout, config, matrices,
        nullptr);
    return matrices;
  }
};

long double quadratic_form(const std::vector<Real>& matrix,
                           const std::vector<Real>& vector) {
  const std::size_t order = vector.size();
  long double result = 0;
  for (std::size_t column = 0; column < order; ++column) {
    for (std::size_t row = 0; row < order; ++row) {
      result += static_cast<long double>(vector[row])
              * matrix[row + column * order]
              * static_cast<long double>(vector[column]);
    }
  }
  return result;
}

TEST(FixedBoundaryDofMap, EliminatesOnlyOuterNormalDisplacement) {
  RadialDomains domains{};
  domains.n_domains = 2;
  domains.breakpoints[0] = Real{0.2};
  domains.breakpoints[1] = Real{0.5};
  domains.breakpoints[2] = Real{0.9};
  domains.resonance_offsets[0] = 0;
  domains.resonance_offsets[1] = 0;
  domains.resonant_m[0] = 1;
  domains.resonant_psi_n[0] = Real{0.5};
  domains.resonance_count = 1;
  domains.resonance_offsets[2] = 1;
  domains.resonance_offsets[3] = 1;

  const SpectralDofLayout layout{domains, 3, 2, 9};
  const FixedBoundaryDofMap fixed{layout};
  EXPECT_EQ(fixed.complex_dof_count(),
            layout.complex_dof_count() - layout.n_harmonics());

  for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
    const std::size_t inner = 0;
    const std::size_t outer = layout.n_radial(m) - 1;
    EXPECT_NE(fixed.complex_dof(layout, inner, m,
                                DisplacementComponent::psi),
              FixedBoundaryDofMap::eliminated);
    EXPECT_EQ(fixed.complex_dof(layout, outer, m,
                                DisplacementComponent::psi),
              FixedBoundaryDofMap::eliminated);
    EXPECT_NE(fixed.complex_dof(layout, outer, m,
                                DisplacementComponent::theta),
              FixedBoundaryDofMap::eliminated);
    EXPECT_NE(fixed.complex_dof(layout, outer, m,
                                DisplacementComponent::phi),
              FixedBoundaryDofMap::eliminated);
  }
}

TEST(ToroidalEnergyStorage, ReportsDenseArithmeticOverflowWithoutThrowing) {
  std::size_t bytes = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(quasar::stability::toroidal_dense_storage_bytes(2, bytes),
            ToroidalAssemblyStatus::success);
  EXPECT_EQ(bytes, std::size_t{8} * sizeof(Real));

  EXPECT_EQ(quasar::stability::toroidal_dense_storage_bytes(
                std::numeric_limits<std::size_t>::max(), bytes),
            ToroidalAssemblyStatus::storage_overflow);
  EXPECT_EQ(bytes, 0u);
}

TEST(ToroidalEnergyAssembly, RejectsAxisAndAxisymmetricToroidalMode) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem;

  auto n_zero = problem.assemble(Real{5} / Real{3}, 0);
  EXPECT_TRUE(quasar::stability::has_status(
      n_zero.diagnostics.status,
      ToroidalAssemblyStatus::invalid_toroidal_mode));
  EXPECT_TRUE(n_zero.stiffness.empty());

  problem.domains.breakpoints[0] = Real{0};
  auto axis = problem.assemble();
  EXPECT_TRUE(quasar::stability::has_status(
      axis.diagnostics.status,
      ToroidalAssemblyStatus::unsupported_magnetic_axis));
  EXPECT_TRUE(axis.inertia.empty());
}

TEST(ToroidalEnergyAssembly, RejectsMalformedRadialTopologyMetadata) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem;
  const auto expect_invalid = [&](const RadialDomains& domains) {
    ToroidalMatrixPair matrices;
    quasar::stability::launch_assemble_fixed_boundary_toroidal_matrices(
        problem.basis, domains, problem.coordinates, problem.equilibrium,
        problem.layout, problem.assembly_config(), matrices, nullptr);
    EXPECT_TRUE(quasar::stability::has_status(
        matrices.diagnostics.status,
        ToroidalAssemblyStatus::invalid_topology));
    EXPECT_TRUE(matrices.stiffness.empty());
    EXPECT_TRUE(matrices.inertia.empty());
  };

  auto overflowed = problem.domains;
  overflowed.overflow = true;
  expect_invalid(overflowed);

  auto bad_offsets = problem.domains;
  bad_offsets.resonance_offsets[1] = 1;
  expect_invalid(bad_offsets);

  auto bad_breakpoint = problem.domains;
  bad_breakpoint.breakpoints[1] = bad_breakpoint.breakpoints[0];
  expect_invalid(bad_breakpoint);

  auto endpoint_mismatch = problem.domains;
  endpoint_mismatch.breakpoints[1] -= Real{0.01};
  expect_invalid(endpoint_mismatch);

  ConstantToroidalProblem resonant{/*order=*/2, /*m_max=*/1,
                                    /*density=*/Real{3},
                                    /*angular_metric=*/false,
                                    /*n_domains=*/2,
                                    /*resonant_harmonic=*/1};
  auto nonfinite_source = resonant.domains;
  nonfinite_source.resonant_psi_n[0] =
      std::numeric_limits<Real>::quiet_NaN();
  ToroidalMatrixPair matrices;
  quasar::stability::launch_assemble_fixed_boundary_toroidal_matrices(
      resonant.basis, nonfinite_source, resonant.coordinates,
      resonant.equilibrium, resonant.layout, resonant.assembly_config(),
      matrices, nullptr);
  EXPECT_TRUE(quasar::stability::has_status(
      matrices.diagnostics.status, ToroidalAssemblyStatus::invalid_topology));
  EXPECT_TRUE(matrices.stiffness.empty());
  EXPECT_TRUE(matrices.inertia.empty());
}

TEST(ToroidalEnergyAssembly,
     IsExactlySymmetricAndUsesPhysicalCosineSineRealification) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem{/*order=*/2, /*m_max=*/1,
                                  /*density=*/Real{3},
                                  /*angular_metric=*/true};
  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());
  EXPECT_LE(matrices.diagnostics.hermitian_residual,
            ToroidalAssemblyConfig{}.hermitian_tolerance);

  const auto stiffness = download(matrices.stiffness);
  const int complex_order = matrices.complex_order;
  const int real_order = matrices.real_order;
  Real largest_imaginary_block = Real{0};
  for (int column = 0; column < complex_order; ++column) {
    for (int row = 0; row < complex_order; ++row) {
      const int rc = 2 * row;
      const int rs = rc + 1;
      const int cc = 2 * column;
      const int cs = cc + 1;
      const Real a = stiffness[static_cast<std::size_t>(rc)
                               + static_cast<std::size_t>(cc) * real_order];
      const Real b = stiffness[static_cast<std::size_t>(rc)
                               + static_cast<std::size_t>(cs) * real_order];
      EXPECT_EQ(stiffness[static_cast<std::size_t>(rs)
                          + static_cast<std::size_t>(cc) * real_order],
                -b);
      EXPECT_EQ(stiffness[static_cast<std::size_t>(rs)
                          + static_cast<std::size_t>(cs) * real_order],
                a);
      EXPECT_EQ(stiffness[static_cast<std::size_t>(rc)
                          + static_cast<std::size_t>(cc) * real_order],
                stiffness[static_cast<std::size_t>(cc)
                          + static_cast<std::size_t>(rc) * real_order]);
      largest_imaginary_block = std::max(largest_imaginary_block, std::abs(b));
    }
  }
  EXPECT_GT(largest_imaginary_block, Real{1e-6});
}

TEST(ToroidalEnergyAssembly, RealificationProducesDoubledSpectrum) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem{/*order=*/2, /*m_max=*/1,
                                  /*density=*/Real{3},
                                  /*angular_metric=*/true};
  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());
  const auto result =
      quasar::numerics::solve_generalized_symmetric_eigenproblem(
          matrices.stiffness, matrices.inertia, matrices.real_order);
  ASSERT_TRUE(result.ok()) << result.solver_info;
  const auto eigenvalues = download(result.eigenvalues);
  ASSERT_EQ(eigenvalues.size() % 2, 0u);
  for (std::size_t i = 0; i < eigenvalues.size(); i += 2) {
    EXPECT_NEAR(eigenvalues[i], eigenvalues[i + 1],
                Real{2e-8} * (Real{1} + std::abs(eigenvalues[i])))
        << "pair " << i / 2;
  }
}

TEST(ToroidalEnergyAssembly, ParallelDisplacementIsNeutralButHasInertia) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem;
  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());
  const FixedBoundaryDofMap fixed{problem.layout};
  std::vector<Real> displacement(matrices.real_order, Real{0});
  for (std::size_t radial = 0; radial < problem.layout.n_radial(1); ++radial) {
    displacement[fixed.dof(problem.layout, radial, 1,
                            DisplacementComponent::theta,
                            FourierQuadrature::cosine)] = Real{1};
    displacement[fixed.dof(problem.layout, radial, 1,
                            DisplacementComponent::phi,
                            FourierQuadrature::cosine)] = problem.q_value;
  }

  const long double energy =
      quadratic_form(download(matrices.stiffness), displacement);
  const long double inertia =
      quadratic_form(download(matrices.inertia), displacement);
  EXPECT_GT(inertia, 0);
  EXPECT_NEAR(static_cast<double>(energy), 0.0,
              2e-10 * static_cast<double>(inertia));
}

TEST(ToroidalEnergyAssembly, GammaIncrementMatchesPureCompressionTerm) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem;
  auto gamma_one = problem.assemble(Real{1});
  auto gamma_two = problem.assemble(Real{2});
  ASSERT_TRUE(gamma_one.diagnostics.ok());
  ASSERT_TRUE(gamma_two.diagnostics.ok());

  const FixedBoundaryDofMap fixed{problem.layout};
  std::vector<Real> displacement(gamma_one.real_order, Real{0});
  for (std::size_t radial = 0; radial < problem.layout.n_radial(0); ++radial) {
    displacement[fixed.dof(problem.layout, radial, 0,
                            DisplacementComponent::phi,
                            FourierQuadrature::cosine)] = Real{1};
  }
  const long double delta =
      quadratic_form(download(gamma_two.stiffness), displacement)
      - quadratic_form(download(gamma_one.stiffness), displacement);
  const long double expected =
      4.0L * static_cast<long double>(quasar::pi)
      * static_cast<long double>(quasar::pi)
      * static_cast<long double>(problem.domains.breakpoints[1]
                                 - problem.domains.breakpoints[0])
      * static_cast<long double>(problem.jacobian_value)
      * static_cast<long double>(problem.pressure_value);
  EXPECT_NEAR(static_cast<double>(delta), static_cast<double>(expected),
              2e-10 * static_cast<double>(expected));
}

TEST(ToroidalEnergyAssembly, InertiaScalesLinearlyWithDensity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem base{/*order=*/2, /*m_max=*/1,
                               /*density=*/Real{2}};
  ConstantToroidalProblem scaled{/*order=*/2, /*m_max=*/1,
                                 /*density=*/Real{6}};
  auto first = base.assemble();
  auto second = scaled.assemble();
  ASSERT_TRUE(first.diagnostics.ok());
  ASSERT_TRUE(second.diagnostics.ok());
  const auto a = download(first.inertia);
  const auto b = download(second.inertia);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(b[i], Real{3} * a[i],
                Real{2e-13} * (Real{1} + std::abs(b[i])));
  }
}

TEST(ToroidalEnergyAssembly, RegularInterfaceReceivesBothDomainContributions) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem{/*order=*/2, /*m_max=*/1,
                                  /*density=*/Real{3},
                                  /*angular_metric=*/false,
                                  /*n_domains=*/2};
  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());

  const std::size_t interface = problem.layout.global_radial(0, 0, 0);
  ASSERT_EQ(interface,
            problem.layout.global_radial(1, problem.order, 0));
  const FixedBoundaryDofMap fixed{problem.layout};
  const std::size_t dof = fixed.dof(
      problem.layout, interface, 0, DisplacementComponent::phi,
      FourierQuadrature::cosine);
  ASSERT_NE(dof, FixedBoundaryDofMap::eliminated);

  const auto weights = download(problem.basis.weights);
  const std::size_t left_endpoint = 0;
  const std::size_t right_endpoint =
      static_cast<std::size_t>(problem.basis.n_nodes + problem.order);
  const Real expected =
      Real{4} * quasar::pi * quasar::pi * problem.density_value
      * problem.phi_metric_value * problem.jacobian_value
      * (weights[left_endpoint] + weights[right_endpoint]);
  const auto inertia = download(matrices.inertia);
  const Real actual =
      inertia[dof + dof * static_cast<std::size_t>(matrices.real_order)];
  EXPECT_NEAR(actual, expected,
              Real{3e-13} * (Real{1} + std::abs(expected)));
}

TEST(ToroidalEnergyAssembly,
     SplitsOnlyTaggedHarmonicAndRejectsAllRegularTopology) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem{/*order=*/2, /*m_max=*/1,
                                  /*density=*/Real{3},
                                  /*angular_metric=*/false,
                                  /*n_domains=*/2,
                                  /*resonant_harmonic=*/1};
  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());

  const std::size_t resonant_left =
      problem.layout.global_radial(0, 0, 1);
  const std::size_t resonant_right =
      problem.layout.global_radial(1, problem.order, 1);
  EXPECT_NE(resonant_left, resonant_right);
  const std::size_t regular_left = problem.layout.global_radial(0, 0, 0);
  const std::size_t regular_right =
      problem.layout.global_radial(1, problem.order, 0);
  EXPECT_EQ(regular_left, regular_right);

  const FixedBoundaryDofMap fixed{problem.layout};
  const auto occurrence_count = download(matrices.occurrence_count);
  for (int component = 0;
       component < SpectralDofLayout::kComponentCount; ++component) {
    const auto displacement =
        static_cast<DisplacementComponent>(component);
    const std::size_t left = fixed.complex_dof(
        problem.layout, resonant_left, 1, displacement);
    const std::size_t right = fixed.complex_dof(
        problem.layout, resonant_right, 1, displacement);
    ASSERT_NE(left, FixedBoundaryDofMap::eliminated);
    ASSERT_NE(right, FixedBoundaryDofMap::eliminated);
    EXPECT_EQ(occurrence_count[left], 1) << "component " << component;
    EXPECT_EQ(occurrence_count[right], 1) << "component " << component;

    const std::size_t merged = fixed.complex_dof(
        problem.layout, regular_left, 0, displacement);
    ASSERT_NE(merged, FixedBoundaryDofMap::eliminated);
    EXPECT_EQ(occurrence_count[merged], 2) << "component " << component;
  }

  const SpectralDofLayout all_regular{
      problem.domains.n_domains, problem.order, problem.layout.m_max(),
      problem.n_theta};
  ToroidalMatrixPair rejected;
  quasar::stability::launch_assemble_fixed_boundary_toroidal_matrices(
      problem.basis, problem.domains, problem.coordinates,
      problem.equilibrium, all_regular, problem.assembly_config(), rejected,
      nullptr);
  EXPECT_TRUE(quasar::stability::has_status(
      rejected.diagnostics.status, ToroidalAssemblyStatus::invalid_topology));
  EXPECT_TRUE(rejected.stiffness.empty());
  EXPECT_TRUE(rejected.inertia.empty());
}

TEST(ToroidalEnergyAssembly, ComputesRadialAndSourceAngleDerivativeSigns) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem{/*order=*/3, /*m_max=*/1,
                                  /*density=*/Real{3},
                                  /*angular_metric=*/false,
                                  /*n_domains=*/2};

  constexpr Real q_offset = Real{0.7};
  constexpr Real q_slope = Real{0.6};
  constexpr Real jacobian_offset = Real{2.0};
  constexpr Real jacobian_slope = Real{0.4};
  constexpr Real angular_amplitude = Real{0.15};
  const int n_lambda = static_cast<int>(problem.nodes.size());
  const std::size_t geometry =
      static_cast<std::size_t>(n_lambda) * problem.n_theta;
  std::vector<Real> q(n_lambda);
  std::vector<Real> jacobian(geometry);
  std::vector<Real> gpp(geometry);
  std::vector<Real> btheta(geometry);
  std::vector<Real> bphi(geometry);
  for (int radial = 0; radial < n_lambda; ++radial) {
    const Real lambda = problem.nodes[radial];
    q[radial] = q_offset + q_slope * lambda;
    const Real flux_function = Real{0.5} + Real{0.1} * lambda;
    for (int theta = 0; theta < problem.n_theta; ++theta) {
      const Real theta_stored =
          Real{2} * quasar::pi * static_cast<Real>(theta)
          / static_cast<Real>(problem.n_theta);
      const std::size_t index =
          static_cast<std::size_t>(radial) * problem.n_theta + theta;
      jacobian[index] = jacobian_offset + jacobian_slope * lambda
                      + angular_amplitude * std::sin(theta_stored);
      gpp[index] = jacobian[index] / flux_function;
      btheta[index] = problem.equilibrium.signed_flux_scale / jacobian[index];
      bphi[index] = q[radial] * btheta[index];
    }
  }
  upload(problem.coordinates.q, q);
  upload(problem.equilibrium.jacobian, jacobian);
  upload(problem.equilibrium.g_phi_phi, gpp);
  upload(problem.equilibrium.b_theta, btheta);
  upload(problem.equilibrium.b_phi, bphi);

  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());
  const auto q_lambda = download(matrices.q_lambda);
  const auto jacobian_lambda = download(matrices.jacobian_lambda);
  const auto jacobian_theta = download(matrices.jacobian_theta);
  ASSERT_EQ(q_lambda.size(), q.size());
  ASSERT_EQ(jacobian_lambda.size(), geometry);
  ASSERT_EQ(jacobian_theta.size(), geometry);
  for (int radial = 0; radial < n_lambda; ++radial) {
    EXPECT_NEAR(q_lambda[radial], q_slope, Real{3e-13})
        << "radial point " << radial;
    for (int theta = 0; theta < problem.n_theta; ++theta) {
      const Real theta_stored =
          Real{2} * quasar::pi * static_cast<Real>(theta)
          / static_cast<Real>(problem.n_theta);
      const std::size_t index =
          static_cast<std::size_t>(radial) * problem.n_theta + theta;
      EXPECT_NEAR(jacobian_lambda[index], jacobian_slope, Real{5e-13})
          << "geometry point " << index;
      EXPECT_NEAR(jacobian_theta[index],
                  -angular_amplitude * std::cos(theta_stored), Real{5e-13})
          << "geometry point " << index;
    }
  }
}

TEST(ToroidalEnergyAssembly, ConstantGeometryDecouplesDistinctHarmonics) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem{/*order=*/2, /*m_max=*/2};
  auto matrices = problem.assemble();
  ASSERT_TRUE(matrices.diagnostics.ok());

  const auto harmonics = download(matrices.harmonic);
  ASSERT_EQ(harmonics.size(),
            static_cast<std::size_t>(matrices.complex_order));
  const auto check_decoupled = [&](const std::vector<Real>& matrix,
                                   const char* name) {
    Real maximum = Real{0};
    Real maximum_cross_harmonic = Real{0};
    for (int column = 0; column < matrices.complex_order; ++column) {
      for (int row = 0; row < matrices.complex_order; ++row) {
        for (int column_quadrature = 0; column_quadrature < 2;
             ++column_quadrature) {
          for (int row_quadrature = 0; row_quadrature < 2;
               ++row_quadrature) {
            const std::size_t real_row =
                static_cast<std::size_t>(2 * row + row_quadrature);
            const std::size_t real_column =
                static_cast<std::size_t>(2 * column + column_quadrature);
            const Real magnitude = std::abs(
                matrix[real_row
                       + real_column
                             * static_cast<std::size_t>(matrices.real_order)]);
            maximum = std::max(maximum, magnitude);
            if (harmonics[row] != harmonics[column]) {
              maximum_cross_harmonic =
                  std::max(maximum_cross_harmonic, magnitude);
            }
          }
        }
      }
    }
    EXPECT_LE(maximum_cross_harmonic,
              Real{2e-12} * std::max(Real{1}, maximum))
        << name << " maximum entry " << maximum;
  };
  check_decoupled(download(matrices.stiffness), "stiffness");
  check_decoupled(download(matrices.inertia), "inertia");
}

TEST(ToroidalEnergyAssembly, ReportsFiniteInvalidMetricAndFirstPoint) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ConstantToroidalProblem problem;
  const std::size_t geometry =
      problem.nodes.size() * static_cast<std::size_t>(problem.n_theta);
  constexpr std::size_t invalid_point = 3;
  ASSERT_LT(invalid_point, geometry);
  std::vector<Real> gll(geometry, Real{1.5});
  gll[invalid_point] = Real{0};
  upload(problem.equilibrium.g_lambda_lambda, gll);

  auto matrices = problem.assemble();
  EXPECT_TRUE(quasar::stability::has_status(
      matrices.diagnostics.status, ToroidalAssemblyStatus::invalid_metric));
  EXPECT_FALSE(quasar::stability::has_status(
      matrices.diagnostics.status, ToroidalAssemblyStatus::nonfinite_input));
  EXPECT_EQ(matrices.diagnostics.first_invalid_metric_point, invalid_point);
  EXPECT_TRUE(matrices.stiffness.empty());
  EXPECT_TRUE(matrices.inertia.empty());
}

TEST(ToroidalMatrixValidation, ReportsNonHermitianRealMatrix) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ToroidalMatrixPair matrices;
  matrices.complex_order = 1;
  matrices.real_order = 2;
  matrices.stiffness = DeviceBuffer<Real>{4};
  matrices.inertia = DeviceBuffer<Real>{4};
  upload(matrices.stiffness,
         std::vector<Real>{Real{1}, Real{0}, Real{0.25}, Real{1}});
  upload(matrices.inertia,
         std::vector<Real>{Real{1}, Real{0}, Real{0}, Real{1}});

  quasar::stability::launch_validate_toroidal_matrix_pair(
      matrices, Real{64} * std::numeric_limits<Real>::epsilon(), nullptr);
  EXPECT_TRUE(quasar::stability::has_status(
      matrices.diagnostics.status,
      ToroidalAssemblyStatus::nonhermitian_matrix));
  EXPECT_GT(matrices.diagnostics.hermitian_residual, Real{0});
  EXPECT_FALSE(quasar::stability::has_status(
      matrices.diagnostics.status,
      ToroidalAssemblyStatus::mass_not_positive_definite));
}

TEST(ToroidalMatrixValidation,
     DetectsPositiveDiagonalIndefiniteMassAndPreservesInput) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP();
  ToroidalMatrixPair matrices;
  matrices.complex_order = 1;
  matrices.real_order = 2;
  matrices.stiffness = DeviceBuffer<Real>{4};
  matrices.inertia = DeviceBuffer<Real>{4};
  const std::vector<Real> stiffness{Real{1}, Real{0}, Real{0}, Real{1}};
  const std::vector<Real> indefinite_mass{Real{1}, Real{2}, Real{2}, Real{1}};
  upload(matrices.stiffness, stiffness);
  upload(matrices.inertia, indefinite_mass);

  quasar::stability::launch_validate_toroidal_matrix_pair(
      matrices, Real{64} * std::numeric_limits<Real>::epsilon(), nullptr);
  EXPECT_FALSE(quasar::stability::has_status(
      matrices.diagnostics.status,
      ToroidalAssemblyStatus::nonpositive_mass_diagonal));
  EXPECT_TRUE(quasar::stability::has_status(
      matrices.diagnostics.status,
      ToroidalAssemblyStatus::mass_not_positive_definite));
  EXPECT_EQ(matrices.diagnostics.mass_failed_leading_minor_order, 2);
  EXPECT_EQ(download(matrices.inertia), indefinite_mass);
}

}  // namespace
