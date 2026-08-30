// Cylindrical Newcomb coefficients and scalar weak-form assembly.
//
// The coefficient checks pin Newcomb's printed Eq. (18) numerically and its
// separately printed m=0 Eqs. (19)--(20).  The assembly checks are independent:
// a host quadrature oracle builds local matrices and performs the interface
// scatter, while a second test compares x^T K x with direct quadrature of a
// manufactured physical radial displacement xi_r.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/stability/energy_functional.hpp"
#include "quasar/physics/stability/spectral_layout.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::stability::ChebyshevBasis;
using quasar::stability::CylindricalNewcombCoefficients;
using quasar::stability::CylindricalNewcombMatrix;
using quasar::stability::CylindricalNewcombMode;
using quasar::stability::RadialDomains;
using quasar::stability::SpectralDofLayout;

struct BasisData {
  RadialDomains domains{};
  ChebyshevBasis basis{};
  std::vector<Real> nodes;
  std::vector<Real> weights;
  std::vector<Real> diff;
  int order{0};
  int n_nodes{0};
  int n_domains{0};

  BasisData(int order_in, const std::vector<Real>& breakpoints)
    : order{order_in}, n_nodes{order_in + 1},
      n_domains{static_cast<int>(breakpoints.size()) - 1} {
    domains.n_domains = n_domains;
    for (std::size_t i = 0; i < breakpoints.size(); ++i) {
      domains.breakpoints[i] = breakpoints[i];
    }
    DeviceBuffer<RadialDomains> d_domains{1};
    d_domains.copy_from_host(&domains, 1);

    basis.resize(order, n_domains);
    quasar::stability::launch_build_chebyshev_basis(d_domains.device_ptr(),
                                                    basis, nullptr);
    quasar::backend::device_synchronize(nullptr);

    nodes.resize(basis.nodes.size());
    weights.resize(basis.weights.size());
    diff.resize(basis.diff.size());
    basis.nodes.copy_to_host(nodes.data(), nodes.size());
    basis.weights.copy_to_host(weights.data(), weights.size());
    basis.diff.copy_to_host(diff.data(), diff.size());
  }

  std::size_t local_index(int domain, int local_node) const {
    return static_cast<std::size_t>(domain) * n_nodes + local_node;
  }

  Real D(int domain, int row, int column) const {
    return diff[static_cast<std::size_t>(domain) * n_nodes * n_nodes
                + static_cast<std::size_t>(row) * n_nodes + column];
  }

  int global_node(int domain, int local_node) const {
    // Local Chebyshev nodes descend from the upper to the lower endpoint;
    // global nodes ascend in physical radius.
    return domain * order + (order - local_node);
  }

  int n_global() const { return n_domains * order + 1; }

  void set_rational_interfaces(
      const std::map<int, std::vector<int>>& interface_harmonics) {
    domains.resonance_count = 0;
    for (int breakpoint = 0; breakpoint <= n_domains; ++breakpoint) {
      domains.resonance_offsets[breakpoint] = domains.resonance_count;
      const auto found = interface_harmonics.find(breakpoint);
      if (found == interface_harmonics.end()) continue;
      for (const int m : found->second) {
        ASSERT_LT(domains.resonance_count,
                  RadialDomains::kMaxResonanceTags);
        const int tag = domains.resonance_count++;
        domains.resonant_m[tag] = m;
        domains.resonant_psi_n[tag] = domains.breakpoints[breakpoint];
      }
    }
    domains.resonance_offsets[n_domains + 1] = domains.resonance_count;
  }

  SpectralDofLayout layout(int m) const {
    const int m_max = std::abs(m);
    return SpectralDofLayout{domains, order, m_max, 4 * m_max + 1};
  }

  int radial_node(int domain, int local_node, int m) const {
    return static_cast<int>(layout(m).global_radial(domain, local_node, m));
  }

  int free_dof(int domain, int local_node, int m) const {
    const auto spectral = layout(m);
    const int radial =
        static_cast<int>(spectral.global_radial(domain, local_node, m));
    if (radial == 0 || radial == static_cast<int>(spectral.n_radial(m)) - 1) {
      return -1;
    }
    return radial - 1;
  }

  int n_free(int m = 0) const {
    return static_cast<int>(layout(m).n_radial(m)) - 2;
  }
};

DeviceBuffer<Real> to_device(const std::vector<Real>& host) {
  DeviceBuffer<Real> device{host.size()};
  device.copy_from_host(host.data(), host.size());
  return device;
}

std::vector<Real> copy_to_host(const DeviceBuffer<Real>& device) {
  std::vector<Real> host(device.size());
  device.copy_to_host(host.data(), host.size());
  return host;
}

Real local_entry(const BasisData& basis, const std::vector<Real>& f,
                 const std::vector<Real>& g,
                 const std::vector<Real>& dr_dcoordinate, int domain, int a,
                 int b) {
  Real value = Real{0};
  for (int q = 0; q < basis.n_nodes; ++q) {
    const std::size_t iq = basis.local_index(domain, q);
    value += basis.weights[iq] * (f[iq] / dr_dcoordinate[iq])
           * basis.D(domain, q, a) * basis.D(domain, q, b);
    if (q == a && a == b) {
      value += basis.weights[iq] * g[iq] * dr_dcoordinate[iq];
    }
  }
  return value;
}

std::vector<Real> assemble_host_reference(const BasisData& basis,
                                          const std::vector<Real>& f,
                                          const std::vector<Real>& g,
                                          const std::vector<Real>& jacobian,
                                          int m) {
  const int n_free = basis.n_free(m);
  const auto spectral = basis.layout(m);
  std::vector<Real> matrix(
      static_cast<std::size_t>(n_free) * n_free, Real{0});

  const auto free_dof = [&](int domain, int local) {
    const int radial =
        static_cast<int>(spectral.global_radial(domain, local, m));
    return (radial == 0
            || radial == static_cast<int>(spectral.n_radial(m)) - 1)
               ? -1
               : radial - 1;
  };

  for (int domain = 0; domain < basis.n_domains; ++domain) {
    for (int a = 0; a < basis.n_nodes; ++a) {
      const int row = free_dof(domain, a);
      if (row < 0) continue;
      for (int b = 0; b < basis.n_nodes; ++b) {
        const int column = free_dof(domain, b);
        if (column < 0) continue;
        matrix[static_cast<std::size_t>(row)
               + static_cast<std::size_t>(column) * n_free]
            += local_entry(basis, f, g, jacobian, domain, a, b);
      }
    }
  }
  return matrix;
}

TEST(CylindricalNewcombCoefficients, PinsNewcombEquation18InSIUnits) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Order-two Lobatto nodes on [0.4, 1.7] are exactly 1.7, 1.05, 0.4 in the
  // descending local-node convention.
  const BasisData basis{2, {Real{0.4}, Real{1.7}}};
  ASSERT_NEAR(basis.nodes[0], Real{1.7}, Real{1e-15});
  ASSERT_NEAR(basis.nodes[1], Real{1.05}, Real{1e-15});
  ASSERT_NEAR(basis.nodes[2], Real{0.4}, Real{1e-15});

  const std::vector<Real> b_theta{Real{0.55}, Real{-0.25}, Real{0.8}};
  const std::vector<Real> b_z{Real{-0.6}, Real{2.1}, Real{1.3}};
  const std::vector<Real> dp_dr{Real{-725}, Real{3500}, Real{-21000}};
  auto d_r = to_device(basis.nodes);
  auto d_b_theta = to_device(b_theta);
  auto d_b_z = to_device(b_z);
  auto d_dp_dr = to_device(dp_dr);
  CylindricalNewcombCoefficients coefficients{basis.nodes.size()};

  quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
      basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
      CylindricalNewcombMode{2, Real{-0.7}}, coefficients, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto f = copy_to_host(coefficients.f);
  const auto g = copy_to_host(coefficients.g);

  // Independently evaluated from Eq. (18) at 60 decimal digits, then rounded
  // once to binary64.  Keeping literal answers here prevents the test oracle
  // from repeating the production algebra.
  const Real expected_f[] = {
      Real{821915.0795787664}, Real{768513.2213458238},
      Real{119233.21397691061}};
  const Real expected_g[] = {
      Real{1223912.4934691659}, Real{2553303.5401747010},
      Real{2247713.9731362197}};
  for (std::size_t i = 0; i < f.size(); ++i) {
    EXPECT_NEAR(f[i], expected_f[i], Real{3e-13} * std::abs(expected_f[i]))
        << "f at local node " << i;
    EXPECT_NEAR(g[i], expected_g[i], Real{3e-13} * std::abs(expected_g[i]))
        << "g at local node " << i;
  }
}

TEST(CylindricalNewcombCoefficients, ReducesToPrintedMZeroEquations19And20) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const BasisData basis{1, {Real{0.35}, Real{1.1}}};
  // B_theta is deliberately large and changes sign: it must cancel completely
  // when m=0 in Eqs. (19)--(20).
  const std::vector<Real> b_theta{Real{-7}, Real{9}};
  const std::vector<Real> b_z{Real{-0.8}, Real{1.4}};
  const std::vector<Real> dp_dr{Real{450}, Real{-3200}};
  auto d_r = to_device(basis.nodes);
  auto d_b_theta = to_device(b_theta);
  auto d_b_z = to_device(b_z);
  auto d_dp_dr = to_device(dp_dr);
  CylindricalNewcombCoefficients coefficients{basis.nodes.size()};

  quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
      basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
      CylindricalNewcombMode{0, Real{1.25}}, coefficients, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto f = copy_to_host(coefficients.f);
  const auto g = copy_to_host(coefficients.g);
  const Real expected_f[] = {Real{560225.3996834715},
                             Real{545901.4548052010}};
  const Real expected_g[] = {Real{1339248.3850909380},
                             Real{5302909.4297061960}};
  for (std::size_t i = 0; i < f.size(); ++i) {
    EXPECT_NEAR(f[i], expected_f[i], Real{3e-13} * std::abs(expected_f[i]));
    EXPECT_NEAR(g[i], expected_g[i], Real{3e-13} * std::abs(expected_g[i]));
  }
}

TEST(CylindricalNewcombAssembly, IsSymmetricAndMergesInterfaceNodes) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const BasisData basis{5, {Real{0.2}, Real{0.65}, Real{1.4}}};
  std::vector<Real> f(basis.nodes.size());
  std::vector<Real> g(basis.nodes.size());
  for (std::size_t i = 0; i < basis.nodes.size(); ++i) {
    // The explicit i-dependence makes the two local copies of the interface
    // carry distinct coefficients, so dropping either side is observable.
    f[i] = Real{0.7} + Real{0.2} * basis.nodes[i]
         + Real{0.03} * static_cast<Real>(i);
    g[i] = Real{-0.4} + Real{0.1} * basis.nodes[i]
         + Real{0.02} * static_cast<Real>(i);
  }

  CylindricalNewcombCoefficients coefficients{f.size()};
  coefficients.f.copy_from_host(f.data(), f.size());
  coefficients.g.copy_from_host(g.data(), g.size());
  const std::vector<Real> jacobian(f.size(), Real{1});
  auto d_r = to_device(basis.nodes);
  auto d_jacobian = to_device(jacobian);
  const int n_free =
      quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis.basis, basis.domains, /*m=*/0);
  ASSERT_EQ(n_free, basis.n_free(0));
  CylindricalNewcombMatrix matrix{n_free};
  quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
      basis.basis, basis.domains, /*m=*/0, d_r, d_jacobian, coefficients,
      matrix, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto got = copy_to_host(matrix.values);
  const auto expected = assemble_host_reference(basis, f, g, jacobian, 0);
  ASSERT_EQ(got.size(), expected.size());
  for (int column = 0; column < n_free; ++column) {
    for (int row = 0; row < n_free; ++row) {
      const std::size_t index = static_cast<std::size_t>(row)
                              + static_cast<std::size_t>(column) * n_free;
      EXPECT_NEAR(got[index], expected[index],
                  Real{2e-12} * (Real{1} + std::abs(expected[index])))
          << "matrix entry (" << row << ", " << column << ")";
      EXPECT_DOUBLE_EQ(
          got[index],
          got[static_cast<std::size_t>(column)
              + static_cast<std::size_t>(row) * n_free]);
    }
  }

  // The interface at global node `order` is free index order-1.  Its diagonal
  // must be the sum of domain 0's upper endpoint and domain 1's lower endpoint.
  const int interface = basis.order - 1;
  const Real left = local_entry(basis, f, g, jacobian, 0, 0, 0);
  const Real right =
      local_entry(basis, f, g, jacobian, 1, basis.order, basis.order);
  ASSERT_NE(left, Real{0});
  ASSERT_NE(right, Real{0});
  EXPECT_NEAR(got[static_cast<std::size_t>(interface)
                  + static_cast<std::size_t>(interface) * n_free],
              left + right, Real{2e-12} * (Real{1} + std::abs(left + right)));
}

TEST(CylindricalNewcombAssembly,
     QuadraticFormEqualsDirectQuadratureForManufacturedXiR) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const BasisData basis{8, {Real{0.25}, Real{0.8}, Real{1.35}}};
  std::vector<Real> b_theta(basis.nodes.size());
  std::vector<Real> b_z(basis.nodes.size());
  std::vector<Real> dp_dr(basis.nodes.size());
  for (std::size_t i = 0; i < basis.nodes.size(); ++i) {
    const Real r = basis.nodes[i];
    b_theta[i] = Real{0.3} * r;
    b_z[i] = Real{1.2} + Real{0.1} * r;
    dp_dr[i] = Real{-800} * (Real{1} + Real{0.2} * r);
  }

  auto d_r = to_device(basis.nodes);
  auto d_b_theta = to_device(b_theta);
  auto d_b_z = to_device(b_z);
  auto d_dp_dr = to_device(dp_dr);
  CylindricalNewcombCoefficients coefficients{basis.nodes.size()};
  quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
      basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
      CylindricalNewcombMode{2, Real{-0.35}}, coefficients, nullptr);

  const int n_free =
      quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis.basis, basis.domains, /*m=*/2);
  const std::vector<Real> jacobian(basis.nodes.size(), Real{1});
  auto d_jacobian = to_device(jacobian);
  CylindricalNewcombMatrix matrix{n_free};
  quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
      basis.basis, basis.domains, /*m=*/2, d_r, d_jacobian, coefficients,
      matrix, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto f = copy_to_host(coefficients.f);
  const auto g = copy_to_host(coefficients.g);
  const auto k = copy_to_host(matrix.values);

  std::vector<Real> global_r(static_cast<std::size_t>(basis.n_global()),
                             std::numeric_limits<Real>::quiet_NaN());
  for (int domain = 0; domain < basis.n_domains; ++domain) {
    for (int local = 0; local < basis.n_nodes; ++local) {
      const int global = basis.global_node(domain, local);
      const Real radius = basis.nodes[basis.local_index(domain, local)];
      if (std::isfinite(global_r[static_cast<std::size_t>(global)])) {
        EXPECT_EQ(global_r[static_cast<std::size_t>(global)], radius);
      }
      global_r[static_cast<std::size_t>(global)] = radius;
    }
  }

  // A smooth manufactured physical radial displacement.  The endpoint
  // factors impose xi_r(a)=xi_r(b)=0 without changing variables to r*xi_r.
  const Real a = global_r.front();
  const Real b = global_r.back();
  std::vector<Real> xi_global(static_cast<std::size_t>(basis.n_global()));
  for (int global = 0; global < basis.n_global(); ++global) {
    const Real r = global_r[static_cast<std::size_t>(global)];
    xi_global[static_cast<std::size_t>(global)] =
        (r - a) * (b - r) * (Real{1} + Real{0.2} * r);
  }
  EXPECT_EQ(xi_global.front(), Real{0});
  EXPECT_EQ(xi_global.back(), Real{0});

  long double direct = 0;
  for (int domain = 0; domain < basis.n_domains; ++domain) {
    for (int q = 0; q < basis.n_nodes; ++q) {
      Real dxi_dr = Real{0};
      for (int local = 0; local < basis.n_nodes; ++local) {
        dxi_dr += basis.D(domain, q, local)
                 * xi_global[static_cast<std::size_t>(
                     basis.global_node(domain, local))];
      }
      const std::size_t iq = basis.local_index(domain, q);
      const Real xi_r = xi_global[static_cast<std::size_t>(
          basis.global_node(domain, q))];
      direct += static_cast<long double>(basis.weights[iq])
              * (static_cast<long double>(f[iq]) * dxi_dr * dxi_dr
                 + static_cast<long double>(g[iq]) * xi_r * xi_r);
    }
  }

  long double quadratic = 0;
  for (int column = 0; column < n_free; ++column) {
    const Real x_column = xi_global[static_cast<std::size_t>(column + 1)];
    for (int row = 0; row < n_free; ++row) {
      const Real x_row = xi_global[static_cast<std::size_t>(row + 1)];
      quadratic += static_cast<long double>(x_row)
                 * k[static_cast<std::size_t>(row)
                     + static_cast<std::size_t>(column) * n_free]
                 * x_column;
    }
  }

  const long double scale = 1 + std::abs(direct);
  EXPECT_NEAR(static_cast<double>(quadratic), static_cast<double>(direct),
              static_cast<double>(Real{5e-12} * scale));
}

TEST(CylindricalNewcombAssembly,
     RationalInterfaceHasIndependentOneSidedDisplacements) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  BasisData basis{5, {Real{0.3}, Real{0.8}, Real{1.4}}};
  basis.set_rational_interfaces({{1, {2}}});

  std::vector<Real> f(basis.nodes.size());
  std::vector<Real> g(basis.nodes.size());
  const std::vector<Real> jacobian(basis.nodes.size(), Real{1});
  for (std::size_t i = 0; i < basis.nodes.size(); ++i) {
    f[i] = Real{1.1} + Real{0.2} * basis.nodes[i]
         + Real{0.01} * static_cast<Real>(i);
    g[i] = Real{-0.15} + Real{0.03} * static_cast<Real>(i);
  }

  CylindricalNewcombCoefficients coefficients{f.size()};
  coefficients.f.copy_from_host(f.data(), f.size());
  coefficients.g.copy_from_host(g.data(), g.size());
  auto d_r = to_device(basis.nodes);
  auto d_jacobian = to_device(jacobian);

  const int n_free =
      quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis.basis, basis.domains, /*m=*/2);
  EXPECT_EQ(n_free, basis.n_free(0) + 1);
  CylindricalNewcombMatrix matrix{n_free};
  quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
      basis.basis, basis.domains, /*m=*/2, d_r, d_jacobian, coefficients,
      matrix, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto got = copy_to_host(matrix.values);
  const auto expected = assemble_host_reference(basis, f, g, jacobian, 2);
  ASSERT_EQ(got.size(), expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], expected[i],
                Real{2e-12} * (Real{1} + std::abs(expected[i])));
  }

  // Domain 0 local node 0 and domain 1 local node order are the two copies of
  // r=0.8.  They must be distinct for m=2 and have no cross-domain coupling.
  const int left = basis.free_dof(0, 0, 2);
  const int right = basis.free_dof(1, basis.order, 2);
  ASSERT_GE(left, 0);
  ASSERT_GE(right, 0);
  ASSERT_NE(left, right);
  EXPECT_DOUBLE_EQ(
      got[static_cast<std::size_t>(left)
          + static_cast<std::size_t>(right) * n_free],
      Real{0});

  const Real left_energy =
      local_entry(basis, f, g, jacobian, 0, 0, 0);
  const Real right_energy =
      local_entry(basis, f, g, jacobian, 1, basis.order, basis.order);
  EXPECT_NEAR(got[static_cast<std::size_t>(left)
                  + static_cast<std::size_t>(left) * n_free],
              left_energy,
              Real{2e-12} * (Real{1} + std::abs(left_energy)));
  EXPECT_NEAR(got[static_cast<std::size_t>(right)
                  + static_cast<std::size_t>(right) * n_free],
              right_energy,
              Real{2e-12} * (Real{1} + std::abs(right_energy)));

  // A cardinal trial with only the right interface value nonzero has exactly
  // the right-domain energy.  This would be impossible if the two copies were
  // still merged.
  std::vector<Real> one_sided(static_cast<std::size_t>(n_free), Real{0});
  one_sided[static_cast<std::size_t>(right)] = Real{1};
  long double quadratic = 0;
  for (int column = 0; column < n_free; ++column) {
    for (int row = 0; row < n_free; ++row) {
      quadratic += static_cast<long double>(
                       one_sided[static_cast<std::size_t>(row)])
                 * got[static_cast<std::size_t>(row)
                       + static_cast<std::size_t>(column) * n_free]
                 * one_sided[static_cast<std::size_t>(column)];
    }
  }
  EXPECT_NEAR(static_cast<Real>(quadratic), right_energy,
              Real{2e-12} * (Real{1} + std::abs(right_energy)));
}

TEST(CylindricalNewcombAssembly,
     ExplicitJacobianHandlesNonlinearRadiusCoordinate) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // The Chebyshev coordinate is s, while physical radius is a nonlinear map.
  // Treating basis.weights/diff as dr would therefore give the wrong form.
  const BasisData basis{8, {Real{0.2}, Real{0.6}, Real{1.0}}};
  std::vector<Real> radius(basis.nodes.size());
  std::vector<Real> jacobian(basis.nodes.size());
  std::vector<Real> f(basis.nodes.size());
  std::vector<Real> g(basis.nodes.size());
  for (std::size_t i = 0; i < basis.nodes.size(); ++i) {
    const Real s = basis.nodes[i];
    radius[i] = Real{0.7} + s + Real{0.4} * s * s;
    jacobian[i] = Real{1} + Real{0.8} * s;
    f[i] = Real{1.2} + Real{0.3} * radius[i];
    g[i] = Real{-0.4} + Real{0.2} * radius[i];
  }

  CylindricalNewcombCoefficients coefficients{f.size()};
  coefficients.f.copy_from_host(f.data(), f.size());
  coefficients.g.copy_from_host(g.data(), g.size());
  auto d_r = to_device(radius);
  auto d_jacobian = to_device(jacobian);
  const int n_free =
      quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis.basis, basis.domains, /*m=*/0);
  CylindricalNewcombMatrix matrix{n_free};
  quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
      basis.basis, basis.domains, /*m=*/0, d_r, d_jacobian, coefficients,
      matrix, nullptr);
  quasar::backend::device_synchronize(nullptr);
  const auto k = copy_to_host(matrix.values);

  std::vector<Real> global_radius(static_cast<std::size_t>(basis.n_global()));
  for (int domain = 0; domain < basis.n_domains; ++domain) {
    for (int local = 0; local < basis.n_nodes; ++local) {
      global_radius[static_cast<std::size_t>(
          basis.global_node(domain, local))] =
          radius[basis.local_index(domain, local)];
    }
  }
  const Real a = global_radius.front();
  const Real b = global_radius.back();
  std::vector<Real> xi(static_cast<std::size_t>(basis.n_global()));
  for (int radial = 0; radial < basis.n_global(); ++radial) {
    const Real r = global_radius[static_cast<std::size_t>(radial)];
    xi[static_cast<std::size_t>(radial)] =
        (r - a) * (b - r) * (Real{1} + Real{0.1} * r);
  }

  long double direct = 0;
  for (int domain = 0; domain < basis.n_domains; ++domain) {
    for (int q = 0; q < basis.n_nodes; ++q) {
      Real dxi_ds = Real{0};
      for (int local = 0; local < basis.n_nodes; ++local) {
        dxi_ds += basis.D(domain, q, local)
                 * xi[static_cast<std::size_t>(
                     basis.global_node(domain, local))];
      }
      const std::size_t iq = basis.local_index(domain, q);
      const Real value = xi[static_cast<std::size_t>(
          basis.global_node(domain, q))];
      direct += static_cast<long double>(basis.weights[iq])
              * (static_cast<long double>(f[iq] / jacobian[iq])
                     * dxi_ds * dxi_ds
                 + static_cast<long double>(g[iq] * jacobian[iq])
                     * value * value);
    }
  }

  long double quadratic = 0;
  for (int column = 0; column < n_free; ++column) {
    for (int row = 0; row < n_free; ++row) {
      quadratic += static_cast<long double>(
                       xi[static_cast<std::size_t>(row + 1)])
                 * k[static_cast<std::size_t>(row)
                     + static_cast<std::size_t>(column) * n_free]
                 * xi[static_cast<std::size_t>(column + 1)];
    }
  }
  const long double scale = 1 + std::abs(direct);
  EXPECT_NEAR(static_cast<double>(quadratic), static_cast<double>(direct),
              static_cast<double>(Real{5e-12} * scale));
}

TEST(CylindricalNewcombSuydam,
     CoefficientIdentityAndLocalizedNegativeEnergyWhenViolated) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr Real rs = Real{1};
  constexpr Real shear_parameter = Real{2};
  constexpr int m = 1;
  constexpr Real k_axial = Real{-1};
  BasisData basis{32, {Real{0.98}, rs, Real{1.02}}};
  basis.set_rational_interfaces({{1, {m}}});

  std::vector<Real> b_theta(basis.nodes.size());
  std::vector<Real> b_z(basis.nodes.size(), Real{1});
  std::vector<Real> dp_dr(basis.nodes.size());
  const std::vector<Real> jacobian(basis.nodes.size(), Real{1});
  for (std::size_t i = 0; i < basis.nodes.size(); ++i) {
    const Real r = basis.nodes[i];
    const Real h = Real{1} + shear_parameter * (r - rs);
    b_theta[i] = r * h;
    const Real db_theta_dr = h + r * shear_parameter;
    // Exact cylindrical radial force balance for B_z=constant.
    dp_dr[i] = -(b_theta[i] * b_theta[i] / r
                 + b_theta[i] * db_theta_dr)
               / quasar::mu0;
  }

  auto d_r = to_device(basis.nodes);
  auto d_b_theta = to_device(b_theta);
  auto d_b_z = to_device(b_z);
  auto d_dp_dr = to_device(dp_dr);
  auto d_jacobian = to_device(jacobian);
  CylindricalNewcombCoefficients coefficients{basis.nodes.size()};
  quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
      basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
      CylindricalNewcombMode{m, k_axial}, coefficients, nullptr);
  quasar::backend::device_synchronize(nullptr);
  const auto f = copy_to_host(coefficients.f);
  const auto g = copy_to_host(coefficients.g);

  const std::size_t rs_left = basis.local_index(0, 0);
  const std::size_t rs_right = basis.local_index(1, basis.order);
  EXPECT_DOUBLE_EQ(basis.nodes[rs_left], rs);
  EXPECT_DOUBLE_EQ(basis.nodes[rs_right], rs);
  EXPECT_DOUBLE_EQ(f[rs_left], Real{0});
  EXPECT_DOUBLE_EQ(f[rs_right], Real{0});
  EXPECT_DOUBLE_EQ(g[rs_left], g[rs_right]);

  // F = B_theta/r - B_z = shear_parameter*(r-rs), hence
  // alpha = rs^3 F'(rs)^2 / (mu0 D_s) in f=alpha*(r-rs)^2+...
  const Real d_s = Real{m * m} + k_axial * k_axial * rs * rs;
  const Real alpha = rs * rs * rs * shear_parameter * shear_parameter
                   / (quasar::mu0 * d_s);
  const Real q_prime_over_q = -shear_parameter;
  const Real suydam =
      rs / (Real{8} * quasar::mu0)
          * q_prime_over_q * q_prime_over_q
      + dp_dr[rs_left];
  const Real positive_prefactor =
      Real{2} * k_axial * k_axial * rs * rs / d_s;
  const Real identity_left = alpha / Real{4} + g[rs_left];
  const Real identity_right = positive_prefactor * suydam;
  EXPECT_LT(suydam, Real{0});
  EXPECT_DOUBLE_EQ(positive_prefactor, Real{1});
  EXPECT_NEAR(identity_left, suydam,
              Real{2e-12} * (Real{1} + std::abs(suydam)));
  EXPECT_NEAR(identity_left, identity_right,
              Real{2e-12} * (Real{1} + std::abs(identity_right)));

  const int n_free =
      quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis.basis, basis.domains, m);
  CylindricalNewcombMatrix matrix{n_free};
  quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
      basis.basis, basis.domains, m, d_r, d_jacobian, coefficients, matrix,
      nullptr);
  quasar::backend::device_synchronize(nullptr);
  const auto assembled = copy_to_host(matrix.values);

  // Localize on the narrow right side only.  The trial is one at rs+ and
  // falls linearly to zero at the conducting outer edge; every left-domain DOF
  // stays exactly zero.  The rational split makes this admissible.
  std::vector<Real> trial(static_cast<std::size_t>(n_free), Real{0});
  const Real width = basis.domains.breakpoints[2] - rs;
  for (int local = 0; local < basis.n_nodes; ++local) {
    const int dof = basis.free_dof(1, local, m);
    if (dof < 0) continue;
    const Real r = basis.nodes[basis.local_index(1, local)];
    trial[static_cast<std::size_t>(dof)] =
        Real{1} - (r - rs) / width;
  }

  long double energy = 0;
  for (int column = 0; column < n_free; ++column) {
    for (int row = 0; row < n_free; ++row) {
      energy += static_cast<long double>(
                    trial[static_cast<std::size_t>(row)])
              * assembled[static_cast<std::size_t>(row)
                          + static_cast<std::size_t>(column) * n_free]
              * trial[static_cast<std::size_t>(column)];
    }
  }
  EXPECT_LT(energy, 0.0L);
}

TEST(CylindricalNewcomb, RejectsInvalidModesAndStorageShapes) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const BasisData basis{2, {Real{0.4}, Real{1.0}}};
  const std::vector<Real> ones(basis.nodes.size(), Real{1});
  auto d_r = to_device(basis.nodes);
  auto d_b_theta = to_device(ones);
  auto d_b_z = to_device(ones);
  auto d_dp_dr = to_device(ones);
  auto d_jacobian = to_device(ones);
  CylindricalNewcombCoefficients coefficients{basis.nodes.size()};

  EXPECT_THROW(
      quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
          basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
          CylindricalNewcombMode{0, Real{0}}, coefficients, nullptr),
      std::invalid_argument);

  std::vector<Real> reaches_axis = basis.nodes;
  reaches_axis.back() = Real{0};
  auto d_reaches_axis = to_device(reaches_axis);
  EXPECT_THROW(
      quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
          basis.basis, d_reaches_axis, d_b_theta, d_b_z, d_dp_dr,
          CylindricalNewcombMode{1, Real{0}}, coefficients, nullptr),
      std::invalid_argument);

  const int valid_count =
      quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis.basis, basis.domains, /*m=*/1);
  CylindricalNewcombMatrix valid_matrix{valid_count};
  EXPECT_THROW(quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
                   basis.basis, basis.domains, /*m=*/1, d_reaches_axis,
                   d_jacobian, coefficients, valid_matrix, nullptr),
               std::invalid_argument);

  std::vector<Real> zero_jacobian = ones;
  zero_jacobian[0] = Real{0};
  auto d_zero_jacobian = to_device(zero_jacobian);
  EXPECT_THROW(quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
                   basis.basis, basis.domains, /*m=*/1, d_r,
                   d_zero_jacobian, coefficients, valid_matrix, nullptr),
               std::invalid_argument);
  EXPECT_THROW(
      quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
          basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
          CylindricalNewcombMode{
              1, std::numeric_limits<Real>::quiet_NaN()},
          coefficients, nullptr),
      std::invalid_argument);

  CylindricalNewcombCoefficients short_output{basis.nodes.size() - 1};
  EXPECT_THROW(
      quasar::stability::launch_evaluate_cylindrical_newcomb_coefficients(
          basis.basis, d_r, d_b_theta, d_b_z, d_dp_dr,
          CylindricalNewcombMode{1, Real{0}}, short_output, nullptr),
      std::invalid_argument);

  CylindricalNewcombMatrix wrong_matrix{2};
  EXPECT_THROW(quasar::stability::launch_assemble_cylindrical_newcomb_matrix(
                   basis.basis, basis.domains, /*m=*/1, d_r, d_jacobian,
                   coefficients, wrong_matrix, nullptr),
               std::invalid_argument);
}

TEST(CylindricalNewcomb, RejectsUnrepresentableBasisNodeMetadataOnHost) {
  ChebyshevBasis basis;
  basis.order = std::numeric_limits<int>::max();
  basis.n_nodes = std::numeric_limits<int>::min();
  basis.n_domains = 2;

  EXPECT_THROW(
      (void)quasar::stability::cylindrical_newcomb_fixed_boundary_dof_count(
          basis),
      std::invalid_argument);
}

}  // namespace
