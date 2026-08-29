// Exact-Chebyshev/Fourier PEST geometry for the toroidal stability operator.
//
// The synthetic equilibrium is deliberately analytic.  Its flux surfaces are
// concentric circles with a radius that is quadratic in psi_N, so the local
// Chebyshev differentiation matrix must recover radial derivatives to
// rounding.  The field-grid data obey B_phi = R B_pol with B_pol quadratic;
// tensor-product cubic interpolation therefore preserves a constant PEST
// integrand between grid nodes, while bilinear interpolation does not.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"
#include "quasar/physics/stability/kernels.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::GsFluxSurfaces;
using quasar::equilibrium::GsMagneticField;
using quasar::numerics::EllipticGrid;
using quasar::stability::ChebyshevBasis;
using quasar::stability::FluxCoordinateGrid;
using quasar::stability::RadialDomains;

constexpr Real kPi = Real{3.14159265358979323846};
constexpr int kOrder = 6;
constexpr int kDomains = 2;
constexpr int kNodes = kOrder + 1;
constexpr int kLocal = kDomains * kNodes;
constexpr int kContour = 64;
constexpr int kTheta = 64;
constexpr Real kMajorRadius = Real{1.45};
constexpr Real kAxisZ = Real{0.04};

Real minor_radius(Real psi_n) {
  return Real{0.12} + Real{0.22} * psi_n
       + Real{0.04} * psi_n * psi_n;
}

Real minor_radius_derivative(Real psi_n) {
  return Real{0.22} + Real{0.08} * psi_n;
}

template <class T>
std::vector<T> download(const DeviceBuffer<T>& buffer) {
  std::vector<T> host(buffer.size());
  buffer.copy_to_host(host.data(), host.size());
  return host;
}

struct SyntheticGeometry {
  // The minimum legal cubic grid exercises the shifted boundary stencil for
  // every sample rather than only the centered interior case.
  EllipticGrid grid{4, 4, Real{0.65}, Real{2.25}, Real{-0.8}, Real{0.8}};
  ChebyshevBasis basis{};
  GsMagneticField field{grid};
  int n_contour{kContour};
  int n_theta{kTheta};
  GsFluxSurfaces surfaces{};
  FluxCoordinateGrid coords{};
  std::vector<Real> nodes{};

  explicit SyntheticGeometry(int contour = kContour, int theta = kTheta)
      : n_contour{contour}, n_theta{theta}, surfaces{kLocal, contour},
        coords{kLocal, theta} {
    RadialDomains domains{};
    domains.n_domains = kDomains;
    domains.breakpoints[0] = Real{0.11};
    domains.breakpoints[1] = Real{0.43};
    domains.breakpoints[2] = Real{0.91};
    DeviceBuffer<RadialDomains> d_domains{1};
    d_domains.copy_from_host(&domains, 1);

    basis.resize(kOrder, kDomains);
    quasar::stability::launch_build_chebyshev_basis(d_domains.device_ptr(),
                                                     basis, nullptr);
    quasar::backend::device_synchronize(nullptr);
    nodes = download(basis.nodes);

    std::vector<Real> b_pol(grid.size());
    std::vector<Real> b_phi(grid.size());
    for (int j = 0; j < grid.nz; ++j) {
      const Real z = grid.z(j);
      for (int i = 0; i < grid.nr; ++i) {
        const Real r = grid.r(i);
        const std::size_t k = grid.index(i, j);
        // Positive everywhere on the grid and polynomial of degree <= 2 in
        // each coordinate.  R*B_pol is degree <= 3, so cubic tensor-product
        // interpolation preserves B_phi/(R B_pol) == 1 off-grid.
        b_pol[k] = Real{1.2} + Real{0.09} * r * r
                 + Real{0.07} * z * z + Real{0.025} * r * z;
        b_phi[k] = r * b_pol[k];
      }
    }
    field.b_poloidal.copy_from_host(b_pol.data(), b_pol.size());
    field.b_phi.copy_from_host(b_phi.data(), b_phi.size());

    std::vector<Real> contour_r(static_cast<std::size_t>(kLocal) * n_contour);
    std::vector<Real> contour_z(contour_r.size());
    std::vector<int> counts(kLocal, n_contour);
    std::vector<int> closed(kLocal, 1);
    for (int s = 0; s < kLocal; ++s) {
      const Real a = minor_radius(nodes[static_cast<std::size_t>(s)]);
      const std::size_t base = static_cast<std::size_t>(s) * n_contour;
      for (int j = 0; j < n_contour; ++j) {
        const Real theta = Real{2} * kPi * static_cast<Real>(j)
                         / static_cast<Real>(n_contour);
        contour_r[base + j] = kMajorRadius + a * std::cos(theta);
        contour_z[base + j] = kAxisZ + a * std::sin(theta);
      }
    }
    surfaces.r.copy_from_host(contour_r.data(), contour_r.size());
    surfaces.z.copy_from_host(contour_z.data(), contour_z.size());
    surfaces.count.copy_from_host(counts.data(), counts.size());
    surfaces.closed.copy_from_host(closed.data(), closed.size());
    surfaces.psi_n.copy_from_host(nodes.data(), nodes.size());
  }

  void build() {
    quasar::stability::launch_build_spectral_flux_coordinates(
        grid, surfaces, field, basis, coords, nullptr);
  }
};

TEST(SpectralFluxCoordinates, RecoversAnalyticChebyshevFourierGeometry) {
  SyntheticGeometry fixture;
  ASSERT_NO_THROW(fixture.build());

  const auto psi_n = download(fixture.coords.psi_n);
  const auto q = download(fixture.coords.q);
  const auto valid = download(fixture.coords.valid);
  const auto r = download(fixture.coords.r);
  const auto z = download(fixture.coords.z);
  const auto r_p = download(fixture.coords.dr_dpsi);
  const auto z_p = download(fixture.coords.dz_dpsi);
  const auto r_t = download(fixture.coords.dr_dtheta);
  const auto z_t = download(fixture.coords.dz_dtheta);
  const auto jac = download(fixture.coords.jacobian);
  const auto g_pp = download(fixture.coords.g_psipsi);
  const auto g_pt = download(fixture.coords.g_psitheta);
  const auto g_tt = download(fixture.coords.g_thetatheta);

  for (int s = 0; s < kLocal; ++s) {
    ASSERT_EQ(valid[static_cast<std::size_t>(s)], 1) << "surface " << s;
    EXPECT_DOUBLE_EQ(psi_n[static_cast<std::size_t>(s)],
                     fixture.nodes[static_cast<std::size_t>(s)]);

    const Real psi = fixture.nodes[static_cast<std::size_t>(s)];
    const Real a = minor_radius(psi);
    const Real a_p = minor_radius_derivative(psi);
    const Real polygon_q = static_cast<Real>(kContour) * Real{2} * a
                         * std::sin(kPi / static_cast<Real>(kContour))
                         / (Real{2} * kPi);
    EXPECT_NEAR(q[static_cast<std::size_t>(s)], polygon_q, Real{2e-12})
        << "surface " << s
        << ": cubic sampling did not preserve B_phi/(R B_pol) = 1";

    for (int j = 0; j < kTheta; ++j) {
      const Real theta = Real{2} * kPi * static_cast<Real>(j)
                       / static_cast<Real>(kTheta);
      const Real c = std::cos(theta);
      const Real sn = std::sin(theta);
      const Real want_r = kMajorRadius + a * c;
      const Real want_z = kAxisZ + a * sn;
      const Real want_r_p = a_p * c;
      const Real want_z_p = a_p * sn;
      const Real want_r_t = -a * sn;
      const Real want_z_t = a * c;
      const Real want_jac = want_r * a * a_p;
      const std::size_t k = static_cast<std::size_t>(s) * kTheta + j;

      EXPECT_NEAR(r[k], want_r, Real{3e-12});
      EXPECT_NEAR(z[k], want_z, Real{3e-12});
      EXPECT_NEAR(r_p[k], want_r_p, Real{2e-9});
      EXPECT_NEAR(z_p[k], want_z_p, Real{2e-9});
      EXPECT_NEAR(r_t[k], want_r_t, Real{2e-10});
      EXPECT_NEAR(z_t[k], want_z_t, Real{2e-10});
      EXPECT_NEAR(jac[k], want_jac, Real{3e-9});
      EXPECT_NEAR(g_pp[k], Real{1} / (a_p * a_p), Real{5e-8});
      EXPECT_NEAR(g_pt[k], Real{0}, Real{5e-8});
      EXPECT_NEAR(g_tt[k], Real{1} / (a * a), Real{5e-8});

      const Real cov_pp = r_p[k] * r_p[k] + z_p[k] * z_p[k];
      const Real cov_pt = r_p[k] * r_t[k] + z_p[k] * z_t[k];
      const Real cov_tt = r_t[k] * r_t[k] + z_t[k] * z_t[k];
      EXPECT_NEAR(g_pp[k] * cov_pp + g_pt[k] * cov_pt, Real{1},
                  Real{2e-12});
      EXPECT_NEAR(g_pp[k] * cov_pt + g_pt[k] * cov_tt, Real{0},
                  Real{2e-12});
      EXPECT_NEAR(g_pt[k] * cov_pp + g_tt[k] * cov_pt, Real{0},
                  Real{2e-12});
      EXPECT_NEAR(g_pt[k] * cov_pt + g_tt[k] * cov_tt, Real{1},
                  Real{2e-12});
    }
  }
}

template <class T>
void expect_bitwise_equal(const DeviceBuffer<T>& lhs,
                          const DeviceBuffer<T>& rhs) {
  const auto a = download(lhs);
  const auto b = download(rhs);
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(T)), 0);
}

TEST(SpectralFluxCoordinates, RetainsEqualCopiesOfAChebyshevInterface) {
  SyntheticGeometry fixture;
  fixture.build();

  const int left_upper = 0;
  const int right_lower = 2 * kNodes - 1;
  ASSERT_DOUBLE_EQ(fixture.nodes[static_cast<std::size_t>(left_upper)],
                   fixture.nodes[static_cast<std::size_t>(right_lower)]);

  const auto psi_n = download(fixture.coords.psi_n);
  const auto q = download(fixture.coords.q);
  const auto r = download(fixture.coords.r);
  const auto z = download(fixture.coords.z);
  const auto r_p = download(fixture.coords.dr_dpsi);
  const auto z_p = download(fixture.coords.dz_dpsi);
  const auto r_t = download(fixture.coords.dr_dtheta);
  const auto z_t = download(fixture.coords.dz_dtheta);
  const auto jac = download(fixture.coords.jacobian);
  const auto g_pp = download(fixture.coords.g_psipsi);
  const auto g_pt = download(fixture.coords.g_psitheta);
  const auto g_tt = download(fixture.coords.g_thetatheta);

  EXPECT_DOUBLE_EQ(psi_n[static_cast<std::size_t>(left_upper)],
                   psi_n[static_cast<std::size_t>(right_lower)]);
  EXPECT_DOUBLE_EQ(q[static_cast<std::size_t>(left_upper)],
                   q[static_cast<std::size_t>(right_lower)]);
  for (int j = 0; j < kTheta; ++j) {
    const std::size_t left = static_cast<std::size_t>(left_upper) * kTheta + j;
    const std::size_t right = static_cast<std::size_t>(right_lower) * kTheta + j;
    EXPECT_DOUBLE_EQ(r[left], r[right]);
    EXPECT_DOUBLE_EQ(z[left], z[right]);
    EXPECT_DOUBLE_EQ(r_t[left], r_t[right]);
    EXPECT_DOUBLE_EQ(z_t[left], z_t[right]);
    EXPECT_NEAR(r_p[left], r_p[right], Real{2e-11});
    EXPECT_NEAR(z_p[left], z_p[right], Real{2e-11});
    EXPECT_NEAR(jac[left], jac[right], Real{2e-11});
    EXPECT_NEAR(g_pp[left], g_pp[right], Real{2e-9});
    EXPECT_NEAR(g_pt[left], g_pt[right], Real{2e-9});
    EXPECT_NEAR(g_tt[left], g_tt[right], Real{2e-9});
  }
}

TEST(SpectralFluxCoordinates, OddFourierGridDifferentiatesPeriodicModes) {
  constexpr int odd_theta = 63;
  SyntheticGeometry fixture{odd_theta, odd_theta};
  fixture.build();

  const auto r_t = download(fixture.coords.dr_dtheta);
  const auto z_t = download(fixture.coords.dz_dtheta);
  for (int s = 0; s < kLocal; ++s) {
    const Real a = minor_radius(fixture.nodes[static_cast<std::size_t>(s)]);
    for (int j = 0; j < odd_theta; ++j) {
      const Real theta = Real{2} * kPi * static_cast<Real>(j)
                       / static_cast<Real>(odd_theta);
      const std::size_t k = static_cast<std::size_t>(s) * odd_theta + j;
      EXPECT_NEAR(r_t[k], -a * std::sin(theta), Real{2e-10});
      EXPECT_NEAR(z_t[k], a * std::cos(theta), Real{2e-10});
    }
  }
}

TEST(SpectralFluxCoordinates, RepeatedBuildIsBitwiseDeterministic) {
  SyntheticGeometry fixture;
  FluxCoordinateGrid second{kLocal, kTheta};
  fixture.build();
  quasar::stability::launch_build_spectral_flux_coordinates(
      fixture.grid, fixture.surfaces, fixture.field, fixture.basis, second,
      nullptr);

  expect_bitwise_equal(fixture.coords.psi_n, second.psi_n);
  expect_bitwise_equal(fixture.coords.q, second.q);
  expect_bitwise_equal(fixture.coords.valid, second.valid);
  expect_bitwise_equal(fixture.coords.r, second.r);
  expect_bitwise_equal(fixture.coords.z, second.z);
  expect_bitwise_equal(fixture.coords.dr_dpsi, second.dr_dpsi);
  expect_bitwise_equal(fixture.coords.dz_dpsi, second.dz_dpsi);
  expect_bitwise_equal(fixture.coords.dr_dtheta, second.dr_dtheta);
  expect_bitwise_equal(fixture.coords.dz_dtheta, second.dz_dtheta);
  expect_bitwise_equal(fixture.coords.jacobian, second.jacobian);
  expect_bitwise_equal(fixture.coords.g_psipsi, second.g_psipsi);
  expect_bitwise_equal(fixture.coords.g_psitheta, second.g_psitheta);
  expect_bitwise_equal(fixture.coords.g_thetatheta, second.g_thetatheta);
}

TEST(SpectralFluxCoordinates, RejectsShapeAndOrderingMismatches) {
  SyntheticGeometry fixture;

  FluxCoordinateGrid missing_node{kLocal - 1, kTheta};
  EXPECT_THROW(quasar::stability::launch_build_spectral_flux_coordinates(
                   fixture.grid, fixture.surfaces, fixture.field,
                   fixture.basis, missing_node, nullptr),
               std::invalid_argument);

  FluxCoordinateGrid too_few_fourier_points{kLocal, 2};
  EXPECT_THROW(quasar::stability::launch_build_spectral_flux_coordinates(
                   fixture.grid, fixture.surfaces, fixture.field,
                   fixture.basis, too_few_fourier_points, nullptr),
               std::invalid_argument);

  auto wrong_order = fixture.nodes;
  wrong_order[3] += Real{1e-12};
  fixture.surfaces.psi_n.copy_from_host(wrong_order.data(), wrong_order.size());
  EXPECT_THROW(quasar::stability::launch_build_spectral_flux_coordinates(
                   fixture.grid, fixture.surfaces, fixture.field,
                   fixture.basis, fixture.coords, nullptr),
               std::invalid_argument);

  fixture.surfaces.psi_n.copy_from_host(fixture.nodes.data(),
                                        fixture.nodes.size());
  auto malformed_basis_nodes = fixture.nodes;
  std::swap(malformed_basis_nodes[2], malformed_basis_nodes[3]);
  fixture.basis.nodes.copy_from_host(malformed_basis_nodes.data(),
                                     malformed_basis_nodes.size());
  fixture.surfaces.psi_n.copy_from_host(malformed_basis_nodes.data(),
                                        malformed_basis_nodes.size());
  EXPECT_THROW(quasar::stability::launch_build_spectral_flux_coordinates(
                   fixture.grid, fixture.surfaces, fixture.field,
                   fixture.basis, fixture.coords, nullptr),
               std::invalid_argument);

  fixture.basis.nodes.copy_from_host(fixture.nodes.data(), fixture.nodes.size());
  fixture.surfaces.psi_n.copy_from_host(fixture.nodes.data(),
                                        fixture.nodes.size());
  std::vector<int> invalid_counts(kLocal, fixture.n_contour);
  invalid_counts[0] = fixture.n_contour + 1;
  fixture.surfaces.count.copy_from_host(invalid_counts.data(),
                                        invalid_counts.size());
  EXPECT_THROW(quasar::stability::launch_build_spectral_flux_coordinates(
                   fixture.grid, fixture.surfaces, fixture.field,
                   fixture.basis, fixture.coords, nullptr),
               std::invalid_argument);
}

TEST(SpectralFluxCoordinates, InvalidatesAnEntireCoupledChebyshevDomain) {
  SyntheticGeometry fixture;
  std::vector<int> closed(kLocal, 1);
  closed[2] = 0;
  fixture.surfaces.closed.copy_from_host(closed.data(), closed.size());
  fixture.build();

  const auto valid = download(fixture.coords.valid);
  const auto r_p = download(fixture.coords.dr_dpsi);
  const auto r_t = download(fixture.coords.dr_dtheta);
  const auto jac = download(fixture.coords.jacobian);
  for (int s = 0; s < kNodes; ++s) {
    EXPECT_EQ(valid[static_cast<std::size_t>(s)], 0);
    for (int j = 0; j < kTheta; ++j) {
      const std::size_t k = static_cast<std::size_t>(s) * kTheta + j;
      EXPECT_DOUBLE_EQ(r_p[k], Real{0});
      EXPECT_DOUBLE_EQ(r_t[k], Real{0});
      EXPECT_DOUBLE_EQ(jac[k], Real{0});
    }
  }
  for (int s = kNodes; s < kLocal; ++s) {
    EXPECT_EQ(valid[static_cast<std::size_t>(s)], 1);
  }
}

}  // namespace
