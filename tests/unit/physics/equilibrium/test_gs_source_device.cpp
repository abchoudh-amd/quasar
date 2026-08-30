// Host-vs-device equivalence for the Grad-Shafranov source terms and the
// sixth-order derivative fields.
//
// Every kernel covered here is elementwise or a per-line solve, so all of these
// are EQUALITY assertions.
//
// One caveat about the references. build_current and build_jacobian_diagonal
// are private members of GsSolver, so they cannot be called directly and the
// expected values below re-express them. Those two references are transcribed
// from gs_solver.hpp and must be kept in step with it; the remaining references
// (compute_derivatives, and the trivial field helpers) call the real host code.
// When the host implementation is deleted at the end of the port, these
// transcriptions become the specification rather than a copy of one.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::ProfileCoefficients;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

constexpr Real kMu0 = quasar::equilibrium::kMu0;

std::size_t bitwise_mismatches(const ScalarField& a, const ScalarField& b) {
  std::size_t n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::memcmp(&a[k], &b[k], sizeof(Real)) != 0) ++n;
  }
  return n;
}

Real max_abs_difference(const ScalarField& a, const ScalarField& b) {
  Real m = Real{0};
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

// A flux-like field with a genuine interior extremum, so normalized_flux spans
// the full [0, 1] range and both the inside-plasma and outside-plasma branches
// of the current kernel are exercised.
ScalarField flux_field(const EllipticGrid& g) {
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real dr = (g.r(i) - Real{1.15}) / Real{0.55};
      const Real dz = (g.z(j) - Real{0.05}) / Real{0.50};
      psi[g.index(i, j)] = Real{0.24} * std::exp(-(dr * dr + dz * dz))
                         + Real{0.03} * std::sin(Real{2.0} * g.r(i));
    }
  }
  return psi;
}

// Deliberately multi-term and asymmetric between p' and FF': a one-term profile
// would not distinguish Horner from a plain linear evaluation, and identical
// coefficient lists would hide a p/f swap in the kernel.
quasar::equilibrium::PolynomialProfile test_profile() {
  return quasar::equilibrium::PolynomialProfile{
      {Real{1.0}, Real{-0.6}, Real{0.25}, Real{-0.05}},
      {Real{0.8}, Real{-1.1}, Real{0.4}}};
}

DeviceBuffer<Real> upload(const ScalarField& f) {
  DeviceBuffer<Real> d{f.size()};
  d.copy_from_host(f.data(), f.size());
  return d;
}

DeviceBuffer<int> upload_mask(const std::vector<int>& mask) {
  DeviceBuffer<int> device{mask.size()};
  device.copy_from_host(mask.data(), mask.size());
  return device;
}

ScalarField download(const DeviceBuffer<Real>& d, std::size_t n) {
  ScalarField h(n, Real{0});
  d.copy_to_host(h.data(), n);
  return h;
}

TEST(GsSourceDevice, ProfileLoweringPreservesCoefficients) {
  const auto profile = test_profile();
  const ProfileCoefficients pod = quasar::equilibrium::to_coefficients(profile);

  ASSERT_EQ(pod.n_p, 4);
  ASSERT_EQ(pod.n_f, 3);
  for (int k = 0; k < pod.n_p; ++k) {
    EXPECT_EQ(pod.p_coeffs[k], profile.p_coefficients()[static_cast<std::size_t>(k)]);
  }
  for (int k = 0; k < pod.n_f; ++k) {
    EXPECT_EQ(pod.f_coeffs[k], profile.f_coefficients()[static_cast<std::size_t>(k)]);
  }
}

TEST(GsSourceDevice, ProfileLoweringRejectsOversizedProfile) {
  std::vector<Real> too_many(ProfileCoefficients::kMaxCoefficients + 1, Real{1});
  const quasar::equilibrium::PolynomialProfile big{too_many, {Real{1}}};
  EXPECT_THROW((void)quasar::equilibrium::to_coefficients(big),
               std::invalid_argument);
}

TEST(GsSourceDevice, DerivativeFieldsMatchHostBitExactly) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.2}, Real{-0.8}, Real{0.8}};
  const ScalarField psi = flux_field(g);

  const quasar::equilibrium::DerivativeFields host =
      quasar::equilibrium::compute_derivatives(g, psi);

  auto d_psi = upload(psi);
  quasar::equilibrium::GsDerivativeFields dev{g};
  quasar::equilibrium::GsOperatorScratch scratch{g};
  quasar::equilibrium::launch_gs_compute_derivatives(g, d_psi.device_ptr(), dev,
                                                     scratch, nullptr);
  quasar::backend::device_synchronize(nullptr);

  EXPECT_EQ(bitwise_mismatches(host.d_r, download(dev.d_r, g.size())), 0u)
      << "d_r";
  EXPECT_EQ(bitwise_mismatches(host.d_z, download(dev.d_z, g.size())), 0u)
      << "d_z";
  EXPECT_EQ(bitwise_mismatches(host.d_rr, download(dev.d_rr, g.size())), 0u)
      << "d_rr";
  EXPECT_EQ(bitwise_mismatches(host.d_zz, download(dev.d_zz, g.size())), 0u)
      << "d_zz";
  // The mixed derivative is the one that catches a composition-order mistake:
  // d_rz must be d_r differentiated along z, not d_z differentiated along r.
  EXPECT_EQ(bitwise_mismatches(host.d_rz, download(dev.d_rz, g.size())), 0u)
      << "d_rz (composition order)";
}

TEST(GsSourceDevice, BuildCurrentMatchesHostBitExactly) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.2}, Real{-0.8}, Real{0.8}};
  const ScalarField psi = flux_field(g);
  const auto profile = test_profile();

  const Real psi_axis = Real{0.27};
  const Real psi_boundary = Real{0.08};

  // Reference transcribed from GsSolver::build_current.
  ScalarField host = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real pn = quasar::equilibrium::normalized_flux(
          psi[g.index(i, j)], psi_axis, psi_boundary);
      if (pn >= Real{1}) continue;
      const Real r = g.r(i);
      host[g.index(i, j)] = r * profile.dp_dpsi(pn)
                          + profile.ff_prime(pn) / (kMu0 * r);
    }
  }

  auto d_psi = upload(psi);
  auto d_mask = upload_mask(std::vector<int>(g.size(), 1));
  DeviceBuffer<Real> d_j{g.size()};
  quasar::equilibrium::launch_gs_build_current(
      g, d_psi.device_ptr(), quasar::equilibrium::to_coefficients(profile),
      psi_axis, psi_boundary, d_mask.device_ptr(), d_j.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  const ScalarField dev = download(d_j, g.size());

  // Both branches must actually be taken, or the equality is only covering one.
  int inside = 0, outside = 0;
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real pn = quasar::equilibrium::normalized_flux(
          psi[g.index(i, j)], psi_axis, psi_boundary);
      (pn >= Real{1} ? outside : inside)++;
    }
  }
  ASSERT_GT(inside, 0) << "no in-plasma nodes: current branch untested";
  ASSERT_GT(outside, 0) << "no out-of-plasma nodes: cutoff branch untested";

  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

TEST(GsSourceDevice, JacobianDiagonalMatchesHostBitExactly) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.2}, Real{-0.8}, Real{0.8}};
  const ScalarField psi = flux_field(g);
  const auto profile = test_profile();

  const Real psi_axis = Real{0.27};
  const Real psi_boundary = Real{0.08};
  const Real profile_scale = Real{2.2365837665};
  const Real denom = psi_boundary - psi_axis;

  // Reference transcribed from GsSolver::build_jacobian_diagonal.
  ScalarField host = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real pn = quasar::equilibrium::normalized_flux(
          psi[g.index(i, j)], psi_axis, psi_boundary);
      if (pn >= Real{1} || pn <= Real{0}) continue;
      const Real r = g.r(i);
      host[g.index(i, j)] = profile_scale
          * (-kMu0 * r * r * profile.d2p_dpsi2(pn)
             - profile.ff_prime_prime(pn)) / denom;
    }
  }

  auto d_psi = upload(psi);
  auto d_mask = upload_mask(std::vector<int>(g.size(), 1));
  DeviceBuffer<Real> d_jac{g.size()};
  quasar::equilibrium::launch_gs_build_jacobian_diagonal(
      g, d_psi.device_ptr(), quasar::equilibrium::to_coefficients(profile),
      psi_axis, psi_boundary, profile_scale, d_mask.device_ptr(),
      d_jac.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  EXPECT_EQ(bitwise_mismatches(host, download(d_jac, g.size())), 0u);
}

// A zero flux span makes normalized_flux degenerate. The host returns psi_N = 1
// (no current) and the Jacobian bails out entirely; the device must do both,
// not produce NaN or Inf.
TEST(GsSourceDevice, DegenerateFluxSpanProducesNoCurrentOrJacobian) {
  const EllipticGrid g{33, 33, Real{0.4}, Real{1.8}, Real{-0.6}, Real{0.6}};
  const ScalarField psi = flux_field(g);
  const auto profile = test_profile();
  const ProfileCoefficients pod = quasar::equilibrium::to_coefficients(profile);

  auto d_psi = upload(psi);
  auto d_mask = upload_mask(std::vector<int>(g.size(), 1));
  DeviceBuffer<Real> d_j{g.size()};
  DeviceBuffer<Real> d_jac{g.size()};

  quasar::equilibrium::launch_gs_build_current(g, d_psi.device_ptr(), pod,
                                               Real{0.5}, Real{0.5},
                                               d_mask.device_ptr(),
                                               d_j.device_ptr(), nullptr);
  quasar::equilibrium::launch_gs_build_jacobian_diagonal(
      g, d_psi.device_ptr(), pod, Real{0.5}, Real{0.5}, Real{1},
      d_mask.device_ptr(), d_jac.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  for (const Real v : download(d_j, g.size())) {
    ASSERT_TRUE(std::isfinite(v));
    EXPECT_EQ(v, Real{0});
  }
  for (const Real v : download(d_jac, g.size())) {
    ASSERT_TRUE(std::isfinite(v));
    EXPECT_EQ(v, Real{0});
  }
}

TEST(GsSourceDevice, InvalidNormalizedFluxProducesNoCurrentOrJacobian) {
  const EllipticGrid g{9, 9, Real{0.4}, Real{1.8}, Real{-0.6}, Real{0.6}};
  const ProfileCoefficients profile =
      quasar::equilibrium::to_coefficients(test_profile());
  auto d_mask = upload_mask(std::vector<int>(g.size(), 1));
  DeviceBuffer<Real> d_j{g.size()};
  DeviceBuffer<Real> d_jac{g.size()};

  const Real limit = std::numeric_limits<Real>::max();
  const Real infinity = std::numeric_limits<Real>::infinity();
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real tiny = std::numeric_limits<Real>::denorm_min();
  struct Case {
    Real psi;
    Real axis;
    Real boundary;
  };
  const Case cases[] = {
      {nan, Real{0}, Real{1}},
      {infinity, Real{0}, Real{1}},
      {Real{0}, nan, Real{1}},
      {Real{0}, Real{0}, infinity},
      {Real{0}, -limit, limit},
      {limit, -limit, Real{0}},
      {-limit, Real{0}, tiny},
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(testing::Message{}
                 << "psi=" << test_case.psi << " axis=" << test_case.axis
                 << " boundary=" << test_case.boundary);
    const ScalarField psi(g.size(), test_case.psi);
    auto d_psi = upload(psi);
    quasar::equilibrium::launch_gs_build_current(
        g, d_psi.device_ptr(), profile, test_case.axis, test_case.boundary,
        d_mask.device_ptr(), d_j.device_ptr(), nullptr);
    quasar::equilibrium::launch_gs_build_jacobian_diagonal(
        g, d_psi.device_ptr(), profile, test_case.axis, test_case.boundary,
        Real{1}, d_mask.device_ptr(), d_jac.device_ptr(), nullptr);
    quasar::backend::device_synchronize(nullptr);

    for (const Real value : download(d_j, g.size())) EXPECT_EQ(value, Real{0});
    for (const Real value : download(d_jac, g.size())) {
      EXPECT_EQ(value, Real{0});
    }
  }
}

TEST(GsSourceDevice, PlasmaMaskBlocksAnOffGridSaddleBetweenFluxLobes) {
  const EllipticGrid g{81, 41, Real{0.6}, Real{2.4}, Real{-0.5}, Real{0.5}};
  // The saddle lies halfway between radial nodes 40 and 41. Both endpoint
  // values are inside the scalar cutoff, so a node-only BFS crosses this edge.
  constexpr Real center = Real{1.51125};
  constexpr Real half_separation = Real{0.45};
  constexpr Real axis_r = center - half_separation;
  constexpr Real axis_z = Real{0};
  constexpr Real psi_axis = Real{0};
  constexpr Real psi_boundary =
      -half_separation * half_separation * half_separation * half_separation;

  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real x = g.r(i) - center;
      const Real z = g.z(j);
      const Real well = x * x - half_separation * half_separation;
      psi[g.index(i, j)] = -(well * well + z * z);
    }
  }

  const std::vector<int> host_mask =
      quasar::equilibrium::axis_connected_plasma_mask(
          g, psi, axis_r, axis_z, psi_axis, psi_boundary);
  const int left_i = 20;
  const int right_i = 60;
  const int mid_j = 20;
  const int bridge_left_i = 40;
  const int bridge_right_i = 41;
  ASSERT_EQ(center, Real{0.5}
                        * (g.r(bridge_left_i) + g.r(bridge_right_i)));
  EXPECT_LT(quasar::equilibrium::normalized_flux(
                psi[g.index(bridge_left_i, mid_j)], psi_axis, psi_boundary),
            Real{1});
  EXPECT_LT(quasar::equilibrium::normalized_flux(
                psi[g.index(bridge_right_i, mid_j)], psi_axis, psi_boundary),
            Real{1});
  EXPECT_EQ(host_mask[g.index(left_i, mid_j)], 1);
  EXPECT_EQ(host_mask[g.index(right_i, mid_j)], 0);

  auto d_psi = upload(psi);
  quasar::equilibrium::GsDerivativeFields derivatives{g};
  quasar::equilibrium::GsOperatorScratch operator_scratch{g};
  quasar::equilibrium::launch_gs_compute_derivatives(
      g, d_psi.device_ptr(), derivatives, operator_scratch, nullptr);
  quasar::equilibrium::GsPlasmaMaskScratch scratch{g};
  quasar::equilibrium::launch_gs_build_plasma_mask(
      g, d_psi.device_ptr(), derivatives, axis_r, axis_z, psi_axis,
      psi_boundary, scratch, nullptr);
  quasar::backend::device_synchronize(nullptr);
  std::vector<int> device_mask(g.size(), 0);
  scratch.mask.copy_to_host(device_mask.data(), device_mask.size());
  EXPECT_EQ(device_mask, host_mask);

  DeviceBuffer<Real> d_current{g.size()};
  quasar::equilibrium::launch_gs_build_current(
      g, d_psi.device_ptr(),
      quasar::equilibrium::to_coefficients(test_profile()), psi_axis,
      psi_boundary, scratch.mask.device_ptr(), d_current.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  const ScalarField current = download(d_current, g.size());
  EXPECT_NE(current[g.index(left_i, mid_j)], Real{0});
  EXPECT_EQ(current[g.index(right_i, mid_j)], Real{0});
}

TEST(GsSourceDevice, PlasmaMaskRejectsOverflowedFluxSpanIdentically) {
  const EllipticGrid g{17, 17, Real{0.6}, Real{2.4}, Real{-0.5}, Real{0.5}};
  const ScalarField psi(g.size(), Real{0});
  const Real limit = std::numeric_limits<Real>::max();
  const Real psi_axis = -limit;
  const Real psi_boundary = limit;
  ASSERT_TRUE(std::isfinite(psi_axis));
  ASSERT_TRUE(std::isfinite(psi_boundary));
  ASSERT_FALSE(std::isfinite(psi_boundary - psi_axis));

  const std::vector<int> host_mask =
      quasar::equilibrium::axis_connected_plasma_mask(
          g, psi, Real{1.5}, Real{0}, psi_axis, psi_boundary);
  for (const int value : host_mask) EXPECT_EQ(value, 0);

  auto d_psi = upload(psi);
  quasar::equilibrium::GsDerivativeFields derivatives{g};
  quasar::equilibrium::GsPlasmaMaskScratch scratch{g};
  quasar::equilibrium::launch_gs_build_plasma_mask(
      g, d_psi.device_ptr(), derivatives, Real{1.5}, Real{0}, psi_axis,
      psi_boundary, scratch, nullptr);
  quasar::backend::device_synchronize(nullptr);
  std::vector<int> device_mask(g.size(), 1);
  scratch.mask.copy_to_host(device_mask.data(), device_mask.size());
  EXPECT_EQ(device_mask, host_mask);
}

TEST(GsSourceDevice, FieldHelpersMatchHost) {
  const EllipticGrid g{41, 27, Real{0.35}, Real{1.85}, Real{-0.7}, Real{0.9}};
  const ScalarField base = flux_field(g);

  // -- rhs = -mu0 r j_phi on the interior, zero on the boundary --------------
  {
    ScalarField host = quasar::numerics::make_field(g);
    for (int j = 1; j < g.nz - 1; ++j) {
      for (int i = 1; i < g.nr - 1; ++i) {
        host[g.index(i, j)] = -kMu0 * g.r(i) * base[g.index(i, j)];
      }
    }
    auto d_j = upload(base);
    DeviceBuffer<Real> d_rhs{g.size()};
    quasar::equilibrium::launch_gs_build_rhs(g, d_j.device_ptr(),
                                             d_rhs.device_ptr(), nullptr);
    quasar::backend::device_synchronize(nullptr);
    EXPECT_EQ(bitwise_mismatches(host, download(d_rhs, g.size())), 0u) << "rhs";
  }

  // -- in-place scale --------------------------------------------------------
  {
    const Real scale = Real{2.2365837665};
    ScalarField host = base;
    for (auto& v : host) v *= scale;

    auto d = upload(base);
    quasar::equilibrium::launch_gs_scale_field(g, scale, d.device_ptr(),
                                               nullptr);
    quasar::backend::device_synchronize(nullptr);
    EXPECT_EQ(bitwise_mismatches(host, download(d, g.size())), 0u) << "scale";
  }

  // -- blend: target += weight * (candidate - target) ------------------------
  {
    ScalarField candidate = quasar::numerics::make_field(g);
    for (std::size_t k = 0; k < candidate.size(); ++k) {
      candidate[k] = Real{0.7} * base[k] + Real{0.1};
    }
    const Real weight = Real{0.65};

    ScalarField host = base;
    for (std::size_t k = 0; k < host.size(); ++k) {
      host[k] += weight * (candidate[k] - host[k]);
    }

    auto d_t = upload(base);
    auto d_c = upload(candidate);
    quasar::equilibrium::launch_gs_blend(g, d_t.device_ptr(), d_c.device_ptr(),
                                         weight, nullptr);
    quasar::backend::device_synchronize(nullptr);
    EXPECT_EQ(bitwise_mismatches(host, download(d_t, g.size())), 0u) << "blend";
  }

  // -- restore_boundary: boundary from source, interior untouched ------------
  {
    ScalarField source = quasar::numerics::make_field(g);
    for (std::size_t k = 0; k < source.size(); ++k) source[k] = Real{7.25} + Real(k);

    ScalarField host = base;
    for (int j = 0; j < g.nz; ++j) {
      for (int i = 0; i < g.nr; ++i) {
        if (g.on_boundary(i, j)) host[g.index(i, j)] = source[g.index(i, j)];
      }
    }

    auto d_s = upload(source);
    auto d_t = upload(base);
    quasar::equilibrium::launch_gs_restore_boundary(g, d_s.device_ptr(),
                                                    d_t.device_ptr(), nullptr);
    quasar::backend::device_synchronize(nullptr);
    EXPECT_EQ(bitwise_mismatches(host, download(d_t, g.size())), 0u)
        << "restore_boundary";
  }
}

}  // namespace
