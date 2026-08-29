// Host-vs-device equivalence for the derived diagnostics: F profile, magnetic
// field, traced flux surfaces, safety factor, and the aggregate shape bundle.
//
// The F profile and the magnetic field are bitwise equalities: pure elementwise
// arithmetic over already-bit-exact derivative fields.
//
// The traced surfaces are NOT, and the reason is outside the port's control.
// Ray directions come from cos/sin of the ray angle, and host libm and device
// libm do not agree bit-for-bit on every argument -- measured here, 7 of the
// 128 ray angles differ by one ulp. That perturbs those rays' marching paths
// and their crossing points at the 1e-16 level. It is a library difference, not
// an algorithmic one, and no amount of care in the kernel removes it.
//
// So the discrete outcomes -- whether a ray hit, how many points a surface has,
// whether it is closed -- are asserted exactly, because those are decisions
// rather than values. The geometry is asserted to a tight relative tolerance.
//
// The surfaces are traced from a real solved equilibrium rather than an
// analytic stand-in. A synthetic psi would trace clean circles and would not
// exercise the case that actually matters -- rays that find no crossing, which
// mark a surface open and change the point count the polygon integrals run
// over.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/flux_surfaces.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::GsConfig;
using quasar::equilibrium::GsResult;
using quasar::equilibrium::GsSolver;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

constexpr int kFSamples = 257;
constexpr int kNSurfaces = 32;
constexpr int kNTheta = 128;
constexpr Real kFVacuum = Real{5};

std::size_t bitwise_mismatches(const std::vector<Real>& a,
                               const std::vector<Real>& b) {
  std::size_t n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::memcmp(&a[k], &b[k], sizeof(Real)) != 0) ++n;
  }
  return n;
}

Real max_abs_difference(const std::vector<Real>& a, const std::vector<Real>& b) {
  Real m = Real{0};
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

// A short solve is enough: the diagnostics only need a psi with a genuine
// interior extremum and closed surfaces around it, not a converged one.
struct Fixture {
  EllipticGrid grid{33, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  GsResult result{};
  PolynomialProfile profile{std::vector<Real>{Real{1}, Real{-1}},
                            std::vector<Real>{Real{1}, Real{-1}}};

  Fixture() {
    GsConfig cfg;
    cfg.grid = grid;
    cfg.coils = {
        CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
        CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
        CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
    };
    cfg.plasma_current = Real{1e6};
    cfg.max_iterations = 40;
    result = GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}.solve();
  }
};

std::vector<Real> download(const DeviceBuffer<Real>& d, std::size_t n) {
  std::vector<Real> h(n, Real{0});
  d.copy_to_host(h.data(), n);
  return h;
}

TEST(GsContourDevice, FProfileMatchesHostBitExactly) {
  const Fixture fx;
  ASSERT_TRUE(fx.result.critical.axis.valid);

  const std::vector<Real> host = quasar::equilibrium::integrate_f_profile(
      fx.profile, kFVacuum, fx.result.critical.psi_axis,
      fx.result.critical.psi_boundary, fx.result.profile_scale, kFSamples);

  DeviceBuffer<Real> d_f{static_cast<std::size_t>(kFSamples)};
  quasar::equilibrium::launch_gs_integrate_f_profile(
      quasar::equilibrium::to_coefficients(fx.profile), kFVacuum,
      fx.result.critical.psi_axis, fx.result.critical.psi_boundary,
      fx.result.profile_scale, kFSamples, d_f.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  const std::vector<Real> dev = download(d_f, kFSamples);
  ASSERT_EQ(host.size(), dev.size());
  Real peak = Real{0};
  for (const Real v : dev) peak = std::max(peak, std::abs(v));
  ASSERT_GT(peak, Real{1}) << "F profile is trivially small";

  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

TEST(GsContourDevice, MagneticFieldMatchesHostBitExactly) {
  const Fixture fx;
  ASSERT_TRUE(fx.result.critical.axis.valid);
  const EllipticGrid& g = fx.grid;

  const std::vector<Real> f_table = quasar::equilibrium::integrate_f_profile(
      fx.profile, kFVacuum, fx.result.critical.psi_axis,
      fx.result.critical.psi_boundary, fx.result.profile_scale, kFSamples);
  const auto f_of = [&](Real pn) {
    const Real clamped = std::min(std::max(pn, Real{0}), Real{1});
    const auto k = static_cast<std::size_t>(
        clamped * static_cast<Real>(f_table.size() - 1) + Real{0.5});
    return f_table[k];
  };
  const auto host = quasar::equilibrium::compute_field(g, fx.result.psi,
                                                       fx.result.critical, f_of);

  DeviceBuffer<Real> d_psi{fx.result.psi.size()};
  d_psi.copy_from_host(fx.result.psi.data(), fx.result.psi.size());
  DeviceBuffer<Real> d_f{f_table.size()};
  d_f.copy_from_host(f_table.data(), f_table.size());

  quasar::equilibrium::GsOperatorScratch op{g};
  quasar::equilibrium::GsDerivativeFields deriv{g};
  quasar::equilibrium::launch_gs_compute_derivatives(g, d_psi.device_ptr(),
                                                     deriv, op, nullptr);

  quasar::equilibrium::GsMagneticField dev{g};
  quasar::equilibrium::launch_gs_compute_field(
      g, d_psi.device_ptr(), deriv, d_f.device_ptr(), kFSamples,
      fx.result.critical.psi_axis, fx.result.critical.psi_boundary, dev,
      nullptr);
  quasar::backend::device_synchronize(nullptr);

  EXPECT_EQ(bitwise_mismatches(host.b_r, download(dev.b_r, g.size())), 0u) << "b_r";
  EXPECT_EQ(bitwise_mismatches(host.b_z, download(dev.b_z, g.size())), 0u) << "b_z";
  EXPECT_EQ(bitwise_mismatches(host.b_phi, download(dev.b_phi, g.size())), 0u)
      << "b_phi";
  EXPECT_EQ(bitwise_mismatches(host.b_poloidal,
                               download(dev.b_poloidal, g.size())), 0u)
      << "b_poloidal";
}

TEST(GsContourDevice, TracedSurfacesAndDiagnosticsMatchHost) {
  const Fixture fx;
  ASSERT_TRUE(fx.result.critical.axis.valid);
  const EllipticGrid& g = fx.grid;

  const std::vector<Real> f_table = quasar::equilibrium::integrate_f_profile(
      fx.profile, kFVacuum, fx.result.critical.psi_axis,
      fx.result.critical.psi_boundary, fx.result.profile_scale, kFSamples);
  const auto f_of = [&](Real pn) {
    const Real clamped = std::min(std::max(pn, Real{0}), Real{1});
    const auto k = static_cast<std::size_t>(
        clamped * static_cast<Real>(f_table.size() - 1) + Real{0.5});
    return f_table[k];
  };
  const auto host = quasar::equilibrium::compute_diagnostics(
      g, fx.result.psi, fx.result.critical, f_of, kNSurfaces);

  // -- device ----------------------------------------------------------------
  DeviceBuffer<Real> d_psi{fx.result.psi.size()};
  d_psi.copy_from_host(fx.result.psi.data(), fx.result.psi.size());
  DeviceBuffer<Real> d_f{f_table.size()};
  d_f.copy_from_host(f_table.data(), f_table.size());

  quasar::equilibrium::GsOperatorScratch op{g};
  quasar::equilibrium::GsDerivativeFields deriv{g};
  quasar::equilibrium::launch_gs_compute_derivatives(g, d_psi.device_ptr(),
                                                     deriv, op, nullptr);
  quasar::equilibrium::GsMagneticField field{g};
  quasar::equilibrium::launch_gs_compute_field(
      g, d_psi.device_ptr(), deriv, d_f.device_ptr(), kFSamples,
      fx.result.critical.psi_axis, fx.result.critical.psi_boundary, field,
      nullptr);

  quasar::equilibrium::GsFluxSurfaces surfaces{kNSurfaces, kNTheta};
  quasar::equilibrium::launch_gs_trace_surfaces(
      g, d_psi.device_ptr(), field, fx.result.critical.axis.r,
      fx.result.critical.axis.z, fx.result.critical.psi_axis,
      fx.result.critical.psi_boundary, surfaces, nullptr);

  DeviceBuffer<quasar::equilibrium::GsDiagnostics> d_diag{1};
  quasar::equilibrium::launch_gs_reduce_diagnostics(surfaces,
                                                    d_diag.device_ptr(),
                                                    nullptr);
  const auto dev = quasar::equilibrium::copy_diagnostics_to_host(d_diag, nullptr);
  quasar::backend::device_synchronize(nullptr);

  // -- per-surface -----------------------------------------------------------
  std::vector<int> closed(kNSurfaces), count(kNSurfaces);
  surfaces.closed.copy_to_host(closed.data(), closed.size());
  surfaces.count.copy_to_host(count.data(), count.size());
  const std::vector<Real> q = download(surfaces.q, kNSurfaces);
  const std::vector<Real> volume = download(surfaces.volume, kNSurfaces);
  const std::vector<Real> pts_r =
      download(surfaces.r, static_cast<std::size_t>(kNSurfaces) * kNTheta);
  const std::vector<Real> pts_z =
      download(surfaces.z, static_cast<std::size_t>(kNSurfaces) * kNTheta);

  ASSERT_EQ(host.surfaces.size(), static_cast<std::size_t>(kNSurfaces));

  // Measured divergence is ~1e-16 relative, from the libm difference described
  // at the top. 1e-13 is four orders of margin on that and still far too tight
  // to absorb a real geometric error.
  constexpr Real kRelTol = Real{1e-13};
  // z passes through zero on the midplane, where a relative bound is
  // meaningless; the floor is scaled to the domain height rather than to z.
  constexpr Real kAbsFloor = Real{1e-13};

  int n_closed = 0;
  for (int s = 0; s < kNSurfaces; ++s) {
    const auto& hs = host.surfaces[static_cast<std::size_t>(s)];
    EXPECT_EQ(hs.closed, closed[s] != 0) << "surface " << s << ": closed flag";
    EXPECT_EQ(hs.r.size(), static_cast<std::size_t>(count[s]))
        << "surface " << s << ": point count";
    EXPECT_NEAR(q[s], hs.q, std::abs(hs.q) * kRelTol)
        << "surface " << s << ": q";
    EXPECT_NEAR(volume[s], hs.volume, std::abs(hs.volume) * kRelTol)
        << "surface " << s << ": volume";

    const std::size_t base = static_cast<std::size_t>(s) * kNTheta;
    for (std::size_t k = 0; k < hs.r.size(); ++k) {
      ASSERT_NEAR(pts_r[base + k], hs.r[k], std::abs(hs.r[k]) * kRelTol)
          << "surface " << s << " point " << k << " (r)";
      ASSERT_NEAR(pts_z[base + k], hs.z[k],
                  std::abs(hs.z[k]) * kRelTol + kAbsFloor)
          << "surface " << s << " point " << k << " (z)";
    }
    if (hs.closed) ++n_closed;
  }
  ASSERT_GT(n_closed, 2) << "too few closed surfaces to exercise the aggregates";

  // -- aggregates ------------------------------------------------------------
  // psi_n_boundary and n_open_surfaces are discrete selections, so they must
  // match exactly; everything else is geometry.
  EXPECT_EQ(host.psi_n_boundary, dev.psi_n_boundary);
  EXPECT_EQ(host.n_open_surfaces, dev.n_open_surfaces);

  EXPECT_NEAR(dev.q_axis, host.q_axis, std::abs(host.q_axis) * kRelTol);
  EXPECT_NEAR(dev.q_95, host.q_95, std::abs(host.q_95) * kRelTol);
  EXPECT_NEAR(dev.total_volume, host.total_volume,
              std::abs(host.total_volume) * kRelTol);
  EXPECT_NEAR(dev.r_major, host.boundary_shape.r_major,
              std::abs(host.boundary_shape.r_major) * kRelTol);
  EXPECT_NEAR(dev.r_minor, host.boundary_shape.r_minor,
              std::abs(host.boundary_shape.r_minor) * kRelTol);
  EXPECT_NEAR(dev.elongation, host.boundary_shape.elongation,
              std::abs(host.boundary_shape.elongation) * kRelTol);
  EXPECT_NEAR(dev.triangularity, host.boundary_shape.triangularity,
              std::abs(host.boundary_shape.triangularity) * kRelTol + kAbsFloor);

  ASSERT_NE(dev.q_axis, Real{0}) << "q_axis is trivially zero";
  ASSERT_GT(dev.total_volume, Real{0}) << "volume is trivially zero";
}

}  // namespace
