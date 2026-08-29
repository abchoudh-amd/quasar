// Derived equilibrium quantities: field components, flux-surface tracing,
// safety factor, and shaping.
//
// Wherever possible these are checked against an ANALYTIC field rather than
// against the solver's own output, so a bug in the diagnostics cannot be masked
// by a matching bug in the solve. The exception is the end-to-end test, which
// deliberately runs the real pipeline and checks physical plausibility
// (monotonic q, closed surfaces, geometry consistent with the axis).

#include "quasar/physics/equilibrium/flux_surfaces.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

namespace {

using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;
using namespace quasar::equilibrium;

EllipticGrid diag_grid(int n = 65) {
  return EllipticGrid{n, n, Real{0.4}, Real{2.0}, Real{-0.8}, Real{0.8}};
}

// A concentric-circular test equilibrium: psi = -((r-r0)^2 + (z-z0)^2).
// Negated so psi is maximal on axis, matching the solver's sign convention.
struct CircularTest {
  Real r0{Real{1.2}};
  Real z0{Real{0.0}};
  Real psi(Real r, Real z) const {
    const Real dr = r - r0;
    const Real dz = z - z0;
    return -(dr * dr + dz * dz);
  }
};

CriticalPointSet circular_criticals(const CircularTest& c, Real psi_edge) {
  CriticalPointSet cps;
  cps.axis.valid = true;
  cps.axis.kind = CriticalKind::o_point;
  cps.axis.r = c.r0;
  cps.axis.z = c.z0;
  cps.axis.psi = Real{0};
  cps.psi_axis = Real{0};
  cps.psi_boundary = psi_edge;
  cps.has_closed_surface = true;
  return cps;
}

ScalarField circular_field(const EllipticGrid& g, const CircularTest& c) {
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) psi[g.index(i, j)] = c.psi(g.r(i), g.z(j));
  }
  return psi;
}

}  // namespace

TEST(MagneticField, MatchesTheAnalyticCurlOfPsi) {
  // B_r = -(1/r) dpsi/dz, B_z = (1/r) dpsi/dr. For the circular test field
  // dpsi/dr = -2(r-r0) and dpsi/dz = -2(z-z0), both known exactly.
  const EllipticGrid g = diag_grid();
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-0.25});

  const auto b = compute_field(g, psi, cps, [](Real) { return Real{0}; });

  for (int j = 4; j < g.nz - 4; j += 7) {
    for (int i = 4; i < g.nr - 4; i += 7) {
      const Real r = g.r(i);
      const Real z = g.z(j);
      const Real want_br = Real{2} * (z - c.z0) / r;
      const Real want_bz = Real{-2} * (r - c.r0) / r;
      EXPECT_NEAR(b.b_r[g.index(i, j)], want_br, 1e-9) << "i=" << i << " j=" << j;
      EXPECT_NEAR(b.b_z[g.index(i, j)], want_bz, 1e-9) << "i=" << i << " j=" << j;
    }
  }
}

TEST(MagneticField, ToroidalComponentFollowsFOverR) {
  const EllipticGrid g = diag_grid();
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-0.25});
  const Real f_const = Real{3.5};

  const auto b = compute_field(g, psi, cps, [&](Real) { return f_const; });
  for (int j = 2; j < g.nz - 2; j += 9) {
    for (int i = 2; i < g.nr - 2; i += 9) {
      EXPECT_NEAR(b.b_phi[g.index(i, j)], f_const / g.r(i), 1e-12);
    }
  }
}

TEST(FluxSurfaces, TraceRecoversAKnownCircle) {
  // psi_N = 0.25 of psi_edge = -0.25 corresponds to psi = -0.0625, i.e. a
  // circle of radius 0.25 about the axis. Geometry is exactly known.
  const EllipticGrid g = diag_grid(129);
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-0.25});

  const FluxSurface s = trace_surface(g, psi, cps, Real{0.25});
  ASSERT_TRUE(s.closed);
  ASSERT_GE(s.r.size(), 32u);
  for (std::size_t k = 0; k < s.r.size(); ++k) {
    const Real dr = s.r[k] - c.r0;
    const Real dz = s.z[k] - c.z0;
    EXPECT_NEAR(std::sqrt(dr * dr + dz * dz), Real{0.25}, 2e-3)
        << "contour point " << k << " off the known circle";
  }
}

TEST(FluxSurfaces, GeometryMatchesTheAnalyticCircleAndPappusVolume) {
  const EllipticGrid g = diag_grid(129);
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-0.25});

  FluxSurface s = trace_surface(g, psi, cps, Real{0.25});
  compute_surface_geometry(s);

  const Real a = Real{0.25};
  const Real pi = Real{3.14159265358979323846};
  EXPECT_NEAR(s.area, pi * a * a, 1e-3);
  // Pappus: a torus of minor radius a at major radius r0.
  EXPECT_NEAR(s.volume, Real{2} * pi * c.r0 * pi * a * a, 1e-2);
}

TEST(FluxSurfaces, ShapeOfACircleIsUnitElongationAndZeroTriangularity) {
  const EllipticGrid g = diag_grid(129);
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-0.25});

  FluxSurface s = trace_surface(g, psi, cps, Real{0.25});
  const SurfaceShape sh = compute_shape(s);
  EXPECT_NEAR(sh.r_major, c.r0, 5e-3);
  EXPECT_NEAR(sh.r_minor, Real{0.25}, 5e-3);
  EXPECT_NEAR(sh.elongation, Real{1}, 1e-2);
  EXPECT_NEAR(sh.triangularity, Real{0}, 1e-2);
}

TEST(FluxSurfaces, DetectsAnOpenSurface) {
  // A surface whose contour leaves the domain must be reported open rather than
  // silently returning a partial contour that downstream geometry would treat
  // as closed.
  const EllipticGrid g = EllipticGrid{65, 65, Real{1.0}, Real{1.4},
                                      Real{-0.2}, Real{0.2}};
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  CriticalPointSet cps = circular_criticals(c, Real{-4.0});
  const FluxSurface s = trace_surface(g, psi, cps, Real{0.9});
  EXPECT_FALSE(s.closed);
}

TEST(SafetyFactor, MatchesTheClosedFormValueForACircularSurface) {
  // For psi = -((r-r0)^2 + (z-z0)^2) the poloidal field is B_pol = 2a/r on the
  // circle of radius a, and B_phi = F/r. The contour integral is then
  //
  //   q = (1/2pi) * closed_int [ (F/r) / (r * 2a/r) ] dl
  //     = (1/2pi) * (F/2) * closed_int dtheta / (r0 + a cos theta)
  //     = F / (2 sqrt(r0^2 - a^2))
  //
  // using the standard integral closed_int dtheta/(A + B cos theta)
  // = 2pi/sqrt(A^2 - B^2). Note the 1/r inside the integrand: dropping it gives
  // the plausible-looking but wrong F/(2a), which differs by nearly 5x here.
  const EllipticGrid g = diag_grid(129);
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-0.25});
  const Real f_const = Real{3.0};
  const auto b = compute_field(g, psi, cps, [&](Real) { return f_const; });

  FluxSurface s = trace_surface(g, psi, cps, Real{0.25});
  const Real q = compute_q(g, b, s);
  const Real a = Real{0.25};
  const Real expected =
      f_const / (Real{2} * std::sqrt(c.r0 * c.r0 - a * a));
  EXPECT_NEAR(q, expected, 1e-3)
      << "q does not match the closed-form value for this field";
}

TEST(FProfile, IntegratesToTheVacuumValueAtTheBoundary) {
  const PolynomialProfile prof;
  const Real f_vac = Real{4.2};
  const auto f = integrate_f_profile(prof, f_vac, Real{1}, Real{0}, Real{1});
  ASSERT_FALSE(f.empty());
  // Index 0 is psi_N = 0 (axis), last index is psi_N = 1 (boundary).
  EXPECT_NEAR(f.back(), f_vac, 1e-12);
  // F must remain real and finite everywhere.
  for (const Real v : f) EXPECT_TRUE(std::isfinite(v));
}

TEST(FProfile, PreservesTheNegativeVacuumFieldBranch) {
  const PolynomialProfile prof;
  const Real magnitude = Real{4.2};
  const auto positive =
      integrate_f_profile(prof, magnitude, Real{1}, Real{0}, Real{1});
  const auto negative =
      integrate_f_profile(prof, -magnitude, Real{1}, Real{0}, Real{1});

  ASSERT_EQ(negative.size(), positive.size());
  EXPECT_DOUBLE_EQ(negative.back(), -magnitude);
  for (std::size_t k = 0; k < negative.size(); ++k) {
    EXPECT_DOUBLE_EQ(negative[k], -positive[k]) << "sample " << k;
    EXPECT_TRUE(std::signbit(negative[k])) << "sample " << k;
  }
}

TEST(FProfile, AppliesTheProfileNormalizationScale) {
  const PolynomialProfile prof;
  const Real f_vac = Real{20};
  const auto unit = integrate_f_profile(prof, f_vac, Real{1}, Real{0}, Real{1});
  const auto scaled =
      integrate_f_profile(prof, f_vac, Real{1}, Real{0}, Real{2.5});
  ASSERT_EQ(unit.size(), scaled.size());
  for (std::size_t k = 0; k < unit.size(); k += 31) {
    const Real delta_unit = unit[k] * unit[k] - f_vac * f_vac;
    const Real delta_scaled = scaled[k] * scaled[k] - f_vac * f_vac;
    EXPECT_NEAR(delta_scaled, Real{2.5} * delta_unit,
                Real{1e-11} * std::max(Real{1}, std::abs(delta_scaled)));
  }
}

TEST(Diagnostics, OpenSurfacesAreExcludedFromAggregates) {
  const EllipticGrid g{65, 65, Real{1.0}, Real{1.4}, Real{-0.2}, Real{0.2}};
  const CircularTest c;
  const ScalarField psi = circular_field(g, c);
  const auto cps = circular_criticals(c, Real{-4.0});
  const auto d = compute_diagnostics(g, psi, cps, [](Real) { return Real{3}; }, 16);

  EXPECT_GT(d.n_open_surfaces, 0);
  EXPECT_EQ(d.surfaces.size(),
            d.q_profile.size() + static_cast<std::size_t>(d.n_open_surfaces));
  EXPECT_EQ(d.psi_n_grid.size(), d.q_profile.size());
  for (const auto& s : d.surfaces) {
    if (s.closed) continue;
    EXPECT_EQ(s.area, Real{0});
    EXPECT_EQ(s.volume, Real{0});
    EXPECT_EQ(s.q, Real{0});
    EXPECT_EQ(compute_shape(s).r_minor, Real{0});
  }
  if (!d.psi_n_grid.empty()) {
    EXPECT_EQ(d.psi_n_boundary, d.psi_n_grid.back());
  }
}

TEST(Diagnostics, EndToEndOnASolvedEquilibriumIsPhysicallySensible) {
  GsConfig cfg;
  cfg.grid = EllipticGrid{33, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  cfg.coils = {
      {Real{2.4}, Real{0.9},  Real{-3.0e5}},
      {Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      {Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1.0e6};
  cfg.max_iterations = 400;
  cfg.tolerance = Real{1e-9};

  auto prof = std::make_shared<PolynomialProfile>();
  GsSolver solver{cfg, prof};
  const GsResult r = solver.solve();
  ASSERT_EQ(r.status, GsStatus::converged);

  const auto ftab = integrate_f_profile(*prof, Real{5}, r.critical.psi_axis,
                                        r.critical.psi_boundary, r.profile_scale);
  const auto f_of = [&](Real pn) {
    int k = static_cast<int>(pn * static_cast<Real>(ftab.size() - 1) + Real{0.5});
    k = std::min<int>(std::max(k, 0), static_cast<int>(ftab.size()) - 1);
    return ftab[static_cast<std::size_t>(k)];
  };

  const auto d = compute_diagnostics(cfg.grid, r.psi, r.critical, f_of, 16);

  // Every traced surface must close for a confined equilibrium.
  EXPECT_EQ(d.n_open_surfaces, 0);
  ASSERT_FALSE(d.psi_n_grid.empty());
  EXPECT_EQ(d.psi_n_boundary, d.psi_n_grid.back());
  for (const auto& s : d.surfaces) EXPECT_TRUE(s.closed);

  // q must be positive and rise from axis to edge (normal magnetic shear).
  EXPECT_GT(d.q_axis, Real{0});
  EXPECT_GT(d.q_95, d.q_axis) << "q profile has the wrong shear direction";
  for (std::size_t k = 1; k < d.q_profile.size(); ++k) {
    EXPECT_GE(d.q_profile[k], d.q_profile[k - 1] - Real{1e-6})
        << "q not monotonic at index " << k;
  }

  // Geometry must be consistent with the located axis and a positive volume.
  EXPECT_NEAR(d.boundary_shape.r_major, r.critical.axis.r, Real{0.15});
  EXPECT_GT(d.boundary_shape.r_minor, Real{0});
  EXPECT_GT(d.total_volume, Real{0});
  EXPECT_GT(d.boundary_shape.elongation, Real{0.5});
  EXPECT_LT(d.boundary_shape.elongation, Real{3});
}

TEST(Diagnostics, ReturnsEmptyProfilesWithoutAValidAxis) {
  const EllipticGrid g = diag_grid(33);
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) psi[g.index(i, j)] = g.r(i);
  }
  CriticalPointSet cps;  // axis.valid == false
  const auto d = compute_diagnostics(g, psi, cps, [](Real) { return Real{1}; });
  EXPECT_TRUE(d.surfaces.empty());
  EXPECT_TRUE(d.q_profile.empty());
}
