// Grad-Shafranov physics verification: Solov'ev exact solution, profiles, and
// critical-point location.
//
// These tests check the PHYSICS layer -- that Delta* is being fed the right
// r^2-weighted source, that the sign conventions are consistent, and that the
// axis/X-point search recovers known geometry. Scheme ORDER is established by
// the manufactured-solution tests in tests/unit/numerics; see the caveat on
// SolovevSolution about why a Solov'ev order study cannot do that job.

#include "quasar/numerics/defect_correction.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/geometric_multigrid.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;
using quasar::equilibrium::SolovevSolution;

EllipticGrid make_grid(int n) {
  return EllipticGrid{n, n, Real{0.4}, Real{1.6}, Real{-0.6}, Real{0.6}};
}

}  // namespace

TEST(Solovev, SatisfiesTheGradShafranovEquationAnalytically) {
  // Independent check of the closed form: differentiate psi numerically with a
  // fine central difference and confirm Delta* psi equals the stated source.
  // This catches an algebra slip in the analytic Delta* without relying on the
  // solver machinery under test.
  const SolovevSolution sol{Real{1.3}, Real{0.7}, Real{0.2}, Real{-0.4}};
  const Real h = 1e-4;
  for (const Real r : {Real{0.6}, Real{1.0}, Real{1.4}}) {
    for (const Real z : {Real{-0.3}, Real{0.0}, Real{0.45}}) {
      const Real p_rr = (sol.psi(r + h, z) - Real{2} * sol.psi(r, z)
                       + sol.psi(r - h, z)) / (h * h);
      const Real p_r  = (sol.psi(r + h, z) - sol.psi(r - h, z)) / (Real{2} * h);
      const Real p_zz = (sol.psi(r, z + h) - Real{2} * sol.psi(r, z)
                       + sol.psi(r, z - h)) / (h * h);
      const Real numeric = p_rr - p_r / r + p_zz;
      EXPECT_NEAR(numeric, sol.delta_star(r, z), 1e-4)
          << "at r=" << r << " z=" << z;
    }
  }
}

TEST(Solovev, IsRecoveredByTheDefectCorrectedSolver) {
  // End-to-end physics check: seed exact Dirichlet data and the exact source,
  // solve, and compare. Solov'ev is polynomial so a sixth-order scheme should
  // reproduce it essentially exactly -- that is the point of this test, and
  // also the reason it cannot measure convergence order.
  const SolovevSolution sol{Real{1.3}, Real{0.7}, Real{0.2}, Real{-0.4}};
  const EllipticGrid g = make_grid(65);

  ScalarField x = quasar::numerics::make_field(g);
  ScalarField b = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (g.on_boundary(i, j)) {
        x[g.index(i, j)] = sol.psi(g.r(i), g.z(j));
      } else {
        b[g.index(i, j)] = sol.delta_star(g.r(i), g.z(j));
      }
    }
  }

  quasar::numerics::GsMultigrid mg{g};
  const auto rep = quasar::numerics::solve_defect_corrected(g, x, b, mg);
  ASSERT_TRUE(rep.converged);

  Real err = Real{0};
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      err = std::max(err, std::abs(x[g.index(i, j)] - sol.psi(g.r(i), g.z(j))));
    }
  }
  // Scale-relative: psi is O(1) here.
  EXPECT_LT(err, 1e-10) << "Solov'ev not reproduced; err=" << err;
}

TEST(Solovev, SourceCarriesTheRSquaredWeighting) {
  // The defining structural feature of Grad-Shafranov versus Poisson: the
  // pressure term enters as r^2 p'. If the r^2 were dropped, delta_star would be
  // constant in r and this test fails.
  const SolovevSolution sol{Real{2}, Real{0}, Real{0}, Real{0}};
  EXPECT_DOUBLE_EQ(sol.delta_star(Real{1}, Real{0}), Real{2});
  EXPECT_DOUBLE_EQ(sol.delta_star(Real{2}, Real{0}), Real{8});
  EXPECT_DOUBLE_EQ(sol.delta_star(Real{3}, Real{0}), Real{18});
}

TEST(PolynomialProfile, DefaultVanishesAtThePlasmaBoundary) {
  // p' and FF' must go to zero at psi_N = 1 so no current is driven outside the
  // plasma. The default p' = 1 - psi_N does this.
  const quasar::equilibrium::PolynomialProfile prof;
  EXPECT_DOUBLE_EQ(prof.dp_dpsi(Real{1}), Real{0});
  EXPECT_DOUBLE_EQ(prof.ff_prime(Real{1}), Real{0});
  EXPECT_DOUBLE_EQ(prof.dp_dpsi(Real{0}), Real{1});
  EXPECT_GT(prof.dp_dpsi(Real{0.5}), Real{0});
}

TEST(PolynomialProfile, DerivativesAreConsistentWithFiniteDifferences) {
  // The Newton Jacobian depends on the second derivatives; an inconsistency
  // here degrades Newton to a slow, wrong-direction iteration.
  const quasar::equilibrium::PolynomialProfile prof{
      {Real{1}, Real{-0.5}, Real{0.25}}, {Real{0.8}, Real{-1.1}, Real{0.3}}};
  const Real h = 1e-6;
  for (const Real x : {Real{0.1}, Real{0.5}, Real{0.9}}) {
    const Real fd_p = (prof.dp_dpsi(x + h) - prof.dp_dpsi(x - h)) / (Real{2} * h);
    const Real fd_f = (prof.ff_prime(x + h) - prof.ff_prime(x - h)) / (Real{2} * h);
    EXPECT_NEAR(prof.d2p_dpsi2(x), fd_p, 1e-6) << "at psi_n=" << x;
    EXPECT_NEAR(prof.ff_prime_prime(x), fd_f, 1e-6) << "at psi_n=" << x;
  }
}

TEST(CriticalPoints, LocatesAKnownOPointToHighAccuracy) {
  // Analytic test field with a single minimum at a known, deliberately
  // off-grid location. Accuracy well below one cell proves the Newton refinement
  // is doing real work rather than snapping to the nearest node.
  const EllipticGrid g = make_grid(65);
  const Real r0 = Real{1.037};
  const Real z0 = Real{0.111};
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real dr = g.r(i) - r0;
      const Real dz = g.z(j) - z0;
      psi[g.index(i, j)] = dr * dr + Real{1.7} * dz * dz;
    }
  }

  const auto cps = quasar::equilibrium::find_critical_points(g, psi);
  ASSERT_TRUE(cps.axis.valid) << "axis not found";
  EXPECT_EQ(cps.axis.kind, quasar::equilibrium::CriticalKind::o_point);
  EXPECT_NEAR(cps.axis.r, r0, 1e-6);
  EXPECT_NEAR(cps.axis.z, z0, 1e-6);
  EXPECT_LT(std::abs(cps.axis.r - r0), g.dr() / Real{100})
      << "axis located no better than a grid cell";
}

TEST(CriticalPoints, ClassifiesASaddleAsAnXPoint) {
  // psi = (r-r0)^2 - (z-z0)^2 is a pure saddle: Hessian determinant < 0.
  const EllipticGrid g = make_grid(65);
  const Real r0 = Real{1.0};
  const Real z0 = Real{0.0};
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real dr = g.r(i) - r0;
      const Real dz = g.z(j) - z0;
      psi[g.index(i, j)] = dr * dr - dz * dz;
    }
  }
  const auto cps = quasar::equilibrium::find_critical_points(g, psi);
  ASSERT_FALSE(cps.x_points.empty()) << "saddle not classified as an X-point";
  EXPECT_NEAR(cps.x_points.front().r, r0, 1e-6);
  EXPECT_NEAR(cps.x_points.front().z, z0, 1e-6);
}

TEST(CriticalPoints, ReportsNoAxisForAMonotonicField) {
  // A field with no interior extremum must not invent one. This is the
  // "no confined plasma" case that the solver maps to a diagnosed failure.
  const EllipticGrid g = make_grid(33);
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      psi[g.index(i, j)] = Real{2} * g.r(i) + Real{0.5} * g.z(j);
    }
  }
  const auto cps = quasar::equilibrium::find_critical_points(g, psi);
  EXPECT_FALSE(cps.axis.valid);
}

TEST(NormalizedFlux, IsZeroOnAxisAndOneAtTheBoundary) {
  EXPECT_DOUBLE_EQ(
      quasar::equilibrium::normalized_flux(Real{-1}, Real{-1}, Real{0}), Real{0});
  EXPECT_DOUBLE_EQ(
      quasar::equilibrium::normalized_flux(Real{0}, Real{-1}, Real{0}), Real{1});
  EXPECT_DOUBLE_EQ(
      quasar::equilibrium::normalized_flux(Real{-0.5}, Real{-1}, Real{0}),
      Real{0.5});
  // Outside the plasma, psi_N clamps to 1 so the default profiles drive no
  // current there.
  EXPECT_DOUBLE_EQ(
      quasar::equilibrium::normalized_flux(Real{5}, Real{-1}, Real{0}), Real{1});
  // Degenerate normalization must not produce NaN.
  EXPECT_DOUBLE_EQ(
      quasar::equilibrium::normalized_flux(Real{1}, Real{2}, Real{2}), Real{1});
}
