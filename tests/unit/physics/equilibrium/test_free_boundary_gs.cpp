// Free-boundary Grad-Shafranov: Green's-function kernel, coil field, and the
// nonlinear outer loop with its failure contract.
//
// The kernel tests are exact-value checks against independently known physics
// (published elliptic-integral values, the analytic on-axis field of a circular
// loop, and the vacuum property Delta* G = 0). The solver tests cover both the
// success path and the diagnosed failure modes, because with an optimizer as a
// downstream consumer the failure contract is as much a deliverable as
// convergence.

#include "quasar/physics/equilibrium/gs_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;
using namespace quasar::equilibrium;

// A coil set and grid known to admit a confined equilibrium. The coil currents
// are deliberately matched in scale to the plasma current: an equilibrium needs
// the vacuum field's flux variation to be comparable to mu0*I_p, and a coil set
// an order of magnitude too strong simply has no confined solution.
// n=33 by default: the solve is a few-hundred-iteration nonlinear loop, and the
// equilibrium it finds is resolution-independent to four digits (axis at
// r=1.2222 on 33^2 vs r=1.2228 on 65^2), so the coarse grid tests the same
// physics roughly 7x faster. One test below pins that resolution independence.
GsConfig converging_case(int n = 33) {
  GsConfig cfg;
  cfg.grid = EllipticGrid{n, n, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  cfg.coils = {
      {Real{2.4}, Real{0.9},  Real{-3.0e5}},
      {Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      {Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1.0e6};
  cfg.max_iterations = 400;
  cfg.tolerance = Real{1e-9};
  return cfg;
}

}  // namespace

// -- Green's function kernel --------------------------------------------------

TEST(EllipticIntegrals, MatchPublishedValues) {
  struct Ref { Real m, k, e; };
  const Ref refs[] = {
      {Real{0.0},  Real{1.5707963267948966}, Real{1.5707963267948966}},
      {Real{0.1},  Real{1.6124413487202193}, Real{1.5307576368977633}},
      {Real{0.5},  Real{1.8540746773013719}, Real{1.3506438810476755}},
      {Real{0.9},  Real{2.5780921133481733}, Real{1.1047747327040733}},
      {Real{0.99}, Real{3.6956373629898746}, Real{1.0159935450252239}},
  };
  for (const auto& r : refs) {
    const auto ke = complete_elliptic_ke(r.m);
    EXPECT_NEAR(ke.k, r.k, 1e-14) << "K at m=" << r.m;
    EXPECT_NEAR(ke.e, r.e, 1e-13) << "E at m=" << r.m;
  }
}

TEST(EllipticIntegrals, HandleDegenerateParametersWithoutNaN) {
  // m -> 1 is the coincident-source limit where K diverges. Returning a large
  // finite value beats letting a NaN propagate silently into a residual.
  for (const Real m : {Real{-0.5}, Real{0.0}, Real{1.0}, Real{2.0}}) {
    const auto ke = complete_elliptic_ke(m);
    EXPECT_TRUE(std::isfinite(ke.k)) << "K not finite at m=" << m;
    EXPECT_TRUE(std::isfinite(ke.e)) << "E not finite at m=" << m;
  }
}

TEST(GreensFunction, IsAVacuumSolutionAwayFromTheSource) {
  // Delta* G = 0 except at the filament. This is the single most important
  // property of the kernel: it is what makes superposing it a valid way to set
  // the boundary condition.
  const Real rp = Real{1};
  const Real zp = Real{0};
  const Real h = Real{1e-4};
  for (const Real r : {Real{0.5}, Real{0.7}, Real{1.5}, Real{2.0}}) {
    for (const Real z : {Real{0.0}, Real{0.5}}) {
      const Real g_rr = (greens_function(r + h, z, rp, zp)
                       - Real{2} * greens_function(r, z, rp, zp)
                       + greens_function(r - h, z, rp, zp)) / (h * h);
      const Real g_r = (greens_function(r + h, z, rp, zp)
                      - greens_function(r - h, z, rp, zp)) / (Real{2} * h);
      const Real g_zz = (greens_function(r, z + h, rp, zp)
                       - Real{2} * greens_function(r, z, rp, zp)
                       + greens_function(r, z - h, rp, zp)) / (h * h);
      EXPECT_NEAR(g_rr - g_r / r + g_zz, Real{0}, 1e-10)
          << "Delta* G nonzero at r=" << r << " z=" << z;
    }
  }
}

TEST(GreensFunction, ReproducesTheAnalyticOnAxisLoopField) {
  // B_z on the axis of a circular loop is mu0 I R^2 / (2 (R^2+z^2)^{3/2}).
  // Near the axis psi ~ B_z r^2 / 2, so 2 psi / r^2 recovers B_z.
  const Real R = Real{1};
  const Real I = Real{1};
  const Real r = Real{1e-3};
  for (const Real z : {Real{0.0}, Real{0.3}, Real{0.8}}) {
    const Real psi = I * greens_function(r, z, R, Real{0});
    const Real bz_num = Real{2} * psi / (r * r);
    const Real bz_exact =
        kMu0 * I * R * R / (Real{2} * std::pow(R * R + z * z, Real{1.5}));
    // Tolerance is set by the O(r^2) truncation of the near-axis expansion,
    // not by the kernel's accuracy.
    EXPECT_NEAR(bz_num / bz_exact, Real{1}, 1e-5) << "at z=" << z;
  }
}

TEST(GreensFunction, IsSymmetricUnderSourceFieldExchange) {
  // Reciprocity. A violation would mean the kernel is not a Green's function.
  EXPECT_NEAR(greens_function(Real{0.8}, Real{0.2}, Real{1.4}, Real{-0.3}),
              greens_function(Real{1.4}, Real{-0.3}, Real{0.8}, Real{0.2}),
              1e-18);
}

TEST(GreensFunction, VanishesForDegenerateGeometry) {
  EXPECT_DOUBLE_EQ(greens_function(Real{0}, Real{0}, Real{1}, Real{0}), Real{0});
  EXPECT_DOUBLE_EQ(greens_function(Real{1}, Real{0}, Real{0}, Real{0}), Real{0});
  // Coincident source and field point: guarded rather than divergent.
  EXPECT_TRUE(std::isfinite(
      greens_function(Real{1}, Real{0}, Real{1}, Real{0})));
}

TEST(CoilField, HasNoInteriorExtremum) {
  // Delta* psi = 0 admits no interior maximum or minimum, so a pure vacuum coil
  // field never has an O-point. This is exactly why the solver must seed a
  // plasma column to bootstrap -- the property is load-bearing, so it is pinned.
  const GsConfig cfg = converging_case();
  ScalarField psi;
  evaluate_coil_field(cfg.grid, cfg.coils, psi);
  const auto cps = find_critical_points(cfg.grid, psi);
  EXPECT_FALSE(cps.axis.valid)
      << "a vacuum field must not contain a magnetic axis";
}

// -- Nonlinear solver ---------------------------------------------------------

TEST(GsSolver, ConvergesToAConfinedEquilibrium) {
  const GsConfig cfg = converging_case();
  GsSolver solver{cfg, std::make_shared<PolynomialProfile>()};
  const GsResult r = solver.solve();

  ASSERT_EQ(r.status, GsStatus::converged)
      << "status=" << to_string(r.status) << " resid=" << r.residual;
  EXPECT_LE(r.residual, cfg.tolerance);
  EXPECT_TRUE(r.critical.axis.valid);

  // The axis must sit inside the domain, away from the walls.
  EXPECT_GT(r.critical.axis.r, cfg.grid.r_min + Real{0.2});
  EXPECT_LT(r.critical.axis.r, cfg.grid.r_max - Real{0.2});

  // Sign convention: with positive plasma current psi attains a MAXIMUM on the
  // magnetic axis. Getting this backwards is a subtle bug that still "converges"
  // to something, so it is asserted explicitly.
  EXPECT_GT(r.critical.psi_axis, r.critical.psi_boundary)
      << "psi should be maximal on axis for positive plasma current";
}

TEST(GsSolver, MeetsTheRequestedPlasmaCurrent) {
  // The I_p normalization is what makes the problem well-posed; without it the
  // profile amplitude is unconstrained.
  for (const Real target : {Real{5.0e5}, Real{1.0e6}}) {
    GsConfig cfg = converging_case();
    cfg.plasma_current = target;
    GsSolver solver{cfg, std::make_shared<PolynomialProfile>()};
    const GsResult r = solver.solve();
    ASSERT_EQ(r.status, GsStatus::converged) << "target=" << target;
    EXPECT_NEAR(r.achieved_current / target, Real{1}, 1e-9);
    EXPECT_TRUE(std::isfinite(r.profile_scale));
    EXPECT_GT(std::abs(r.profile_scale), Real{0});
  }
}

TEST(GsSolver, NewtonIsDisabledByDefaultAndDamagesConvergenceWhenEnabled) {
  // Pins the measured decision recorded in GsConfig::enable_newton. The
  // implemented Newton step carries only the diagonal profile Jacobian; on a
  // free-boundary problem the boundary condition is itself a dense functional
  // of psi, so the Newton direction is wrong by an O(1) amount near
  // convergence.
  //
  // If a future change completes the Jacobian (JFNK, or von Hagenow to make the
  // boundary dependence local), this test should be inverted -- that is the
  // signal the accelerator is finally worth enabling.
  GsConfig base = converging_case();
  EXPECT_FALSE(base.enable_newton) << "Newton must stay off until its Jacobian "
                                      "includes the boundary coupling";

  GsSolver picard{base, std::make_shared<PolynomialProfile>()};
  const GsResult rp = picard.solve();
  ASSERT_EQ(rp.status, GsStatus::converged);
  EXPECT_EQ(rp.newton_steps, 0);

  GsConfig with_newton = base;
  with_newton.enable_newton = true;
  GsSolver newton{with_newton, std::make_shared<PolynomialProfile>()};
  const GsResult rn = newton.solve();
  EXPECT_NE(rn.status, GsStatus::converged)
      << "Newton unexpectedly converged; if its Jacobian was completed, invert "
         "this test and re-enable the default";
}

TEST(GsSolver, FindsTheSameEquilibriumAtHigherResolution) {
  // Justifies running the rest of the solver tests on a coarse grid: the
  // converged equilibrium is a property of the physics, not the mesh.
  GsSolver coarse{converging_case(33), std::make_shared<PolynomialProfile>()};
  GsSolver fine{converging_case(65), std::make_shared<PolynomialProfile>()};
  const GsResult rc = coarse.solve();
  const GsResult rf = fine.solve();
  ASSERT_EQ(rc.status, GsStatus::converged);
  ASSERT_EQ(rf.status, GsStatus::converged);
  EXPECT_NEAR(rc.critical.axis.r, rf.critical.axis.r, 5e-3);
  EXPECT_NEAR(rc.critical.axis.z, rf.critical.axis.z, 5e-3);
}

TEST(GsSolver, ResidualHistoryIsRecordedAndDecreases) {
  const GsConfig cfg = converging_case();
  GsSolver solver{cfg, std::make_shared<PolynomialProfile>()};
  const GsResult r = solver.solve();
  ASSERT_EQ(r.status, GsStatus::converged);
  ASSERT_GE(r.residual_history.size(), 2u);
  EXPECT_LT(r.residual_history.back(), r.residual_history.front());
}

// -- Failure contract ---------------------------------------------------------

TEST(GsSolver, ReportsNoClosedSurfaceForOverwhelmingCoils) {
  // A coil set far too strong for the requested current admits no confined
  // plasma. That is a physical answer, not an error: the solver must report it
  // and return the partial state so an optimizer can score the configuration.
  GsConfig cfg = converging_case();
  for (auto& c : cfg.coils) c.current *= Real{20};
  cfg.max_iterations = 40;
  GsSolver solver{cfg, std::make_shared<PolynomialProfile>()};
  const GsResult r = solver.solve();

  EXPECT_NE(r.status, GsStatus::converged);
  EXPECT_FALSE(r.ok());
  // Best-effort state is retained regardless of outcome.
  EXPECT_EQ(r.psi.size(), cfg.grid.size());
}

TEST(GsSolver, ReportsIterationLimitRatherThanClaimingSuccess) {
  GsConfig cfg = converging_case();
  cfg.max_iterations = 3;          // deliberately too few
  cfg.tolerance = Real{1e-14};
  GsSolver solver{cfg, std::make_shared<PolynomialProfile>()};
  const GsResult r = solver.solve();
  EXPECT_NE(r.status, GsStatus::converged);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.psi.size(), cfg.grid.size());
}

TEST(GsSolver, RejectsMalformedConfiguration) {
  // Bad INPUT throws; a non-existent equilibrium does not. The distinction is
  // deliberate -- one is a programming error, the other is physics.
  GsConfig cfg = converging_case();
  EXPECT_THROW((GsSolver{cfg, nullptr}), std::invalid_argument);

  GsConfig zero_current = converging_case();
  zero_current.plasma_current = Real{0};
  EXPECT_THROW((GsSolver{zero_current, std::make_shared<PolynomialProfile>()}),
               std::invalid_argument);

  GsConfig tiny = converging_case();
  tiny.grid = EllipticGrid{9, 9, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  EXPECT_THROW((GsSolver{tiny, std::make_shared<PolynomialProfile>()}),
               std::invalid_argument);
}

TEST(GsStatusStrings, CoverEveryEnumerator) {
  // Guards against a new failure mode being added without a diagnostic string,
  // which would surface to a user as "unknown".
  EXPECT_STREQ(to_string(GsStatus::converged), "converged");
  EXPECT_STREQ(to_string(GsStatus::no_closed_surface), "no_closed_surface");
  EXPECT_STREQ(to_string(GsStatus::axis_lost), "axis_lost");
  EXPECT_STREQ(to_string(GsStatus::residual_stalled), "residual_stalled");
  EXPECT_STREQ(to_string(GsStatus::iteration_limit), "iteration_limit");
}
