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
#include "quasar/physics/equilibrium/flux_surfaces.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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

TEST(CriticalPoints, RejectsFluxFieldsWhoseSizeDoesNotMatchTheGrid) {
  const EllipticGrid g = make_grid(33);
  const ScalarField undersized(g.size() - 1, Real{0});

  EXPECT_THROW((void)quasar::equilibrium::compute_derivatives(g, undersized),
               std::invalid_argument);
  EXPECT_THROW((void)quasar::equilibrium::find_critical_points(g, undersized),
               std::invalid_argument);

  const quasar::equilibrium::CriticalPointSet critical;
  EXPECT_THROW(
      (void)quasar::equilibrium::compute_field(
          g, undersized, critical, [](Real) { return Real{1}; }),
      std::invalid_argument);
}

TEST(CriticalPoints, RejectsNewtonCycleWhenIterationBudgetIsExhausted) {
  const EllipticGrid g = make_grid(17);
  const ScalarField psi(g.size(), Real{1});
  quasar::equilibrium::DerivativeFields derivatives;
  derivatives.d_r.assign(g.size(), Real{0});
  derivatives.d_z.assign(g.size(), Real{0});
  derivatives.d_rr.assign(g.size(), Real{0});
  derivatives.d_zz.assign(g.size(), Real{1});
  derivatives.d_rz.assign(g.size(), Real{0});
  const Real origin = g.r(g.nr / 2);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real x = (g.r(i) - origin) / g.dr();
      derivatives.d_r[g.index(i, j)] = x * x * x - Real{2} * x + Real{2};
      derivatives.d_rr[g.index(i, j)] =
          (Real{3} * x * x - Real{2}) / g.dr();
    }
  }

  const auto point = quasar::equilibrium::refine_critical_point(
      g, psi, derivatives, origin, g.z(g.nz / 2));

  EXPECT_FALSE(point.valid)
      << "the exact Newton two-cycle x=0 -> 1 -> 0 was accepted after "
         "exhausting the iteration budget";
}

TEST(CriticalPoints, RejectsOverflowedDerivativesFromFiniteField) {
  const EllipticGrid g{33, 33, Real{0.9}, Real{1.1}, Real{-0.1}, Real{0.1}};
  const Real amplitude =
      Real{0.3} * std::numeric_limits<Real>::max();
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real offset = g.r(i) - Real{1};
      psi[g.index(i, j)] = amplitude * (offset * offset);
      ASSERT_TRUE(std::isfinite(psi[g.index(i, j)]));
    }
  }

  const auto cps = quasar::equilibrium::find_critical_points(g, psi);
  EXPECT_FALSE(cps.axis.valid);
  EXPECT_TRUE(cps.x_points.empty());
}

TEST(GsFluxSurfaces, RejectsNegativeDimensionsBeforeMutation) {
  quasar::equilibrium::GsFluxSurfaces surfaces{0, 4};
  ASSERT_EQ(surfaces.n_surfaces, 0);
  ASSERT_EQ(surfaces.n_theta, 4);

  EXPECT_THROW(surfaces.resize(-1, 8), std::invalid_argument);
  EXPECT_THROW(surfaces.resize(8, -1), std::invalid_argument);

  EXPECT_EQ(surfaces.n_surfaces, 0);
  EXPECT_EQ(surfaces.n_theta, 4);
  EXPECT_TRUE(surfaces.r.empty());
  EXPECT_TRUE(surfaces.z.empty());
  EXPECT_TRUE(surfaces.count.empty());
  EXPECT_TRUE(surfaces.closed.empty());
  EXPECT_TRUE(surfaces.q.empty());
  EXPECT_TRUE(surfaces.area.empty());
  EXPECT_TRUE(surfaces.volume.empty());
  EXPECT_TRUE(surfaces.psi_n.empty());
}

TEST(PlasmaMask, BlocksAnOffGridSaddleBetweenFluxLobes) {
  const EllipticGrid g{81, 41, Real{0.6}, Real{2.4}, Real{-0.5}, Real{0.5}};
  // The separatrix saddle lies halfway between radial nodes 40 and 41. Both
  // nodes pass a scalar psi_N < 1 cutoff, so only edge-aware connectivity can
  // keep the two lobes separate.
  constexpr Real center = Real{1.51125};
  constexpr Real half_separation = Real{0.45};
  constexpr Real axis_r = center - half_separation;
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

  constexpr int mid_j = 20;
  constexpr int bridge_left_i = 40;
  constexpr int bridge_right_i = 41;
  ASSERT_EQ(center, Real{0.5}
                        * (g.r(bridge_left_i) + g.r(bridge_right_i)));
  EXPECT_LT(quasar::equilibrium::normalized_flux(
                psi[g.index(bridge_left_i, mid_j)], psi_axis, psi_boundary),
            Real{1});
  EXPECT_LT(quasar::equilibrium::normalized_flux(
                psi[g.index(bridge_right_i, mid_j)], psi_axis, psi_boundary),
            Real{1});

  const std::vector<int> mask =
      quasar::equilibrium::axis_connected_plasma_mask(
          g, psi, axis_r, Real{0}, psi_axis, psi_boundary);
  EXPECT_EQ(mask[g.index(20, mid_j)], 1);
  EXPECT_EQ(mask[g.index(60, mid_j)], 0);

  std::size_t connected = 0;
  for (const int value : mask) connected += value != 0 ? 1u : 0u;
  EXPECT_EQ(connected, 310u);
}

TEST(PlasmaMask, RejectsAnOverflowedFluxSpan) {
  const EllipticGrid g{17, 17, Real{0.6}, Real{2.4}, Real{-0.5}, Real{0.5}};
  const ScalarField psi(g.size(), Real{0});
  const Real limit = std::numeric_limits<Real>::max();

  const std::vector<int> mask =
      quasar::equilibrium::axis_connected_plasma_mask(
          g, psi, Real{1.5}, Real{0}, -limit, limit);
  for (const int value : mask) EXPECT_EQ(value, 0);
}

TEST(PlasmaMask, ValidatesGridAndPsiBeforeDeviceAccess) {
  EllipticGrid malformed;
  malformed.nr = -1;
  malformed.nz = 9;
  malformed.r_min = Real{0.4};
  malformed.r_max = Real{1.8};
  malformed.z_min = Real{-0.6};
  malformed.z_max = Real{0.6};
  const quasar::equilibrium::GsDerivativeFields derivatives;
  quasar::equilibrium::GsPlasmaMaskScratch scratch;

  try {
    quasar::equilibrium::launch_gs_build_plasma_mask(
        malformed, nullptr, derivatives, Real{1}, Real{0}, Real{0}, Real{1},
        scratch, nullptr);
    FAIL() << "malformed grid was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("EllipticGrid"),
              std::string::npos)
        << "validation reached pointer or device storage before the grid";
  }

  const EllipticGrid valid = make_grid(9);
  try {
    quasar::equilibrium::launch_gs_build_plasma_mask(
        valid, nullptr, derivatives, Real{1}, Real{0}, Real{0}, Real{1},
        scratch, nullptr);
    FAIL() << "null psi buffer was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("psi buffer"), std::string::npos)
        << "validation reached device storage before the input pointer";
  }
}

TEST(PlasmaMask, ResizeRejectsMalformedAndOverflowedGridsBeforeAllocation) {
  quasar::equilibrium::GsPlasmaMaskScratch scratch;

  EllipticGrid malformed;
  malformed.nr = -1;
  malformed.nz = 9;
  malformed.r_min = Real{0.4};
  malformed.r_max = Real{1.8};
  malformed.z_min = Real{-0.6};
  malformed.z_max = Real{0.6};
  EXPECT_THROW(scratch.resize(malformed), std::invalid_argument);
  EXPECT_TRUE(scratch.mask.empty());
  EXPECT_TRUE(scratch.queue.empty());

  const int huge = std::numeric_limits<int>::max();
  const EllipticGrid overflowed{huge, huge, Real{0.4}, Real{1.8},
                                Real{-0.6}, Real{0.6}};
  EXPECT_THROW(scratch.resize(overflowed), std::length_error);
  EXPECT_TRUE(scratch.mask.empty());
  EXPECT_TRUE(scratch.queue.empty());
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

TEST(NormalizedFlux, RejectsNonFiniteAndOverflowedNormalization) {
  const Real limit = std::numeric_limits<Real>::max();
  const Real infinity = std::numeric_limits<Real>::infinity();
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real tiny = std::numeric_limits<Real>::denorm_min();

  EXPECT_EQ(quasar::equilibrium::normalized_flux(nan, Real{0}, Real{1}),
            Real{1});
  EXPECT_EQ(quasar::equilibrium::normalized_flux(infinity, Real{0}, Real{1}),
            Real{1});
  EXPECT_EQ(quasar::equilibrium::normalized_flux(
                Real{0}, nan, Real{1}),
            Real{1});
  EXPECT_EQ(quasar::equilibrium::normalized_flux(
                Real{0}, Real{0}, infinity),
            Real{1});
  EXPECT_EQ(quasar::equilibrium::normalized_flux(
                Real{0}, -limit, limit),
            Real{1});
  EXPECT_EQ(quasar::equilibrium::normalized_flux(
                limit, -limit, Real{0}),
            Real{1});
  EXPECT_EQ(quasar::equilibrium::normalized_flux(
                -limit, Real{0}, tiny),
            Real{1});
}

TEST(GsFluxSurfaces, RejectsMalformedGridBeforeDeviceAccess) {
  EllipticGrid malformed;
  malformed.nr = -1;
  malformed.nz = 9;
  malformed.r_min = Real{0.4};
  malformed.r_max = Real{1.8};
  malformed.z_min = Real{-0.6};
  malformed.z_max = Real{0.6};

  const quasar::equilibrium::GsMagneticField field;
  quasar::equilibrium::GsFluxSurfaces surfaces;
  try {
    quasar::equilibrium::launch_gs_trace_surfaces(
        malformed, nullptr, field, Real{1}, Real{0}, Real{0}, Real{1},
        surfaces, nullptr);
    FAIL() << "malformed grid was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("EllipticGrid"),
              std::string::npos)
        << "validation reached pointer or device storage before the grid";
  }
}
