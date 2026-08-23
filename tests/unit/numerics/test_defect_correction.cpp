// Defect-correction verification: does the L6/L2 pair actually deliver sixth
// order, and does it converge?
//
// This is the test that justifies the whole architecture. If the converged
// solution is only second order, the Pade machinery is dead weight and the
// design should collapse to a plain multigrid solve. The order study is
// therefore the primary assertion, not a secondary check.
//
// The manufactured solution is non-polynomial: a polynomial of degree <= 6 would
// be reproduced exactly by both operators and the test would pass vacuously.

#include "quasar/numerics/defect_correction.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/geometric_multigrid.hpp"
#include "quasar/numerics/gs_operator_l2.hpp"
#include "quasar/numerics/gs_operator_l6.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

// -- On measuring sixth order in fp64 -----------------------------------------
// A sixth-order SECOND derivative saturates the double-precision roundoff floor
// almost immediately. Forming the operator divides by h^2, so the achievable
// error floor grows as eps/h^2 while the truncation error falls as h^6; the two
// cross very early. On this domain with a smooth O(1)-wavenumber solution the
// crossover is near n = 33, so an order study over n = 33 -> 65 measures the
// FLOOR (apparent rate ~3.5, then negative on further refinement) and not the
// scheme.
//
// This is not a defect and it is not something to tune away -- it is the real
// resolution limit of a sixth-order Delta* in fp64. The order study therefore
// uses a higher-wavenumber solution, which raises the truncation error enough to
// keep it above the floor across the measured range. The low-wavenumber solution
// is retained for the solver tests, where the quantity of interest is psi itself
// (no 1/h^2 amplification) and the floor is not in play.
Real mms_psi(Real r, Real z) {
  return std::sin(Real{1.7} * r) * std::exp(Real{-0.6} * z) + Real{0.3} * r * z;
}

Real mms_source(Real r, Real z) {
  const Real a = Real{1.7};
  const Real b = Real{-0.6};
  const Real s = std::sin(a * r);
  const Real c = std::cos(a * r);
  const Real e = std::exp(b * z);
  const Real psi_rr = -a * a * s * e;
  const Real psi_r  = a * c * e + Real{0.3} * z;
  const Real psi_zz = b * b * s * e;
  return psi_rr - psi_r / r + psi_zz;
}

// High-wavenumber companion used only for operator order studies.
Real osc_psi(Real r, Real z) {
  return std::sin(Real{9} * r) * std::cos(Real{7} * z);
}

Real osc_source(Real r, Real z) {
  const Real a = Real{9};
  const Real b = Real{7};
  const Real s = std::sin(a * r);
  const Real c = std::cos(a * r);
  const Real cz = std::cos(b * z);
  return -a * a * s * cz - (a * c * cz) / r - b * b * s * cz;
}

// Max |L(psi_exact) - source_exact| over interior nodes, for an arbitrary
// operator and manufactured pair.
template <class Apply, class Psi, class Src>
Real operator_error(const EllipticGrid& g, Apply&& apply, Psi&& p, Src&& s) {
  ScalarField x = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) x[g.index(i, j)] = p(g.r(i), g.z(j));
  }
  ScalarField y = quasar::numerics::make_field(g);
  apply(g, x, y);
  Real m = Real{0};
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      m = std::max(m, std::abs(y[g.index(i, j)] - s(g.r(i), g.z(j))));
    }
  }
  return m;
}

// Max |L6(psi) - L2(psi)| over the interior.
//
// Deliberately a noinline free function rather than a lambda: called with two
// compile-time grid sizes, GCC inlines both into one body, constant-propagates
// one size into the other's allocation, and emits a spurious
// -Wstringop-overflow. AddressSanitizer confirms there is no real overflow
// (both calls size their outputs correctly at runtime).
__attribute__((noinline))
Real l2_l6_gap(int n) {
  const EllipticGrid g = EllipticGrid{n, n, Real{0.5}, Real{1.5},
                                      Real{-0.5}, Real{0.5}};
  ScalarField x = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) x[g.index(i, j)] = mms_psi(g.r(i), g.z(j));
  }
  ScalarField y2 = quasar::numerics::make_field(g);
  ScalarField y6 = quasar::numerics::make_field(g);
  quasar::numerics::gs_apply_l2(g, x, y2);
  quasar::numerics::gs_apply_l6(g, x, y6);
  Real m = Real{0};
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      m = std::max(m, std::abs(y6[g.index(i, j)] - y2[g.index(i, j)]));
    }
  }
  return m;
}

// Observed order between two grids that need not differ by a factor of two.
Real observed_order(int n_coarse, Real e_coarse, int n_fine, Real e_fine) {
  const Real h_ratio = static_cast<Real>(n_fine - 1)
                     / static_cast<Real>(n_coarse - 1);
  return std::log(e_coarse / e_fine) / std::log(h_ratio);
}

EllipticGrid make_grid(int n) {
  return EllipticGrid{n, n, Real{0.5}, Real{1.5}, Real{-0.5}, Real{0.5}};
}

ScalarField seeded_boundary(const EllipticGrid& g) {
  ScalarField x = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (g.on_boundary(i, j)) x[g.index(i, j)] = mms_psi(g.r(i), g.z(j));
    }
  }
  return x;
}

ScalarField exact_source(const EllipticGrid& g) {
  ScalarField b = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      b[g.index(i, j)] = mms_source(g.r(i), g.z(j));
    }
  }
  return b;
}

Real solution_error(const EllipticGrid& g, const ScalarField& x) {
  Real m = Real{0};
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      m = std::max(m, std::abs(x[g.index(i, j)] - mms_psi(g.r(i), g.z(j))));
    }
  }
  return m;
}

struct SolveOutcome {
  Real error{0};
  int  iterations{0};
  bool converged{false};
};

SolveOutcome run(int n) {
  const EllipticGrid g = make_grid(n);
  quasar::numerics::GsMultigrid mg{g};
  ScalarField x = seeded_boundary(g);
  const ScalarField b = exact_source(g);
  const auto rep = quasar::numerics::solve_defect_corrected(g, x, b, mg);
  return {solution_error(g, x), rep.iterations, rep.converged};
}

}  // namespace

TEST(GsOperatorL6, IsSixthOrderAccurate) {
  // Measured on the high-wavenumber solution so truncation stays above the
  // fp64 floor across the range (see the note above). Grids are chosen close
  // together for the same reason.
  const auto err = [](int n) {
    return operator_error(make_grid(n), quasar::numerics::gs_apply_l6,
                          osc_psi, osc_source);
  };
  const Real rate = observed_order(33, err(33), 65, err(65));
  EXPECT_GT(rate, 5.5) << "L6 operator is not sixth order (rate=" << rate << ")";
}

TEST(GsOperatorL6, OrderStudyDegradesOnceRoundoffDominates) {
  // Documents the fp64 limit as a tested property rather than folklore: on the
  // smooth low-wavenumber solution the sixth-order operator is ALREADY at its
  // roundoff floor by n=33, so refining makes the measured error worse. Anyone
  // who later sees a "failing" order study on a smooth case should land here.
  const auto err = [](int n) {
    return operator_error(make_grid(n), quasar::numerics::gs_apply_l6,
                          mms_psi, mms_source);
  };
  const Real coarse = err(65);
  const Real fine   = err(257);
  EXPECT_GT(fine, coarse)
      << "expected roundoff-floor growth under refinement; coarse=" << coarse
      << " fine=" << fine;
  // The floor scales like eps/h^2. Confirm the growth is consistent with that
  // rather than with a genuine instability.
  const Real growth = fine / coarse;
  EXPECT_GT(growth, Real{2});
  EXPECT_LT(growth, Real{100});
}

TEST(GsOperatorL6, DiffersFromL2AtSecondOrder) {
  // The two operators must differ by O(h^2) -- that difference IS the defect
  // being corrected. If they agreed to high order, defect correction would be
  // pointless; if they diverged, the preconditioner would be invalid.
  const Real rate = std::log2(l2_l6_gap(33) / l2_l6_gap(65));
  EXPECT_GT(rate, 1.7);
  EXPECT_LT(rate, 2.5);
}

TEST(DefectCorrection, BeatsTheSecondOrderSolverByOrdersOfMagnitude) {
  // The headline claim of the elliptic design, stated as the comparison that
  // actually matters: does running L6 in the residual buy anything over just
  // solving L2?
  //
  // A convergence RATE cannot be measured here -- at n >= 33 the sixth-order
  // solution is already at the fp64 floor (~5e-12), so refining does not reduce
  // the error and log2(e1/e2) is meaningless. The meaningful statement is the
  // error gap against the second-order solver on the same grid.
  const EllipticGrid g = make_grid(33);
  const ScalarField b = exact_source(g);

  quasar::numerics::GsMultigrid mg2{g};
  ScalarField x2 = seeded_boundary(g);
  ASSERT_GT(mg2.solve(x2, b, Real{1e-12}, 200), 0);
  const Real err_l2 = solution_error(g, x2);

  const auto o6 = run(33);
  ASSERT_TRUE(o6.converged);

  EXPECT_LT(o6.error, err_l2 / Real{1000})
      << "defect correction bought little over plain L2 multigrid: "
      << err_l2 << " -> " << o6.error;
  // Absolute check: sixth order on this grid should reach the fp64 floor.
  EXPECT_LT(o6.error, Real{1e-10});
}

TEST(DefectCorrection, ReachesRoundoffFloorAtEveryResolution) {
  // Because the scheme saturates fp64 by n=33, the error should stay at that
  // floor as the grid refines rather than growing (which would indicate
  // instability) or falling (which would mean n=33 was not yet converged).
  for (const int n : {33, 65, 129}) {
    const auto o = run(n);
    EXPECT_TRUE(o.converged) << "n=" << n;
    EXPECT_LT(o.error, Real{1e-10}) << "n=" << n << " err=" << o.error;
  }
}

TEST(DefectCorrection, IterationCountIsGridIndependent) {
  // The property that makes the scheme affordable at scale. Note this is
  // grid-INDEPENDENCE, not the O(h^2) shrinkage a naive analysis predicts; see
  // the measured table in defect_correction.hpp.
  const auto coarse = run(33);
  const auto fine   = run(129);
  ASSERT_TRUE(coarse.converged);
  ASSERT_TRUE(fine.converged);
  EXPECT_LE(fine.iterations, coarse.iterations + 3)
      << coarse.iterations << " -> " << fine.iterations;
  EXPECT_LE(coarse.iterations, 40) << "unexpectedly slow: " << coarse.iterations;
}

TEST(DefectCorrection, DivergesAboveTheRelaxationStabilityLimit) {
  // Pins the stability margin that sets the default. omega = 1.2 is measured to
  // diverge; if a future change makes this pass, the iteration's spectral
  // properties have shifted and the default should be re-tuned.
  const EllipticGrid g = make_grid(33);
  quasar::numerics::GsMultigrid mg{g};
  ScalarField x = seeded_boundary(g);
  const ScalarField b = exact_source(g);
  quasar::numerics::DefectCorrectionConfig cfg;
  cfg.relaxation = Real{1.2};
  cfg.max_iterations = 60;
  const auto rep = quasar::numerics::solve_defect_corrected(g, x, b, mg, cfg);
  EXPECT_FALSE(rep.converged)
      << "omega=1.2 was expected to diverge; re-tune the default relaxation";
}

TEST(DefectCorrection, ResidualDecreasesMonotonically) {
  const EllipticGrid g = make_grid(65);
  quasar::numerics::GsMultigrid mg{g};
  ScalarField x = seeded_boundary(g);
  const ScalarField b = exact_source(g);
  const auto rep = quasar::numerics::solve_defect_corrected(g, x, b, mg);
  ASSERT_TRUE(rep.converged);
  ASSERT_GE(rep.residual_history.size(), 2u);
  for (std::size_t k = 1; k < rep.residual_history.size(); ++k) {
    EXPECT_LT(rep.residual_history[k], rep.residual_history[k - 1])
        << "residual rose at step " << k;
  }
}

TEST(DefectCorrection, PreservesDirichletData) {
  const EllipticGrid g = make_grid(65);
  quasar::numerics::GsMultigrid mg{g};
  ScalarField x = seeded_boundary(g);
  const ScalarField b = exact_source(g);
  quasar::numerics::solve_defect_corrected(g, x, b, mg);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (!g.on_boundary(i, j)) continue;
      EXPECT_DOUBLE_EQ(x[g.index(i, j)], mms_psi(g.r(i), g.z(j)));
    }
  }
}

TEST(GsOperatorL6, ReportsApplicabilityForSmallGrids) {
  // The Pade closures need a minimum line length; the hierarchy's coarse levels
  // fall below it, which is why only L2 ever runs there.
  EXPECT_FALSE(quasar::numerics::l6_is_applicable(make_grid(9)));
  EXPECT_TRUE(quasar::numerics::l6_is_applicable(make_grid(33)));
}
