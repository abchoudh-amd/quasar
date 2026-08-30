// Multigrid convergence and second-order-operator verification.
//
// The property that matters for multigrid is not accuracy but GRID-INDEPENDENT
// convergence: the V-cycle residual reduction factor must not degrade as the
// grid refines. A solver whose iteration count grows with resolution is not
// multigrid, it is an expensive relaxation scheme, and the failure is invisible
// on a single grid size. Every convergence test here therefore sweeps at least
// three resolutions and compares.
//
// The operator itself is only second order by design (it is the smoothable half
// of the defect-correction pair), so its order study expects 2, not 6.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/geometric_multigrid.hpp"
#include "quasar/numerics/gs_operator_l2.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

// Manufactured solution on an annular domain. Non-polynomial and non-separable
// so no term of Delta* is accidentally zero.
Real mms_psi(Real r, Real z) {
  return std::sin(Real{1.7} * r) * std::exp(Real{-0.6} * z) + Real{0.3} * r * z;
}

// Delta* psi = psi_rr - psi_r / r + psi_zz, evaluated analytically.
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

EllipticGrid make_grid(int n) {
  return EllipticGrid{n, n, Real{0.5}, Real{1.5}, Real{-0.5}, Real{0.5}};
}

// Seed `x` with exact Dirichlet boundary data and zero interior.
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

}  // namespace

TEST(EllipticGrid, RejectsAxisTouchingDomain) {
  // The annular decision is enforced, not merely documented: r_min = 0 would
  // make the 1/r term in Delta* singular.
  EXPECT_THROW((EllipticGrid{9, 9, Real{0}, Real{1}, Real{0}, Real{1}}),
               std::invalid_argument);
  EXPECT_THROW((EllipticGrid{9, 9, Real{-0.5}, Real{1}, Real{0}, Real{1}}),
               std::invalid_argument);
}

TEST(EllipticGrid, RejectsNonFiniteSpansAndCollapsedSpacing) {
  const Real limit = std::numeric_limits<Real>::max();
  const Real tiny = std::numeric_limits<Real>::denorm_min();

  EXPECT_THROW((EllipticGrid{9, 9, Real{1}, Real{2}, -limit, limit}),
               std::invalid_argument);
  EXPECT_THROW((EllipticGrid{3, 3, tiny, Real{2} * tiny, Real{-1}, Real{1}}),
               std::invalid_argument);
  EXPECT_THROW((EllipticGrid{3, 3, Real{1}, Real{2}, Real{0}, tiny}),
               std::invalid_argument);
}

TEST(EllipticGrid, CoarsensCleanlyForPowerOfTwoPlusOne) {
  const EllipticGrid g = make_grid(65);
  EXPECT_TRUE(g.can_coarsen());
  const EllipticGrid c = g.coarsen();
  EXPECT_EQ(c.nr, 33);
  EXPECT_EQ(c.nz, 33);
  // Physical extents are preserved under coarsening; only resolution changes.
  EXPECT_DOUBLE_EQ(c.r_min, g.r_min);
  EXPECT_DOUBLE_EQ(c.r_max, g.r_max);
  EXPECT_GE(g.coarsenable_levels(), 4);
}

TEST(GsOperatorL2, AnnihilatesTheHomogeneousSolution) {
  // psi = r^2 has Delta*(r^2) = 2 - (2r)/r = 0 exactly, and it is quadratic so
  // the second-order operator reproduces it to round-off. This is the sharpest
  // available check that the 1/r weighting carries the right sign and scale --
  // a Poisson-style discretization (no 1/r term) would return 2, not 0.
  const EllipticGrid g = make_grid(33);
  ScalarField x = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) x[g.index(i, j)] = g.r(i) * g.r(i);
  }
  ScalarField y = quasar::numerics::make_field(g);
  quasar::numerics::gs_apply_l2(g, x, y);
  EXPECT_LT(quasar::numerics::interior_max_norm(g, y), 1e-10);
}

TEST(GsOperatorL2, IsSecondOrderAccurate) {
  const EllipticGrid g1 = make_grid(33);
  const EllipticGrid g2 = make_grid(65);

  const auto op_error = [](const EllipticGrid& g) {
    ScalarField x = quasar::numerics::make_field(g);
    for (int j = 0; j < g.nz; ++j) {
      for (int i = 0; i < g.nr; ++i) x[g.index(i, j)] = mms_psi(g.r(i), g.z(j));
    }
    ScalarField y = quasar::numerics::make_field(g);
    quasar::numerics::gs_apply_l2(g, x, y);
    Real m = Real{0};
    for (int j = 1; j < g.nz - 1; ++j) {
      for (int i = 1; i < g.nr - 1; ++i) {
        m = std::max(m, std::abs(y[g.index(i, j)] - mms_source(g.r(i), g.z(j))));
      }
    }
    return m;
  };

  const Real rate = std::log2(op_error(g1) / op_error(g2));
  EXPECT_GT(rate, 1.8) << "L2 operator lost its design order";
  EXPECT_LT(rate, 2.4) << "L2 is unexpectedly better than second order; "
                          "the test may not be measuring truncation";
}

TEST(GsMultigrid, RestrictionRequiresAnExactlyNestedPair) {
  const EllipticGrid fine = make_grid(33);
  const EllipticGrid coarse = fine.coarsen();
  ScalarField rf(fine.size(), Real{1});
  ScalarField rc;

  EXPECT_NO_THROW(quasar::numerics::restrict_full_weighting(fine, coarse, rf, rc));
  for (int j = 1; j < coarse.nz - 1; ++j) {
    for (int i = 1; i < coarse.nr - 1; ++i) {
      EXPECT_NEAR(rc[coarse.index(i, j)], Real{1}, Real{1e-14});
    }
  }

  const EllipticGrid wrong_size = make_grid(17);
  EXPECT_THROW(quasar::numerics::restrict_full_weighting(
                   fine, wrong_size.coarsen(), rf, rc),
               std::invalid_argument);

  const EllipticGrid wrong_extent{coarse.nr, coarse.nz, Real{0.6}, Real{1.6},
                                  coarse.z_min, coarse.z_max};
  EXPECT_THROW(quasar::numerics::restrict_full_weighting(
                   fine, wrong_extent, rf, rc),
               std::invalid_argument);

  rf.pop_back();
  EXPECT_THROW(quasar::numerics::restrict_full_weighting(fine, coarse, rf, rc),
               std::invalid_argument);
}

TEST(GsMultigrid, VCycleRateIsGridIndependent) {
  // The defining multigrid property. If this test passes at one size and fails
  // at another, the transfer operators are not a variational pair.
  std::vector<Real> rates;
  for (const int n : {33, 65, 129}) {
    const EllipticGrid g = make_grid(n);
    quasar::numerics::GsMultigrid mg{g};
    ScalarField x = seeded_boundary(g);
    const ScalarField b = exact_source(g);
    // Measure after one cycle so the rate reflects the asymptotic smoother
    // behaviour rather than the first transient.
    mg.v_cycle(x, b);
    rates.push_back(mg.cycle_reduction(x, b));
  }
  for (std::size_t k = 0; k < rates.size(); ++k) {
    EXPECT_LT(rates[k], 0.35) << "V-cycle reduction too weak at index " << k
                              << " (rate=" << rates[k] << ")";
  }
  // Grid independence: the finest rate must not be materially worse than the
  // coarsest. A growing rate is the signature of a broken hierarchy.
  EXPECT_LT(rates.back(), rates.front() * Real{2.5})
      << "V-cycle rate degrades with refinement: " << rates.front() << " -> "
      << rates.back();
}

TEST(GsMultigrid, CycleReductionPropagatesNonFiniteInitialResidual) {
  const EllipticGrid g = make_grid(9);
  quasar::numerics::GsMultigrid mg{g};
  const ScalarField x = seeded_boundary(g);
  ScalarField b = exact_source(g);
  const std::size_t center = g.index(g.nr / 2, g.nz / 2);

  b[center] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_TRUE(std::isnan(mg.cycle_reduction(x, b)));

  b[center] = std::numeric_limits<Real>::infinity();
  const Real reduction = mg.cycle_reduction(x, b);
  EXPECT_TRUE(std::isinf(reduction));
  EXPECT_GT(reduction, Real{0});
}

TEST(GsMultigrid, SolvesToToleranceInBoundedCycles) {
  std::vector<int> counts;
  for (const int n : {33, 65, 129}) {
    const EllipticGrid g = make_grid(n);
    quasar::numerics::GsMultigrid mg{g};
    ScalarField x = seeded_boundary(g);
    const ScalarField b = exact_source(g);
    const int its = mg.solve(x, b, Real{1e-10}, 60);
    ASSERT_GT(its, 0) << "multigrid failed to converge at n=" << n;
    counts.push_back(its);
  }
  // Iteration count must be bounded independent of resolution.
  for (const int c : counts) EXPECT_LE(c, 30);
  EXPECT_LE(counts.back(), counts.front() + 5)
      << "iteration count grows with resolution: " << counts.front() << " -> "
      << counts.back();
}

TEST(GsMultigrid, ConvergedSolutionIsSecondOrderAccurate) {
  // Solving the discrete system to tight tolerance must recover the continuum
  // solution at the operator's order. This links the solver back to the PDE:
  // a solver can converge perfectly to the wrong discrete system.
  const auto solved_error = [](int n) {
    const EllipticGrid g = make_grid(n);
    quasar::numerics::GsMultigrid mg{g};
    ScalarField x = seeded_boundary(g);
    const ScalarField b = exact_source(g);
    const int its = mg.solve(x, b, Real{1e-12}, 200);
    EXPECT_GT(its, 0);
    return solution_error(g, x);
  };
  const Real rate = std::log2(solved_error(33) / solved_error(65));
  EXPECT_GT(rate, 1.8) << "converged solution is not second-order accurate";
}

TEST(GsMultigrid, PreservesExactDirichletBoundaryData) {
  const EllipticGrid g = make_grid(33);
  quasar::numerics::GsMultigrid mg{g};
  ScalarField x = seeded_boundary(g);
  const ScalarField b = exact_source(g);
  mg.solve(x, b, Real{1e-10}, 50);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (!g.on_boundary(i, j)) continue;
      EXPECT_DOUBLE_EQ(x[g.index(i, j)], mms_psi(g.r(i), g.z(j)))
          << "boundary node (" << i << "," << j << ") was modified";
    }
  }
}
