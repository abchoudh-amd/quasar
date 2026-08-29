// Solver-level invariants that need no reference implementation.
//
// This file replaced a host-vs-device equivalence suite. Those comparisons did
// their job -- they caught a real defect during the port, where the failure
// exits returned zeros for j_phi instead of the partial current -- and then
// became unrunnable when the host solver was deleted.
//
// What is kept here is everything that was never really about the host: the
// properties a correct free-boundary solve must have regardless of who computes
// it. Where a host comparison is still the right tool, it lives in the
// per-kernel tests, which compare against the reference implementations
// deliberately retained under numerics/. Where only the assembled solver can be
// checked, that is the port gate against the pre-port recorded equilibrium.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using quasar::Real;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::GsConfig;
using quasar::equilibrium::GsResult;
using quasar::equilibrium::GsSolver;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;

Real peak_magnitude(const std::vector<Real>& a) {
  Real m = Real{0};
  for (const Real v : a) m = std::max(m, std::abs(v));
  return m;
}

std::size_t bitwise_mismatches(const std::vector<Real>& a,
                               const std::vector<Real>& b) {
  std::size_t n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::memcmp(&a[k], &b[k], sizeof(Real)) != 0) ++n;
  }
  return n;
}

// The reference deck's coil set, which is known to confine.
GsConfig base_config(int nr, int nz) {
  GsConfig cfg;
  cfg.grid = EllipticGrid{nr, nz, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  cfg.coils = {
      CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
      CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1e6};
  cfg.tolerance = Real{1e-9};
  cfg.max_iterations = 400;
  return cfg;
}

std::shared_ptr<quasar::equilibrium::IEquilibriumProfile> make_profile() {
  return std::make_shared<PolynomialProfile>(
      std::vector<Real>{Real{1}, Real{-1}},
      std::vector<Real>{Real{1}, Real{-1}});
}

// Determinism is the property the port promises in place of bitwise agreement
// with a host reference: no atomics, fixed-order reductions, and a fixed tree
// shape, so the same input must give the identical answer every time. Without
// this, every other tolerance in the suite would be measuring noise.
TEST(GsSolverInvariants, RepeatedSolvesAreBitwiseIdentical) {
  const GsConfig cfg = base_config(33, 33);

  const GsResult first = GsSolver{cfg, make_profile()}.solve();
  ASSERT_EQ(first.status, GsStatus::converged);

  for (int trial = 0; trial < 3; ++trial) {
    const GsResult again = GsSolver{cfg, make_profile()}.solve();
    EXPECT_EQ(again.status, first.status) << "trial " << trial;
    EXPECT_EQ(again.iterations, first.iterations) << "trial " << trial;
    EXPECT_EQ(bitwise_mismatches(first.psi, again.psi), 0u) << "trial " << trial;
    EXPECT_EQ(bitwise_mismatches(first.j_phi, again.j_phi), 0u)
        << "trial " << trial;
    EXPECT_EQ(first.critical.axis.r, again.critical.axis.r) << "trial " << trial;
  }
}

// Regression, and the reason the equivalence suite earned its keep: a solve
// that loses the plasma must still return the current that was present when it
// failed. Returning zeros reads as "no current anywhere" and silently breaks
// the contract GsResult documents, that a failed configuration stays scoreable
// by an optimizer.
TEST(GsSolverInvariants, FailedSolveRetainsPartialFields) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 10;
  // A coil set that confines briefly and then loses the plasma.
  cfg.coils = {
      CoilFilament{Real{1.75}, Real{0.60}, Real{3.6e5}},
      CoilFilament{Real{1.75}, Real{-0.60}, Real{3.6e5}},
      CoilFilament{Real{0.42}, Real{0.0}, Real{-1.2e5}},
  };

  const GsResult res = GsSolver{cfg, make_profile()}.solve();

  ASSERT_NE(res.status, GsStatus::converged)
      << "this configuration is supposed to fail; the test proves nothing "
         "if it converges";
  EXPECT_GT(peak_magnitude(res.j_phi), Real{1})
      << "partial current was discarded on the failure path";
  EXPECT_GT(peak_magnitude(res.psi), Real{0})
      << "partial psi was discarded on the failure path";
  for (const Real v : res.j_phi) ASSERT_TRUE(std::isfinite(v));
  for (const Real v : res.psi) ASSERT_TRUE(std::isfinite(v));
}

// Picard relaxation changes the path to the solution, not the solution. A
// damped and an undamped solve must land on the same equilibrium, which is a
// far stronger statement than either converging on its own: it says the fixed
// point is a property of the problem rather than of the iteration.
TEST(GsSolverInvariants, RelaxationChangesThePathNotTheFixedPoint) {
  GsConfig undamped = base_config(33, 33);
  undamped.picard_relaxation = Real{1};
  GsConfig damped = undamped;
  damped.picard_relaxation = Real{0.6};

  const GsResult a = GsSolver{undamped, make_profile()}.solve();
  const GsResult b = GsSolver{damped, make_profile()}.solve();

  ASSERT_EQ(a.status, GsStatus::converged);
  ASSERT_EQ(b.status, GsStatus::converged);
  EXPECT_GT(b.iterations, a.iterations)
      << "damping should cost iterations; if not, the relaxation is not "
         "being applied";

  // Both converge to the same tolerance, so they agree to about that tolerance
  // rather than to machine precision.
  const Real tol = Real{1e-6};
  EXPECT_NEAR(b.critical.axis.r, a.critical.axis.r,
              std::abs(a.critical.axis.r) * tol);
  EXPECT_NEAR(b.critical.psi_axis, a.critical.psi_axis,
              std::abs(a.critical.psi_axis) * tol);
  EXPECT_NEAR(b.achieved_current, a.achieved_current,
              std::abs(a.achieved_current) * tol);
}

// Reversing the plasma current must mirror the equilibrium, not break it. The
// seeded well has to invert with the current: seeding the wrong sign makes the
// well pass through zero as it relaxes and the axis vanishes mid-iteration.
TEST(GsSolverInvariants, ReversedCurrentInvertsTheFluxWell) {
  GsConfig forward = base_config(33, 33);
  GsConfig reversed = forward;
  reversed.plasma_current = -forward.plasma_current;
  for (auto& c : reversed.coils) c.current = -c.current;

  const GsResult a = GsSolver{forward, make_profile()}.solve();
  const GsResult b = GsSolver{reversed, make_profile()}.solve();

  ASSERT_EQ(a.status, GsStatus::converged);
  ASSERT_EQ(b.status, GsStatus::converged);

  // The axis sits in the same place; only the sign of the flux flips.
  EXPECT_NEAR(b.critical.axis.r, a.critical.axis.r,
              std::abs(a.critical.axis.r) * Real{1e-6});
  EXPECT_LT(a.critical.psi_axis * b.critical.psi_axis, Real{0})
      << "psi_axis did not change sign with the current";
  EXPECT_NEAR(b.achieved_current, -a.achieved_current,
              std::abs(a.achieved_current) * Real{1e-9});
}

TEST(GsSolverInvariants, NonSquareGridConverges) {
  const GsConfig cfg = base_config(65, 33);
  const GsResult res = GsSolver{cfg, make_profile()}.solve();

  ASSERT_EQ(res.status, GsStatus::converged);
  EXPECT_TRUE(res.critical.axis.valid);
  EXPECT_GT(res.critical.axis.r, cfg.grid.r_min);
  EXPECT_LT(res.critical.axis.r, cfg.grid.r_max);
  EXPECT_NEAR(res.achieved_current / cfg.plasma_current, Real{1}, Real{1e-9});
}

TEST(GsSolverInvariants, RejectsNonPolynomialProfile) {
  struct Unsupported : quasar::equilibrium::IEquilibriumProfile {
    Real dp_dpsi(Real) const override { return Real{1}; }
    Real ff_prime(Real) const override { return Real{1}; }
    Real d2p_dpsi2(Real) const override { return Real{0}; }
    Real ff_prime_prime(Real) const override { return Real{0}; }
  };

  const GsConfig cfg = base_config(33, 33);
  EXPECT_THROW(GsSolver(cfg, std::make_shared<Unsupported>()),
               std::invalid_argument);
}

TEST(GsSolverInvariants, RejectsMalformedConfiguration) {
  EXPECT_THROW(GsSolver(base_config(33, 33), nullptr), std::invalid_argument);

  GsConfig zero_current = base_config(33, 33);
  zero_current.plasma_current = Real{0};
  EXPECT_THROW(GsSolver(zero_current, make_profile()), std::invalid_argument);

  // Below the two-sided Pade closure width, the sixth-order operator cannot be
  // formed at all.
  GsConfig tiny = base_config(33, 33);
  tiny.grid = EllipticGrid{9, 9, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  EXPECT_THROW(GsSolver(tiny, make_profile()), std::invalid_argument);
}

}  // namespace
