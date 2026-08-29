// End-to-end equivalence between the host and device Grad-Shafranov solvers.
//
// This is the port's acceptance gate. The per-kernel tests establish that each
// piece matches in isolation; this establishes that the assembled Picard loop
// takes the same path -- same iteration count, same status, same axis, same
// converged field.
//
// Iteration count is asserted, not just the final state. A device solve that
// reached the same answer by a different route would mean some branch (the
// convergence test, the stall window, the geometry-stability check) is seeing
// different scalars, and that would surface on a harder case even if it does
// not here.
//
// Full solves are expensive -- the host reference takes ~30 s at 65x65 -- so
// the grids here are small and the iteration caps modest. The 65x65 reference
// configuration is covered separately by the port gate against the recorded
// run.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/gs_solver_device.hpp"

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
using quasar::equilibrium::GsSolverDevice;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;

std::size_t bitwise_mismatches(const std::vector<Real>& a,
                               const std::vector<Real>& b) {
  std::size_t n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::memcmp(&a[k], &b[k], sizeof(Real)) != 0) ++n;
  }
  return n;
}

Real peak_magnitude(const std::vector<Real>& a) {
  Real m = Real{0};
  for (const Real v : a) m = std::max(m, std::abs(v));
  return m;
}

// Count nodes where exactly one side placed current. The psi_N >= 1 cutoff is a
// discrete branch, so a disagreement here is categorically different from a
// rounding difference.
int cutoff_flips(const std::vector<Real>& a, const std::vector<Real>& b) {
  int n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if ((a[k] == Real{0}) != (b[k] == Real{0})) ++n;
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

GsConfig base_config(int nr, int nz) {
  GsConfig cfg;
  cfg.grid = EllipticGrid{nr, nz, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  // The reference deck's coil set (runs/grad_shafranov_tokamak/input.yaml).
  // This matters: an arbitrary coil set can fail to confine at all, and a
  // comparison between two solves that both lost the plasma on iteration 3
  // tests almost nothing.
  cfg.coils = {
      CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
      CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1e6};
  cfg.tolerance = Real{1e-9};
  cfg.picard_relaxation = Real{1};
  return cfg;
}

std::shared_ptr<quasar::equilibrium::IEquilibriumProfile> make_profile() {
  return std::make_shared<PolynomialProfile>(
      std::vector<Real>{Real{1}, Real{-1}},
      std::vector<Real>{Real{1}, Real{-1}});
}

void compare_solves(GsConfig cfg, const char* label) {
  SCOPED_TRACE(label);

  const GsResult host = GsSolver{cfg, make_profile()}.solve();
  const GsResult dev = GsSolverDevice{cfg, make_profile()}.solve();

  EXPECT_EQ(host.status, dev.status)
      << "host " << quasar::equilibrium::to_string(host.status) << " vs device "
      << quasar::equilibrium::to_string(dev.status);
  EXPECT_EQ(host.iterations, dev.iterations) << "iteration count";
  EXPECT_EQ(host.newton_steps, dev.newton_steps);

  // Scalars agree to rounding, not bitwise. The plasma-current integral is the
  // one operation that cannot be bit-exact against a naive sequential host sum
  // (see launch_gs_total_plasma_current), and profile_scale is derived directly
  // from it, so a last-bit difference there seeds everything downstream.
  // Measured relative divergence on these cases is ~1e-15; 1e-12 leaves room
  // without being permissive enough to hide a real defect.
  constexpr Real kRelTol = Real{1e-12};
  EXPECT_NEAR(dev.profile_scale, host.profile_scale,
              std::abs(host.profile_scale) * kRelTol);

  // The residual needs an ABSOLUTE floor, not just a relative bound, and the
  // reason is worth recording. res.residual is a normalized norm of b - A*psi:
  // a difference of two larger, nearly equal quantities. Cancellation means its
  // error is inherited from the field-level rounding in absolute terms and does
  // not shrink as the residual does. Measured divergence is ~7e-14 absolute
  // across every case here and is flat in the residual's own magnitude, which
  // is exactly what that model predicts; a purely relative tolerance would
  // therefore tighten without limit as the solve converges and fail for a
  // reason that has nothing to do with the port.
  constexpr Real kResidualFloor = Real{1e-12};
  EXPECT_NEAR(dev.residual, host.residual,
              std::abs(host.residual) * kRelTol + kResidualFloor);

  EXPECT_EQ(host.critical.axis.valid, dev.critical.axis.valid);
  if (host.critical.axis.valid) {
    EXPECT_NEAR(dev.critical.axis.r, host.critical.axis.r,
                std::abs(host.critical.axis.r) * kRelTol);
    EXPECT_NEAR(dev.critical.axis.z, host.critical.axis.z,
                std::abs(host.critical.axis.z) * kRelTol + Real{1e-12});
    EXPECT_NEAR(dev.critical.psi_axis, host.critical.psi_axis,
                std::abs(host.critical.psi_axis) * kRelTol);
    EXPECT_NEAR(dev.critical.psi_boundary, host.critical.psi_boundary,
                std::abs(host.critical.psi_boundary) * kRelTol);
  }

  ASSERT_EQ(host.psi.size(), dev.psi.size());
  EXPECT_LE(max_abs_difference(host.psi, dev.psi),
            peak_magnitude(host.psi) * kRelTol)
      << "psi diverged beyond rounding";
  EXPECT_LE(max_abs_difference(host.j_phi, dev.j_phi),
            peak_magnitude(host.j_phi) * kRelTol)
      << "j_phi diverged beyond rounding";

  // No node may cross the plasma-boundary cutoff differently. That test is
  // discrete, so a flip is not a rounding difference -- it means one side put
  // current where the other put none.
  EXPECT_EQ(cutoff_flips(host.j_phi, dev.j_phi), 0)
      << "nodes disagree on whether they are inside the plasma";

  ASSERT_EQ(host.residual_history.size(), dev.residual_history.size());
  for (std::size_t k = 0; k < host.residual_history.size(); ++k) {
    EXPECT_NEAR(dev.residual_history[k], host.residual_history[k],
                std::abs(host.residual_history[k]) * kRelTol + kResidualFloor)
        << "residual history diverges at iteration " << k;
  }
}

// A short run that does not converge: exercises the main loop body and the
// iteration-limit exit without paying for a full solve.
TEST(GsSolverDevice, PartialSolveTracksHostIterationForIteration) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 6;
  compare_solves(cfg, "33x33, 6 iterations");
}

TEST(GsSolverDevice, NonSquareGridTracksHost) {
  GsConfig cfg = base_config(41, 33);
  cfg.max_iterations = 5;
  compare_solves(cfg, "41x33, 5 iterations");
}

TEST(GsSolverDevice, DampedPicardTracksHost) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 8;
  cfg.picard_relaxation = Real{0.7};
  compare_solves(cfg, "damped Picard, relaxation 0.7");
}

// Reversed plasma current flips the seed sign, which the host notes is
// load-bearing: seeding the wrong sign makes the well pass through zero and the
// axis vanish mid-iteration.
TEST(GsSolverDevice, ReversedCurrentTracksHost) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 5;
  cfg.plasma_current = Real{-1e6};
  compare_solves(cfg, "reversed plasma current");
}

// The Newton path is off by default and known to stall on this problem, but it
// must stall identically on both sides.
TEST(GsSolverDevice, NewtonPathTracksHost) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 8;
  cfg.enable_newton = true;
  cfg.newton_residual_threshold = Real{1e30};  // force the branch to be taken
  compare_solves(cfg, "Newton enabled");
}

// Failure is a reportable outcome, not an exception: an optimizer must be able
// to score a configuration with no confined plasma and continue. Both solvers
// must agree that this one has none.
TEST(GsSolverDevice, UnconfinedConfigurationFailsIdentically) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 10;
  cfg.coils = {CoilFilament{Real{1.0}, Real{0.0}, Real{1.0e3}}};
  cfg.seed.depth = Real{0};  // no seeded well: nothing creates an O-point

  const GsResult host = GsSolver{cfg, make_profile()}.solve();
  const GsResult dev = GsSolverDevice{cfg, make_profile()}.solve();

  EXPECT_EQ(host.status, dev.status);
  EXPECT_NE(dev.status, GsStatus::converged);
  EXPECT_EQ(host.iterations, dev.iterations);
}

// Regression: the failure paths must retain BOTH partial fields. An earlier
// version of the device solver read back only psi when the plasma was lost,
// leaving j_phi as the zeros it was constructed with -- which reads as "no
// current anywhere" rather than "the current present when confinement failed",
// and silently breaks the optimizer-scoring contract GsResult documents.
TEST(GsSolverDevice, FailedSolveStillReturnsPartialCurrent) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 10;
  // A coil set that confines briefly and then loses the plasma, so the solve
  // reaches the failure exit with a nonzero current already computed.
  cfg.coils = {
      CoilFilament{Real{1.75}, Real{0.60}, Real{3.6e5}},
      CoilFilament{Real{1.75}, Real{-0.60}, Real{3.6e5}},
      CoilFilament{Real{0.42}, Real{0.0}, Real{-1.2e5}},
  };

  const GsResult host = GsSolver{cfg, make_profile()}.solve();
  const GsResult dev = GsSolverDevice{cfg, make_profile()}.solve();

  ASSERT_EQ(host.status, GsStatus::no_closed_surface)
      << "this case is supposed to lose the plasma";
  EXPECT_EQ(dev.status, host.status);
  EXPECT_EQ(dev.iterations, host.iterations);

  ASSERT_GT(peak_magnitude(host.j_phi), Real{1})
      << "host retained no current either; the test proves nothing";
  EXPECT_GT(peak_magnitude(dev.j_phi), Real{1})
      << "device discarded the partial current on the failure path";
  EXPECT_LE(max_abs_difference(host.j_phi, dev.j_phi),
            peak_magnitude(host.j_phi) * Real{1e-12});
  EXPECT_GT(peak_magnitude(dev.psi), Real{0})
      << "device discarded the partial psi on the failure path";
}

TEST(GsSolverDevice, RejectsNonPolynomialProfile) {
  struct Unsupported : quasar::equilibrium::IEquilibriumProfile {
    Real dp_dpsi(Real) const override { return Real{1}; }
    Real ff_prime(Real) const override { return Real{1}; }
    Real d2p_dpsi2(Real) const override { return Real{0}; }
    Real ff_prime_prime(Real) const override { return Real{0}; }
  };

  const GsConfig cfg = base_config(33, 33);
  EXPECT_THROW(GsSolverDevice(cfg, std::make_shared<Unsupported>()),
               std::invalid_argument);
}

TEST(GsSolverDevice, RejectsNullProfile) {
  const GsConfig cfg = base_config(33, 33);
  EXPECT_THROW(GsSolverDevice(cfg, nullptr), std::invalid_argument);
}

// A converged solve, at the smallest grid that reaches tolerance in reasonable
// time. This is the one that proves the loop terminates the same way, not just
// that it steps the same way.
TEST(GsSolverDevice, ConvergedSolveMatchesHost) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 400;

  const GsResult host = GsSolver{cfg, make_profile()}.solve();
  ASSERT_EQ(host.status, GsStatus::converged)
      << "host did not converge; test cannot check convergence agreement";

  const GsResult dev = GsSolverDevice{cfg, make_profile()}.solve();
  EXPECT_EQ(dev.status, GsStatus::converged);
  EXPECT_EQ(host.iterations, dev.iterations);
  EXPECT_LE(max_abs_difference(host.psi, dev.psi),
            peak_magnitude(host.psi) * Real{1e-12});
  EXPECT_NEAR(dev.achieved_current, host.achieved_current,
              std::abs(host.achieved_current) * Real{1e-12});
}

}  // namespace
