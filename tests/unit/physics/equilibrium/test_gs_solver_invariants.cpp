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
#include "quasar/numerics/gs_operator_l6.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/free_boundary.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
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
using quasar::numerics::ScalarField;

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

void expect_same_critical_points(
    const quasar::equilibrium::CriticalPointSet& expected,
    const quasar::equilibrium::CriticalPointSet& actual) {
  EXPECT_EQ(actual.axis.kind, expected.axis.kind);
  EXPECT_EQ(actual.axis.r, expected.axis.r);
  EXPECT_EQ(actual.axis.z, expected.axis.z);
  EXPECT_EQ(actual.axis.psi, expected.axis.psi);
  EXPECT_EQ(actual.axis.valid, expected.axis.valid);
  EXPECT_EQ(actual.psi_axis, expected.psi_axis);
  EXPECT_EQ(actual.psi_boundary, expected.psi_boundary);
  EXPECT_EQ(actual.has_closed_surface, expected.has_closed_surface);
  EXPECT_EQ(actual.critical_point_overflow,
            expected.critical_point_overflow);
  ASSERT_EQ(actual.x_points.size(), expected.x_points.size());
  for (std::size_t k = 0; k < expected.x_points.size(); ++k) {
    EXPECT_EQ(actual.x_points[k].kind, expected.x_points[k].kind);
    EXPECT_EQ(actual.x_points[k].r, expected.x_points[k].r);
    EXPECT_EQ(actual.x_points[k].z, expected.x_points[k].z);
    EXPECT_EQ(actual.x_points[k].psi, expected.x_points[k].psi);
    EXPECT_EQ(actual.x_points[k].valid, expected.x_points[k].valid);
  }
}

ScalarField initial_seeded_flux(const GsConfig& cfg) {
  const EllipticGrid& g = cfg.grid;
  ScalarField psi;
  quasar::equilibrium::evaluate_coil_field(g, cfg.coils, psi);

  const Real r_center = cfg.seed.r_center > Real{0}
                            ? cfg.seed.r_center
                            : g.r_min + Real{0.5} * (g.r_max - g.r_min);
  const Real z_center = cfg.seed.z_center != Real{0}
                            ? cfg.seed.z_center
                            : g.z_min + Real{0.5} * (g.z_max - g.z_min);
  const Real minor_radius = cfg.seed.minor_radius > Real{0}
                                ? cfg.seed.minor_radius
                                : Real{0.25} * std::min(g.r_max - g.r_min,
                                                        g.z_max - g.z_min);
  const Real depth = cfg.seed.depth * quasar::equilibrium::kMu0
                   * std::abs(cfg.plasma_current);
  const Real sign = cfg.plasma_current >= Real{0} ? Real{1} : Real{-1};

  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real dr = (g.r(i) - r_center) / minor_radius;
      const Real dz = (g.z(j) - z_center) / minor_radius;
      const Real s2 = dr * dr + dz * dz;
      if (s2 >= Real{1}) continue;
      const Real weight = Real{1} - s2;
      psi[g.index(i, j)] += depth * weight * weight * sign;
    }
  }
  return psi;
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
TEST(GsSolverInvariants, IterationLimitRetainsPartialFields) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 10;
  // This slowly converging coil set reaches the iteration limit after several
  // coherent states have been evaluated.
  cfg.coils = {
      CoilFilament{Real{1.75}, Real{0.60}, Real{3.6e5}},
      CoilFilament{Real{1.75}, Real{-0.60}, Real{3.6e5}},
      CoilFilament{Real{0.42}, Real{0.0}, Real{-1.2e5}},
  };

  const PolynomialProfile profile{
      std::vector<Real>{Real{1}, Real{-1}},
      std::vector<Real>{Real{1}, Real{-1}}};
  const GsResult res =
      GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}.solve();

  ASSERT_EQ(res.status, GsStatus::iteration_limit)
      << "this configuration must remain unconverged at the deliberately "
         "small iteration budget";
  EXPECT_GT(peak_magnitude(res.j_phi), Real{1})
      << "partial current was discarded on the failure path";
  EXPECT_GT(peak_magnitude(res.psi), Real{0})
      << "partial psi was discarded on the failure path";
  for (const Real v : res.j_phi) ASSERT_TRUE(std::isfinite(v));
  for (const Real v : res.psi) ASSERT_TRUE(std::isfinite(v));

  ASSERT_FALSE(res.residual_history.empty());
  const std::vector<int> mask =
      quasar::equilibrium::axis_connected_plasma_mask(
          cfg.grid, res.psi, res.critical.axis.r, res.critical.axis.z,
          res.critical.psi_axis, res.critical.psi_boundary);
  ScalarField expected_current = quasar::numerics::make_field(cfg.grid);
  for (int j = 1; j < cfg.grid.nz - 1; ++j) {
    for (int i = 1; i < cfg.grid.nr - 1; ++i) {
      const std::size_t k = cfg.grid.index(i, j);
      if (mask[k] == 0) continue;
      const Real psi_n = quasar::equilibrium::normalized_flux(
          res.psi[k], res.critical.psi_axis, res.critical.psi_boundary);
      if (psi_n >= Real{1}) continue;
      const Real r = cfg.grid.r(i);
      Real current = r * profile.dp_dpsi(psi_n)
                   + profile.ff_prime(psi_n)
                         / (quasar::equilibrium::kMu0 * r);
      expected_current[k] = res.profile_scale * current;
    }
  }
  EXPECT_EQ(bitwise_mismatches(expected_current, res.j_phi), 0u)
      << "failure returned psi from a different iteration than its current "
         "and metadata";
}

// An iteration-limit exit returns the state that was actually evaluated: its
// interior is the current Picard iterate, its boundary was generated from the
// returned current, and its residual was measured before any subsequent
// interior update. Critical points are intentionally those of the pre-boundary
// iterate; only boundary nodes change during the free-boundary refresh.
TEST(GsSolverInvariants, OneIterationReturnsTheEvaluatedPicardState) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 1;
  const PolynomialProfile profile{
      std::vector<Real>{Real{1}, Real{-1}},
      std::vector<Real>{Real{1}, Real{-1}}};

  const GsResult res =
      GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}.solve();
  ASSERT_EQ(res.status, GsStatus::iteration_limit);
  ASSERT_EQ(res.iterations, 1);
  ASSERT_EQ(res.residual_history.size(), 1u);
  EXPECT_EQ(res.residual, Real{1});
  EXPECT_EQ(res.residual_history.front(), Real{1});

  const ScalarField initial = initial_seeded_flux(cfg);
  const auto expected_critical =
      quasar::equilibrium::find_critical_points(cfg.grid, initial);
  ASSERT_TRUE(expected_critical.axis.valid);
  expect_same_critical_points(expected_critical, res.critical);

  const std::vector<int> mask =
      quasar::equilibrium::axis_connected_plasma_mask(
          cfg.grid, res.psi, res.critical.axis.r, res.critical.axis.z,
          res.critical.psi_axis, res.critical.psi_boundary);
  ScalarField expected_current = quasar::numerics::make_field(cfg.grid);
  for (int j = 1; j < cfg.grid.nz - 1; ++j) {
    for (int i = 1; i < cfg.grid.nr - 1; ++i) {
      const std::size_t k = cfg.grid.index(i, j);
      if (mask[k] == 0) continue;
      const Real psi_n = quasar::equilibrium::normalized_flux(
          res.psi[k], res.critical.psi_axis, res.critical.psi_boundary);
      if (psi_n >= Real{1}) continue;
      const Real r = cfg.grid.r(i);
      Real current = r * profile.dp_dpsi(psi_n)
                   + profile.ff_prime(psi_n)
                         / (quasar::equilibrium::kMu0 * r);
      current *= res.profile_scale;
      expected_current[k] = current;
    }
  }
  EXPECT_EQ(bitwise_mismatches(expected_current, res.j_phi), 0u)
      << "returned current does not match returned interior psi and metadata";

  ScalarField expected_state = initial;
  quasar::equilibrium::apply_coil_boundary(cfg.grid, cfg.coils,
                                            expected_state);
  quasar::equilibrium::add_plasma_boundary(cfg.grid, res.j_phi,
                                            expected_state);
  EXPECT_EQ(bitwise_mismatches(expected_state, res.psi), 0u)
      << "returned psi is not the boundary-refreshed state evaluated by the "
         "solver";

  ScalarField rhs = quasar::numerics::make_field(cfg.grid);
  for (int j = 1; j < cfg.grid.nz - 1; ++j) {
    for (int i = 1; i < cfg.grid.nr - 1; ++i) {
      const std::size_t k = cfg.grid.index(i, j);
      rhs[k] = -quasar::equilibrium::kMu0 * cfg.grid.r(i) * res.j_phi[k];
    }
  }
  ScalarField residual;
  quasar::numerics::gs_residual_l6(cfg.grid, res.psi, rhs, residual);
  const Real absolute_residual =
      quasar::numerics::interior_max_norm(cfg.grid, residual);
  EXPECT_TRUE(std::isfinite(absolute_residual));
  EXPECT_GT(absolute_residual, Real{0});
  // The first nonlinear residual is its own normalization denominator.
  EXPECT_EQ(absolute_residual / absolute_residual, res.residual);
}

TEST(GsSolverInvariants, ReportsCurrentNormalizationOverflow) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 1;
  const Real tiny = std::numeric_limits<Real>::min() / Real{100};
  const auto profile = std::make_shared<PolynomialProfile>(
      std::vector<Real>{Real{0}}, std::vector<Real>{tiny});

  const GsResult res = GsSolver{cfg, profile}.solve();

  EXPECT_EQ(res.status, GsStatus::numerical_failure);
  EXPECT_TRUE(res.residual_history.empty());
  EXPECT_TRUE(std::isfinite(res.profile_scale));
  for (const Real value : res.psi) EXPECT_TRUE(std::isfinite(value));
  for (const Real value : res.j_phi) EXPECT_TRUE(std::isfinite(value));
}

TEST(GsSolverInvariants, ReportsNonFiniteRawCurrentAsNumericalFailure) {
  GsConfig cfg = base_config(33, 33);
  cfg.max_iterations = 1;
  const Real limit = std::numeric_limits<Real>::max();
  const auto profile = std::make_shared<PolynomialProfile>(
      std::vector<Real>{limit}, std::vector<Real>{Real{0}});

  const GsResult res = GsSolver{cfg, profile}.solve();

  EXPECT_EQ(res.status, GsStatus::numerical_failure);
  EXPECT_TRUE(res.residual_history.empty());
  EXPECT_TRUE(res.critical.axis.valid);
  EXPECT_EQ(peak_magnitude(res.j_phi), Real{0});
  for (const Real value : res.psi) EXPECT_TRUE(std::isfinite(value));
}

TEST(GsSolverInvariants, ReportsOverflowedResolvedSeedAsNumericalFailure) {
  GsConfig cfg = base_config(33, 33);
  cfg.coils.clear();
  cfg.plasma_current = std::numeric_limits<Real>::max();
  cfg.seed.depth = std::numeric_limits<Real>::max();

  const GsResult res = GsSolver{cfg, make_profile()}.solve();

  EXPECT_EQ(res.status, GsStatus::numerical_failure);
  EXPECT_EQ(res.iterations, 0);
  EXPECT_TRUE(res.residual_history.empty());
  EXPECT_EQ(peak_magnitude(res.psi), Real{0});
  EXPECT_EQ(peak_magnitude(res.j_phi), Real{0});
}

TEST(GsSolverInvariants, DefaultSeedCenterDoesNotOverflowFiniteGridExtents) {
  const Real limit = std::numeric_limits<Real>::max();
  GsConfig cfg = base_config(33, 33);
  cfg.grid = EllipticGrid{33, 33, Real{0.5} * limit, Real{0.75} * limit,
                          Real{0.5} * limit, Real{0.75} * limit};
  cfg.coils.clear();
  cfg.max_iterations = 1;

  const GsResult res = GsSolver{cfg, make_profile()}.solve();

  EXPECT_NE(res.status, GsStatus::numerical_failure);
  EXPECT_GT(peak_magnitude(res.psi), Real{0})
      << "the default radial center overflowed and the seed was not added";
  for (const Real value : res.psi) EXPECT_TRUE(std::isfinite(value));
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

  const Real inf = std::numeric_limits<Real>::infinity();
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const auto expect_invalid = [&](const GsConfig& cfg) {
    EXPECT_THROW((void)GsSolver(cfg, make_profile()), std::invalid_argument);
  };

  for (const Real value : {inf, -inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.plasma_current = value;
    expect_invalid(cfg);
  }
  for (const int value : {0, -1}) {
    GsConfig cfg = base_config(33, 33);
    cfg.max_iterations = value;
    expect_invalid(cfg);
  }
  for (const Real value : {Real{0}, Real{-1}, inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.tolerance = value;
    expect_invalid(cfg);
  }
  for (const Real value : {Real{0}, Real{-0.1}, Real{1.1}, inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.picard_relaxation = value;
    expect_invalid(cfg);
  }
  for (const Real value : {Real{-1}, inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.newton_residual_threshold = value;
    expect_invalid(cfg);
    cfg = base_config(33, 33);
    cfg.newton_geometry_tolerance = value;
    expect_invalid(cfg);
  }

  {
    GsConfig cfg = base_config(33, 33);
    cfg.seed.r_center = cfg.grid.r_max;
    expect_invalid(cfg);
    cfg = base_config(33, 33);
    cfg.seed.z_center = cfg.grid.z_min;
    expect_invalid(cfg);
  }
  for (const Real value : {Real{-1}, inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.seed.r_center = value;
    expect_invalid(cfg);
    cfg = base_config(33, 33);
    cfg.seed.minor_radius = value;
    expect_invalid(cfg);
  }
  for (const Real value : {inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.seed.z_center = value;
    expect_invalid(cfg);
  }
  for (const Real value : {Real{0}, Real{-1}, inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.seed.depth = value;
    expect_invalid(cfg);
  }

  for (const Real value : {Real{0}, Real{-1}, inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.coils.front().r = value;
    expect_invalid(cfg);
  }
  for (const Real value : {inf, nan}) {
    GsConfig cfg = base_config(33, 33);
    cfg.coils.front().z = value;
    expect_invalid(cfg);
    cfg = base_config(33, 33);
    cfg.coils.front().current = value;
    expect_invalid(cfg);
  }

  EXPECT_THROW((void)EllipticGrid(33, 33, Real{0.3}, inf, Real{-0.8},
                                  Real{0.8}),
               std::invalid_argument);
  EXPECT_THROW((void)EllipticGrid(33, 33, Real{0.3}, Real{1.9}, nan,
                                  Real{0.8}),
               std::invalid_argument);

  EXPECT_THROW((void)PolynomialProfile(std::vector<Real>{nan},
                                       std::vector<Real>{Real{1}}),
               std::invalid_argument);
  EXPECT_THROW((void)PolynomialProfile(std::vector<Real>{Real{1}},
                                       std::vector<Real>{inf}),
               std::invalid_argument);

  // Below the two-sided Pade closure width, the sixth-order operator cannot be
  // formed at all.
  GsConfig tiny = base_config(33, 33);
  tiny.grid = EllipticGrid{9, 9, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  EXPECT_THROW(GsSolver(tiny, make_profile()), std::invalid_argument);
}

}  // namespace
