// Conservation + CFL-guard contract for the high-order ideal-MHD solver.
//
// A periodic Cartesian run has no boundary flux, so the discrete finite-volume
// scheme must conserve, to round-off, the volume integrals of the conserved
// variables: total mass (sum of rho), each momentum component (sum of mx/my/mz),
// and total energy (sum of energy), over the interior cells. We seed a smooth,
// non-trivial state (a Gaussian density+pressure bump on a uniform background, a
// small shear velocity, and a weak uniform B), advance several full RK steps, and
// require the interior sums to be unchanged within a tight tolerance.
//
// Conventions assumed of the public interface (documented so the blind
// implementer matches):
//   * MhdConfig.grid carries nghost >= 4 (mp7 reconstruction needs 4 ghost cells).
//   * seed_state(component, host_buf) takes a full storage-sized host vector
//     (length == grid.storage_size()) addressed by Grid2D::index(i,j); cell-
//     centered components are rho,mx,my,mz,energy,bz. Face components bx,by are
//     also seeded over the same storage layout (bx on the left x-face of cell i,
//     by on the lower y-face of cell j) — for a *uniform* B the face and centered
//     samples coincide, which is why a constant B is used here.
//   * state_component_to_host(component) returns a storage-sized host vector with
//     the current value of that component; we sum only over interior cells
//     [0,nx)x[0,ny).
//   * cfl_limit() returns the max stable dt for the currently-seeded state
//     (|v| + c_fast based); it is positive and finite.
//   * step(dt) with dt > cfl_limit() is rejected by throwing.
//
// Energy/momentum conservation only holds for an unlimited smooth flow; the seed
// is smooth and well-resolved so the positivity/troubled-cell limiter never
// trips, leaving the scheme in its conservative high-order branch.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quasar::Real;

// Build a periodic Cartesian config on a modest grid with enough ghost cells for
// the widest (mp7) reconstruction stencil.
quasar::mhd::MhdConfig make_periodic_config() {
  quasar::Grid2D g{32, 32, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.geometry = "cartesian";
  cfg.reconstruction = "mp7";
  cfg.riemann = "hlld";
  cfg.integrator = "ssprk3";
  cfg.ct = "fd_ct_christlieb";
  cfg.positivity = "troubled_cell";
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = "periodic";
    cfg.boundary.field[s] = "periodic";
  }
  return cfg;
}

// Seed a smooth, periodic, non-trivial MHD state into the solver. Returns nothing;
// fills every conserved component over full storage so ghost cells are also
// initialized (the periodic BC will overwrite them, but seeding them avoids NaNs).
void seed_smooth_state(quasar::mhd::MhdSolver2D& solver, const quasar::Grid2D& g) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, 0.0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);

  const Real gamma = 5.0 / 3.0;
  const Real rho0 = 1.0, drho = 0.3;       // smooth density bump
  const Real p0 = 1.0, dp = 0.3;           // smooth pressure bump
  const Real cx = 0.5, cy = 0.5, sig = 0.15;
  const Real B0 = 0.05;                     // weak uniform B (face == centered)

  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      // Use periodic-wrapped centers so the bump is smooth and periodic.
      const Real x = g.x_at_cell_center(g.wrap_i(i));
      const Real y = g.y_at_cell_center(g.wrap_j(j));
      const Real r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const Real bump = std::exp(-r2 / (2.0 * sig * sig));
      const Real rho_ij = rho0 + drho * bump;
      const Real p_ij = p0 + dp * bump;
      // Small smooth divergence-free-ish shear velocity (periodic).
      const Real vx = 0.1 * std::sin(2.0 * quasar::pi * y);
      const Real vy = 0.1 * std::sin(2.0 * quasar::pi * x);
      const Real vz = 0.0;
      const Real bxv = B0, byv = B0, bzv = 0.0;

      rho[k] = rho_ij;
      mx[k] = rho_ij * vx;
      my[k] = rho_ij * vy;
      mz[k] = rho_ij * vz;
      const Real kinetic = 0.5 * rho_ij * (vx * vx + vy * vy + vz * vz);
      const Real magnetic = 0.5 * (bxv * bxv + byv * byv + bzv * bzv);
      en[k] = p_ij / (gamma - 1.0) + kinetic + magnetic;
      bx[k] = bxv;
      by[k] = byv;
      bz[k] = bzv;
    }
  }

  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", en);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

// Interior sum of a storage-sized component vector.
Real interior_sum(const std::vector<Real>& comp, const quasar::Grid2D& g) {
  Real s = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      s += comp[g.index(i, j)];
    }
  }
  return s;
}

}  // namespace

// cfl_limit() is a pure-host computation over the seeded state, so we can probe
// it without a device; it must be positive and finite.
TEST(MhdSolverConservation, CflLimitIsPositiveFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_periodic_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth_state(solver, cfg.grid);

  const Real dt_max = solver.cfl_limit();
  EXPECT_GT(dt_max, 0.0);
  EXPECT_TRUE(std::isfinite(dt_max));
}

// step(dt) with dt larger than the CFL bound must be rejected.
TEST(MhdSolverConservation, OverCflStepThrows) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_periodic_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth_state(solver, cfg.grid);

  const Real dt_max = solver.cfl_limit();
  EXPECT_THROW(solver.step(2.0 * dt_max), std::exception);
}

// Periodic ⇒ no boundary flux ⇒ interior sums of conserved variables are
// invariant under the update, to round-off.
TEST(MhdSolverConservation, PeriodicConservesMassMomentumEnergy) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_periodic_config();
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth_state(solver, g);

  const Real mass0 = interior_sum(solver.state_component_to_host("rho"), g);
  const Real mx0 = interior_sum(solver.state_component_to_host("mx"), g);
  const Real my0 = interior_sum(solver.state_component_to_host("my"), g);
  const Real mz0 = interior_sum(solver.state_component_to_host("mz"), g);
  const Real en0 = interior_sum(solver.state_component_to_host("energy"), g);

  // Advance several full steps at a safe fraction of the CFL bound.
  const Real dt = 0.4 * solver.cfl_limit();
  for (int s = 0; s < 8; ++s) {
    solver.step(dt);
  }

  const Real mass1 = interior_sum(solver.state_component_to_host("rho"), g);
  const Real mx1 = interior_sum(solver.state_component_to_host("mx"), g);
  const Real my1 = interior_sum(solver.state_component_to_host("my"), g);
  const Real mz1 = interior_sum(solver.state_component_to_host("mz"), g);
  const Real en1 = interior_sum(solver.state_component_to_host("energy"), g);

  // Conservation is exact up to floating-point accumulation; scale the tolerance
  // by the magnitude of each conserved total.
  const Real tol_mass = 1e-10 * std::abs(mass0) + 1e-12;
  const Real tol_mom = 1e-10 * (std::abs(mass0)) + 1e-10;  // momenta near zero
  const Real tol_en = 1e-10 * std::abs(en0) + 1e-12;

  EXPECT_NEAR(mass1, mass0, tol_mass);
  EXPECT_NEAR(mx1, mx0, tol_mom);
  EXPECT_NEAR(my1, my0, tol_mom);
  EXPECT_NEAR(mz1, mz0, tol_mom);
  EXPECT_NEAR(en1, en0, tol_en);
}

// advance(t_end, dt) drives a sequence of steps to t_end and must conserve the
// same invariants as repeated step() calls.
TEST(MhdSolverConservation, AdvanceConservesMass) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_periodic_config();
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth_state(solver, g);

  const Real mass0 = interior_sum(solver.state_component_to_host("rho"), g);
  const Real dt = 0.4 * solver.cfl_limit();
  solver.advance(5.0 * dt, dt);
  const Real mass1 = interior_sum(solver.state_component_to_host("rho"), g);

  EXPECT_NEAR(mass1, mass0, 1e-10 * std::abs(mass0) + 1e-12);
}

TEST(MhdSolverConservation, RejectsNonfiniteTimeArguments) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_periodic_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth_state(solver, cfg.grid);
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real inf = std::numeric_limits<Real>::infinity();

  EXPECT_THROW(solver.step(nan), std::invalid_argument);
  EXPECT_THROW(solver.step_unchecked(nan), std::invalid_argument);
  EXPECT_THROW(solver.advance(nan, Real{1e-4}), std::invalid_argument);
  EXPECT_THROW(solver.advance(inf, Real{1e-4}), std::invalid_argument);
  EXPECT_THROW(solver.advance(Real{-1}, Real{1e-4}), std::invalid_argument);
  EXPECT_THROW(solver.advance(Real{1}, inf), std::invalid_argument);
}

TEST(MhdSolverConservation, AdvanceClipsExactlyToFinalStep) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_periodic_config();
  quasar::mhd::MhdSolver2D via_advance{cfg};
  quasar::mhd::MhdSolver2D via_step{cfg};
  seed_smooth_state(via_advance, cfg.grid);
  seed_smooth_state(via_step, cfg.grid);

  const Real clipped = Real{0.5} * via_advance.cfl_limit();
  via_advance.advance(clipped, Real{2} * clipped);
  via_step.step(clipped);
  EXPECT_EQ(via_advance.state_component_to_host("rho"),
            via_step.state_component_to_host("rho"));
  EXPECT_EQ(via_advance.state_component_to_host("energy"),
            via_step.state_component_to_host("energy"));
}
