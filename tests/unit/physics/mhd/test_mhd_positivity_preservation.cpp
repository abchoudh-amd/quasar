// Solver-level positivity preservation through a step on a stiff near-floor MHD
// state for the conservative positivity controller.
//
// The seed is a LOW-PRESSURE (low plasma beta) Orszag-Tang-like vortex: a strong,
// sharp velocity shear plus a divergence-free field whose By ~ sin(4 pi x) is a
// marginally-resolved feature. As the vortex steepens into thin current sheets,
// high-order MP7 reconstruction can overshoot at interfaces and drive a stage
// outside the strictly positive set. A legacy cell-local pressure repair would
// re-derive energy upward and pump total energy. The solver-level positivity
// controller instead finds a
// per-cell convex admissible fraction theta in [0,1], rejects an inadmissible
// SSP-RK stage before it is accepted, and conservatively subcycles the requested
// interval at a smaller CFL fraction. No evolved cell is mass/energy-floored.
//
// This test pins the OBSERVABLE consequence on a stiff-but-initially-positive,
// divergence-free periodic problem:
//   (a) every interior cell keeps density > 0 and gas pressure > 0 (finite);
//   (b) total energy (sum of state energy over the interior) does NOT grow by
//       more than a small tolerance of its initial value -- no large per-step
//       energy injection;
//   (c) the field stays finite throughout (no NaN/Inf);
//   (d) div(B) stays at round-off (CT sanity check; the seed is div-free).
//
// The INITIAL state is unambiguously positive (uniform density, uniform low gas
// pressure, both well above the floors) and divergence-free (B = curl of a
// periodic vector potential A_z, so div B = 0 to round-off). The energy-growth
// assertion (b) ensures the conservative retry path does not introduce the
// mass/energy injection associated with a cell-local repair.
//
// All device-touching work is guarded by has_hip_runtime()/GTEST_SKIP.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Grid2D;
using quasar::Real;

// Test-only integrator that performs a complete, valid SSP-RK3 update and then
// reports an unrelated failure. The solver must roll the live state back and
// clear all positivity-controller overrides before propagating that exception.
class ThrowAfterFullStepIntegrator final
    : public quasar::numerics::ISsprkIntegrator {
 public:
  int n_stages() const override { return 3; }

  void advance(quasar::mhd::MhdSolver2D& solver, Real dt) const override {
    for (int stage = 0; stage < n_stages(); ++stage) {
      solver.compute_residual(solver.rk_register(stage),
                              solver.residual_register());
      solver.combine_stage(stage, dt);
    }
    // Model a failing numerical kernel that poisoned its scratch before the
    // host-side exception. Rollback must not evaluate 0*NaN through the RK
    // arithmetic path.
    std::vector<Real> poison(
        solver.grid().storage_size(), std::numeric_limits<Real>::quiet_NaN());
    solver.residual_register().rho.copy_from_host(poison.data(), poison.size());
    throw std::runtime_error{"test integrator failure after live-state update"};
  }
};

// Forces two positivity rejections so the controller shortens the requested
// interval, accepts one valid SSP-RK piece, then fails after updating the next
// piece. This distinguishes whole-request rollback from the ordinary
// per-piece backup (which by then contains the already-advanced state).
class ThrowAfterAcceptedPieceIntegrator final
    : public quasar::numerics::ISsprkIntegrator {
 public:
  int n_stages() const override { return 3; }

  void advance(quasar::mhd::MhdSolver2D& solver, Real dt) const override {
    ++calls_;
    if (calls_ <= 2) {
      const std::size_t n = solver.grid().storage_size();
      // stage 0 would map rho=1 to rho=-1. Both configured- and low-order
      // attempts reject it with theta near 1/2; the second rejection makes the
      // controller reduce `trial`, leaving a nonzero interval after the first
      // subsequent accepted call.
      const std::vector<Real> rho_rate(n, -Real{2} / dt);
      const std::vector<Real> zero(n, Real{0});
      auto& residual = solver.residual_register();
      residual.rho.copy_from_host(rho_rate.data(), n);
      residual.mx.copy_from_host(zero.data(), n);
      residual.my.copy_from_host(zero.data(), n);
      residual.mz.copy_from_host(zero.data(), n);
      residual.energy.copy_from_host(zero.data(), n);
      residual.bx_face.copy_from_host(zero.data(), n);
      residual.by_face.copy_from_host(zero.data(), n);
      residual.bz_cell.copy_from_host(zero.data(), n);
      solver.combine_stage(0, dt);  // throws the controller's retry signal
      throw std::logic_error{"test candidate unexpectedly remained positive"};
    }

    for (int stage = 0; stage < n_stages(); ++stage) {
      solver.compute_residual(solver.rk_register(stage),
                              solver.residual_register());
      solver.combine_stage(stage, dt);
    }
    if (calls_ >= 4) {
      throw std::runtime_error{
          "test integrator failure after an earlier accepted piece"};
    }
  }

 private:
  mutable int calls_{0};
};

// Completes eight public requests normally, then performs the ninth full update
// and throws. On the eighth request it models a one-ulp CT/RK storage residual
// through the integrator's internal state access. The same bytes therefore
// have solver-owned provenance until a mutable view escapes, giving the
// rollback regression a deterministic witness for the validation privilege.
class ThrowOnNinthRequestIntegrator final
    : public quasar::numerics::ISsprkIntegrator {
 public:
  int n_stages() const override { return 3; }

  void advance(quasar::mhd::MhdSolver2D& solver, Real dt) const override {
    for (int stage = 0; stage < n_stages(); ++stage) {
      solver.compute_residual(solver.rk_register(stage),
                              solver.residual_register());
      solver.combine_stage(stage, dt);
    }
    ++requests_;
    if (requests_ == 8) {
      auto& live = solver.rk_register(0);
      std::vector<Real> bx(live.grid.storage_size());
      live.bx_face.copy_to_host(bx.data(), bx.size());
      const std::size_t witness = live.grid.index(3, 4);
      bx[witness] = std::nextafter(
          bx[witness], std::numeric_limits<Real>::infinity());
      live.bx_face.copy_from_host(bx.data(), bx.size());
    }
    if (requests_ == 9) {
      throw std::runtime_error{
          "test integrator failure on ninth public request"};
    }
  }

 private:
  mutable int requests_{0};
};

quasar::mhd::MhdConfig stiff_config() {
  // Unit periodic domain on a modest 64x64 grid, nghost=4 to admit MP7.
  Grid2D g{64, 64, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.gamma = 5.0 / 3.0;
  cfg.geometry = "cartesian";
  cfg.reconstruction = "mp7";
  cfg.riemann = "hlld";
  cfg.integrator = "ssprk3";
  cfg.ct = "fd_ct_christlieb";
  cfg.positivity = "troubled_cell";
  cfg.cfl = 0.4;
  // Configured floors are explicit-repair thresholds. Automatic evolution
  // enforces strict positivity and the seed begins well above both thresholds.
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = "periodic";
    cfg.boundary.field[s] = "periodic";
  }
  return cfg;
}

// Corner-staggered periodic vector potential A_z at the lower-left corner of cell
// (i,j): corner position is (origin_x + i*dx, origin_y + j*dy). Its discrete curl
// gives a face-staggered B that is divergence-free to round-off by construction.
//
// This is the Orszag-Tang vortex potential A_z = (B0/4pi) cos(4 pi x)
// + (B0/2pi) cos(2 pi y), whose curl yields the canonical OT field
//   Bx = -B0 sin(2 pi y),  By = B0 sin(4 pi x).
// The sin(4 pi x) doubling makes By a sharp, marginally-resolved feature that
// MP7 overshoots once the vortex develops thin current sheets.
Real Az_corner(const Grid2D& g, int i, int j) {
  const Real x = g.origin_x + static_cast<Real>(g.wrap_i(i)) * g.dx();
  const Real y = g.origin_y + static_cast<Real>(g.wrap_j(j)) * g.dy();
  const Real b0 = 1.0 / std::sqrt(4.0 * quasar::pi);
  return b0 / (4.0 * quasar::pi) * std::cos(4.0 * quasar::pi * x) +
         b0 / (2.0 * quasar::pi) * std::cos(2.0 * quasar::pi * y);
}

// Seed a LOW-PRESSURE Orszag-Tang-like vortex: a strong, sharp velocity shear
// (the OT vortex flow) plus a divergence-free field whose By ~ sin(4 pi x) is a
// marginally-resolved feature. The gas pressure is set LOW (low plasma beta) so
// that as the vortex steepens into thin current sheets/shocks, MP7 interface
// reconstruction can make a stage inadmissible. The conservative controller
// must reject and retry that stage without locally changing mass or energy.
//
//   rho      = rho0                                 (constant, strictly positive)
//   vx       = -v0 sin(2 pi y)                       (OT vortex flow)
//   vy       =  v0 sin(2 pi x)
//   p        = p0                                    (LOW, low-beta)
//   B        = curl(A_z e_z)                         (OT field, div-free)
//
// rho is uniform so density positivity is trivially seeded; pressure is the
// restrictive admissibility condition as current sheets develop.
struct StiffSeed {
  Real rho_min{};  // smallest interior density seeded
  Real p_min{};    // smallest interior gas pressure seeded
};

StiffSeed seed_stiff(quasar::mhd::MhdSolver2D& solver, const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, 0.0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);

  const Real rho0 = 1.0;   // uniform density (positivity trivially seeded)
  const Real p0 = 0.04;    // LOW gas pressure (low plasma beta -> stiff)
  const Real v0 = 1.0;     // OT vortex flow amplitude (strong shear)

  const Real inv_dx = 1.0 / g.dx();
  const Real inv_dy = 1.0 / g.dy();

  Real rho_min = rho0;
  Real p_min = p0;

  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real x = g.x_at_cell_center(g.wrap_i(i));
      const Real y = g.y_at_cell_center(g.wrap_j(j));

      const Real rk = rho0;
      const Real pk = p0;
      const Real vx = -v0 * std::sin(2.0 * quasar::pi * y);
      const Real vy = v0 * std::sin(2.0 * quasar::pi * x);
      const Real vz = 0.0;

      // Divergence-free field from curl(A_z e_z) (Orszag-Tang field).
      const Real bxk = (Az_corner(g, i, j + 1) - Az_corner(g, i, j)) * inv_dy;
      const Real byk = -(Az_corner(g, i + 1, j) - Az_corner(g, i, j)) * inv_dx;
      const Real bzk = 0.0;

      rho[k] = rk;
      mx[k] = rk * vx;
      my[k] = rk * vy;
      mz[k] = rk * vz;
      const Real kinetic = 0.5 * rk * (vx * vx + vy * vy + vz * vz);
      // Magnetic energy is assembled in a second pass after all staggered faces
      // exist, using the same face-to-cell collocation as the solver EOS.
      en[k] = pk / (gamma - 1.0) + kinetic;
      bx[k] = bxk;
      by[k] = byk;
      bz[k] = bzk;

      if (i >= 0 && i < g.nx && j >= 0 && j < g.ny) {
        if (rk < rho_min) rho_min = rk;
        if (pk < p_min) p_min = pk;
      }
    }
  }

  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real bxc = quasar::mhd::cell_bx(g, bx.data(), i, j);
      const Real byc = quasar::mhd::cell_by(g, by.data(), i, j);
      en[k] += quasar::numerics::half_squared_norm3(bxc, byc, bz[k]);
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

  return StiffSeed{rho_min, p_min};
}

void seed_uniform_static_field(
    quasar::mhd::MhdSolver2D& solver, const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  const Real density = Real{1};
  const Real pressure = Real{1};
  const Real bx_value = Real{0.05};
  const Real by_value = Real{0.05};
  std::vector<Real> rho(n, density), mx(n, Real{0}), my(n, Real{0});
  std::vector<Real> mz(n, Real{0});
  std::vector<Real> energy(
      n, pressure / (gamma - Real{1})
          + Real{0.5} * (bx_value * bx_value + by_value * by_value));
  std::vector<Real> bx(n, bx_value), by(n, by_value), bz(n, Real{0});
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

bool all_finite(const std::vector<Real>& v) {
  for (const Real x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// Sum of total energy over interior cells from the staged "energy" component.
Real interior_energy_sum(const Grid2D& g, const std::vector<Real>& en) {
  Real sum = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      sum += en[g.index(i, j)];
    }
  }
  return sum;
}

}  // namespace

QUASAR_REGISTER_INTEGRATOR("test_throw_after_full_step",
                           ThrowAfterFullStepIntegrator)
QUASAR_REGISTER_INTEGRATOR("test_throw_after_accepted_piece",
                           ThrowAfterAcceptedPieceIntegrator)
QUASAR_REGISTER_INTEGRATOR("test_throw_on_ninth_request",
                           ThrowOnNinthRequestIntegrator)

// ---------------------------------------------------------------------------
// Stiff near-floor periodic MHD: after a sequence of high-order steps the state
// stays strictly positive and finite, div B stays at round-off, AND total energy
// is approximately conserved (no large per-step injection). The energy-growth
// bound is the load-bearing assertion the PP limiter must satisfy.
// ---------------------------------------------------------------------------
// Short robustness contract for the conservative update controller.
TEST(MhdPositivityPreservation, StiffStateStaysFiniteAndPositive) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = stiff_config();
  const Grid2D& g = cfg.grid;
  const Real gamma = cfg.gamma;

  quasar::mhd::MhdSolver2D solver{cfg};
  const StiffSeed seed = seed_stiff(solver, g, gamma);
  ASSERT_GT(seed.rho_min, cfg.rho_floor);
  ASSERT_GT(seed.p_min, cfg.p_floor);

  // A shorter run than the conservation test, still long enough to exercise
  // update rejection/subcycling in the developing current sheets.
  const int n_steps = 60;
  for (int s = 0; s < n_steps; ++s) {
    const Real dt = 0.4 * solver.cfl_limit();
    ASSERT_GT(dt, 0.0) << "step " << s << " produced a non-positive CFL dt";
    ASSERT_TRUE(std::isfinite(dt)) << "step " << s << " produced a non-finite dt";
    ASSERT_NO_THROW(solver.step(dt)) << "step " << s << " threw";
  }

  const std::vector<Real> rho = solver.state_component_to_host("rho");
  const std::vector<Real> mx = solver.state_component_to_host("mx");
  const std::vector<Real> my = solver.state_component_to_host("my");
  const std::vector<Real> mz = solver.state_component_to_host("mz");
  const std::vector<Real> en = solver.state_component_to_host("energy");
  const std::vector<Real> bx = solver.state_component_to_host("bx");
  const std::vector<Real> by = solver.state_component_to_host("by");
  const std::vector<Real> bz = solver.state_component_to_host("bz");

  EXPECT_TRUE(all_finite(rho) && all_finite(mx) && all_finite(my) &&
              all_finite(mz) && all_finite(en) && all_finite(bx) &&
              all_finite(by) && all_finite(bz))
      << "stiff run produced a non-finite component";

  Real min_rho = std::numeric_limits<Real>::infinity();
  Real min_p = std::numeric_limits<Real>::infinity();
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      quasar::numerics::MhdState u;
      u.rho = rho[k]; u.mx = mx[k]; u.my = my[k]; u.mz = mz[k];
      u.energy = en[k]; u.bx = bx[k]; u.by = by[k]; u.bz = bz[k];
      const Real p = quasar::numerics::pressure(u, gamma);
      if (rho[k] < min_rho) min_rho = rho[k];
      if (p < min_p) min_p = p;
    }
  }
  EXPECT_GT(min_rho, 0.0) << "an interior cell lost positivity of density";
  EXPECT_GT(min_p, 0.0) << "an interior cell lost positivity of gas pressure";

  const Real div1 = solver.divergence_b_max();
  EXPECT_LT(div1, 1e-9) << "FD-CT must hold div B at round-off through the run";
}

// Full conservative-update contract. Positivity comes from conservative SSP-RK
// retries/subcycling, never from changing one cell's mass or energy, so periodic
// total energy stays at flux-conservation error instead of being pumped.
TEST(MhdPositivityPreservation, StiffStateConservesEnergyAndStaysPositive) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = stiff_config();
  const Grid2D& g = cfg.grid;
  const Real gamma = cfg.gamma;

  quasar::mhd::MhdSolver2D solver{cfg};
  const StiffSeed seed = seed_stiff(solver, g, gamma);

  // The initial state is unambiguously separated from zero. These comparisons
  // characterize the seed; the configured thresholds are not evolution bounds.
  ASSERT_GT(seed.rho_min, cfg.rho_floor)
      << "seed density must start strictly above rho_floor";
  ASSERT_GT(seed.p_min, cfg.p_floor)
      << "seed pressure must start strictly above p_floor";

  // Seed is divergence-free to round-off (B = curl of periodic A_z).
  const Real div0 = solver.divergence_b_max();
  EXPECT_LT(div0, 1e-10) << "stiff seed must be divergence-free to round-off";

  // Initial interior total energy reference.
  const std::vector<Real> en0 = solver.state_component_to_host("energy");
  ASSERT_TRUE(all_finite(en0)) << "seed energy must be finite";
  const Real energy0 = interior_energy_sum(g, en0);
  ASSERT_GT(energy0, 0.0);

  // Adaptive step: recompute the stable dt = 0.4*cfl_limit() each step so the
  // run stays CFL-stable as the vortex steepens (the signal speed grows as the
  // current sheets form). This isolates conservative positivity behavior from a
  // trivial fixed-dt CFL overshoot.
  const Real dt0 = 0.4 * solver.cfl_limit();
  ASSERT_GT(dt0, 0.0) << "cfl_limit() must yield a positive step on a live state";

  const int n_steps = 250;
  for (int s = 0; s < n_steps; ++s) {
    const Real dt = 0.4 * solver.cfl_limit();
    ASSERT_GT(dt, 0.0) << "step " << s << " produced a non-positive CFL dt";
    ASSERT_TRUE(std::isfinite(dt)) << "step " << s << " produced a non-finite dt";
    ASSERT_NO_THROW(solver.step(dt)) << "step " << s << " threw";
  }

  // Stage everything needed to reconstruct interior gas pressure on the host.
  const std::vector<Real> rho = solver.state_component_to_host("rho");
  const std::vector<Real> mx = solver.state_component_to_host("mx");
  const std::vector<Real> my = solver.state_component_to_host("my");
  const std::vector<Real> mz = solver.state_component_to_host("mz");
  const std::vector<Real> en = solver.state_component_to_host("energy");
  const std::vector<Real> bx = solver.state_component_to_host("bx");
  const std::vector<Real> by = solver.state_component_to_host("by");
  const std::vector<Real> bz = solver.state_component_to_host("bz");

  // (c) Everything stayed finite.
  EXPECT_TRUE(all_finite(rho)) << "stiff run produced non-finite rho";
  EXPECT_TRUE(all_finite(mx)) << "stiff run produced non-finite mx";
  EXPECT_TRUE(all_finite(my)) << "stiff run produced non-finite my";
  EXPECT_TRUE(all_finite(mz)) << "stiff run produced non-finite mz";
  EXPECT_TRUE(all_finite(en)) << "stiff run produced non-finite energy";
  EXPECT_TRUE(all_finite(bx)) << "stiff run produced non-finite bx";
  EXPECT_TRUE(all_finite(by)) << "stiff run produced non-finite by";
  EXPECT_TRUE(all_finite(bz)) << "stiff run produced non-finite bz";

  // (a) Strict positivity of density and gas pressure on every interior cell.
  Real min_rho = std::numeric_limits<Real>::infinity();
  Real min_p = std::numeric_limits<Real>::infinity();
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      quasar::numerics::MhdState u;
      u.rho = rho[k];
      u.mx = mx[k];
      u.my = my[k];
      u.mz = mz[k];
      u.energy = en[k];
      u.bx = bx[k];
      u.by = by[k];
      u.bz = bz[k];
      const Real p = quasar::numerics::pressure(u, gamma);
      if (rho[k] < min_rho) min_rho = rho[k];
      if (p < min_p) min_p = p;
    }
  }
  EXPECT_GT(min_rho, 0.0) << "an interior cell lost positivity of density";
  EXPECT_GT(min_p, 0.0) << "an interior cell lost positivity of gas pressure";
  EXPECT_TRUE(std::isfinite(min_rho));
  EXPECT_TRUE(std::isfinite(min_p));

  // (d) div B stays at round-off (CT sanity check).
  const Real div1 = solver.divergence_b_max();
  EXPECT_LT(div1, 1e-9) << "FD-CT must hold div B at round-off through the run";

  // (b) Periodic total energy must not exhibit the large growth caused by a
  // nonconservative cell-local pressure repair.
  const Real energy1 = interior_energy_sum(g, en);
  const Real growth = (energy1 - energy0) / energy0;

  std::cout << "[diag] energy0=" << energy0 << " energy1=" << energy1
            << " growth=" << (growth * 100.0) << "%"
            << " min_rho=" << min_rho << " min_p=" << min_p
            << " div1=" << div1 << std::endl;

  EXPECT_LT(growth, 0.10)
      << "total energy grew by " << (growth * 100.0)
      << "% over " << n_steps
      << " steps -- a PP limiter must prevent floor-driven energy injection";
}

TEST(MhdPositivityPreservation,
     ConfiguredFloorIsNotTreatedAsAConservativeInvariant) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = stiff_config();
  cfg.grid = Grid2D{32, 8, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  const Grid2D& g = cfg.grid;
  const Real gamma = cfg.gamma;
  const Real p0 = Real{0.1};
  cfg.p_floor = p0;

  quasar::mhd::MhdSolver2D solver{cfg};
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{1}), mx(n), zero(n, Real{0}), en(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real x = g.x_at_cell_center(g.wrap_i(i));
      // dvx/dx is positive near x=0: the pressure derivative there points
      // below the configured p_floor immediately, even though p remains >0.
      const Real vx = Real{0.8} * std::sin(Real{2} * quasar::pi * x);
      mx[k] = rho[k] * vx;
      en[k] = p0 / (gamma - Real{1}) +
              quasar::numerics::kinetic_from_velocity(
                  rho[k], vx, Real{0}, Real{0});
    }
  }
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", en);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);

  const Real energy0 = interior_energy_sum(g, en);
  // Deliberately exceed the ordinary one-step CFL bound. The positivity
  // controller must split this requested interval conservatively rather than
  // freezing at the non-invariant configured pressure floor.
  const Real requested = Real{32} * solver.cfl_limit();
  ASSERT_NO_THROW(solver.step_unchecked(requested));
  EXPECT_GT(solver.last_positivity_substeps(), 1);

  const auto rho1 = solver.state_component_to_host("rho");
  const auto mx1 = solver.state_component_to_host("mx");
  const auto my1 = solver.state_component_to_host("my");
  const auto mz1 = solver.state_component_to_host("mz");
  const auto en1 = solver.state_component_to_host("energy");
  Real min_rho = std::numeric_limits<Real>::infinity();
  Real min_p = std::numeric_limits<Real>::infinity();
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      quasar::numerics::MhdState u{};
      u.rho = rho1[k]; u.mx = mx1[k]; u.my = my1[k]; u.mz = mz1[k];
      u.energy = en1[k];
      min_rho = std::min(min_rho, u.rho);
      min_p = std::min(min_p, quasar::numerics::pressure(u, gamma));
    }
  }
  EXPECT_GT(min_rho, Real{0});
  EXPECT_GT(min_p, Real{0});
  EXPECT_LT(min_p, cfg.p_floor)
      << "an expanding state at the configured floor must make finite progress";
  const Real energy1 = interior_energy_sum(g, en1);
  EXPECT_NEAR(energy1, energy0,
              Real{2e-11} * std::max(Real{1}, std::abs(energy0)))
      << "periodic total energy must remain conservative across subcycling";
}

TEST(MhdPositivityPreservation, UnrelatedIntegratorExceptionRollsBackState) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = stiff_config();
  cfg.integrator = "test_throw_after_full_step";
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_stiff(solver, cfg.grid, cfg.gamma);

  const std::vector<std::string> components = {
      "rho", "mx", "my", "mz", "energy", "bx_face", "by_face", "bz"};
  std::vector<std::vector<Real>> before;
  before.reserve(components.size());
  for (const auto& component : components) {
    before.push_back(solver.state_component_to_host(component));
  }

  const Real dt = Real{0.1} * solver.cfl_limit();
  EXPECT_THROW(solver.step_unchecked(dt), std::runtime_error);
  EXPECT_EQ(solver.last_positivity_substeps(), 0);

  const Grid2D& g = cfg.grid;
  for (std::size_t c = 0; c < components.size(); ++c) {
    const auto after = solver.state_component_to_host(components[c]);
    for (int j = 0; j < g.ny; ++j) {
      for (int i = 0; i < g.nx; ++i) {
        const std::size_t k = g.index(i, j);
        EXPECT_EQ(after[k], before[c][k])
            << "component=" << components[c] << " i=" << i << " j=" << j;
      }
    }
  }
}

TEST(MhdPositivityPreservation,
     ExceptionAfterAcceptedPieceRollsBackWholeRequestedInterval) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = stiff_config();
  cfg.grid = Grid2D{16, 16, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  cfg.integrator = "test_throw_after_accepted_piece";
  quasar::mhd::MhdSolver2D solver{cfg};

  const std::size_t n = cfg.grid.storage_size();
  const std::vector<Real> one(n, Real{1});
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> energy(
      n, Real{1} / (cfg.gamma - Real{1}));
  solver.seed_state("rho", one);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);

  const std::vector<std::string> components = {
      "rho", "mx", "my", "mz", "energy", "bx_face", "by_face", "bz"};
  std::vector<std::vector<Real>> before;
  for (const auto& component : components) {
    before.push_back(solver.state_component_to_host(component));
  }
  const int controller_before = solver.last_positivity_substeps();

  const Real requested = Real{0.2} * solver.cfl_limit();
  EXPECT_THROW(solver.step_unchecked(requested), std::runtime_error);
  EXPECT_EQ(solver.last_positivity_substeps(), controller_before);

  const Grid2D& g = cfg.grid;
  for (std::size_t c = 0; c < components.size(); ++c) {
    const auto after = solver.state_component_to_host(components[c]);
    for (int j = 0; j < g.ny; ++j) {
      for (int i = 0; i < g.nx; ++i) {
        const std::size_t k = g.index(i, j);
        EXPECT_EQ(after[k], before[c][k])
            << "component=" << components[c] << " i=" << i << " j=" << j;
      }
    }
  }
}

TEST(MhdPositivityPreservation,
     RollbackRestoresSolverOwnedSolenoidalityProvenance) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = stiff_config();
  cfg.grid = Grid2D{32, 32, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  cfg.integrator = "test_throw_on_ninth_request";
  quasar::mhd::MhdSolver2D solver{cfg};
  quasar::mhd::MhdSolver2D exposed_twin{cfg};
  seed_uniform_static_field(solver, cfg.grid, cfg.gamma);
  seed_uniform_static_field(exposed_twin, cfg.grid, cfg.gamma);

  const Real dt = Real{0.4} * solver.cfl_limit();
  for (int step = 0; step < 8; ++step) {
    ASSERT_NO_THROW(solver.step_unchecked(dt));
  }
  ASSERT_NO_THROW((void)solver.cfl_limit());

  const std::vector<std::string> components = {
      "rho", "mx", "my", "mz", "energy", "bx_face", "by_face", "bz"};
  std::vector<std::vector<Real>> before;
  for (const auto& component : components) {
    before.push_back(solver.state_component_to_host(component));
    exposed_twin.seed_state(component, before.back());
    ASSERT_EQ(exposed_twin.state_component_to_host(component), before.back());
  }

  // The eighth internally owned update leaves the controlled one-ulp storage
  // witness above. Copying those exact buffers through the external seed API
  // and then escaping a mutable view denies solver-owned provenance, proving
  // that the strict external-data predicate rejects the saved rollback bytes.
  (void)exposed_twin.state();
  ASSERT_THROW((void)exposed_twin.cfl_limit(), std::invalid_argument);

  EXPECT_THROW(solver.step_unchecked(dt), std::runtime_error);
  for (std::size_t c = 0; c < components.size(); ++c) {
    EXPECT_EQ(solver.state_component_to_host(components[c]), before[c]);
  }
  // The request-start state was solver-owned. Rollback must restore that
  // provenance transactionally along with the component buffers.
  EXPECT_NO_THROW((void)solver.cfl_limit());
}

// A captured rotor-stage stencil exposed a structural problem in the original
// retry path: selecting piecewise-constant reconstruction did not change the
// MP7 face-to-cell magnetic recovery, so the nominally first-order candidate
// could remain inadmissible solely because its eight-face quadrature overshot.
// Pin both the host recovery and the device positivity mode. The adjacent-face
// recovery is the bounded cell magnetic field used by the low-order HLL/CT
// operator, independent of the configured MP halo width.
TEST(MhdPositivityPreservation,
     LowOrderRetryUsesBoundedAdjacentFaceCollocation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g{12, 12, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  const std::size_t n = g.storage_size();
  constexpr int ci = 5;
  constexpr int cj = 6;
  constexpr Real gamma = Real{5} / Real{3};

  // Representative values from the first rejected rotor stage. The stored
  // energy is chosen a small, deterministic distance below the MP7 magnetic
  // energy threshold; the adjacent-face state retains an O(1e-1) internal
  // energy margin.
  constexpr Real rho0 = 7.8790776884;
  constexpr Real mx0 = 12.6919487609;
  constexpr Real my0 = 1.87738977855;
  constexpr Real bx_adjacent = 1.44368724614;
  constexpr Real by_adjacent = 1.18279804799;
  constexpr Real bx_mp7 = 1.49036572369;
  constexpr Real by_mp7 = 1.23153546530;

  std::vector<Real> bx(n, bx_adjacent);
  std::vector<Real> by(n, by_adjacent);
  // In the MP7 cell-average formula the (cell-1) face coefficient is
  // -9531/120960. Perturb only that non-bounding face so the two adjacent faces
  // remain exactly bx_adjacent/by_adjacent while the MP7 quadrature recovers the
  // captured overshoot.
  constexpr Real remote_coeff = -Real{9531} / Real{120960};
  bx[g.index(ci - 1, cj)] += (bx_mp7 - bx_adjacent) / remote_coeff;
  by[g.index(ci, cj - 1)] += (by_mp7 - by_adjacent) / remote_coeff;

  const Real kinetic = (mx0 * mx0 + my0 * my0) / (Real{2} * rho0);
  const Real magnetic_mp7 =
      Real{0.5} * (bx_mp7 * bx_mp7 + by_mp7 * by_mp7);
  const Real energy0 = kinetic + magnetic_mp7 - Real{1e-6};

  std::vector<Real> rho(n, rho0);
  std::vector<Real> mx(n, mx0);
  std::vector<Real> my(n, my0);
  std::vector<Real> zero(n, Real{0});
  std::vector<Real> energy(n, energy0);

  const auto high = quasar::mhd::load_cell_state(
      g, rho.data(), mx.data(), my.data(), zero.data(), energy.data(),
      bx.data(), by.data(), zero.data(), ci, cj,
      /*collocation_order=*/0);
  const auto low = quasar::mhd::load_cell_state(
      g, rho.data(), mx.data(), my.data(), zero.data(), energy.data(),
      bx.data(), by.data(), zero.data(), ci, cj,
      /*collocation_order=*/1);

  EXPECT_NEAR(high.bx, bx_mp7, Real{2e-13});
  EXPECT_NEAR(high.by, by_mp7, Real{2e-13});
  EXPECT_NEAR(low.bx, bx_adjacent, Real{2e-15});
  EXPECT_NEAR(low.by, by_adjacent, Real{2e-15});
  EXPECT_LT(quasar::numerics::pressure(high, gamma), Real{0});
  EXPECT_GT(quasar::numerics::pressure(low, gamma), Real{0.08});

  quasar::mhd::MhdField2D<Real> field{g};
  field.rho.copy_from_host(rho.data(), n);
  field.mx.copy_from_host(mx.data(), n);
  field.my.copy_from_host(my.data(), n);
  field.mz.copy_from_host(zero.data(), n);
  field.energy.copy_from_host(energy.data(), n);
  field.bx_face.copy_from_host(bx.data(), n);
  field.by_face.copy_from_host(by.data(), n);
  field.bz_cell.copy_from_host(zero.data(), n);

  auto limiter =
      quasar::Registry<quasar::numerics::IPositivityLimiter>::instance().create(
          "troubled_cell");
  ASSERT_NE(limiter, nullptr);
  EXPECT_EQ(limiter->admissible_fraction(
                field, field, Real{0}, Real{0}, gamma,
                /*collocation_order=*/0),
            Real{0});
  EXPECT_EQ(limiter->admissible_fraction(
                field, field, Real{0}, Real{0}, gamma,
                /*collocation_order=*/1),
            Real{1});
}
