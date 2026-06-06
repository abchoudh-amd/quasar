// RED-phase tests for the SSP-RK3 time integrator.
//
// Targets the blind contract in include/quasar/numerics/ssprk_integrator.hpp:
//
//   class ISsprkIntegrator {
//    public: virtual ~ISsprkIntegrator() = default;
//     virtual int  n_stages() const = 0;
//     virtual void advance(class quasar::mhd::MhdSolver2D& solver, Real dt) const = 0;
//   };
//
// Registry name "ssprk3" (3 stages), obtained via
//   quasar::Registry<quasar::numerics::ISsprkIntegrator>::instance().create("ssprk3").
//
// advance() is a thin loop over the solver's RK-register API
// (compute_residual + combine_stage). We exercise it through a minimally
// constructed MhdSolver2D (MhdConfig with a tiny periodic grid).
//
// ASSUMED CONTRACT (documented so the blind implementer matches it):
//   * quasar::mhd::MhdConfig { Grid2D grid; Real gamma; std::string geometry,
//       reconstruction, riemann, integrator, ct, positivity; Real rho_floor,
//       p_floor; quasar::boundary::MhdBoundarySpec boundary; } with the defaults
//       in the task contract; quasar::mhd::MhdSolver2D(MhdConfig).
//   * MhdSolver2D::seed_state("rho"|"mx"|"my"|"mz"|"energy"|"bx"|"by"|"bz",
//       host_buf) seeds the named conserved component from a host vector of
//       length grid.storage_size() (interior + ghosts; "bx"/"by" address the
//       face-staggered buffers).
//   * MhdSolver2D::state_component_to_host(name) returns that component as a
//       host std::vector<Real> of length grid.storage_size().
//   * MhdSolver2D::n_stages-equivalent register count via n_rk_registers().
//   * quasar::boundary::MhdBoundarySpec defaults all sides to "periodic", so a
//       default-constructed MhdConfig.boundary is a fully periodic problem.
//
// Device-touching assertions are guarded with has_hip_runtime() / GTEST_SKIP.
// The pure-host probe (n_stages / registry presence) fails by missing symbol.

#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/backend/device.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::ISsprkIntegrator;

std::unique_ptr<ISsprkIntegrator> make_integrator() {
  return quasar::Registry<ISsprkIntegrator>::instance().create("ssprk3");
}

// A tiny periodic solver with nghost>=4 (mp7 default reconstruction needs 4).
quasar::mhd::MhdConfig tiny_config() {
  quasar::mhd::MhdConfig cfg{};
  cfg.grid = Grid2D{16, 16, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  cfg.gamma = Real{5} / Real{3};
  // Default reconstruction "mp7"/riemann "hlld"/integrator "ssprk3"/ct/positivity
  // and a default (periodic) boundary are taken from the contract.
  return cfg;
}

// Seed a spatially-uniform, quiescent, positive conserved state across the whole
// storage (interior + ghosts). Such a state is a fixed point of the residual.
void seed_uniform(quasar::mhd::MhdSolver2D& solver, const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  const Real rho0 = Real{1.5}, p0 = Real{2.0};
  std::vector<Real> rho(n, rho0), zero(n, Real{0});
  std::vector<Real> en(n, p0 / (gamma - Real{1}));  // at rest, B = 0
  solver.seed_state("rho", rho);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", en);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);
}

Real max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b) {
  Real m = Real{0};
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t k = 0; k < n; ++k) m = std::max(m, std::abs(a[k] - b[k]));
  return m;
}

}  // namespace

TEST(MhdSsprk3Integrator, IsRegistered) {
  EXPECT_TRUE(quasar::Registry<ISsprkIntegrator>::instance().contains("ssprk3"));
}

TEST(MhdSsprk3Integrator, HasThreeStages) {
  EXPECT_EQ(make_integrator()->n_stages(), 3);
}

// A spatially-uniform constant state is a fixed point of the residual, so one
// SSP-RK3 advance must leave every conserved component unchanged to round-off.
TEST(MhdSsprk3Integrator, ConstantStateIsFixedPoint) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  auto cfg = tiny_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform(solver, cfg.grid, cfg.gamma);

  const auto rho_before = solver.state_component_to_host("rho");
  const auto en_before = solver.state_component_to_host("energy");

  auto integ = make_integrator();
  integ->advance(solver, /*dt=*/Real{1e-3});

  const auto rho_after = solver.state_component_to_host("rho");
  const auto en_after = solver.state_component_to_host("energy");

  EXPECT_LT(max_abs_diff(rho_before, rho_after), Real{1e-12});
  EXPECT_LT(max_abs_diff(en_before, en_after), Real{1e-12});
}

// SSP-RK3 is third-order accurate in time. On a smooth periodic problem the
// self-convergence error (vs a reference taken with a much smaller dt) shrinks
// like dt^3 under halving: the observed order across a halved-dt pair must be
// near 3. We integrate a smooth standing structure for a fixed short interval
// at dt and dt/2 and compare each to a fine reference; the error ratio gives the
// order. The state is built so the residual is non-trivial (a sinusoid in rho,
// pressure and a transverse momentum), exercising the full RK combination.
TEST(MhdSsprk3Integrator, ThirdOrderUnderTimestepRefinement) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Grid2D g{16, 16, Real{1}, Real{1}, Real{0}, Real{0}, 4};

  auto seed_smooth = [&](quasar::mhd::MhdSolver2D& s) {
    const std::size_t n = g.storage_size();
    std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
    std::vector<Real> en(n, Real{0}), zero(n, Real{0});
    for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
      for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
        const Real x = g.x_at_cell_center(i);
        const Real y = g.y_at_cell_center(j);
        const std::size_t k = g.index(i, j);
        const Real r = Real{2} + Real{0.3} * std::sin(Real{2} * M_PI * x);
        const Real p = Real{3} + Real{0.2} * std::cos(Real{2} * M_PI * y);
        const Real vx = Real{0.1} * std::sin(Real{2} * M_PI * (x + y));
        rho[k] = r;
        mx[k] = r * vx;
        en[k] = p / (gamma - Real{1}) + Real{0.5} * r * vx * vx;
      }
    }
    s.seed_state("rho", rho);
    s.seed_state("mx", mx);
    s.seed_state("my", my);
    s.seed_state("mz", mz);
    s.seed_state("energy", en);
    s.seed_state("bx", zero);
    s.seed_state("by", zero);
    s.seed_state("bz", zero);
  };

  auto integrate_n = [&](Real dt, int nsteps) {
    quasar::mhd::MhdConfig cfg{};
    cfg.grid = g;
    cfg.gamma = gamma;
    quasar::mhd::MhdSolver2D s{cfg};
    seed_smooth(s);
    auto integ = make_integrator();
    for (int n = 0; n < nsteps; ++n) integ->advance(s, dt);
    return s.state_component_to_host("rho");
  };

  // Fixed time interval T = 4e-3, integrated with three resolutions.
  const Real T = Real{4e-3};
  const Real dt1 = T / Real{4};   // 4 steps
  const Real dt2 = T / Real{8};   // 8 steps
  const Real dt_ref = T / Real{64};  // reference

  const auto u_ref = integrate_n(dt_ref, 64);
  const auto u1 = integrate_n(dt1, 4);
  const auto u2 = integrate_n(dt2, 8);

  const Real e1 = max_abs_diff(u1, u_ref);
  const Real e2 = max_abs_diff(u2, u_ref);
  ASSERT_GT(e1, Real{0});
  ASSERT_GT(e2, Real{0});
  const Real order = std::log2(e1 / e2);
  // A generous lower bound: 3rd-order temporal accuracy should give ~3; require
  // clearly above 2 to distinguish from a 2nd-order scheme.
  EXPECT_GT(order, Real{2.5}) << "observed temporal order " << order << " (design 3)";
}
