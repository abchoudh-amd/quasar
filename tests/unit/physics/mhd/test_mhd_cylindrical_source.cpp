// Axisymmetric (r,z) ideal-MHD geometric-source contract.
//
// In cylindrical (r,z), m=0 geometry the conservation laws pick up geometric
// source terms from the divergence in curvilinear coordinates (e.g. the
// p_total/r and rho*v_phi^2/r radial source, and the B-tension source). The
// solver must (a) construct without throwing for a grid that includes the axis
// (origin_x == 0, i.e. r_min = 0), (b) hold a balanced equilibrium stationary to
// truncation order, (c) keep the CT divergence at machine epsilon in the
// cylindrical discretization, and (d) report a positive, finite cylindrical CFL
// built from the (dr, dz) spacing.
//
// EQUILIBRIUM SEEDED (documented so the implementer's geometric-source math
// targets the same state):
//   A uniform, static, force-free trivial equilibrium:
//     rho   = rho0           (uniform)
//     v     = 0              (no momentum)
//     p     = p0             (uniform ⇒ zero pressure gradient)
//     B_r   = 0, B_phi = 0, B_z = B0  (uniform axial field)
//   With v = 0, B_phi = 0, and dp/dr = 0, every radial geometric source term
//   vanishes identically: the p_total/r term is balanced by the divergence of the
//   uniform-pressure flux in the finite-volume cylindrical discretization, and a
//   uniform axial B_z carries no curvature source. Hence the state is a discrete
//   equilibrium and must remain stationary to truncation order after several
//   steps. This is the minimal seed that still exercises the cylindrical flux/
//   source path (non-zero B_z, non-zero pressure, axis column included) without
//   requiring the implementer to match a non-trivial radial profile.
//
// Staggering/conventions assumed (same as the Cartesian tests, with x≡r, y≡z):
//   * grid.origin_x == 0 places the i=0 left edge on the axis (r_at_edge(0)==0).
//   * seed_state writes full-storage host vectors indexed by Grid2D::index(i,j);
//     bx≡B_r on radial faces, by≡B_z on axial faces, bz_cell≡B_phi at centers.
//   * state_component_to_host reads components back over the same layout.
//   * divergence_b_max() uses the cylindrical discrete divergence; for the seeded
//     uniform field it is zero to round-off.
//   * cfl_limit() in cylindrical mode uses dr=grid.dx() and dz=grid.dy().

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;

constexpr Real kRho0 = 1.0;
constexpr Real kP0 = 0.5;
constexpr Real kB0 = 0.2;  // uniform axial B_z
constexpr Real kGamma = 5.0 / 3.0;

// (r,z) grid with the axis on the i=0 left edge: origin_x = 0 ⇒ r_at_edge(0)=0.
quasar::mhd::MhdConfig make_cyl_config() {
  quasar::Grid2D g{32, 32, /*lx=lr=*/1.0, /*ly=lz=*/1.0,
                   /*origin_x=r_min=*/0.0, /*origin_y=*/0.0, /*nghost=*/4};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.geometry = "cylindrical";
  cfg.reconstruction = "mp7";
  cfg.riemann = "hlld";
  cfg.integrator = "ssprk3";
  cfg.ct = "fd_ct_christlieb";
  cfg.positivity = "troubled_cell";
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = "outflow";
    cfg.boundary.field[s] = "outflow";
  }
  return cfg;
}

void seed_uniform_equilibrium(quasar::mhd::MhdSolver2D& solver,
                              const quasar::Grid2D& g) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, kRho0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, kB0), bz(n, 0.0);  // B_r=0, B_z=B0, B_phi=0

  const Real magnetic = 0.5 * kB0 * kB0;
  const Real e_cell = kP0 / (kGamma - 1.0) + magnetic;  // v=0 ⇒ no kinetic part
  for (std::size_t k = 0; k < n; ++k) en[k] = e_cell;

  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", en);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

Real max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b,
                  const quasar::Grid2D& g) {
  Real m = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      m = std::max(m, std::abs(a[g.index(i, j)] - b[g.index(i, j)]));
    }
  }
  return m;
}

}  // namespace

// Constructing a cylindrical solver with the axis included (origin_x == 0) must
// not throw — this is a pure-host construction probe.
TEST(MhdCylindricalSource, ConstructsOnAxisWithoutThrowing) {
  const auto cfg = make_cyl_config();
  EXPECT_NO_THROW({ quasar::mhd::MhdSolver2D solver{cfg}; });
}

// The cylindrical CFL limit must be positive and finite and derive from the
// (dr, dz) grid spacing.
TEST(MhdCylindricalSource, CylindricalCflIsPositiveFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_equilibrium(solver, cfg.grid);

  const Real dt_max = solver.cfl_limit();
  EXPECT_GT(dt_max, 0.0);
  EXPECT_TRUE(std::isfinite(dt_max));
}

// The uniform, radially-balanced equilibrium must stay stationary to truncation
// order: with v=0, B_phi=0 and dp/dr=0 the geometric source vanishes, so the
// state should barely move after several steps.
TEST(MhdCylindricalSource, BalancedEquilibriumStaysStationary) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_equilibrium(solver, g);

  const std::vector<Real> rho0 = solver.state_component_to_host("rho");
  const std::vector<Real> en0 = solver.state_component_to_host("energy");

  const Real dt = 0.4 * solver.cfl_limit();
  for (int s = 0; s < 6; ++s) solver.step(dt);

  const std::vector<Real> rho1 = solver.state_component_to_host("rho");
  const std::vector<Real> en1 = solver.state_component_to_host("energy");

  // Equilibrium ⇒ the state holds to truncation order. The seed is uniform, so a
  // correctly-cancelling geometric source leaves it essentially unchanged.
  EXPECT_LT(max_abs_diff(rho1, rho0, g), 1e-8);
  EXPECT_LT(max_abs_diff(en1, en0, g), 1e-8);
}

// CT divergence stays at machine epsilon in the cylindrical discretization for
// the uniform (divergence-free) field.
TEST(MhdCylindricalSource, DivergenceStaysAtMachineEpsilon) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_equilibrium(solver, cfg.grid);

  EXPECT_LT(solver.divergence_b_max(), 1e-10);

  const Real dt = 0.4 * solver.cfl_limit();
  for (int s = 0; s < 6; ++s) solver.step(dt);

  EXPECT_LT(solver.divergence_b_max(), 1e-9);
}
