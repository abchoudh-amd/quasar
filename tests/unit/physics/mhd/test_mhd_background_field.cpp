// Static field-split (B = B0 + b) contract for the ideal-MHD solver.
//
// The solver may carry a STATIC uniform background magnetic field B0 and evolve
// only the perturbation b. The conserved `energy` slot stores the
// PERTURBATION-only magnetic energy (E = p/(gamma-1) + 0.5*rho|v|^2 + 0.5|b|^2);
// B0 enters the total magnetic pressure / fast speed / energy flux but is NEVER
// constrained-transport-updated. This test pins four observable consequences of
// that split (see the build plan "Behavioral acceptance criteria"):
//
//   1. Guide-field equivalence. A "split" solver (background.enabled=true with a
//      uniform B0, perturbation b + fluid seeded) reproduces, after one step at
//      the SAME dt, the conserved totals of an equivalent "unsplit" reference
//      solver (background.enabled=false, total field B0+b seeded directly into B,
//      same fluid). Density, momentum, gas pressure, and the TOTAL in-plane field
//      (split A's b + B0 vs reference B's b) agree to a loose physical tolerance.
//
//   2. CFL tightening. For the SAME seeded perturbation+fluid state, cfl_limit()
//      with a nonzero uniform B0 is strictly LESS than cfl_limit() with B0=0 (the
//      fast magnetosonic speed uses the TOTAL field B0+b, so |B| rises).
//
//   3. CT never touches B0 (checked indirectly, since the plan's MhdSolver2D API
//      exposes no background read-back accessor): with a div-free seed, after a
//      step divergence_b_max() (which measures div(B0+b)=div(b) because div(B0)=0)
//      stays at round-off; and a second identical run (re-seeding the same B0 and
//      state) reproduces the post-step state bit-for-bit (B0 is static, so the
//      update is deterministic and carries no hidden B0 evolution).
//
//   4. has_background() reflects the config (true when enabled, false otherwise).
//
// Conventions match tests/unit/physics/mhd/test_mhd_divergence_free.cpp:
// HIP-guarded skip, the same includes, a make_config(...) helper, a div-free seed
// built from a corner vector potential A_z, seed_state(...), and
// dt = 0.4 * cfl_limit(). Grids are kept small (32x32, nghost=4).
//
// Interface targeted (do NOT implement here; just call):
//   * quasar::mhd::MhdBackgroundSpec { bool enabled; std::string profile;
//                                      Real bx0,by0,bz0; }  (mhd_background.hpp)
//   * quasar::mhd::MhdConfig::background  (NEW field of the above type)
//   * MhdSolver2D::seed_background("b0x"/"b0y"/"b0z", host_buf)
//   * MhdSolver2D::has_background()
//   * existing MhdSolver2D::{seed_state, step, cfl_limit, divergence_b_max,
//                            state_component_to_host}

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;

constexpr Real kGamma = Real{5} / Real{3};
// A small uniform guide field keeps the split/unsplit equivalence clean.
constexpr Real kB0x = 0.02;
constexpr Real kB0y = 0.0;
constexpr Real kB0z = 0.0;

// Corner-staggered periodic vector potential A_z at the lower-left corner of cell
// (i,j): corner position is (origin_x + i*dx, origin_y + j*dy). Its discrete curl
// gives a non-trivial divergence-free perturbation field b.
Real Az_corner(const quasar::Grid2D& g, int i, int j) {
  const Real x = g.origin_x + static_cast<Real>(g.wrap_i(i)) * g.dx();
  const Real y = g.origin_y + static_cast<Real>(g.wrap_j(j)) * g.dy();
  return 0.05 * std::sin(2.0 * quasar::pi * x) * std::cos(2.0 * quasar::pi * y);
}

// Base periodic Cartesian config; background defaults to disabled.
quasar::mhd::MhdConfig make_config() {
  quasar::Grid2D g{32, 32, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.gamma = kGamma;
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

void set_all_boundaries(quasar::mhd::MhdConfig& cfg,
                        const std::string& name) {
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.fluid[side] = name;
    cfg.boundary.field[side] = name;
  }
}

// The smooth periodic perturbation field b (face-staggered, div-free) and the
// fluid (rho, momentum, gas pressure). Returned as host buffers so both the split
// and unsplit runs can be seeded from the SAME physical perturbation + fluid.
struct SeedFields {
  std::vector<Real> rho, mx, my, mz;
  std::vector<Real> bx, by, bz;   // perturbation b (div-free)
  std::vector<Real> p;            // gas pressure (for energy assembly)
  std::vector<Real> vx, vy, vz;   // velocity (for energy assembly)
};

SeedFields build_seed(const quasar::Grid2D& g) {
  const std::size_t n = g.storage_size();
  SeedFields s;
  s.rho.assign(n, 0.0);
  s.mx.assign(n, 0.0);
  s.my.assign(n, 0.0);
  s.mz.assign(n, 0.0);
  s.bx.assign(n, 0.0);
  s.by.assign(n, 0.0);
  s.bz.assign(n, 0.0);
  s.p.assign(n, 0.0);
  s.vx.assign(n, 0.0);
  s.vy.assign(n, 0.0);
  s.vz.assign(n, 0.0);

  const Real rho0 = 1.0, drho = 0.2;
  const Real p0 = 1.0, dp = 0.2;
  const Real cx = 0.5, cy = 0.5, sig = 0.15;
  const Real inv_dx = 1.0 / g.dx();
  const Real inv_dy = 1.0 / g.dy();

  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real x = g.x_at_cell_center(g.wrap_i(i));
      const Real y = g.y_at_cell_center(g.wrap_j(j));
      const Real r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const Real bump = std::exp(-r2 / (2.0 * sig * sig));
      const Real rho_ij = rho0 + drho * bump;
      const Real p_ij = p0 + dp * bump;
      const Real vx = 0.1 * std::sin(2.0 * quasar::pi * y);
      const Real vy = 0.1 * std::sin(2.0 * quasar::pi * x);
      const Real vz = 0.0;
      // Perturbation b = curl(A_z e_z) on the faces -> discretely div-free.
      const Real bxv = (Az_corner(g, i, j + 1) - Az_corner(g, i, j)) * inv_dy;
      const Real byv = -(Az_corner(g, i + 1, j) - Az_corner(g, i, j)) * inv_dx;
      const Real bzv = 0.0;

      s.rho[k] = rho_ij;
      s.vx[k] = vx;
      s.vy[k] = vy;
      s.vz[k] = vz;
      s.mx[k] = rho_ij * vx;
      s.my[k] = rho_ij * vy;
      s.mz[k] = rho_ij * vz;
      s.bx[k] = bxv;
      s.by[k] = byv;
      s.bz[k] = bzv;
      s.p[k] = p_ij;
    }
  }
  return s;
}

// Build the stored conserved energy from gas pressure, velocity, and the magnetic
// field that the run's `energy` convention uses: the SPLIT run stores
// 0.5*|b|^2 (perturbation-only); the UNSPLIT reference stores 0.5*|B0+b|^2 (total).
std::vector<Real> assemble_energy(const SeedFields& s, const quasar::Grid2D& g,
                                  bool include_b0) {
  const std::size_t n = g.storage_size();
  std::vector<Real> en(n, 0.0);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      // Energy is a cell average, while bx/by are staggered face values. Use the
      // same face-to-cell collocation as the solver EOS; using the low-face slot
      // directly gives split and unsplit seeds different gas pressures through
      // the B0.b cross term before either solver takes a step.
      const Real bx = quasar::mhd::cell_bx(g, s.bx.data(), i, j)
                    + (include_b0 ? kB0x : 0.0);
      const Real by = quasar::mhd::cell_by(g, s.by.data(), i, j)
                    + (include_b0 ? kB0y : 0.0);
      const Real bz = s.bz[k] + (include_b0 ? kB0z : 0.0);
      const Real kinetic = quasar::numerics::kinetic_from_velocity(
          s.rho[k], s.vx[k], s.vy[k], s.vz[k]);
      const Real magnetic = quasar::numerics::half_squared_norm3(bx, by, bz);
      en[k] = s.p[k] / (kGamma - 1.0) + kinetic + magnetic;
    }
  }
  return en;
}

// Seed the perturbation b + fluid into a SPLIT solver (energy = 0.5|b|^2 only),
// then seed the uniform background B0 via seed_background.
void seed_split_solver(quasar::mhd::MhdSolver2D& solver, const SeedFields& s,
                       const quasar::Grid2D& g) {
  solver.seed_state("rho", s.rho);
  solver.seed_state("mx", s.mx);
  solver.seed_state("my", s.my);
  solver.seed_state("mz", s.mz);
  solver.seed_state("energy", assemble_energy(s, g, /*include_b0=*/false));
  solver.seed_state("bx", s.bx);
  solver.seed_state("by", s.by);
  solver.seed_state("bz", s.bz);

  const std::size_t n = g.storage_size();
  solver.seed_background("b0x", std::vector<Real>(n, kB0x));
  solver.seed_background("b0y", std::vector<Real>(n, kB0y));
  solver.seed_background("b0z", std::vector<Real>(n, kB0z));
}

// Seed the TOTAL field B0+b + fluid into an UNSPLIT reference solver
// (energy = 0.5|B0+b|^2). No background is seeded (enabled=false).
void seed_unsplit_solver(quasar::mhd::MhdSolver2D& solver, const SeedFields& s,
                         const quasar::Grid2D& g) {
  const std::size_t n = g.storage_size();
  std::vector<Real> bx(n), by(n), bz(n);
  for (std::size_t k = 0; k < n; ++k) {
    bx[k] = s.bx[k] + kB0x;
    by[k] = s.by[k] + kB0y;
    bz[k] = s.bz[k] + kB0z;
  }
  solver.seed_state("rho", s.rho);
  solver.seed_state("mx", s.mx);
  solver.seed_state("my", s.my);
  solver.seed_state("mz", s.mz);
  solver.seed_state("energy", assemble_energy(s, g, /*include_b0=*/true));
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

// Recover gas pressure at cell k from a run's readback, given the magnetic field
// that run's `energy` convention stores (perturbation b for split + add B0 to get
// total; total field directly for unsplit). p = (g-1)*(E - 0.5 rho|v|^2 - 0.5|B_stored|^2)
// where B_stored is the field used to build the stored energy.
Real gas_pressure(Real rho, Real mx, Real my, Real mz, Real en, Real bx_stored,
                  Real by_stored, Real bz_stored) {
  const Real inv_rho = 1.0 / rho;
  const Real kinetic = 0.5 * inv_rho * (mx * mx + my * my + mz * mz);
  const Real magnetic = 0.5 * (bx_stored * bx_stored + by_stored * by_stored +
                               bz_stored * bz_stored);
  return (kGamma - 1.0) * (en - kinetic - magnetic);
}

Real max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b,
                  const quasar::Grid2D& g) {
  Real m = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      m = std::max(m, std::abs(a[k] - b[k]));
    }
  }
  return m;
}

void seed_uniform_split_state(quasar::mhd::MhdSolver2D& solver,
                              const quasar::Grid2D& g,
                              Real vx, Real vy, Real vz,
                              Real bx, Real by, Real bz) {
  const std::size_t n = g.storage_size();
  const std::vector<Real> rho(n, Real{1});
  const std::vector<Real> mx(n, vx);
  const std::vector<Real> my(n, vy);
  const std::vector<Real> mz(n, vz);
  const std::vector<Real> bx_values(n, bx);
  const std::vector<Real> by_values(n, by);
  const std::vector<Real> bz_values(n, bz);
  const Real energy = Real{1} / (kGamma - Real{1}) +
      quasar::numerics::kinetic_from_velocity(
          Real{1}, vx, vy, vz) +
      quasar::numerics::half_squared_norm3(bx, by, bz);
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", std::vector<Real>(n, energy));
  solver.seed_state("bx", bx_values);
  solver.seed_state("by", by_values);
  solver.seed_state("bz", bz_values);
}

quasar::mhd::MhdConfig dominant_linear_background_config(Real amplitude) {
  auto cfg = make_config();
  set_all_boundaries(cfg, "outflow");
  cfg.background.enabled = true;
  cfg.background.profile = "linear_vacuum";
  cfg.background.params = {
      {"gradient", amplitude}, {"shear", Real{0}}};
  return cfg;
}

}  // namespace

// has_background() must reflect the config: true when enabled, false otherwise.
TEST(MhdBackgroundField, HasBackgroundReflectsConfig) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg_off = make_config();
  cfg_off.background.enabled = false;
  quasar::mhd::MhdSolver2D solver_off{cfg_off};
  EXPECT_FALSE(solver_off.has_background());

  auto cfg_on = make_config();
  cfg_on.background.enabled = true;
  cfg_on.background.profile = "uniform";
  cfg_on.background.bx0 = kB0x;
  cfg_on.background.by0 = kB0y;
  cfg_on.background.bz0 = kB0z;
  quasar::mhd::MhdSolver2D solver_on{cfg_on};
  EXPECT_TRUE(solver_on.has_background());
}

// Turning on a nonzero uniform B0 lowers the stable dt for the SAME seeded
// perturbation+fluid state (the fast speed rises with |B0+b|).
TEST(MhdBackgroundField, NonzeroBackgroundTightensCfl) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto base = make_config();
  const auto seed = build_seed(base.grid);

  // Background OFF: stored energy uses perturbation b only; cfl uses |b|.
  auto cfg_off = make_config();
  cfg_off.background.enabled = false;
  quasar::mhd::MhdSolver2D solver_off{cfg_off};
  solver_off.seed_state("rho", seed.rho);
  solver_off.seed_state("mx", seed.mx);
  solver_off.seed_state("my", seed.my);
  solver_off.seed_state("mz", seed.mz);
  solver_off.seed_state("energy", assemble_energy(seed, cfg_off.grid, false));
  solver_off.seed_state("bx", seed.bx);
  solver_off.seed_state("by", seed.by);
  solver_off.seed_state("bz", seed.bz);
  const Real cfl_off = solver_off.cfl_limit();

  // Background ON: identical state + buffers, but a nonzero uniform B0 so the
  // fast speed uses the total field B0+b.
  auto cfg_on = make_config();
  cfg_on.background.enabled = true;
  cfg_on.background.profile = "uniform";
  cfg_on.background.bx0 = kB0x;
  cfg_on.background.by0 = kB0y;
  cfg_on.background.bz0 = kB0z;
  quasar::mhd::MhdSolver2D solver_on{cfg_on};
  seed_split_solver(solver_on, seed, cfg_on.grid);
  const Real cfl_on = solver_on.cfl_limit();

  EXPECT_GT(cfl_off, 0.0);
  EXPECT_TRUE(std::isfinite(cfl_off));
  EXPECT_TRUE(std::isfinite(cfl_on));
  // A larger total |B| => larger fast speed => strictly smaller stable dt.
  EXPECT_LT(cfl_on, cfl_off);
}

TEST(MhdBackgroundField, NativeConfigSamplesUniformBackgroundAtConstruction) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto seed = build_seed(make_config().grid);
  auto cfg_off = make_config();
  quasar::mhd::MhdSolver2D off{cfg_off};
  off.seed_state("rho", seed.rho);
  off.seed_state("mx", seed.mx);
  off.seed_state("my", seed.my);
  off.seed_state("mz", seed.mz);
  off.seed_state("energy", assemble_energy(seed, cfg_off.grid, false));
  off.seed_state("bx", seed.bx);
  off.seed_state("by", seed.by);
  off.seed_state("bz", seed.bz);

  auto cfg_on = make_config();
  cfg_on.background.enabled = true;
  cfg_on.background.profile = "uniform";
  cfg_on.background.bx0 = Real{0.8};
  quasar::mhd::MhdSolver2D on{cfg_on};
  // Seed only the evolved perturbation.  No seed_background() call is made:
  // bx0/by0/bz0 in the native configuration must already have populated B0.
  on.seed_state("rho", seed.rho);
  on.seed_state("mx", seed.mx);
  on.seed_state("my", seed.my);
  on.seed_state("mz", seed.mz);
  on.seed_state("energy", assemble_energy(seed, cfg_on.grid, false));
  on.seed_state("bx", seed.bx);
  on.seed_state("by", seed.by);
  on.seed_state("bz", seed.bz);

  const Real dt_off = off.cfl_limit();
  const Real dt_on = on.cfl_limit();
  ASSERT_TRUE(std::isfinite(dt_off));
  ASSERT_TRUE(std::isfinite(dt_on));
  EXPECT_LT(dt_on, dt_off);
}

TEST(MhdBackgroundField, NativeConfigAppliesGenericAnalyticProfileParameters) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto seed = build_seed(make_config().grid);
  auto seed_state_only = [&](quasar::mhd::MhdSolver2D& solver,
                             const quasar::Grid2D& g) {
    solver.seed_state("rho", seed.rho);
    solver.seed_state("mx", seed.mx);
    solver.seed_state("my", seed.my);
    solver.seed_state("mz", seed.mz);
    solver.seed_state("energy", assemble_energy(seed, g, false));
    solver.seed_state("bx", seed.bx);
    solver.seed_state("by", seed.by);
    solver.seed_state("bz", seed.bz);
  };

  auto zero_cfg = make_config();
  set_all_boundaries(zero_cfg, "outflow");
  zero_cfg.background.enabled = true;
  zero_cfg.background.profile = "linear_vacuum";
  zero_cfg.background.params = {{"gradient", Real{0}}, {"shear", Real{0}}};
  quasar::mhd::MhdSolver2D zero{zero_cfg};
  seed_state_only(zero, zero_cfg.grid);

  auto configured_cfg = make_config();
  set_all_boundaries(configured_cfg, "outflow");
  configured_cfg.background.enabled = true;
  configured_cfg.background.profile = "linear_vacuum";
  configured_cfg.background.params = {
      {"gradient", Real{1.25}}, {"shear", Real{-0.4}}};
  quasar::mhd::MhdSolver2D configured{configured_cfg};
  seed_state_only(configured, configured_cfg.grid);

  const Real dt_zero = zero.cfl_limit();
  const Real dt_configured = configured.cfl_limit();
  ASSERT_TRUE(std::isfinite(dt_zero));
  ASSERT_TRUE(std::isfinite(dt_configured));
  EXPECT_LT(dt_configured, dt_zero);
}

TEST(MhdBackgroundField, RejectsBackgroundIncompatibleWithPeriodicSeam) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = make_config();
  cfg.background.enabled = true;
  cfg.background.profile = "linear_vacuum";
  cfg.background.params = {
      {"gradient", Real{1.25}}, {"shear", Real{-0.4}}};
  EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
}

TEST(MhdBackgroundField, EnforcesWallNormalAndParityCompatibility) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto tangent = make_config();
  set_all_boundaries(tangent, "outflow");
  for (int side : {0, 1}) {
    tangent.boundary.fluid[side] = "wall";
    tangent.boundary.field[side] = "wall";
  }
  tangent.background.enabled = true;
  tangent.background.profile = "uniform";
  tangent.background.by0 = Real{0.3};
  tangent.background.bz0 = Real{-0.2};
  EXPECT_NO_THROW(quasar::mhd::MhdSolver2D{tangent});

  auto normal = tangent;
  normal.background.bx0 = Real{0.1};
  EXPECT_THROW(quasar::mhd::MhdSolver2D{normal}, std::invalid_argument);
}

TEST(MhdBackgroundField, EnforcesCylindricalAxisBackgroundParity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto axial = make_config();
  axial.geometry = "cylindrical";
  axial.reconstruction = "muscl_minmod";
  set_all_boundaries(axial, "outflow");
  axial.boundary.fluid[0] = "axis";
  axial.boundary.field[0] = "axis";
  axial.background.enabled = true;
  axial.background.profile = "uniform";
  axial.background.by0 = Real{0.3};
  EXPECT_NO_THROW(quasar::mhd::MhdSolver2D{axial});

  auto radial = axial;
  radial.background.bx0 = Real{0.1};
  EXPECT_THROW(quasar::mhd::MhdSolver2D{radial}, std::invalid_argument);

  auto toroidal = axial;
  toroidal.background.bz0 = Real{-0.2};
  EXPECT_THROW(quasar::mhd::MhdSolver2D{toroidal}, std::invalid_argument);
}

TEST(MhdBackgroundField, ExplicitSeedIsRevalidatedBeforeUse) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = make_config();
  set_all_boundaries(cfg, "outflow");
  cfg.background.enabled = true;
  cfg.background.profile = "uniform";
  quasar::mhd::MhdSolver2D solver{cfg};
  const auto seed = build_seed(cfg.grid);
  solver.seed_state("rho", seed.rho);
  solver.seed_state("mx", seed.mx);
  solver.seed_state("my", seed.my);
  solver.seed_state("mz", seed.mz);
  solver.seed_state("energy", assemble_energy(seed, cfg.grid, false));
  solver.seed_state("bx", seed.bx);
  solver.seed_state("by", seed.by);
  solver.seed_state("bz", seed.bz);

  const std::size_t n = cfg.grid.storage_size();
  std::vector<Real> b0x(n), b0y(n, Real{0}), b0z(n, Real{1e200});
  for (int j = -cfg.grid.nghost; j < cfg.grid.ny + cfg.grid.nghost; ++j) {
    for (int i = -cfg.grid.nghost; i < cfg.grid.nx + cfg.grid.nghost; ++i) {
      b0x[cfg.grid.index(i, j)] =
          cfg.grid.origin_x + static_cast<Real>(i) * cfg.grid.dx();
    }
  }
  solver.seed_background("b0x", b0x);
  solver.seed_background("b0y", b0y);
  solver.seed_background("b0z", b0z);

  // div(B0)=1. A huge toroidal component must not relax the in-plane
  // derivative-scaled tolerance, and the post-construction overwrite must be
  // rejected before a kernel consumes it.
  EXPECT_THROW((void)solver.cfl_limit(), std::invalid_argument);
}

TEST(MhdBackgroundField,
     DominantCurlFreeBackgroundHasNoSpuriousSelfForce) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr Real amplitude = Real{1e100};
  auto cfg = dominant_linear_background_config(amplitude);
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_split_state(
      solver, cfg.grid, Real{0}, Real{0}, Real{0},
      Real{0}, Real{0}, Real{0});

  const auto rho0 = solver.state_component_to_host("rho");
  const auto mx0 = solver.state_component_to_host("mx");
  const auto my0 = solver.state_component_to_host("my");
  const auto mz0 = solver.state_component_to_host("mz");
  const auto energy0 = solver.state_component_to_host("energy");
  const auto bx0 = solver.state_component_to_host("bx");
  const auto by0 = solver.state_component_to_host("by");
  const auto bz0 = solver.state_component_to_host("bz");

  // B0=(A*x,-A*y,0) is exactly curl-free and divergence-free in the staggered
  // linear stencil, hence -div(T0)=0. Separate O(A^2) directional reductions
  // would leave an O(epsilon*A^2) false force; this tiny physical step would
  // still amplify that defect far above a round-off-level state change.
  const Real dt = Real{1e-6} / amplitude;
  ASSERT_LT(dt, solver.cfl_limit());
  solver.step(dt);

  constexpr Real roundoff_tolerance = Real{2e-12};
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("rho"), rho0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("mx"), mx0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("my"), my0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("mz"), mz0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("energy"), energy0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("bx"), bx0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("by"), by0,
                         cfg.grid), roundoff_tolerance);
  EXPECT_LT(max_abs_diff(solver.state_component_to_host("bz"), bz0,
                         cfg.grid), roundoff_tolerance);
}

TEST(MhdBackgroundField,
     DominantBackgroundRetainsFusedSplitEnergySurvivor) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr Real amplitude = Real{1e100};
  auto cfg = dominant_linear_background_config(amplitude);
  quasar::mhd::MhdSolver2D solver{cfg};
  // b=(0,1,0), v=(0,1,0), rho=p=1. The stored energy is
  // E'=p/(gamma-1)+rho|v|^2/2+|b|^2/2, with no B0 baseline.
  seed_uniform_split_state(
      solver, cfg.grid, Real{0}, Real{1}, Real{0},
      Real{0}, Real{1}, Real{0});
  const auto energy0 = solver.state_component_to_host("energy");

  const Real dt = Real{1e-6} / amplitude;
  ASSERT_LT(dt, solver.cfl_limit());
  solver.step(dt);
  const auto energy1 = solver.state_component_to_host("energy");

  // The analytic initial split-energy rate is +A, hence Delta E'=+1e-6.
  // This is the finite survivor of O(A^2) face/FV/CT cancellations; a
  // sequential correction rounds it to zero long before the RK update.
  constexpr Real expected_increment = Real{1e-6};
  for (int j = 5; j < cfg.grid.ny - 5; ++j) {
    for (int i = 5; i < cfg.grid.nx - 5; ++i) {
      const std::size_t k = cfg.grid.index(i, j);
      EXPECT_NEAR(energy1[k] - energy0[k], expected_increment, Real{2e-10})
          << "i=" << i << " j=" << j;
    }
  }
}

TEST(MhdBackgroundField,
     DominantTangentialBackgroundRetainsGasPressureGradient) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr Real background = Real{1e100};
  auto cfg = make_config();
  cfg.reconstruction = "muscl_minmod";
  cfg.grid.nghost = 2;
  set_all_boundaries(cfg, "outflow");
  cfg.background.enabled = true;
  cfg.background.profile = "uniform";
  cfg.background.bz0 = background;
  const auto& g = cfg.grid;
  const std::size_t n = g.storage_size();

  std::vector<Real> rho(n, Real{1}), zero(n, Real{0});
  std::vector<Real> pressure(n), energy(n);
  std::vector<Real> bphi(n, Real{1});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      pressure[k] = Real{1} + g.x_at_cell_center(i);
      energy[k] = pressure[k] / (kGamma - Real{1}) + Real{0.5};
    }
  }

  quasar::mhd::MhdSolver2D solver{cfg};
  solver.seed_state("rho", rho);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", bphi);

  // B0 and b are uniform and purely tangential, so their Maxwell-stress
  // divergence is exactly zero. The only radial force is -dp/dx=-1. Although
  // each face F_x,mx contains the common O(B0*b)=1e100 cross stress, its
  // cancellation across adjacent faces must not erase the O(1) pressure
  // difference. Starting from mx=0 makes the tiny dt-sized response observable.
  const Real dt = Real{1e-6} / background;
  ASSERT_LT(dt, solver.cfl_limit());
  solver.step(dt);
  const auto mx = solver.state_component_to_host("mx");
  for (int j = 5; j < g.ny - 5; ++j) {
    for (int i = 5; i < g.nx - 5; ++i) {
      EXPECT_NEAR(mx[g.index(i, j)] / dt, Real{-1}, Real{2e-8})
          << "i=" << i << " j=" << j;
    }
  }
}

// Guide-field equivalence: a split run (b evolved over a static uniform B0) and an
// unsplit reference run (total field B0+b evolved directly) agree after one step
// at the SAME dt on density, momentum, gas pressure, and the TOTAL in-plane field.
TEST(MhdBackgroundField, GuideFieldEquivalenceAfterOneStep) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto base = make_config();
  const auto& g = base.grid;
  const auto seed = build_seed(g);

  auto cfg_split = make_config();
  cfg_split.background.enabled = true;
  cfg_split.background.profile = "uniform";
  cfg_split.background.bx0 = kB0x;
  cfg_split.background.by0 = kB0y;
  cfg_split.background.bz0 = kB0z;
  quasar::mhd::MhdSolver2D split{cfg_split};
  seed_split_solver(split, seed, g);

  auto cfg_ref = make_config();
  cfg_ref.background.enabled = false;
  quasar::mhd::MhdSolver2D ref{cfg_ref};
  seed_unsplit_solver(ref, seed, g);

  // Use a dt safe for BOTH runs (the unsplit run sees the larger total field).
  const Real dt = 0.4 * std::min(split.cfl_limit(), ref.cfl_limit());
  ASSERT_GT(dt, 0.0);
  split.step(dt);
  ref.step(dt);

  const auto rho_s = split.state_component_to_host("rho");
  const auto mx_s = split.state_component_to_host("mx");
  const auto my_s = split.state_component_to_host("my");
  const auto mz_s = split.state_component_to_host("mz");
  const auto en_s = split.state_component_to_host("energy");
  const auto bx_s = split.state_component_to_host("bx");
  const auto by_s = split.state_component_to_host("by");
  const auto bz_s = split.state_component_to_host("bz");

  const auto rho_r = ref.state_component_to_host("rho");
  const auto mx_r = ref.state_component_to_host("mx");
  const auto my_r = ref.state_component_to_host("my");
  const auto mz_r = ref.state_component_to_host("mz");
  const auto en_r = ref.state_component_to_host("energy");
  const auto bx_r = ref.state_component_to_host("bx");
  const auto by_r = ref.state_component_to_host("by");
  const auto bz_r = ref.state_component_to_host("bz");

  // Density and momentum agree directly. The two formulations reconstruct
  // different stored energy variables, so the comparison allows their smooth
  // one-step truncation difference while remaining small relative to O(1)
  // state magnitudes.
  const Real tol = 1e-4;
  EXPECT_LT(max_abs_diff(rho_s, rho_r, g), tol);
  EXPECT_LT(max_abs_diff(mx_s, mx_r, g), tol);
  EXPECT_LT(max_abs_diff(my_s, my_r, g), tol);
  EXPECT_LT(max_abs_diff(mz_s, mz_r, g), tol);

  // Total in-plane field: split (b + B0) vs reference (b == total). And gas
  // pressure recovered with each run's stored-energy convention.
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      // Total field comparison.
      EXPECT_NEAR(bx_s[k] + kB0x, bx_r[k], tol) << "i=" << i << " j=" << j;
      EXPECT_NEAR(by_s[k] + kB0y, by_r[k], tol) << "i=" << i << " j=" << j;
      EXPECT_NEAR(bz_s[k] + kB0z, bz_r[k], tol) << "i=" << i << " j=" << j;

      // Gas pressure: split stores 0.5|b|^2, ref stores 0.5|B0+b|^2.
      const Real p_split = gas_pressure(rho_s[k], mx_s[k], my_s[k], mz_s[k],
                                        en_s[k], bx_s[k], by_s[k], bz_s[k]);
      const Real p_ref = gas_pressure(rho_r[k], mx_r[k], my_r[k], mz_r[k],
                                      en_r[k], bx_r[k], by_r[k], bz_r[k]);
      EXPECT_NEAR(p_split, p_ref, tol) << "i=" << i << " j=" << j;
    }
  }
}

// CT never touches B0 (checked indirectly: the plan exposes no background
// read-back accessor). With a div-free seed, after a step the discrete div(B0+b)
// = div(b) stays at round-off; and a second identical run reproduces the post-step
// state bit-for-bit (the static-B0 update is deterministic and carries no hidden
// B0 evolution).
TEST(MhdBackgroundField, BackgroundStaticDivergenceAndDeterminism) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto base = make_config();
  const auto& g = base.grid;
  const auto seed = build_seed(g);

  auto cfg = make_config();
  cfg.background.enabled = true;
  cfg.background.profile = "uniform";
  cfg.background.bx0 = kB0x;
  cfg.background.by0 = kB0y;
  cfg.background.bz0 = kB0z;

  quasar::mhd::MhdSolver2D run_a{cfg};
  seed_split_solver(run_a, seed, g);

  // div(B0+b) = div(b) is round-off at the seed (b is curl of A_z; B0 uniform).
  EXPECT_LT(run_a.divergence_b_max(), 1e-9);

  const Real dt = 0.4 * run_a.cfl_limit();
  ASSERT_GT(dt, 0.0);
  run_a.step(dt);

  // A static discretely divergence-free B0 contributes exactly zero to the
  // divergence stencil, while CT keeps div(b) at round-off.
  EXPECT_LT(run_a.divergence_b_max(), 1e-9);

  const auto rho_a = run_a.state_component_to_host("rho");
  const auto en_a = run_a.state_component_to_host("energy");
  const auto bx_a = run_a.state_component_to_host("bx");
  const auto by_a = run_a.state_component_to_host("by");

  // A second identical run (same B0, same seed, same dt) reproduces the post-step
  // state exactly: the static background introduces no run-to-run drift.
  quasar::mhd::MhdSolver2D run_b{cfg};
  seed_split_solver(run_b, seed, g);
  run_b.step(dt);

  const auto rho_b = run_b.state_component_to_host("rho");
  const auto en_b = run_b.state_component_to_host("energy");
  const auto bx_b = run_b.state_component_to_host("bx");
  const auto by_b = run_b.state_component_to_host("by");

  EXPECT_EQ(max_abs_diff(rho_a, rho_b, g), 0.0);
  EXPECT_EQ(max_abs_diff(en_a, en_b, g), 0.0);
  EXPECT_EQ(max_abs_diff(bx_a, bx_b, g), 0.0);
  EXPECT_EQ(max_abs_diff(by_a, by_b, g), 0.0);
}

TEST(MhdBackgroundField, VariableLinearBackgroundMatchesUnsplitChangeOfVariables) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto split_cfg = make_config();
  split_cfg.background.enabled = true;
  split_cfg.background.profile = "linear_vacuum";
  auto ref_cfg = split_cfg;
  ref_cfg.background.enabled = false;
  for (int side = 0; side < 4; ++side) {
    split_cfg.boundary.fluid[side] = "outflow";
    split_cfg.boundary.field[side] = "outflow";
    ref_cfg.boundary.fluid[side] = "outflow";
    ref_cfg.boundary.field[side] = "outflow";
  }
  const auto& g = split_cfg.grid;
  const std::size_t n = g.storage_size();
  constexpr Real a = Real{0.12};
  constexpr Real b = Real{-0.07};
  constexpr Real bx_pert = Real{0.03};
  constexpr Real by_pert = Real{-0.02};
  constexpr Real bz_pert = Real{0.01};

  std::vector<Real> rho(n), mx(n), my(n), mz(n), pressure(n);
  std::vector<Real> bx(n, bx_pert), by(n, by_pert), bz(n, bz_pert);
  std::vector<Real> b0x(n), b0y(n), b0z(n, Real{0});
  std::vector<Real> total_bx(n), total_by(n), total_bz(n, bz_pert);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real xc = g.x_at_cell_center(i);
      const Real yc = g.y_at_cell_center(j);
      const Real xf = g.origin_x + static_cast<Real>(i) * g.dx();
      const Real yf = g.origin_y + static_cast<Real>(j) * g.dy();
      // B0=(a*x+b*y, b*x-a*y,0): div(B0)=curl(B0)=0 analytically and under the
      // face-difference CT divergence operator.
      b0x[k] = a * xf + b * yc;
      b0y[k] = b * xc - a * yf;
      total_bx[k] = b0x[k] + bx[k];
      total_by[k] = b0y[k] + by[k];
      rho[k] = Real{1} + Real{0.04} * xc;
      const Real vx = Real{0.11};
      const Real vy = Real{-0.08};
      const Real vz = Real{0.03};
      mx[k] = rho[k] * vx;
      my[k] = rho[k] * vy;
      mz[k] = rho[k] * vz;
      pressure[k] = Real{1} + Real{0.03} * yc;
    }
  }

  std::vector<Real> split_energy(n), total_energy(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real kinetic = quasar::numerics::kinetic_from_momentum(
          mx[k], my[k], mz[k], rho[k]);
      const Real bc_x = quasar::mhd::cell_bx(g, bx.data(), i, j);
      const Real bc_y = quasar::mhd::cell_by(g, by.data(), i, j);
      const Real tc_x = quasar::mhd::cell_bx(g, total_bx.data(), i, j);
      const Real tc_y = quasar::mhd::cell_by(g, total_by.data(), i, j);
      split_energy[k] = pressure[k] / (kGamma - Real{1}) + kinetic +
                        quasar::numerics::half_squared_norm3(
                            bc_x, bc_y, bz[k]);
      total_energy[k] = pressure[k] / (kGamma - Real{1}) + kinetic +
                        quasar::numerics::half_squared_norm3(
                            tc_x, tc_y, total_bz[k]);
    }
  }

  quasar::mhd::MhdSolver2D split{split_cfg};
  quasar::mhd::MhdSolver2D ref{ref_cfg};
  for (auto* solver : {&split, &ref}) {
    solver->seed_state("rho", rho);
    solver->seed_state("mx", mx);
    solver->seed_state("my", my);
    solver->seed_state("mz", mz);
  }
  split.seed_state("energy", split_energy);
  split.seed_state("bx", bx); split.seed_state("by", by); split.seed_state("bz", bz);
  split.seed_background("b0x", b0x);
  split.seed_background("b0y", b0y);
  split.seed_background("b0z", b0z);
  ref.seed_state("energy", total_energy);
  ref.seed_state("bx", total_bx);
  ref.seed_state("by", total_by);
  ref.seed_state("bz", total_bz);

  const Real dt = Real{0.02} * std::min(split.cfl_limit(), ref.cfl_limit());
  split.step(dt);
  ref.step(dt);

  const auto rho_s = split.state_component_to_host("rho");
  const auto mx_s = split.state_component_to_host("mx");
  const auto my_s = split.state_component_to_host("my");
  const auto mz_s = split.state_component_to_host("mz");
  const auto en_s = split.state_component_to_host("energy");
  const auto bx_s = split.state_component_to_host("bx");
  const auto by_s = split.state_component_to_host("by");
  const auto bz_s = split.state_component_to_host("bz");
  const auto rho_r = ref.state_component_to_host("rho");
  const auto mx_r = ref.state_component_to_host("mx");
  const auto my_r = ref.state_component_to_host("my");
  const auto mz_r = ref.state_component_to_host("mz");
  const auto en_r = ref.state_component_to_host("energy");
  const auto bx_r = ref.state_component_to_host("bx");
  const auto by_r = ref.state_component_to_host("by");
  const auto bz_r = ref.state_component_to_host("bz");

  constexpr Real tol = Real{3e-7};
  for (int j = 5; j < g.ny - 5; ++j) {
    for (int i = 5; i < g.nx - 5; ++i) {
      const std::size_t k = g.index(i, j);
      const Real b0xc = quasar::mhd::cell_bx(g, b0x.data(), i, j);
      const Real b0yc = quasar::mhd::cell_by(g, b0y.data(), i, j);
      EXPECT_NEAR(rho_s[k], rho_r[k], tol);
      EXPECT_NEAR(mx_s[k], mx_r[k], tol);
      EXPECT_NEAR(my_s[k], my_r[k], tol);
      EXPECT_NEAR(mz_s[k], mz_r[k], tol);
      EXPECT_NEAR(bx_s[k] + b0xc, bx_r[k], tol);
      EXPECT_NEAR(by_s[k] + b0yc, by_r[k], tol);
      EXPECT_NEAR(bz_s[k], bz_r[k], tol);

      const Real transformed_energy = en_s[k] + b0xc * bx_s[k] + b0yc * by_s[k] +
          quasar::numerics::half_squared_norm3(b0xc, b0yc, Real{0});
      EXPECT_NEAR(transformed_energy, en_r[k], tol);
    }
  }
}
