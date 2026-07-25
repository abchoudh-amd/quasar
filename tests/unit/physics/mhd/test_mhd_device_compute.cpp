// Solver-level device-compute observables for the ideal-MHD slice (RED).
//
// Pins the post-port contract of the SOLVER reductions and the high-order
// end-to-end step, exercised only through public MhdSolver2D observables:
//   * cfl_limit() matches the four-incident-face finite-volume Courant
//     coefficient, including exact CT-normal face fields that cancel in a cell
//     average.
//   * Strict initial admissibility rejects pressureless or zero-density states
//     before they can masquerade as a zero CFL signal rate.
//   * divergence_b_max() stays at round-off through a FULL step(dt) for a
//     divergence-free seed (stable dt below cfl_limit()).
//   * A deck with reconstruction "mp7", positivity "troubled_cell", wall
//     boundaries runs one full step() with finite state and round-off div B.
//
// Seeds use CONSTANT rho / velocity / B so the host reference is exact and
// independent of how seed_state / state_component_to_host stage faces vs cells.
// (A uniform B is trivially divergence-free; a curl-of-A_z seed is used for the
// non-uniform divergence test.)

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;
using quasar::Grid2D;

quasar::mhd::MhdConfig base_config() {
  Grid2D g{16, 24, 1.0, 1.5, 0.0, 0.0, /*nghost=*/4};  // dx=1/16, dy=1.5/24=1/16
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
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = "periodic";
    cfg.boundary.field[s] = "periodic";
  }
  return cfg;
}

// Seed a uniform conserved MHD state from primitive constants.
void seed_uniform(quasar::mhd::MhdSolver2D& solver, const Grid2D& g, Real gamma,
                  Real rho, Real vx, Real vy, Real vz, Real p,
                  Real bx, Real by, Real bz) {
  const std::size_t n = g.storage_size();
  const Real kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
  const Real magnetic = 0.5 * (bx * bx + by * by + bz * bz);
  const Real en = p / (gamma - 1.0) + kinetic + magnetic;

  std::vector<Real> vrho(n, rho), vmx(n, rho * vx), vmy(n, rho * vy),
      vmz(n, rho * vz), ven(n, en);
  std::vector<Real> vbx(n, bx), vby(n, by), vbz(n, bz);

  solver.seed_state("rho", vrho);
  solver.seed_state("mx", vmx);
  solver.seed_state("my", vmy);
  solver.seed_state("mz", vmz);
  solver.seed_state("energy", ven);
  solver.seed_state("bx", vbx);
  solver.seed_state("by", vby);
  solver.seed_state("bz", vbz);
}

// Independent host reference for cfl_limit() over a UNIFORM state: the per-cell
// signal speed is identical everywhere, so the interior max collapses to the
// single-cell value over both directions.
Real host_cfl_limit_uniform(const Grid2D& g, Real cfl, Real gamma, Real rho,
                            Real vx, Real vy, Real p, Real bx, Real by, Real bz) {
  quasar::numerics::MhdPrim w;
  w.rho = rho; w.vx = vx; w.vy = vy; w.vz = 0.0; w.p = p;
  w.bx = bx; w.by = by; w.bz = bz;
  const quasar::numerics::MhdState u = quasar::numerics::to_conserved(w, gamma);

  const Real cfx = quasar::numerics::fast_magnetosonic_speed(u, /*dir=*/0, gamma);
  const Real cfy = quasar::numerics::fast_magnetosonic_speed(u, /*dir=*/1, gamma);
  // Additive multidimensional Courant rate (matches the unsplit residual): the
  // stable step is cfl / max_cells((|vx|+cfx)/dx + (|vy|+cfy)/dy). For a uniform
  // state every cell has the same rate, so the reduction is just this single sum.
  const Real rate = (std::abs(vx) + cfx) / g.dx() + (std::abs(vy) + cfy) / g.dy();
  return cfl / rate;
}

// Corner-staggered periodic A_z giving a discretely divergence-free face B.
Real Az_corner(const Grid2D& g, int i, int j) {
  const Real x = g.origin_x + static_cast<Real>(g.wrap_i(i)) * g.dx();
  const Real y = g.origin_y + static_cast<Real>(g.wrap_j(j)) * g.dy();
  return 0.1 * std::sin(2.0 * quasar::pi * x) * std::cos(2.0 * quasar::pi * y);
}

void seed_divergence_free(quasar::mhd::MhdSolver2D& solver, const Grid2D& g,
                          Real gamma) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, 1.0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);
  const Real p0 = 1.0;
  const Real inv_dx = 1.0 / g.dx();
  const Real inv_dy = 1.0 / g.dy();
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      bx[k] = (Az_corner(g, i, j + 1) - Az_corner(g, i, j)) * inv_dy;
      by[k] = -(Az_corner(g, i + 1, j) - Az_corner(g, i, j)) * inv_dx;
      bz[k] = 0.0;
      const Real magnetic = 0.5 * (bx[k] * bx[k] + by[k] * by[k]);
      rho[k] = 1.0;
      en[k] = p0 / (gamma - 1.0) + magnetic;
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

// Corner potential for a closed rectangular conducting wall.  The polynomial
// vanishes exactly (rather than only to sin(pi) roundoff) on all four physical
// edges, so its discrete curl has B.n == 0 on every wall face while retaining
// a non-trivial, exactly compatible interior magnetic field.
Real Az_wall_corner(const Grid2D& g, int i, int j) {
  const Real x = static_cast<Real>(i) / static_cast<Real>(g.nx);
  const Real y = static_cast<Real>(j) / static_cast<Real>(g.ny);
  const Real x_edge = x * (Real{1} - x);
  const Real y_edge = y * (Real{1} - y);
  return Real{0.1} * x_edge * x_edge * y_edge * y_edge;
}

void seed_wall_divergence_free(quasar::mhd::MhdSolver2D& solver,
                               const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{1}), zero(n, Real{0}), energy(n);
  std::vector<Real> bx(n), by(n), bz(n, Real{0});
  const Real inv_dx = Real{1} / g.dx();
  const Real inv_dy = Real{1} / g.dy();
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      bx[k] = (Az_wall_corner(g, i, j + 1) -
               Az_wall_corner(g, i, j)) * inv_dy;
      by[k] = -(Az_wall_corner(g, i + 1, j) -
                Az_wall_corner(g, i, j)) * inv_dx;
      const Real magnetic = Real{0.5} * (bx[k] * bx[k] + by[k] * by[k]);
      energy[k] = Real{1} / (gamma - Real{1}) + magnetic;
    }
  }
  solver.seed_state("rho", rho);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
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

}  // namespace

TEST(MhdConfigValidation, RejectsInvalidThermodynamicAndCflScalars) {
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real inf = std::numeric_limits<Real>::infinity();
  for (const Real bad : {Real{1}, Real{0.5}, nan, inf}) {
    auto cfg = base_config();
    cfg.gamma = bad;
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
  }
  for (const Real bad : {Real{0}, Real{-1}, nan, inf}) {
    auto rho = base_config();
    rho.rho_floor = bad;
    EXPECT_THROW(quasar::mhd::MhdSolver2D{rho}, std::invalid_argument);
    auto pressure = base_config();
    pressure.p_floor = bad;
    EXPECT_THROW(quasar::mhd::MhdSolver2D{pressure}, std::invalid_argument);
  }
  for (const Real bad : {Real{0}, Real{-0.1}, Real{1.01}, nan, inf}) {
    auto cfg = base_config();
    cfg.cfl = bad;
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
  }
}

TEST(MhdDeviceCompute, RkStageKeepsFiniteSurvivorOfOverflowingProducts) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g{2, 2, Real{1}, Real{1}, Real{0}, Real{0}, 2};
  quasar::mhd::MhdField2D<Real> un{g};
  quasar::mhd::MhdField2D<Real> stage{g};
  quasar::mhd::MhdField2D<Real> rate{g};
  quasar::mhd::MhdField2D<Real> out{g};
  const std::size_t n = g.storage_size();
  const std::vector<Real> positive(n, Real{1e308});
  const std::vector<Real> negative(n, Real{-1e308});
  const std::vector<Real> survivor(n, Real{3});

  auto fill = [n](quasar::mhd::MhdField2D<Real>& field,
                  const std::vector<Real>& values) {
    field.rho.copy_from_host(values.data(), n);
    field.mx.copy_from_host(values.data(), n);
    field.my.copy_from_host(values.data(), n);
    field.mz.copy_from_host(values.data(), n);
    field.energy.copy_from_host(values.data(), n);
    field.bx_face.copy_from_host(values.data(), n);
    field.by_face.copy_from_host(values.data(), n);
    field.bz_cell.copy_from_host(values.data(), n);
  };
  fill(un, positive);
  fill(stage, negative);
  fill(rate, survivor);

  // Each of 2*(+/-1e308) overflows when formed directly, but the opposing
  // products cancel and leave the representable residual contribution +3.
  quasar::mhd::launch_mhd_rk_stage(
      out, un, stage, rate, Real{2}, Real{2}, Real{1}, nullptr);
  quasar::backend::device_synchronize(nullptr);

  auto read = [n](const quasar::backend::DeviceBuffer<Real>& buffer) {
    std::vector<Real> values(n);
    buffer.copy_to_host(values.data(), n);
    return values;
  };
  const auto rho = read(out.rho);
  const auto mx = read(out.mx);
  const auto my = read(out.my);
  const auto mz = read(out.mz);
  const auto energy = read(out.energy);
  const auto bx = read(out.bx_face);
  const auto by = read(out.by_face);
  const auto bz = read(out.bz_cell);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      EXPECT_EQ(rho[k], Real{3});
      EXPECT_EQ(mx[k], Real{3});
      EXPECT_EQ(my[k], Real{3});
      EXPECT_EQ(mz[k], Real{3});
      EXPECT_EQ(energy[k], Real{3});
      EXPECT_EQ(bx[k], Real{3});
      EXPECT_EQ(by[k], Real{3});
      EXPECT_EQ(bz[k], Real{3});
    }
  }
  for (int j = 0; j < g.ny; ++j) {
    EXPECT_EQ(bx[g.index(g.nx, j)], Real{3});
  }
  for (int i = 0; i < g.nx; ++i) {
    EXPECT_EQ(by[g.index(i, g.ny)], Real{3});
  }
}

TEST(MhdDeviceCompute, RkStageDoesNotReadZeroWeightOperand) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g{2, 2, Real{1}, Real{1}, Real{0}, Real{0}, 2};
  quasar::mhd::MhdField2D<Real> un{g};
  quasar::mhd::MhdField2D<Real> unused_stage{g};
  quasar::mhd::MhdField2D<Real> rate{g};
  quasar::mhd::MhdField2D<Real> out{g};
  const std::size_t n = g.storage_size();
  const std::vector<Real> base(n, Real{2});
  const std::vector<Real> poison(
      n, std::numeric_limits<Real>::quiet_NaN());
  const std::vector<Real> increment(n, Real{3});

  auto fill = [n](quasar::mhd::MhdField2D<Real>& field,
                  const std::vector<Real>& values) {
    field.rho.copy_from_host(values.data(), n);
    field.mx.copy_from_host(values.data(), n);
    field.my.copy_from_host(values.data(), n);
    field.mz.copy_from_host(values.data(), n);
    field.energy.copy_from_host(values.data(), n);
    field.bx_face.copy_from_host(values.data(), n);
    field.by_face.copy_from_host(values.data(), n);
    field.bz_cell.copy_from_host(values.data(), n);
  };
  fill(un, base);
  fill(unused_stage, poison);
  fill(rate, increment);

  // SSP-RK stage zero has weights (1,0,dt). A rejected prior attempt may have
  // poisoned the reused stage register, but its exactly-zero coefficient makes
  // that operand mathematically inactive; 0*NaN must never enter the result.
  quasar::mhd::launch_mhd_rk_stage(
      out, un, unused_stage, rate, Real{1}, Real{0}, Real{1}, nullptr);
  quasar::backend::device_synchronize(nullptr);

  auto expect_five = [n, &g](
                         const quasar::backend::DeviceBuffer<Real>& buffer,
                         bool bx_faces, bool by_faces) {
    std::vector<Real> values(n);
    buffer.copy_to_host(values.data(), n);
    const int i_hi = bx_faces ? g.nx + 1 : g.nx;
    const int j_hi = by_faces ? g.ny + 1 : g.ny;
    for (int j = 0; j < j_hi; ++j) {
      for (int i = 0; i < i_hi; ++i) {
        EXPECT_EQ(values[g.index(i, j)], Real{5})
            << "i=" << i << " j=" << j;
      }
    }
  };
  expect_five(out.rho, false, false);
  expect_five(out.mx, false, false);
  expect_five(out.my, false, false);
  expect_five(out.mz, false, false);
  expect_five(out.energy, false, false);
  expect_five(out.bz_cell, false, false);
  expect_five(out.bx_face, true, false);
  expect_five(out.by_face, false, true);
}

TEST(MhdConfigValidation, RejectsInvalidBackgroundSpecification) {
  auto unknown = base_config();
  unknown.background.enabled = true;
  unknown.background.profile = "not_registered";
  EXPECT_THROW(quasar::mhd::MhdSolver2D{unknown}, std::invalid_argument);

  auto nonfinite = base_config();
  nonfinite.background.enabled = true;
  nonfinite.background.profile = "uniform";
  nonfinite.background.bx0 = std::numeric_limits<Real>::infinity();
  EXPECT_THROW(quasar::mhd::MhdSolver2D{nonfinite}, std::invalid_argument);

  auto mismatched = base_config();
  mismatched.background.enabled = true;
  mismatched.background.profile = "linear_vacuum";
  mismatched.background.bx0 = Real{1};
  EXPECT_THROW(quasar::mhd::MhdSolver2D{mismatched}, std::invalid_argument);

  auto nonfinite_param = base_config();
  nonfinite_param.background.enabled = true;
  nonfinite_param.background.profile = "linear_vacuum";
  nonfinite_param.background.params["gradient"] =
      std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW(quasar::mhd::MhdSolver2D{nonfinite_param}, std::invalid_argument);

  auto unknown_param = base_config();
  unknown_param.background.enabled = true;
  unknown_param.background.profile = "linear_vacuum";
  unknown_param.background.params["not_a_parameter"] = Real{1};
  EXPECT_THROW(quasar::mhd::MhdSolver2D{unknown_param}, std::invalid_argument);
}

TEST(MhdConfigValidation, RejectsUnknownSchemesAndBoundaryTopology) {
  {
    auto cfg = base_config();
    cfg.integrator = "not_registered";
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
  }
  {
    auto cfg = base_config();
    cfg.positivity = "not_registered";
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
  }
  {
    auto cfg = base_config();
    cfg.boundary.fluid[0] = "not_registered";
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
  }
  {
    auto cfg = base_config();
    cfg.boundary.fluid[1] = "outflow";
    cfg.boundary.field[1] = "outflow";
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument)
        << "a periodic axis must wrap at both ends";
  }
  {
    auto cfg = base_config();
    cfg.boundary.field[0] = "outflow";
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument)
        << "fluid and field periodicity must agree per side";
  }
  {
    auto cfg = base_config();
    cfg.geometry = "cylindrical";
    EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument)
        << "cylindrical MP7 must be rejected before it can advertise Cartesian order";
  }
}

TEST(MhdConfigValidation, RejectsAnnulusWhoseReconstructionHaloCrossesAxis) {
  auto cfg = base_config();
  cfg.geometry = "cylindrical";
  cfg.reconstruction = "muscl_minmod";
  cfg.grid.origin_x = Real{0.1};
  cfg.boundary.fluid[0] = "wall";
  cfg.boundary.field[0] = "wall";
  ASSERT_LE(cfg.grid.origin_x - Real{4} * cfg.grid.dx(), Real{0});
  EXPECT_THROW(quasar::mhd::MhdSolver2D{cfg}, std::invalid_argument);
}

TEST(MhdDeviceCompute, MagneticReadbackUsesOrderMatchedCellCollocation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  const Grid2D& g = cfg.grid;
  const std::size_t n = g.storage_size();
  std::vector<Real> bx_face(n), by_face(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real x = static_cast<Real>(i);
      const Real y = static_cast<Real>(j);
      bx_face[k] = Real{1} + x + Real{0.1} * x * x * x;
      by_face[k] = Real{-2} + y - Real{0.05} * y * y * y;
    }
  }

  quasar::mhd::MhdSolver2D solver{cfg};
  solver.seed_state("bx_face", bx_face);
  solver.seed_state("by_face", by_face);
  const auto raw_bx = solver.state_component_to_host("bx_face");
  const auto raw_by = solver.state_component_to_host("by_face");
  const auto cell_bx = solver.state_component_to_host("bx");
  const auto cell_by = solver.state_component_to_host("by");
  EXPECT_EQ(raw_bx, bx_face);
  EXPECT_EQ(raw_by, by_face);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      EXPECT_DOUBLE_EQ(cell_bx[k],
                       quasar::mhd::cell_bx(g, bx_face.data(), i, j));
      EXPECT_DOUBLE_EQ(cell_by[k],
                       quasar::mhd::cell_by(g, by_face.data(), i, j));
    }
  }
}

// ---------------------------------------------------------------------------
// cfl_limit() matches an independent host computation over both directions for a
// non-trivial uniform state with nonzero velocity AND nonzero B.
// ---------------------------------------------------------------------------
TEST(MhdDeviceCompute, CflLimitMatchesHostReductionBothDirections) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  const Grid2D& g = cfg.grid;
  const Real gamma = cfg.gamma;

  // Non-trivial flow: anisotropic velocity and an anisotropic field so the two
  // directional speeds differ and the max() over directions is exercised.
  const Real rho = 1.3, vx = 0.7, vy = -0.4, vz = 0.0, p = 0.9;
  const Real bx = 0.5, by = 0.2, bz = 0.3;

  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform(solver, g, gamma, rho, vx, vy, vz, p, bx, by, bz);

  const Real got = solver.cfl_limit();
  const Real ref =
      host_cfl_limit_uniform(
          g, std::min(cfg.cfl, Real{0.2}), gamma,
          rho, vx, vy, p, bx, by, bz);

  ASSERT_TRUE(std::isfinite(got));
  EXPECT_NEAR(got, ref, 1e-10 * ref)
      << "cfl_limit() must equal cfl / ((|vx|+c_fast,x)/dx + (|vy|+c_fast,y)/dy)";
}

TEST(MhdDeviceCompute, MpCourantFactorSaturatesAtMonotonicityBound) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto capped_cfg = base_config();
  capped_cfg.cfl = Real{0.2};
  auto larger_cfg = capped_cfg;
  larger_cfg.cfl = Real{0.9};
  auto smaller_cfg = capped_cfg;
  smaller_cfg.cfl = Real{0.1};

  quasar::mhd::MhdSolver2D capped{capped_cfg};
  quasar::mhd::MhdSolver2D larger{larger_cfg};
  quasar::mhd::MhdSolver2D smaller{smaller_cfg};
  for (auto* solver : {&capped, &larger, &smaller}) {
    seed_uniform(*solver, capped_cfg.grid, capped_cfg.gamma,
                 Real{1}, Real{0.3}, Real{-0.2}, Real{0.1}, Real{1},
                 Real{0.4}, Real{-0.1}, Real{0.2});
  }

  const Real capped_dt = capped.cfl_limit();
  EXPECT_DOUBLE_EQ(larger.cfl_limit(), capped_dt);
  EXPECT_NEAR(smaller.cfl_limit(), Real{0.5} * capped_dt,
              Real{4} * std::numeric_limits<Real>::epsilon() * capped_dt);
}

TEST(MhdDeviceCompute, CflUsesReconstructedMpFaceSpeed) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  cfg.cfl = Real{0.2};
  const Grid2D& g = cfg.grid;
  const std::size_t n = g.storage_size();
  constexpr Real rho0 = Real{1};
  constexpr Real p0 = Real{1};
  constexpr Real amplitude = Real{10};
  const Real sound = std::sqrt(cfg.gamma * p0 / rho0);
  std::vector<Real> rho(n, rho0), mx(n), zero(n, Real{0}), energy(n);
  Real max_cell_speed = Real{0};
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const int wrapped_i = g.wrap_i(i);
      const Real x = (static_cast<Real>(wrapped_i) + Real{0.5}) /
                     static_cast<Real>(g.nx);
      const Real vx = amplitude * std::cos(Real{2} * quasar::pi * x);
      const std::size_t k = g.index(i, j);
      mx[k] = rho0 * vx;
      energy[k] = p0 / (cfg.gamma - Real{1}) +
                  Real{0.5} * rho0 * vx * vx;
      max_cell_speed = std::max(max_cell_speed, std::abs(vx));
    }
  }

  quasar::mhd::MhdSolver2D solver{cfg};
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);

  // Independently expose the reconstructed states and reduce the same physical
  // face speeds on the host.  A single faster face does not necessarily make
  // dt smaller than a cell-only bound: the finite-volume LF self-weight is the
  // half-sum of both incident faces, and the opposite face can be slower.  The
  // old one-face comparison therefore tested two different coefficients.
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::mhd::MhdBackgroundField<Real> inactive_background{};
  quasar::mhd::BoundaryFlags4 periodic_flags{};
  quasar::mhd::launch_mhd_reconstruct(
      solver.state(), inactive_background, /*dir=*/0, ifx,
      /*scheme_order=*/7, periodic_flags, cfg.gamma, nullptr);
  quasar::backend::device_synchronize(nullptr);

  auto copy = [](const quasar::backend::DeviceBuffer<Real>& buffer) {
    std::vector<Real> values(buffer.size());
    buffer.copy_to_host(values.data(), values.size());
    return values;
  };
  const auto lrho = copy(ifx.Lrho);
  const auto lmx = copy(ifx.Lmx);
  const auto lmy = copy(ifx.Lmy);
  const auto lmz = copy(ifx.Lmz);
  const auto lenergy = copy(ifx.Lenergy);
  const auto lbx = copy(ifx.Lbx);
  const auto lby = copy(ifx.Lby);
  const auto lbz = copy(ifx.Lbz);
  const auto rrho = copy(ifx.Rrho);
  const auto rmx = copy(ifx.Rmx);
  const auto rmy = copy(ifx.Rmy);
  const auto rmz = copy(ifx.Rmz);
  const auto renergy = copy(ifx.Renergy);
  const auto rbx = copy(ifx.Rbx);
  const auto rby = copy(ifx.Rby);
  const auto rbz = copy(ifx.Rbz);

  auto side_alpha = [&](int face, bool left) {
    const std::size_t k = g.index(face, 0);
    quasar::numerics::MhdState state;
    state.rho = left ? lrho[k] : rrho[k];
    state.mx = left ? lmx[k] : rmx[k];
    state.my = left ? lmy[k] : rmy[k];
    state.mz = left ? lmz[k] : rmz[k];
    state.energy = left ? lenergy[k] : renergy[k];
    state.bx = left ? lbx[k] : rbx[k];
    state.by = left ? lby[k] : rby[k];
    state.bz = left ? lbz[k] : rbz[k];
    return std::abs(state.mx / state.rho) +
        quasar::numerics::fast_magnetosonic_speed(
            state, /*dir=*/0, cfg.gamma);
  };

  std::vector<Real> face_alpha(static_cast<std::size_t>(g.nx));
  Real max_face_alpha = Real{0};
  for (int face = 0; face < g.nx; ++face) {
    face_alpha[static_cast<std::size_t>(face)] =
        std::max(side_alpha(face, true), side_alpha(face, false));
    max_face_alpha = std::max(
        max_face_alpha, face_alpha[static_cast<std::size_t>(face)]);
  }
  EXPECT_GT(max_face_alpha,
            Real{1.001} * (max_cell_speed + sound))
      << "MP reconstruction must expose the faster face used by the CFL path";

  Real max_reconstructed_rate = Real{0};
  for (int i = 0; i < g.nx; ++i) {
    const Real alpha_lo = face_alpha[static_cast<std::size_t>(i)];
    const Real alpha_hi =
        face_alpha[static_cast<std::size_t>((i + 1) % g.nx)];
    const Real rate = Real{0.5} * (alpha_lo + alpha_hi) / g.dx() +
                      sound / g.dy();
    max_reconstructed_rate = std::max(max_reconstructed_rate, rate);
  }
  const Real expected = cfg.cfl / max_reconstructed_rate;
  const Real got = solver.cfl_limit();
  EXPECT_NEAR(got, expected, Real{3e-13} * expected);
}

// A grid-scale curl field can have zero collocated B in every cell while its
// exact CT faces are arbitrarily large. A cell-only CFL reduction therefore
// misses the actual HLLD/LF wave speed. This checkerboard is the discrete curl
// of A_z and is exactly divergence-free; both adjacent and MP7 face-to-cell
// quadratures cancel, but every normal Riemann face has |B_n|=amplitude.
TEST(MhdDeviceCompute, CflUsesExactIncidentNormalFaces) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  const Grid2D& g = cfg.grid;
  const std::size_t n = g.storage_size();
  constexpr Real amplitude = Real{100};
  constexpr Real rho0 = Real{1};
  constexpr Real p0 = Real{1};

  std::vector<Real> rho(n, rho0), zero(n, Real{0});
  std::vector<Real> energy(n, p0 / (cfg.gamma - Real{1}));
  std::vector<Real> bx(n), by(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const Real sign = ((i + j) % 2 == 0) ? Real{1} : Real{-1};
      bx[g.index(i, j)] = amplitude * sign;
      by[g.index(i, j)] = -amplitude * sign;
    }
  }

  quasar::mhd::MhdSolver2D solver{cfg};
  solver.seed_state("rho", rho);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", zero);

  ASSERT_LT(solver.divergence_b_max(), Real{1e-12});
  const Real sound = std::sqrt(cfg.gamma * p0 / rho0);
  const Real face_speed = std::max(sound, amplitude / std::sqrt(rho0));
  const Real expected =
      std::min(cfg.cfl, Real{0.2}) /
      (face_speed / g.dx() + face_speed / g.dy());
  const Real got = solver.cfl_limit();
  EXPECT_NEAR(got, expected, Real{2e-13} * expected);
  EXPECT_LT(got, Real{0.02} * g.dx())
      << "cell-collocated B is zero, so only exact normal faces can impose this bound";
}

TEST(MhdDeviceCompute, LiveStateRejectsResolvedDiscreteMagneticDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform(solver, cfg.grid, cfg.gamma,
               /*rho=*/Real{1}, /*vx=*/Real{0}, /*vy=*/Real{0},
               /*vz=*/Real{0}, /*p=*/Real{1},
               /*bx=*/Real{0.2}, /*by=*/Real{-0.1}, /*bz=*/Real{0});

  auto bx = solver.state_component_to_host("bx_face");
  bx[cfg.grid.index(3, 4)] += Real{1e-6};
  solver.seed_state("bx_face", bx);

  EXPECT_THROW((void)solver.cfl_limit(), std::invalid_argument);
  EXPECT_THROW(solver.step(Real{1e-6}), std::invalid_argument);
}

TEST(MhdDeviceCompute, LiveStateAcceptsRoundoffLevelMagneticDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform(solver, cfg.grid, cfg.gamma,
               /*rho=*/Real{1}, /*vx=*/Real{0}, /*vy=*/Real{0},
               /*vz=*/Real{0}, /*p=*/Real{1},
               /*bx=*/Real{0.2}, /*by=*/Real{-0.1}, /*bz=*/Real{0});

  auto bx = solver.state_component_to_host("bx_face");
  bx[cfg.grid.index(3, 4)] +=
      Real{32} * std::numeric_limits<Real>::epsilon();
  solver.seed_state("bx_face", bx);

  const Real dt = solver.cfl_limit();
  ASSERT_TRUE(std::isfinite(dt));
  ASSERT_GT(dt, Real{0});
  EXPECT_NO_THROW(solver.step(Real{0.1} * dt));
}

// A conservative state may have finite kinetic/internal energy even when
// |m|/rho and c_fast exceed binary64's range. If the mesh spacing is comparably
// large, their Courant rates remain ordinary finite numbers. The face reduction
// must scale before dividing rather than rejecting the unmaterializable speed.
TEST(MhdDeviceCompute, FaceCflKeepsRepresentableExtremeRates) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  cfg.grid = Grid2D{8, 8, Real{8e307}, Real{8e307},
                    Real{0}, Real{0}, /*nghost=*/4};
  const Grid2D& g = cfg.grid;
  const std::size_t n = g.storage_size();
  const Real rho0 = std::numeric_limits<Real>::denorm_min();
  const Real mx0 = Real{1e-8};
  const Real kinetic = quasar::numerics::kinetic_from_momentum(
      mx0, Real{0}, Real{0}, rho0);
  ASSERT_TRUE(std::isfinite(kinetic));
  const Real internal = Real{0.25} * kinetic;
  const Real energy0 = kinetic + internal;

  quasar::numerics::MhdState host_state{};
  host_state.rho = rho0;
  host_state.mx = mx0;
  host_state.energy = energy0;
  const Real p0 = quasar::numerics::pressure(host_state, cfg.gamma);
  ASSERT_TRUE(std::isfinite(p0));
  ASSERT_GT(p0, Real{0});
  ASSERT_FALSE(std::isfinite(mx0 / rho0));
  ASSERT_FALSE(std::isfinite(
      quasar::numerics::fast_magnetosonic_speed(
          host_state, /*dir=*/0, cfg.gamma)));

  quasar::mhd::MhdSolver2D solver{cfg};
  const std::vector<Real> rho(n, rho0), mx(n, mx0), zero(n, Real{0});
  const std::vector<Real> energy(n, energy0);
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);

  const long double rho_ld = static_cast<long double>(rho0);
  const long double p_ld = static_cast<long double>(p0);
  const long double sound_ld = std::sqrt(
      static_cast<long double>(cfg.gamma) * p_ld / rho_ld);
  const long double vx_ld =
      std::abs(static_cast<long double>(mx0)) / rho_ld;
  const long double rate_ld =
      (vx_ld + sound_ld) / static_cast<long double>(g.dx()) +
      sound_ld / static_cast<long double>(g.dy());
  const Real expected = static_cast<Real>(
      static_cast<long double>(std::min(cfg.cfl, Real{0.2})) / rate_ld);
  const Real got = solver.cfl_limit();
  ASSERT_TRUE(std::isfinite(got));
  ASSERT_GT(got, Real{0});
  EXPECT_NEAR(got, expected, Real{3e-13} * expected);
}

// A pressureless seed is outside the strict ideal-MHD admissible set.  Reject it
// before the CFL kernel can mistake a zero signal rate for a benign fallback.
TEST(MhdDeviceCompute, CflRejectsPressurelessInitialState) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  const Grid2D& g = cfg.grid;
  const Real gamma = cfg.gamma;

  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform(solver, g, gamma, /*rho=*/1.0, /*vx=*/0.0, /*vy=*/0.0, /*vz=*/0.0,
               /*p=*/0.0, /*bx=*/0.0, /*by=*/0.0, /*bz=*/0.0);
  EXPECT_THROW((void)solver.cfl_limit(), std::invalid_argument);
}

TEST(MhdDeviceCompute, CflRejectsZeroDensityInitialState) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  auto cfg = base_config();
  const std::size_t n = cfg.grid.storage_size();
  quasar::mhd::MhdSolver2D solver{cfg};
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> one(n, Real{1});
  solver.seed_state("rho", zero);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", one);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);
  EXPECT_THROW((void)solver.cfl_limit(), std::invalid_argument);
}

// Mutable device-buffer references may outlive a successful preflight. A
// validation result therefore cannot be cached across public operations: the
// retained handle can alter the live state without calling state() again.
TEST(MhdDeviceCompute, RetainedMutableStateHandlesCannotBypassAdmissibility) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  const std::size_t n = cfg.grid.storage_size();
  const std::vector<Real> zero(n, Real{0});

  {
    quasar::mhd::MhdSolver2D solver{cfg};
    seed_uniform(solver, cfg.grid, cfg.gamma,
                 /*rho=*/Real{1}, /*vx=*/Real{0}, /*vy=*/Real{0},
                 /*vz=*/Real{0}, /*p=*/Real{1},
                 /*bx=*/Real{0}, /*by=*/Real{0}, /*bz=*/Real{0});
    auto& retained = solver.state();
    const Real dt = Real{0.1} * solver.cfl_limit();
    retained.rho.copy_from_host(zero.data(), zero.size());
    EXPECT_THROW(solver.step_unchecked(dt), std::invalid_argument);
  }

  {
    quasar::mhd::MhdSolver2D solver{cfg};
    seed_uniform(solver, cfg.grid, cfg.gamma,
                 /*rho=*/Real{1}, /*vx=*/Real{0}, /*vy=*/Real{0},
                 /*vz=*/Real{0}, /*p=*/Real{1},
                 /*bx=*/Real{0}, /*by=*/Real{0}, /*bz=*/Real{0});
    auto& retained = solver.rk_register(0);
    ASSERT_NO_THROW((void)solver.cfl_limit());
    retained.rho.copy_from_host(zero.data(), zero.size());
    EXPECT_THROW((void)solver.cfl_limit(), std::invalid_argument);
  }
}

TEST(MhdDeviceCompute, SeedStateRejectsNonfinitePayload) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  auto cfg = base_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  std::vector<Real> values(cfg.grid.storage_size(), Real{1});
  values[cfg.grid.index(0, 0)] = std::numeric_limits<Real>::infinity();
  EXPECT_THROW(solver.seed_state("rho", values), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// divergence_b_max() stays at round-off through a FULL step for a div-free seed.
// ---------------------------------------------------------------------------
TEST(MhdDeviceCompute, DivergenceBStaysRoundoffThroughStep) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  const Grid2D& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_divergence_free(solver, g, cfg.gamma);

  const Real div0 = solver.divergence_b_max();
  EXPECT_LT(div0, 1e-12) << "seed must be divergence-free to round-off";

  const Real dt = 0.4 * solver.cfl_limit();
  ASSERT_GT(dt, 0.0);
  solver.step(dt);

  const Real div1 = solver.divergence_b_max();
  EXPECT_LT(div1, 1e-10) << "FD-CT must hold div B at round-off through a step";
}

// ---------------------------------------------------------------------------
// End-to-end high-order deck: reconstruction "mp7", positivity "troubled_cell",
// wall boundaries -> one full step runs without error and leaves finite state.
// The seed is the discrete curl of a corner potential that is constant on the
// complete boundary, hence it is both discretely solenoidal and compatible
// with this conducting-wall model's B.n=0 constraint.  A periodic curl seed is
// not a valid substitute: overwriting its generally nonzero normal boundary
// flux would create a resolved divergence defect before the first step.
// ---------------------------------------------------------------------------
TEST(MhdDeviceCompute, Mp7WallEndToEndStepIsFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = base_config();
  cfg.reconstruction = "mp7";
  cfg.positivity = "troubled_cell";
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = "wall";
    cfg.boundary.field[s] = "wall";
  }
  const Grid2D& g = cfg.grid;

  quasar::mhd::MhdSolver2D solver{cfg};
  seed_wall_divergence_free(solver, g, cfg.gamma);

  EXPECT_LT(solver.divergence_b_max(), Real{1e-12});
  const Real dt = 0.4 * solver.cfl_limit();
  ASSERT_GT(dt, 0.0);
  ASSERT_NO_THROW(solver.step(dt));

  const std::vector<Real> rho = solver.state_component_to_host("rho");
  const std::vector<Real> en = solver.state_component_to_host("energy");
  const std::vector<Real> bx = solver.state_component_to_host("bx");
  const std::vector<Real> by = solver.state_component_to_host("by");
  EXPECT_TRUE(all_finite(rho)) << "mp7/wall step produced non-finite rho";
  EXPECT_TRUE(all_finite(en)) << "mp7/wall step produced non-finite energy";
  EXPECT_TRUE(all_finite(bx)) << "mp7/wall step produced non-finite bx";
  EXPECT_TRUE(all_finite(by)) << "mp7/wall step produced non-finite by";
  EXPECT_LT(solver.divergence_b_max(), Real{1e-10});
}
