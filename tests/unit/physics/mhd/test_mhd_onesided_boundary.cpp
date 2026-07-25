// One-sided boundary stencils at NON-PERIODIC physical boundaries (RED).
//
// Today MHD boundaries rely purely on ghost-cell filling. The behavioral change
// pinned here: at NON-PERIODIC physical boundaries (outflow, wall), the
// boundary-face reconstruction uses a one-sided (interior-biased) SLOPE so the
// scheme does not depend on the ghost GRADIENT; periodic boundaries still wrap
// via ghosts and stay numerically unchanged. The wall closure is
// STILL supplied by the mirror ghost VALUES (which are still filled and read) --
// the one-sided change only drops dependence on the ghost gradient, it does NOT
// remove the wall symmetry.
//
// We cannot call the internal one-sided reconstruction directly, so the behavior
// is pinned through OBSERVABLE solver outputs (finiteness, determinism, div-B,
// boundedness near an outflow edge, odd/even symmetry across a wall)
// plus the new boundary classifier free function.
//
// Staggering / index conventions (mirrored from test_mhd_divergence_free.cpp and
// src/physics/mhd/mhd_boundary.cpp):
//   * bx_face(i,j) lives on the left x-face of cell (i,j); by_face(i,j) on the
//     lower y-face. seed_state("bx"/"by",..) write the face arrays directly,
//     indexed by Grid2D::index(i,j) over full padded storage.
//   * B built as the discrete curl of a corner-staggered A_z is divergence-free:
//       bx_face(i,j) = +(A_z(i, j+1) - A_z(i, j)) / dy
//       by_face(i,j) = -(A_z(i+1, j) - A_z(i, j)) / dx
//   * A wall on x_lo fills ghost cell i=-layer from interior cell
//     i=layer-1 (mirror about the i=0 left face): the first interior cell i=0
//     maps to ghost i=-1, i=1 maps to i=-2, etc. Normal components (mx, bx_face)
//     are ODD (sign flip); rho/energy/my/by_face are EVEN. We read these via
//     state_component_to_host (cell-centered sampling) and compare the first
//     interior cell to its mirror ghost layer to confirm the wall symmetry
//     survives a step.
//
// Boundary spec array order is [x_lo, x_hi, y_lo, y_hi] (Side enum order).

#include "quasar/backend/device.hpp"
#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;

// Side indices into the MhdBoundarySpec fluid/field arrays.
constexpr int kXlo = 0;
constexpr int kXhi = 1;
constexpr int kYlo = 2;
constexpr int kYhi = 3;

// Corner-staggered periodic vector potential A_z at the lower-left corner of
// cell (i,j): corner position (origin + i*dx, origin + j*dy). Its curl gives a
// non-trivial discretely divergence-free B (same construction as the CT test).
Real Az_corner(const quasar::Grid2D& g, int i, int j) {
  const Real x = g.origin_x + static_cast<Real>(g.wrap_i(i)) * g.dx();
  const Real y = g.origin_y + static_cast<Real>(g.wrap_j(j)) * g.dy();
  return 0.1 * std::sin(2.0 * quasar::pi * x) * std::cos(2.0 * quasar::pi * y);
}

quasar::mhd::MhdConfig make_config(const std::string& xlo, const std::string& xhi,
                                   const std::string& ylo, const std::string& yhi) {
  quasar::Grid2D g{32, 32, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.geometry = "cartesian";
  cfg.reconstruction = "muscl_minmod";
  cfg.riemann = "hlld";
  cfg.integrator = "ssprk3";
  cfg.ct = "fd_ct_christlieb";
  cfg.positivity = "troubled_cell";
  const std::array<std::string, 4> sides{xlo, xhi, ylo, yhi};
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = sides[s];
    cfg.boundary.field[s] = sides[s];
  }
  return cfg;
}

// Smooth, divergence-free seed: a div-free B from A_z plus a smooth fluid
// profile. `vx` advects the gas (used to push a profile toward an open edge).
void seed_smooth(quasar::mhd::MhdSolver2D& solver, const quasar::Grid2D& g,
                 Real vx, Real vy = Real{0}) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, 1.0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);

  const Real gamma = 5.0 / 3.0;
  const Real p0 = 1.0;
  const Real inv_dx = 1.0 / g.dx();
  const Real inv_dy = 1.0 / g.dy();

  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      bx[k] = (Az_corner(g, i, j + 1) - Az_corner(g, i, j)) * inv_dy;
      by[k] = -(Az_corner(g, i + 1, j) - Az_corner(g, i, j)) * inv_dx;
      bz[k] = 0.0;
      // Smooth density bump and uniform horizontal drift; keeps a profile that
      // gets advected toward the +x edge while staying smooth and positive.
      const Real x = g.x_at_cell_center(i);
      const Real rho_i = 1.0 + 0.25 * std::sin(2.0 * quasar::pi * x);
      const Real magnetic = 0.5 * (bx[k] * bx[k] + by[k] * by[k]);
      rho[k] = rho_i;
      mx[k] = rho_i * vx;
      my[k] = rho_i * vy;
      mz[k] = 0.0;
      en[k] = p0 / (gamma - 1.0) +
              0.5 * rho_i * (vx * vx + vy * vy) + magnetic;
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

bool all_finite(const std::vector<Real>& v) {
  for (const Real x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// 4. Boundary classifier (pure host; no HIP runtime needed).
// ---------------------------------------------------------------------------
TEST(MhdOneSidedBoundary, ClassifierFlagsOnlyPeriodic) {
  EXPECT_TRUE(quasar::boundary::mhd_boundary_is_periodic("periodic"));
  EXPECT_FALSE(quasar::boundary::mhd_boundary_is_periodic("outflow"));
  EXPECT_FALSE(quasar::boundary::mhd_boundary_is_periodic("wall"));
}

// The interior-biased slope at the physical low face extrapolates cell zero
// outward.  With rho(0)=1 and rho(1)=4 that raw order-2 state is
// 1 - (4-1)/2 = -0.5 even though every source cell is admissible.  The local
// reconstruction guard must replace both sides with the adjacent cell averages
// and then reapply the exact CT normal field.
TEST(MhdOneSidedBoundary, InadmissibleOneSidedFaceFallsBackToAdjacentCells) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{4, 2, Real{1}, Real{1}, Real{0}, Real{0},
                         /*nghost=*/2};
  const std::size_t n = g.storage_size();
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real p = Real{1};
  constexpr Real energy0 = p / (gamma - Real{1});

  std::vector<Real> rho(n, Real{1});
  std::vector<Real> mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
  std::vector<Real> energy(n, energy0);
  std::vector<Real> bx(n, Real{0}), by(n, Real{0}), bz(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    rho[g.index(/*i=*/1, j)] = Real{4};
  }

  quasar::mhd::MhdField2D<Real> u{g};
  u.rho.copy_from_host(rho.data(), n);
  u.mx.copy_from_host(mx.data(), n);
  u.my.copy_from_host(my.data(), n);
  u.mz.copy_from_host(mz.data(), n);
  u.energy.copy_from_host(energy.data(), n);
  u.bx_face.copy_from_host(bx.data(), n);
  u.by_face.copy_from_host(by.data(), n);
  u.bz_cell.copy_from_host(bz.data(), n);

  const quasar::mhd::MhdBackgroundField<Real> background{};
  const quasar::mhd::BoundaryFlags4 flags{{/*x_lo=*/1, /*x_hi=*/0,
                                            /*y_lo=*/0, /*y_hi=*/0}};
  quasar::numerics::MhdInterfaceStates<Real> out{g, /*dir=*/0};
  quasar::mhd::launch_mhd_reconstruct(
      u, background, /*dir=*/0, out, /*scheme_order=*/2, flags, gamma,
      /*stream=*/nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto left = out.state_left(/*i=*/0, /*j=*/0);
  const auto right = out.state_right(/*i=*/0, /*j=*/0);
  for (const auto* state : {&left, &right}) {
    EXPECT_EQ(state->rho, Real{1});
    EXPECT_EQ(state->mx, Real{0});
    EXPECT_EQ(state->energy, energy0);
    EXPECT_EQ(state->bx, Real{0});
    EXPECT_GT(quasar::numerics::pressure(*state, gamma), Real{0});
  }
}

// ---------------------------------------------------------------------------
// 1. Periodic path unchanged: finite, deterministic, div-B clean.
// ---------------------------------------------------------------------------
TEST(MhdOneSidedBoundary, PeriodicPathFiniteDeterministicAndDivBClean) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto cfg = make_config("periodic", "periodic", "periodic", "periodic");

  // Run A.
  quasar::mhd::MhdSolver2D solver_a{cfg};
  seed_smooth(solver_a, cfg.grid, /*vx=*/0.0);
  const Real div0 = solver_a.divergence_b_max();
  EXPECT_LT(div0, 1e-10);
  const Real dt = 0.4 * solver_a.cfl_limit();
  solver_a.step(dt);
  const std::vector<Real> rho_a = solver_a.state_component_to_host("rho");
  const std::vector<Real> en_a = solver_a.state_component_to_host("energy");
  EXPECT_TRUE(all_finite(rho_a));
  EXPECT_TRUE(all_finite(en_a));
  // Periodic high faces reuse the matching low-face Riemann/EMF data, so the CT
  // curl telescopes across the seam and preserves the seeded divergence.
  EXPECT_LT(solver_a.divergence_b_max(), 1e-9);

  // Run B: identical config + seed + dt -> must be bit-for-bit identical
  // (the new one-sided code path must not perturb the periodic numerics).
  quasar::mhd::MhdSolver2D solver_b{cfg};
  seed_smooth(solver_b, cfg.grid, /*vx=*/0.0);
  solver_b.step(dt);
  const std::vector<Real> rho_b = solver_b.state_component_to_host("rho");
  const std::vector<Real> en_b = solver_b.state_component_to_host("energy");

  ASSERT_EQ(rho_a.size(), rho_b.size());
  for (std::size_t k = 0; k < rho_a.size(); ++k) {
    EXPECT_EQ(rho_a[k], rho_b[k]) << "rho mismatch at k=" << k;
    EXPECT_EQ(en_a[k], en_b[k]) << "energy mismatch at k=" << k;
  }
}

// ---------------------------------------------------------------------------
// 2. Outflow stability: no NaN/Inf and no spurious extremum at the open edge.
//    A smooth profile advected toward the +x open edge must stay bounded by the
//    interior variation -- a bounded, non-oscillatory open boundary whose
//    stability does NOT depend on a special ghost-gradient symmetry.
// ---------------------------------------------------------------------------
TEST(MhdOneSidedBoundary, OutflowEdgeStaysBounded) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // A periodic axis must wrap at both ends, so use the same physical outflow
  // closure at x_lo while probing boundedness at the downstream x_hi edge.
  const auto cfg = make_config("outflow", "outflow", "periodic", "periodic");
  const quasar::Grid2D& g = cfg.grid;

  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth(solver, g, /*vx=*/0.5);  // drift toward +x (the open edge)

  EXPECT_LT(solver.divergence_b_max(), 1e-10)
      << "outflow ghost fill must preserve the seeded normal boundary face";
  const Real dt = 0.4 * solver.cfl_limit();
  solver.step(dt);
  EXPECT_LT(solver.divergence_b_max(), 1e-9)
      << "outflow ghost fill must not overwrite the CT-evolved high face";

  const std::vector<Real> rho = solver.state_component_to_host("rho");
  ASSERT_TRUE(all_finite(rho)) << "outflow run produced NaN/Inf";

  // Interior min/max over cells NOT in the boundary-edge columns (exclude the
  // last column adjacent to the open edge from the bound's *source*).
  Real interior_min = std::numeric_limits<Real>::infinity();
  Real interior_max = -std::numeric_limits<Real>::infinity();
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx - 1; ++i) {
      const Real r = rho[g.index(i, j)];
      interior_min = std::min(interior_min, r);
      interior_max = std::max(interior_max, r);
    }
  }

  // The edge column (i = nx-1, adjacent to the open boundary) must introduce no
  // spurious extremum: it stays within the interior bounds to tolerance. A
  // ghost-gradient-driven oscillation would push it well outside this band.
  const Real tol = 0.05 * (interior_max - interior_min) + 1e-6;
  for (int j = 0; j < g.ny; ++j) {
    const Real edge = rho[g.index(g.nx - 1, j)];
    EXPECT_GE(edge, interior_min - tol) << "outflow edge undershoot at j=" << j;
    EXPECT_LE(edge, interior_max + tol) << "outflow edge overshoot at j=" << j;
  }
}

// The y-normal CT face has the same physical-high-face extent as its x-normal
// sibling: By(i,ny) is evolved by CT and is not a disposable ghost value. This
// transposed evolution regression catches a y_hi outflow fill that replaces the
// authoritative high face with By(i,ny-1), which immediately creates a boundary
// ring divergence even though the discrete curl update itself telescopes.
TEST(MhdOneSidedBoundary, YOutflowPreservesCtEvolvedHighFace) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto cfg = make_config("periodic", "periodic", "outflow", "outflow");
  const quasar::Grid2D& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth(solver, g, /*vx=*/0.0, /*vy=*/0.5);

  EXPECT_LT(solver.divergence_b_max(), 1e-10)
      << "y-outflow ghost fill must preserve the seeded normal boundary face";
  const Real dt = Real{0.4} * solver.cfl_limit();
  ASSERT_NO_THROW(solver.step(dt));
  EXPECT_LT(solver.divergence_b_max(), 1e-9)
      << "y_hi outflow must not overwrite the CT-evolved By(i,ny) face";
}

// ---------------------------------------------------------------------------
// 3. Wall symmetry preserved across an x_lo wall after a step.
//    The mirror ghost VALUES that impose the wall closure are still filled and
//    read by the boundary-biased reconstruction, so the post-step state keeps
//    the wall symmetry: normal momentum (mx) and normal field (bx) are ODD
//    across the x_lo wall; rho/energy/my/by are EVEN. We compare the first
//    interior cell (i=0) to its mirror ghost cell (i=-1) -- mirror about the
//    i=0 left face per src/physics/mhd/mhd_boundary.cpp (gi=-1 <- si=0).
// ---------------------------------------------------------------------------
TEST(MhdOneSidedBoundary, WallSymmetryPreserved) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // A periodic axis must wrap at both ends, so pair the x_lo wall with an x_hi
  // wall. The assertion below remains local to the x_lo mirror closure.
  const auto cfg = make_config("wall", "wall", "periodic", "periodic");
  const quasar::Grid2D& g = cfg.grid;

  quasar::mhd::MhdSolver2D solver{cfg};
  seed_smooth(solver, g, /*vx=*/0.0);

  const Real dt = 0.4 * solver.cfl_limit();
  solver.step(dt);

  const std::vector<Real> rho = solver.state_component_to_host("rho");
  const std::vector<Real> mx = solver.state_component_to_host("mx");
  const std::vector<Real> my = solver.state_component_to_host("my");
  const std::vector<Real> en = solver.state_component_to_host("energy");
  const std::vector<Real> bx = solver.state_component_to_host("bx");
  const std::vector<Real> by = solver.state_component_to_host("by");

  ASSERT_TRUE(all_finite(rho));
  ASSERT_TRUE(all_finite(mx));
  ASSERT_TRUE(all_finite(bx));

  // Mirror about the x_lo wall: ghost cell i=-1 corresponds to interior i=0.
  // Use a scale-aware tolerance from the wall magnitude.
  for (int j = 0; j < g.ny; ++j) {
    const std::size_t k_in = g.index(0, j);    // first interior cell
    const std::size_t k_gh = g.index(-1, j);   // its mirror ghost cell

    const Real scale =
        1.0 + std::abs(rho[k_in]) + std::abs(en[k_in]) + std::abs(mx[k_in]);
    const Real tol = 1e-6 * scale;

    // EVEN (copy across the wall): rho, energy, tangential momentum my,
    // tangential field by.
    EXPECT_NEAR(rho[k_gh], rho[k_in], tol) << "rho not even at j=" << j;
    EXPECT_NEAR(en[k_gh], en[k_in], tol) << "energy not even at j=" << j;
    EXPECT_NEAR(my[k_gh], my[k_in], tol) << "my not even at j=" << j;
    EXPECT_NEAR(by[k_gh], by[k_in], tol) << "by not even at j=" << j;

    // ODD (sign flip across the wall): normal momentum mx, normal field bx.
    EXPECT_NEAR(mx[k_gh], -mx[k_in], tol) << "mx not odd at j=" << j;
    EXPECT_NEAR(bx[k_gh], -bx[k_in], tol) << "bx not odd at j=" << j;
  }
}
