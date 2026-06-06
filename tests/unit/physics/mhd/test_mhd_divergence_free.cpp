// Constrained-transport divergence-free preservation for the ideal-MHD solver.
//
// The face-staggered B and the FD-CT update (fd_ct_christlieb) must keep the
// discrete divergence of B at machine epsilon: if the seed satisfies div B = 0 to
// round-off, every step must too. We build B from a corner-staggered vector
// potential A_z(x,y) so the discrete curl is divergence-free by construction:
//
//   bx_face(i,j) = +(A_z(i, j+1) - A_z(i, j)) / dy
//   by_face(i,j) = -(A_z(i+1, j) - A_z(i, j)) / dx
//
// with A_z sampled at lower-left cell corners. The discrete divergence the solver
// reports is the standard face-difference stencil
//
//   divB(i,j) = (bx_face(i+1,j) - bx_face(i,j))/dx
//             + (by_face(i,j+1) - by_face(i,j))/dy,
//
// which is identically zero (to round-off) for B = curl(A_z e_z). divergence_b_max()
// returns the L-infinity norm of divB over the interior.
//
// Storage/staggering conventions assumed (documented for the blind implementer):
//   * bx_face(i,j) lives on the left x-face of cell (i,j); by_face(i,j) on the
//     lower y-face. seed_state("bx",..)/("by",..) write the face arrays directly,
//     indexed by Grid2D::index(i,j) over full storage.
//   * The corner A_z used to build the faces is periodic, so the seeded B is a
//     valid periodic divergence-free field.
//   * A smooth uniform-ish background fluid keeps the run stable while the CT
//     update advances B; the fluid state is irrelevant to the div-B invariant.
//
// Repeated over reconstruction = {"mp7","muscl_minmod"} to show the CT constraint
// is independent of the reconstruction used for the fluid fluxes.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;

// Corner-staggered periodic vector potential A_z at the lower-left corner of cell
// (i,j): corner position is (origin_x + i*dx, origin_y + j*dy).
Real Az_corner(const quasar::Grid2D& g, int i, int j) {
  const Real x = g.origin_x + static_cast<Real>(g.wrap_i(i)) * g.dx();
  const Real y = g.origin_y + static_cast<Real>(g.wrap_j(j)) * g.dy();
  // Smooth periodic potential; its curl gives a non-trivial divergence-free B.
  return 0.1 * std::sin(2.0 * quasar::pi * x) * std::cos(2.0 * quasar::pi * y);
}

quasar::mhd::MhdConfig make_config(const std::string& reconstruction) {
  quasar::Grid2D g{32, 32, 1.0, 1.0, 0.0, 0.0, /*nghost=*/4};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.geometry = "cartesian";
  cfg.reconstruction = reconstruction;
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

void seed_divergence_free(quasar::mhd::MhdSolver2D& solver, const quasar::Grid2D& g) {
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
      // B = curl(A_z e_z) on the faces ⇒ discretely divergence-free.
      bx[k] = (Az_corner(g, i, j + 1) - Az_corner(g, i, j)) * inv_dy;
      by[k] = -(Az_corner(g, i + 1, j) - Az_corner(g, i, j)) * inv_dx;
      bz[k] = 0.0;
      // Uniform, static fluid background; energy from p0 + magnetic energy.
      const Real magnetic = 0.5 * (bx[k] * bx[k] + by[k] * by[k]);
      rho[k] = 1.0;
      mx[k] = 0.0;
      my[k] = 0.0;
      mz[k] = 0.0;
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

}  // namespace

TEST(MhdDivergenceFree, SeedIsDivergenceFreeAndStaysSo) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  for (const std::string reconstruction : {std::string{"mp7"},
                                           std::string{"muscl_minmod"}}) {
    const auto cfg = make_config(reconstruction);
    quasar::mhd::MhdSolver2D solver{cfg};
    seed_divergence_free(solver, cfg.grid);

    // The CT seed is divergence-free to round-off from the start.
    const Real div0 = solver.divergence_b_max();
    EXPECT_LT(div0, 1e-10) << "reconstruction=" << reconstruction;

    // FD-CT preserves the constraint: after several steps div B stays at the
    // machine-epsilon floor (scaled by the field magnitude / grid spacing).
    const Real dt = 0.4 * solver.cfl_limit();
    for (int s = 0; s < 6; ++s) {
      solver.step(dt);
    }
    const Real div1 = solver.divergence_b_max();
    EXPECT_LT(div1, 1e-9) << "reconstruction=" << reconstruction;
  }
}
