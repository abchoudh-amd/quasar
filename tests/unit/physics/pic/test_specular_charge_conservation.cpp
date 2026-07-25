// Specular reflecting walls must keep the charge-conserving current deposition
// LOCAL: a particle reflecting off a wall must not teleport current to the far
// (opposite) edge. Before the BC-aware deposit + ghost fold-back, the specular
// kernel mirrored x_prev outside the domain and the Esirkepov deposit wrapped the
// out-of-domain stencil nodes via periodic_index onto the opposite edge, injecting
// spurious current there.
//
// Two checks:
//  1) With a specular BC active, a particle that stays in the bulk deposits a
//     current that still satisfies the discrete continuity relation cell-by-cell
//     (the new ghost-aware indexing reduces to the historical path away from
//     walls, and the fold-back is a no-op when no ghost current is produced).
//  2) A particle that crosses a reflecting wall produces NO current at the far
//     edge (the targeted regression for the teleport bug), reflects, stays alive.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace {

template <int ShapeOrder>
void accumulate_rho(std::vector<double>& rho, const quasar::Grid2D& g,
                    double q, double w, double x, double y) {
  const auto sw = quasar::numerics::shape_weights_2d<ShapeOrder>(x, y, g);
  for (int b = 0; b < sw.ny; ++b) {
    for (int a = 0; a < sw.nx; ++a) {
      rho[g.periodic_index(sw.ix[a], sw.iy[b])] += q * w * sw.wx[a] * sw.wy[b];
    }
  }
}

template <int ShapeOrder>
void accumulate_rho_reflecting(std::vector<double>& rho,
                               const quasar::Grid2D& g, double q, double w,
                               double x, double y) {
  const auto sw = quasar::numerics::shape_weights_2d<ShapeOrder>(x, y, g);
  for (int b = 0; b < sw.ny; ++b) {
    int jj = sw.iy[b];
    if (jj < 0) jj = -jj - 1;
    if (jj >= g.ny) jj = 2 * g.ny - 1 - jj;
    if (jj < 0 || jj >= g.ny) continue;
    for (int a = 0; a < sw.nx; ++a) {
      int ii = sw.ix[a];
      if (ii < 0) ii = -ii - 1;
      if (ii >= g.nx) ii = 2 * g.nx - 1 - ii;
      if (ii < 0 || ii >= g.nx) continue;
      rho[g.index(ii, jj)] +=
          q * w * sw.wx[a] * sw.wy[b] / (g.dx() * g.dy());
    }
  }
}

quasar::pic::EmPicConfig specular_config(const quasar::Grid2D& g, int shape_order) {
  quasar::pic::EmPicConfig cfg{g, 2, shape_order == 1 ? "cic" : "tsc"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "specular";
    cfg.boundary.field[side] = "pec";
  }
  return cfg;
}

}  // namespace

TEST(PicSpecularChargeConservation, BulkDepositStillSatisfiesContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // A particle well inside the domain with a specular BC active: the deposit must
  // match Esirkepov continuity exactly, identical to the periodic-bulk case.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{specular_config(g, 1)};

  const double q = 1.0, m = 1.0, w = 1.0;
  const double x0 = 0.51, y0 = 0.52;
  const double vx = 0.37, vy = -0.29, vz = 0.13;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", q, m, 1}};
  sp.set_host_particles({x0}, {y0}, {vx}, {vy}, {vz}, {w});
  solver.add_species(std::move(sp));

  const double dt = 0.03;
  solver.step(dt);

  const double x1 = x0 + dt * vx;
  const double y1 = y0 + dt * vy;
  std::vector<double> rho_old(g.storage_size(), 0.0);
  std::vector<double> rho_new(g.storage_size(), 0.0);
  accumulate_rho<1>(rho_old, g, q, w, x0, y0);
  accumulate_rho<1>(rho_new, g, q, w, x1, y1);

  auto& J = solver.current();
  std::vector<double> jx(g.storage_size()), jy(g.storage_size());
  J.jx.copy_to_host(jx.data(), jx.size());
  J.jy.copy_to_host(jy.data(), jy.size());

  double max_resid = 0.0, max_jmag = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.periodic_index(i, j);
      const double drho = (rho_new[k] - rho_old[k]) / dt / (g.dx() * g.dy());
      const double divJ =
          (jx[g.periodic_index(i + 1, j)] - jx[k]) / g.dx()
        + (jy[g.periodic_index(i, j + 1)] - jy[k]) / g.dy();
      max_resid = std::max(max_resid, std::abs(drho + divJ));
      max_jmag = std::max(max_jmag, std::max(std::abs(jx[k]), std::abs(jy[k])));
    }
  }
  EXPECT_GT(max_jmag, 1.0e-6);
  EXPECT_LT(max_resid, 1.0e-9) << "bulk continuity residual " << max_resid;
}

TEST(PicSpecularChargeConservation, WallCrossingDepositsNoFarEdgeCurrent) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // A particle crossing the x_hi reflecting wall. The deposit must stay near that
  // wall; the opposite (x_lo) edge columns must carry essentially no current.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{specular_config(g, 1)};

  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, 1}};
  sp.set_host_particles({0.96}, {0.5}, {0.99}, {0.0}, {0.0}, {1.0});
  solver.add_species(std::move(sp));

  const double dt = 0.044;
  solver.step(dt);

  auto& J = solver.current();
  std::vector<double> jx(g.storage_size()), jy(g.storage_size()), jz(g.storage_size());
  J.jx.copy_to_host(jx.data(), jx.size());
  J.jy.copy_to_host(jy.data(), jy.size());
  J.jz.copy_to_host(jz.data(), jz.size());

  double far_edge = 0.0;   // x_lo columns (the cells the wrap bug fed)
  double near_wall = 0.0;  // x_hi columns (where current should be)
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i <= 1; ++i) {
      const std::size_t k = g.index(i, j);
      far_edge = std::max({far_edge, std::abs(jx[k]), std::abs(jy[k]), std::abs(jz[k])});
    }
    for (int i = g.nx - 2; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      near_wall = std::max({near_wall, std::abs(jx[k]), std::abs(jy[k]), std::abs(jz[k])});
    }
  }

  EXPECT_GT(near_wall, 1.0e-6) << "no current deposited near the reflecting wall";
  EXPECT_LT(far_edge, 1.0e-12) << "spurious current teleported to the far edge: "
                               << far_edge;

  // Reflected and still alive.
  auto snap = solver.species()[0].to_host();
  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 1u);
  EXPECT_LT(snap.x[0], 1.0);
  EXPECT_GT(snap.x[0], 0.0);
  EXPECT_LT(snap.vx[0], 0.0);
}

TEST(PicSpecularChargeConservation, WallCrossingSatisfiesCellwiseContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{specular_config(g, 1)};
  constexpr double q = 1.0, w = 1.0, x0 = 0.96, y0 = 0.5;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"q", q, 1.0, 1}};
  sp.set_host_particles({x0}, {y0}, {0.99}, {0.0}, {0.0}, {w});
  solver.add_species(std::move(sp));

  constexpr double dt = 0.044;
  solver.step(dt);
  const auto snap = solver.species()[0].to_host();

  std::vector<double> rho_old(g.storage_size(), 0.0);
  std::vector<double> rho_new(g.storage_size(), 0.0);
  accumulate_rho_reflecting<1>(rho_old, g, q, w, x0, y0);
  accumulate_rho_reflecting<1>(rho_new, g, q, w, snap.x[0], snap.y[0]);

  std::vector<double> jx(g.storage_size()), jy(g.storage_size());
  solver.current().jx.copy_to_host(jx.data(), jx.size());
  solver.current().jy.copy_to_host(jy.data(), jy.size());
  double max_resid = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const auto k = g.index(i, j);
      const double residual =
          (rho_new[k] - rho_old[k]) / dt
        + (jx[g.index(i + 1, j)] - jx[k]) / g.dx()
        + (jy[g.index(i, j + 1)] - jy[k]) / g.dy();
      max_resid = std::max(max_resid, std::abs(residual));
    }
  }
  EXPECT_LT(max_resid, 2.0e-9)
      << "reflecting-wall continuity residual " << max_resid;
}
