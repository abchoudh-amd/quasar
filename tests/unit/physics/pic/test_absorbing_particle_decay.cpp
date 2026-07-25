// Absorbing boundary: a particle pushed out of the domain is marked dead and
// the alive count drops. Exercises the end-to-end step() boundary dispatch plus
// the device alive-count reduction.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/boundary/wall.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace {

template <int ShapeOrder>
void run_absorbing_continuity_case() {
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0,
                   ShapeOrder == 2 ? 2 : 1};
  quasar::pic::EmPicConfig cfg{g, 2,
                               ShapeOrder == 1 ? "cic" : "tsc"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "absorbing";
    cfg.boundary.field[side] = "pec";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  constexpr double q = 1.0, w = 1.0, x0 = 0.99, y0 = 0.5;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"q", q, 1.0, 1}};
  sp.set_host_particles({x0}, {y0}, {0.8}, {0.0}, {0.0}, {w});
  solver.add_species(std::move(sp));

  constexpr double dt = 0.02;
  solver.step(dt);
  ASSERT_EQ(quasar::pic::alive_count(solver.species()[0]), 0u);

  std::vector<double> rho_old(g.storage_size(), 0.0);
  const auto sw = quasar::numerics::shape_weights_2d<ShapeOrder>(x0, y0, g);
  for (int b = 0; b < sw.ny; ++b) {
    if (sw.iy[b] < 0 || sw.iy[b] >= g.ny) continue;
    for (int a = 0; a < sw.nx; ++a) {
      if (sw.ix[a] < 0 || sw.ix[a] >= g.nx) continue;
      rho_old[g.index(sw.ix[a], sw.iy[b])] +=
          q * w * sw.wx[a] * sw.wy[b] / (g.dx() * g.dy());
    }
  }

  std::vector<double> jx(g.storage_size()), jy(g.storage_size());
  solver.current().jx.copy_to_host(jx.data(), jx.size());
  solver.current().jy.copy_to_host(jy.data(), jy.size());
  double max_resid = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const auto k = g.index(i, j);
      const double residual =
          -rho_old[k] / dt
        + (jx[g.index(i + 1, j)] - jx[k]) / g.dx()
        + (jy[g.index(i, j + 1)] - jy[k]) / g.dy();
      max_resid = std::max(max_resid, std::abs(residual));
    }
  }
  EXPECT_LT(max_resid, 3.0e-9)
      << "absorbing-wall continuity residual " << max_resid;
}

}  // namespace

TEST(PicAbsorbingParticleDecay, BoundaryTypeConstructs) {
  quasar::boundary::AbsorbingParticleBC bc;
  (void)bc;
  SUCCEED();
}

TEST(PicAbsorbingParticleDecay, RejectsUnrepresentableLossEndpoint) {
  const double maximum = std::numeric_limits<double>::max();
  const quasar::Grid2D translated{
      1, 1, maximum, 1.0, -maximum, 0.0, 1};
  quasar::pic::EmPicConfig cfg{translated, 2, "cic"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "absorbing";
    cfg.boundary.field[side] = "pec";
  }
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{cfg}, std::overflow_error);
}

TEST(PicAbsorbingParticleDecay, ParticleLeavingDomainIsKilled) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Unit-square grid, origin at 0. Absorbing on every side.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "absorbing";
    cfg.boundary.field[side] = "pec";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  // Two particles: one parked in the interior with zero velocity (survives),
  // one near the x_hi wall moving fast outward (leaves and is absorbed).
  const std::size_t n = 2;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, n}};
  std::vector<quasar::Real> x{0.5, 0.99}, y{0.5, 0.5};
  std::vector<quasar::Real> vx{0.0, 0.8}, vy{0.0, 0.0}, vz{0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 2u);

  // dt large enough that the second particle clears x = 1.0 in one push.
  const quasar::Real dt = 0.02;
  solver.step(dt);

  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 1u);
}

TEST(PicAbsorbingParticleDecay, CicLossSatisfiesCellwiseContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  run_absorbing_continuity_case<1>();
}

TEST(PicAbsorbingParticleDecay, TscLossSatisfiesCellwiseContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  run_absorbing_continuity_case<2>();
}

TEST(PicAbsorbingParticleDecay, BoundaryFluxPreservesGaussLaw) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr int nx = 16, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, 0.25, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.boundary.field[0] = "pec";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[0] = "absorbing";
  cfg.boundary.particle[1] = "absorbing";
  quasar::pic::EmPic2D3V solver{cfg};

  // One identical particle per y cell makes rho and J exactly uniform along y,
  // so the seeded longitudinal Ex is curl-free and isolates the normal boundary
  // face in Ampere's law.
  std::vector<double> x(ny, 0.99), y(ny), vx(ny, 0.8), zero(ny, 0.0),
      weight(ny, 1.0);
  for (int j = 0; j < ny; ++j) y[j] = g.y_at_cell_center(j);
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"q", 1.0, 1.0e12,
                                 static_cast<std::size_t>(ny)}};
  sp.set_host_particles(x, y, vx, zero, zero, weight);
  solver.add_species(std::move(sp));

  std::vector<double> rho(g.storage_size(), 0.0);
  for (int j = 0; j < ny; ++j) {
    const auto sw = quasar::numerics::shape_weights_2d<1>(x[j], y[j], g);
    for (int b = 0; b < sw.ny; ++b) {
      for (int a = 0; a < sw.nx; ++a) {
        if (sw.ix[a] < 0 || sw.ix[a] >= nx) continue;
        rho[g.index(sw.ix[a], g.wrap_j(sw.iy[b]))] +=
            sw.wx[a] * sw.wy[b] / (g.dx() * g.dy());
      }
    }
  }
  std::vector<double> ex(g.storage_size(), 0.0);
  for (int j = 0; j < ny; ++j) {
    ex[g.index(0, j)] = 0.0;
    for (int i = 0; i < nx; ++i) {
      ex[g.index(i + 1, j)] =
          ex[g.index(i, j)] + g.dx() * rho[g.index(i, j)];
    }
  }
  solver.fields().ex.copy_from_host(ex.data(), ex.size());

  solver.step(0.02);
  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 0u);
  EXPECT_LT(quasar::pic::gauss_residual(solver), 2.0e-9)
      << "normal electric wall face was not advanced with boundary current";
}

TEST(PicAbsorbingParticleDecay, CylindricalAnnulusLosesAtBothRadiiAndCorner) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 1.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "absorbing";
    cfg.boundary.field[side] = "pec";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"loss", 0.0, 1.0, 3}};
  // Inner-radius exit, outer-radius exit, and simultaneous outer/z-high exit.
  sp.set_host_particles(
      {1.005, 1.995, 1.995}, {0.5, 0.5, 0.995},
      {-0.6, 0.6, 0.6}, {0.0, 0.0, 0.6}, {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0});
  solver.add_species(std::move(sp));
  solver.step(0.02);

  const auto snap = solver.species()[0].to_host();
  ASSERT_EQ(snap.alive.size(), 3u);
  EXPECT_EQ(snap.alive[0], 0u);
  EXPECT_EQ(snap.alive[1], 0u);
  EXPECT_EQ(snap.alive[2], 0u);
  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 0u);
}
