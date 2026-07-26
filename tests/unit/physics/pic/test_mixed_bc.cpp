#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(PicMixedBc, SpecCanMixPeriodicAndWalls) {
  quasar::boundary::BoundarySpec spec;
  spec.field[2] = "pec";
  spec.particle[3] = "absorbing";
  EXPECT_EQ(spec.field[2], "pec");
  EXPECT_EQ(spec.particle[3], "absorbing");
}

// Behavioral check: a solver configured with periodic walls in x and absorbing
// walls in y must wrap an x-exiting particle (kept alive) but kill a y-exiting
// particle. Exercises per-side dispatch through the registry-built BCs.
TEST(PicMixedBc, PeriodicXWrapsAbsorbingYKills) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.boundary.particle[0] = "periodic";   // x_lo
  cfg.boundary.particle[1] = "periodic";   // x_hi
  cfg.boundary.particle[2] = "absorbing";  // y_lo
  cfg.boundary.particle[3] = "absorbing";  // y_hi
  cfg.boundary.field[2] = "pec";
  cfg.boundary.field[3] = "pec";
  quasar::pic::EmPic2D3V solver{cfg};

  // p0 drifts out the +x side (should wrap, stay alive);
  // p1 drifts out the +y side (should be absorbed, die). Keep per-step motion
  // sub-cell so this test is about mixed BC dispatch, not deposit-window limits.
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, 2}};
  std::vector<quasar::Real> x{0.99, 0.5}, y{0.5, 0.99};
  std::vector<quasar::Real> vx{0.5, 0.0}, vy{0.0, 0.5}, vz{0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  solver.step(0.03);

  const auto snap = solver.species()[0].to_host();
  // p0 wrapped around x: alive, and its x is back inside [0,1).
  EXPECT_EQ(snap.alive[0], 1) << "periodic-x particle should survive";
  EXPECT_GE(snap.x[0], 0.0);
  EXPECT_LT(snap.x[0], 1.0);
  // p1 left through the absorbing +y wall: dead.
  EXPECT_EQ(snap.alive[1], 0) << "absorbing-y particle should be killed";
}

TEST(PicMixedBc, OneSidedPeriodicTopologyIsRejected) {
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.boundary.particle[0] = "periodic";   // x_lo
  cfg.boundary.particle[1] = "absorbing";  // x_hi
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{cfg}, std::invalid_argument);
}

namespace {

void expect_high_x_face_gather(bool periodic_field) {
  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.boundary.particle[0] = "specular";
  cfg.boundary.particle[1] = "specular";
  if (periodic_field) {
    cfg.boundary.particle[0] = "periodic";
    cfg.boundary.particle[1] = "periodic";
  } else {
    cfg.boundary.field[0] = "pec";
    cfg.boundary.field[1] = "pec";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  std::vector<double> ex(g.storage_size(), 0.0);
  const int source_i = periodic_field ? 0 : g.nx;
  for (int j = 0; j < g.ny; ++j) ex[g.index(source_i, j)] = -1.0;
  solver.external_fields().ex.copy_from_host(ex.data(), ex.size());

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"probe", 1.0, 1.0, 1}};
  // Start exactly on the high Ex face. The inward force keeps the particle in
  // the domain, so specular post-processing does not alter the expected kick.
  sp.set_host_particles({1.0}, {0.5}, {0.0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.step(0.01);
  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.vx[0], -0.005, 2.0e-14);
}

}  // namespace

TEST(PicMixedBc, NonperiodicGatherUsesPhysicalHighYeeFace) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  expect_high_x_face_gather(false);
}

TEST(PicMixedBc, PeriodicGatherWrapsConsistentFieldAndParticleTopology) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  expect_high_x_face_gather(true);
}

TEST(PicMixedBc, OrderFourHighFaceAmpereReadsOwnedPeriodicCornerHalo) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // x is PEC and y is periodic. Ex(nx,ny-1) is a physical normal-E face; its
  // order-four y curl reads Bz(nx,ny) and Bz(nx,ny+1), i.e. a physical corner
  // and a halo jointly owned by the x-wall/y-periodic closures.
  quasar::Grid2D g{6, 6, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::pic::EmPicConfig cfg{g, 4, "cic"};
  cfg.boundary.field[0] = "pec";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[0] = "specular";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  std::vector<double> bz(g.storage_size(), 0.0);
  bz[g.index(g.nx, 0)] = 2.0;
  bz[g.index(g.nx, 1)] = 3.0;
  bz[g.index(g.nx, g.ny - 2)] = 0.0;
  bz[g.index(g.nx, g.ny - 1)] = 1.0;
  solver.fields().bz.copy_from_host(bz.data(), bz.size());

  const double dt = 0.01;
  solver.step(dt);
  std::vector<double> ex(g.storage_size(), 0.0);
  solver.fields().ex.copy_to_host(ex.data(), ex.size());
  solver.fields().bz.copy_to_host(bz.data(), bz.size());

  const double stencil = (9.0 / 8.0) * (2.0 - 1.0)
                       - (1.0 / 24.0) * (3.0 - 0.0);
  EXPECT_NEAR(ex[g.index(g.nx, g.ny - 1)],
              dt * stencil / g.dy(), 2.0e-14);
  EXPECT_DOUBLE_EQ(bz[g.index(g.nx, g.ny)], 2.0);
  EXPECT_DOUBLE_EQ(bz[g.index(g.nx, g.ny + 1)], 3.0);
}
