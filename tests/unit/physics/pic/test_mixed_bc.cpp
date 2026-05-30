#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(PicMixedBc, SpecCanMixPeriodicAndWalls) {
  quasar::boundary::BoundarySpec spec;
  spec.field[2] = quasar::boundary::FieldBoundaryKind::pec;
  spec.particle[3] = quasar::boundary::ParticleBoundaryKind::absorbing;
  EXPECT_EQ(spec.field[2], quasar::boundary::FieldBoundaryKind::pec);
  EXPECT_EQ(spec.particle[3], quasar::boundary::ParticleBoundaryKind::absorbing);
}

// Behavioral check: a solver configured with periodic walls in x and absorbing
// walls in y must wrap an x-exiting particle (kept alive) but kill a y-exiting
// particle. Exercises per-side dispatch through the registry-built BCs.
TEST(PicMixedBc, PeriodicXWrapsAbsorbingYKills) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, 1};
  using PK = quasar::boundary::ParticleBoundaryKind;
  cfg.boundary.particle[0] = PK::periodic;   // x_lo
  cfg.boundary.particle[1] = PK::periodic;   // x_hi
  cfg.boundary.particle[2] = PK::absorbing;  // y_lo
  cfg.boundary.particle[3] = PK::absorbing;  // y_hi
  quasar::pic::EmPic2D3V solver{cfg};

  // p0 drifts out the +x side (should wrap, stay alive);
  // p1 drifts out the +y side (should be absorbed, die).
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, 2}};
  std::vector<quasar::Real> x{0.95, 0.5}, y{0.5, 0.95};
  std::vector<quasar::Real> vx{5.0, 0.0}, vy{0.0, 5.0}, vz{0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  for (int s = 0; s < 4; ++s) solver.step(0.05);

  const auto snap = solver.species()[0].to_host();
  // p0 wrapped around x: alive, and its x is back inside [0,1).
  EXPECT_EQ(snap.alive[0], 1) << "periodic-x particle should survive";
  EXPECT_GE(snap.x[0], 0.0);
  EXPECT_LT(snap.x[0], 1.0);
  // p1 left through the absorbing +y wall: dead.
  EXPECT_EQ(snap.alive[1], 0) << "absorbing-y particle should be killed";
}
