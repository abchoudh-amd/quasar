// Absorbing boundary: a particle pushed out of the domain is marked dead and
// the alive count drops. Exercises the end-to-end step() boundary dispatch plus
// the device alive-count reduction.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/boundary/wall.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

TEST(PicAbsorbingParticleDecay, BoundaryTypeConstructs) {
  quasar::boundary::AbsorbingParticleBC bc;
  (void)bc;
  SUCCEED();
}

TEST(PicAbsorbingParticleDecay, ParticleLeavingDomainIsKilled) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Unit-square grid, origin at 0. Absorbing on every side.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "absorbing";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  // Two particles: one parked in the interior with zero velocity (survives),
  // one near the x_hi wall moving fast outward (leaves and is absorbed).
  const std::size_t n = 2;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, n}};
  std::vector<quasar::Real> x{0.5, 0.95}, y{0.5, 0.5};
  std::vector<quasar::Real> vx{0.0, 5.0}, vy{0.0, 0.0}, vz{0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 2u);

  // dt large enough that the second particle clears x = 1.0 in one push.
  const quasar::Real dt = 0.05;
  for (int s = 0; s < 4; ++s) solver.step(dt);

  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 1u);
}
