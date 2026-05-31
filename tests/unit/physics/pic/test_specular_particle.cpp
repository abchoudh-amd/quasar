// Specular boundary: a particle that crosses a wall is reflected back into the
// domain with its normal velocity component flipped. Exercises the end-to-end
// step() boundary dispatch.

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

TEST(PicSpecularParticle, BoundaryTypeConstructs) {
  quasar::boundary::SpecularParticleBC bc;
  (void)bc;
  SUCCEED();
}

TEST(PicSpecularParticle, ParticleReflectsOffWall) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "specular";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  // Single particle near the x_hi wall moving outward; it should be reflected
  // back inside with vx negated and stay alive.
  const std::size_t n = 1;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, n}};
  std::vector<quasar::Real> x{0.95}, y{0.5};
  std::vector<quasar::Real> vx{2.0}, vy{0.0}, vz{0.0};
  std::vector<quasar::Real> w{1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  const quasar::Real dt = 0.05;  // 0.95 + 2*0.05 = 1.05 > 1.0 -> crosses x_hi
  solver.step(dt);

  auto snap = solver.species()[0].to_host();
  ASSERT_EQ(snap.x.size(), 1u);
  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 1u);
  // Reflected back inside the domain.
  EXPECT_LT(snap.x[0], 1.0);
  EXPECT_GT(snap.x[0], 0.0);
  // Normal velocity flipped (was moving +x, now -x).
  EXPECT_LT(snap.vx[0], 0.0);
}
