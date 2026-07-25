#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

TEST(PicSingleParticle, UniformExBProducesChargeIndependentGuidingCenterDrift) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{64, 64, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  constexpr double e_x = 0.05;
  constexpr double b_z = 1.0;
  constexpr double dt = 0.02;
  std::vector<double> ex(g.storage_size(), e_x);
  std::vector<double> bz(g.storage_size(), b_z);
  solver.external_fields().ex.copy_from_host(ex.data(), ex.size());
  solver.external_fields().bz.copy_from_host(bz.data(), bz.size());

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  sp.set_host_particles({2.0}, {2.0}, {0.0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));

  const double discrete_angle = 2.0 * std::atan(0.5 * b_z * dt);
  const int steps = static_cast<int>(std::llround(8.0 * std::acos(-1.0) /
                                                  discrete_angle));
  for (int s = 0; s < steps; ++s) solver.step(dt);
  const auto snap = solver.species()[0].to_host();
  const double measured_drift = (snap.y[0] - 2.0) / (steps * dt);
  const double expected_drift = -e_x / b_z;
  EXPECT_NEAR(measured_drift, expected_drift, 8.0e-4);
  // After four numerical gyroperiods the oscillatory velocity nearly closes.
  EXPECT_LT(std::hypot(snap.vx[0], snap.vy[0]), 2.0e-3);
}
