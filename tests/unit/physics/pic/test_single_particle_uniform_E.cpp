#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace {

void fill(quasar::backend::DeviceBuffer<quasar::Real>& component,
          quasar::Real value) {
  std::vector<quasar::Real> host(component.size(), value);
  component.copy_from_host(host.data(), host.size());
}

}  // namespace

TEST(PicSingleParticleUniformE, FirstStepStartsFromPhysicalIntegerTimeVelocity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  constexpr double electric_field = 0.4;
  constexpr double charge = 1.5;
  constexpr double mass = 2.0;
  constexpr double v0 = 0.2;
  constexpr double x0 = 2.0;
  constexpr double dt = 0.08;
  fill(solver.external_fields().ex, electric_field);

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", charge, mass, 1}};
  // Zero macro-weight removes self-fields while retaining q/m in the pusher.
  sp.set_host_particles({x0}, {2.0}, {v0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.step(dt);

  const double acceleration = charge * electric_field / mass;
  const double expected_v_half = v0 + acceleration * dt / 2.0;
  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.vx[0], expected_v_half, 2.0e-14);
  EXPECT_NEAR(snap.x[0], x0 + expected_v_half * dt, 2.0e-14);
  EXPECT_DOUBLE_EQ(snap.vy[0], 0.0);
  EXPECT_DOUBLE_EQ(snap.vz[0], 0.0);
}

TEST(PicSingleParticleUniformE, ClippedFirstStepUsesHalfOfClippedWidth) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  constexpr double electric_field = 0.5;
  constexpr double v0 = 0.1;
  constexpr double x0 = 1.0;
  constexpr double nominal_dt = 0.20;
  constexpr double t_end = 0.06;
  fill(solver.external_fields().ex, electric_field);

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  sp.set_host_particles({x0}, {1.0}, {v0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.advance(t_end, nominal_dt);

  const double expected_v_half = v0 + electric_field * t_end / 2.0;
  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.vx[0], expected_v_half, 2.0e-14);
  EXPECT_NEAR(snap.x[0], x0 + expected_v_half * t_end, 2.0e-14);
}
