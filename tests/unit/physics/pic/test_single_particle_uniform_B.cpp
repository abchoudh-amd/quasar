#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void fill(quasar::backend::DeviceBuffer<quasar::Real>& component,
          quasar::Real value) {
  std::vector<quasar::Real> host(component.size(), value);
  component.copy_from_host(host.data(), host.size());
}

}  // namespace

TEST(PicSingleParticle, CartesianGyroSignFrequencyAndSpeed) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{64, 64, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  constexpr double b = 1.3;
  constexpr double v0 = 0.2;
  constexpr double dt = 0.02;
  constexpr int steps = 50;
  fill(solver.external_fields().bz, b);

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  // Zero macro-weight removes self-fields while retaining q/m in the pusher.
  sp.set_host_particles({2.0}, {2.0}, {v0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  for (int s = 0; s < steps; ++s) solver.step(dt);

  const auto snap = solver.species()[0].to_host();
  const double theta = 2.0 * std::atan(0.25 * b * dt)
                     + (steps - 1) * 2.0 * std::atan(0.5 * b * dt);
  EXPECT_NEAR(snap.vx[0], v0 * std::cos(theta), 2.0e-13);
  EXPECT_NEAR(snap.vy[0], -v0 * std::sin(theta), 2.0e-13);
  EXPECT_LT(snap.vy[0], 0.0);  // q>0, Bz>0 rotates +x toward -y
  EXPECT_NEAR(std::hypot(snap.vx[0], snap.vy[0]), v0, 2.0e-13);
}

TEST(PicSingleParticle, PhysicalInitialVelocityGivesSecondOrderOrbitConvergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto orbit_error = [](double dt) {
    quasar::Grid2D g{64, 64, 4.0, 4.0, 0.0, 0.0, 1};
    quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
    constexpr double b = 0.7;
    constexpr double v0 = 0.2;
    constexpr double x0 = 2.0;
    constexpr double y0 = 2.0;
    constexpr double t_end = 0.8;
    fill(solver.external_fields().bz, b);

    quasar::pic::ParticleSpecies sp{
        quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
    sp.set_host_particles({x0}, {y0}, {v0}, {0.0}, {0.0}, {0.0});
    solver.add_species(std::move(sp));
    const int steps = static_cast<int>(std::llround(t_end / dt));
    for (int step = 0; step < steps; ++step) solver.step(dt);

    const auto snap = solver.species()[0].to_host();
    const double exact_x = x0 + (v0 / b) * std::sin(b * t_end);
    const double exact_y = y0 + (v0 / b) * (std::cos(b * t_end) - 1.0);
    return std::hypot(snap.x[0] - exact_x, snap.y[0] - exact_y);
  };

  const double coarse_error = orbit_error(0.04);
  const double fine_error = orbit_error(0.02);
  EXPECT_GT(coarse_error, 1.0e-10);
  EXPECT_LT(fine_error, coarse_error / 3.5)
      << "coarse=" << coarse_error << " fine=" << fine_error;
}

TEST(PicSingleParticle, FiniteExtremeMagneticFieldHasFinitePiRotationLimit) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  fill(solver.external_fields().bz, 1.0e300);
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  sp.set_host_particles({0.5}, {0.5}, {0.1}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.step(0.01);

  const auto snap = solver.species()[0].to_host();
  EXPECT_TRUE(std::isfinite(snap.vx[0]));
  EXPECT_TRUE(std::isfinite(snap.vy[0]));
  EXPECT_NEAR(snap.vx[0], -0.1, 2.0e-14);
  EXPECT_NEAR(std::hypot(snap.vx[0], snap.vy[0]), 0.1, 2.0e-14);
}

TEST(PicSingleParticle, ExtremeElectricProductDoesNotUnderflowAtIntermediateStep) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  const double field = std::numeric_limits<double>::max();
  const double qm = std::numeric_limits<double>::denorm_min();
  constexpr double dt = 0.01;
  fill(solver.external_fields().ex, field);

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", qm, 1.0, 1}};
  sp.set_host_particles({0.5}, {0.5}, {0.0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.step(dt);

  // In binary64, (qm*dt/2) underflows to zero even though qm*(dt/2)*E is a
  // normal, representable startup kick. The exponent-scaled pusher must retain
  // that product.
  const double expected = static_cast<double>(
      static_cast<long double>(qm) * (static_cast<long double>(dt) / 2.0L)
      * static_cast<long double>(field));
  const auto snap = solver.species()[0].to_host();
  EXPECT_GT(snap.vx[0], 0.0);
  EXPECT_NEAR(snap.vx[0], expected, std::abs(expected) * 2.0e-13);
  EXPECT_DOUBLE_EQ(snap.vy[0], 0.0);
  EXPECT_DOUBLE_EQ(snap.vz[0], 0.0);
}

TEST(PicSingleParticle, PushRejectsNonfiniteAndSuperluminalCandidateStates) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto make_solver = [](double ex) {
    quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
    quasar::pic::EmPic2D3V solver{
        quasar::pic::EmPicConfig{g, 2, "cic"}};
    fill(solver.external_fields().ex, ex);
    quasar::pic::ParticleSpecies sp{
        quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
    sp.set_host_particles({0.5}, {0.5}, {0.0}, {0.0}, {0.0}, {0.0});
    solver.add_species(std::move(sp));
    return solver;
  };

  auto superluminal = make_solver(200.0);
  EXPECT_THROW(superluminal.step(0.01), std::runtime_error);
  auto nonfinite = make_solver(std::numeric_limits<double>::infinity());
  EXPECT_THROW(nonfinite.step(0.01), std::runtime_error);
}

TEST(PicSingleParticle, CylindricalAxialFieldUsesRightHandedRphiZBasis) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{64, 32, 4.0, 2.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};
  constexpr double b_z = 1.1;
  constexpr double v0 = 0.2;
  constexpr double r0 = 1.5;
  constexpr double dt = 0.02;
  fill(solver.external_fields().by, b_z);  // cylindrical Bz storage slot

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  sp.set_host_particles({r0}, {1.0}, {v0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.step(dt);

  const double theta = 2.0 * std::atan(0.25 * b_z * dt);
  const double vr_prebasis = v0 * std::cos(theta);
  const double vphi_prebasis = -v0 * std::sin(theta);
  const double xx = r0 + dt * vr_prebasis;
  const double yy = dt * vphi_prebasis;
  const double dphi = std::atan2(yy, xx);
  const double expected_vr =
      vr_prebasis * std::cos(dphi) + vphi_prebasis * std::sin(dphi);
  const double expected_vphi =
      -vr_prebasis * std::sin(dphi) + vphi_prebasis * std::cos(dphi);

  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.vx[0], expected_vr, 2.0e-13);
  EXPECT_NEAR(snap.vz[0], expected_vphi, 2.0e-13);
  EXPECT_LT(snap.vz[0], 0.0);  // +vr x +Bz = -vphi for q>0
  EXPECT_NEAR(std::sqrt(snap.vx[0] * snap.vx[0] +
                        snap.vz[0] * snap.vz[0]),
              v0, 2.0e-13);
}
