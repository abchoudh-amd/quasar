// Diagnostics magnitude checks: total_kinetic_energy and total_em_energy must
// return the analytically known values for hand-seeded particle velocities and
// a uniform field, in the solver's normalized natural units (c = eps0 = mu0 = 1,
// so u_em = 0.5*(E^2 + B^2)).

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

TEST(PicEnergyConservation, DiagnosticsStubReturnsFinite) {
  // Default-constructed field (empty buffers) must not read out of bounds.
  EXPECT_DOUBLE_EQ(
      quasar::pic::total_em_energy(quasar::YeeField2D<double>{}, quasar::Grid2D{}), 0.0);
}

TEST(PicEnergyConservation, KineticEnergyMatchesAnalytic) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // 3 particles, mass m, weight w, known speeds. KE = sum 0.5*m*|v|^2*w.
  const quasar::Real m = 2.0;
  const std::size_t n = 3;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"p", 1.0, m, n}};
  std::vector<quasar::Real> x{0.1, 0.2, 0.3}, y{0.1, 0.2, 0.3};
  std::vector<quasar::Real> vx{1.0, 0.0, 3.0}, vy{0.0, 2.0, 4.0}, vz{0.0, 0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 2.0, 0.5};
  sp.set_host_particles(x, y, vx, vy, vz, w);

  // KE = 0.5*2*[ (1)*1 + (4)*2 + (9+16)*0.5 ] = [1 + 8 + 12.5] = 21.5
  const quasar::Real expected = 0.5 * m * (1.0 * 1.0 + 4.0 * 2.0 + 25.0 * 0.5);
  EXPECT_NEAR(quasar::pic::total_kinetic_energy(sp), expected, 1e-9);
}

TEST(PicEnergyConservation, KineticEnergyIgnoresDeadParticles) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Seed two particles in a solver with absorbing walls; push one out so it dies.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = quasar::boundary::ParticleBoundaryKind::absorbing;
  }
  quasar::pic::EmPic2D3V solver{cfg};

  const std::size_t n = 2;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, n}};
  std::vector<quasar::Real> x{0.5, 0.95}, y{0.5, 0.5};
  std::vector<quasar::Real> vx{0.0, 5.0}, vy{0.0, 0.0}, vz{0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  for (int s = 0; s < 4; ++s) solver.step(0.05);

  // The fast particle is dead; KE counts only the (near-stationary) survivor.
  // Its speed stays tiny, so KE must be far below the initial 0.5*1*25 = 12.5.
  EXPECT_LT(quasar::pic::total_kinetic_energy(solver.species()[0]), 1.0);
}

TEST(PicEnergyConservation, EmEnergyMatchesUniformField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 2.0, 2.0, 0.0, 0.0, 1};  // dx=dy=0.25, dA=0.0625
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};

  // Set a uniform Ez = E0 and Bx = B0 over the interior storage.
  const quasar::Real E0 = 3.0, B0 = 2.0;
  auto& f = solver.fields();
  std::vector<quasar::Real> ez(f.ez.size(), 0.0), bx(f.bx.size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ez[g.index(i, j)] = E0;
      bx[g.index(i, j)] = B0;
    }
  }
  f.ez.copy_from_host(ez.data(), ez.size());
  f.bx.copy_from_host(bx.data(), bx.size());

  // u = 0.5*(E^2 + B^2) per cell, times dA, times nx*ny cells.
  const quasar::Real dA = g.dx() * g.dy();
  const quasar::Real expected =
      0.5 * (E0 * E0 + B0 * B0) * dA * static_cast<quasar::Real>(g.nx * g.ny);
  EXPECT_NEAR(quasar::pic::total_em_energy(solver.fields(), solver.grid()),
              expected, 1e-9);
}

TEST(PicGaussResidual, UniformFieldHasZeroDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};

  // Uniform Ex over the (periodically-wrapped) grid: backward differences cancel
  // everywhere, so the discrete divergence — and the residual — must be ~0.
  auto& f = solver.fields();
  std::vector<quasar::Real> ex(f.ex.size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) ex[g.index(i, j)] = 1.7;
  }
  f.ex.copy_from_host(ex.data(), ex.size());

  EXPECT_NEAR(quasar::pic::gauss_residual(solver.fields(), solver.current()),
              0.0, 1e-9);
}

TEST(PicGaussResidual, NonUniformFieldMatchesHandComputed) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{4, 4, 1.0, 1.0, 0.0, 0.0, 1};  // dx = dy = 0.25
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};

  // Put a single nonzero Ex at one interior cell. The residual is the L2 norm of
  // the backward-difference divergence (∂Ex/∂x), which is nonzero only at that
  // cell and its +x periodic neighbour. Compute the expectation directly.
  auto& f = solver.fields();
  std::vector<quasar::Real> ex(f.ex.size(), 0.0);
  const quasar::Real val = 2.0;
  ex[g.index(1, 1)] = val;
  f.ex.copy_from_host(ex.data(), ex.size());

  const quasar::Real inv_dx = 1.0 / g.dx();
  // dExdx at (1,1) = (ex[1,1]-ex[0,1])/dx = +val*inv_dx;
  // dExdx at (2,1) = (ex[2,1]-ex[1,1])/dx = -val*inv_dx; others zero.
  const quasar::Real d1 = val * inv_dx;
  const quasar::Real expected = std::sqrt(d1 * d1 + d1 * d1);

  EXPECT_NEAR(quasar::pic::gauss_residual(solver.fields(), solver.current()),
              expected, 1e-9);
}
