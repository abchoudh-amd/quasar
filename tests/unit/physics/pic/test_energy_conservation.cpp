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
#include <limits>
#include <utility>
#include <vector>

TEST(PicEnergyConservation, DefaultConstructedFieldEnergyIsZero) {
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
  std::vector<quasar::Real> vx{0.1, 0.0, 0.3};
  std::vector<quasar::Real> vy{0.0, 0.2, 0.4};
  std::vector<quasar::Real> vz{0.0, 0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 2.0, 0.5};
  sp.set_host_particles(x, y, vx, vy, vz, w);

  const quasar::Real expected =
      0.5 * m * (0.01 * 1.0 + 0.04 * 2.0 + 0.25 * 0.5);
  EXPECT_NEAR(quasar::pic::total_kinetic_energy(sp), expected, 1e-9);
}

TEST(PicEnergyConservation, KineticEnergyIgnoresDeadParticles) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Seed two particles in a solver with absorbing walls; push one out so it dies.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.particle[side] = "absorbing";
    cfg.boundary.field[side] = "pec";
  }
  quasar::pic::EmPic2D3V solver{cfg};

  const std::size_t n = 2;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"e", -1.0, 1.0, n}};
  std::vector<quasar::Real> x{0.5, 0.99}, y{0.5, 0.5};
  std::vector<quasar::Real> vx{0.0, 0.8}, vy{0.0, 0.0}, vz{0.0, 0.0};
  std::vector<quasar::Real> w{1.0, 1.0};
  sp.set_host_particles(x, y, vx, vy, vz, w);
  solver.add_species(std::move(sp));

  solver.step(0.02);

  // The fast particle is dead; KE counts only the (near-stationary) survivor.
  EXPECT_NEAR(quasar::pic::total_kinetic_energy(solver.species()[0]), 0.0,
              1.0e-12);
}

TEST(PicEnergyConservation, UniformMagneticBorisStepConservesKineticEnergy) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  // A single charged species on the default periodic torus needs an explicit
  // immobile counter-charge.  This test isolates the magnetic Boris rotation,
  // so that is precisely the intended physical model.
  cfg.neutralizing_background = true;
  quasar::pic::EmPic2D3V solver{cfg};
  std::vector<quasar::Real> bz(g.storage_size(), 1.7);
  solver.external_fields().bz.copy_from_host(bz.data(), bz.size());
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"q", 1.0, 1.0, 1}};
  sp.set_host_particles({0.5}, {0.5}, {0.31}, {-0.27}, {0.11}, {1.0});
  const quasar::Real before = quasar::pic::total_kinetic_energy(sp);
  solver.add_species(std::move(sp));
  solver.step(0.02);
  const quasar::Real after =
      quasar::pic::total_kinetic_energy(solver.species()[0]);
  EXPECT_NEAR(after, before, 2.0e-14);
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

TEST(PicEnergyConservation, YeeEnergyRetainsCheckerboardMode) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const quasar::Grid2D g{4, 3, 2.0, 1.5, 0.0, 0.0, 1};
  quasar::YeeField2D<double> f{g};
  std::vector<double> ex(g.storage_size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ex[g.index(i, j)] = (i & 1) == 0 ? 1.0 : -1.0;
    }
  }
  f.ex.copy_from_host(ex.data(), ex.size());

  // Every unique periodic Ex face has |Ex|=1, so the exact Yee norm is
  // 0.5*domain_area. Averaging adjacent faces before squaring would erase this
  // Nyquist mode completely.
  EXPECT_NEAR(quasar::pic::total_em_energy(f, g), 0.5 * g.lx * g.ly,
              2.0e-15);
}

TEST(PicElectricDivergence, UniformFieldHasZeroDivergence) {
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

  EXPECT_NEAR(quasar::pic::electric_divergence_norm(
                  solver.fields(), 2, solver.config().boundary),
              0.0, 1e-9);
}

TEST(PicElectricDivergence, NonUniformFieldMatchesHandComputed) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{4, 4, 1.0, 1.0, 0.0, 0.0, 1};  // dx = dy = 0.25
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};

  // Put a single nonzero Ex at one interior face. The reported norm is the
  // volume-weighted RMS of the forward-difference divergence, nonzero in two of
  // the 16 equal-volume cells.
  auto& f = solver.fields();
  std::vector<quasar::Real> ex(f.ex.size(), 0.0);
  const quasar::Real val = 2.0;
  ex[g.index(1, 1)] = val;
  f.ex.copy_from_host(ex.data(), ex.size());

  const quasar::Real inv_dx = 1.0 / g.dx();
  // dExdx at (0,1)=+val/dx and at (1,1)=-val/dx; others zero.
  const quasar::Real d1 = val * inv_dx;
  const quasar::Real expected = std::sqrt((d1 * d1 + d1 * d1) / 16.0);

  EXPECT_NEAR(quasar::pic::electric_divergence_norm(
                  solver.fields(), 2, solver.config().boundary),
              expected, 1e-9);
}

TEST(PicElectricDivergence, ExtremeCurlAndChargeCancellationStaysFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{1, 1, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::YeeField2D<quasar::Real> fields{g};
  quasar::ScalarGrid2D<quasar::Real> charge{g};
  quasar::boundary::BoundarySpec boundary;
  boundary.field = {"pec", "pec", "pec", "pec"};

  const double a = 0.75 * std::numeric_limits<double>::max();
  std::vector<double> ex(g.storage_size(), 0.0);
  std::vector<double> ey(g.storage_size(), 0.0);
  std::vector<double> rho(g.storage_size(), 0.0);
  ex[g.index(0, 0)] = -0.5 * a;
  ex[g.index(1, 0)] = 0.5 * a;  // dEx/dx = a
  ey[g.index(0, 0)] = -0.5 * a;
  ey[g.index(0, 1)] = 0.5 * a;  // dEy/dy = a
  rho[g.index(0, 0)] = a;
  fields.ex.copy_from_host(ex.data(), ex.size());
  fields.ey.copy_from_host(ey.data(), ey.size());
  charge.values.copy_from_host(rho.data(), rho.size());

  // div(E) is 1.5*DBL_MAX and is not representable by itself, while the complete
  // Gauss residual a + a - a = a is finite.
  EXPECT_DOUBLE_EQ(quasar::pic::gauss_residual(
                       fields, charge, 2, boundary, false),
                   a);
}
