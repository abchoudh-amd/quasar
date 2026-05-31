// Validates the FDTD vacuum dispersion: a single-mode plane wave seeded into the
// Yee fields must propagate at the numerical phase velocity predicted by the 2nd
// order Yee dispersion relation, which is close to c (= 1 in these natural units)
// for a well-resolved mode. With no particles the step() is a pure field solve.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// Numerical phase velocity of the 2nd-order Yee scheme for a wave along x:
//   sin(w dt / 2) / (c dt) = sin(k dx / 2) / dx
// Solve for w, then v_phase = w / k.
double yee_phase_velocity(double k, double dx, double dt, double c) {
  const double rhs = (c * dt / dx) * std::sin(k * dx / 2.0);
  const double w = 2.0 * std::asin(rhs) / dt;
  return w / k;
}

}  // namespace

TEST(PicEmWaveDispersion, PlaneWaveTravelsAtYeePhaseVelocity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // 1D-along-x wave on a tall-enough grid; periodic in both directions.
  const int nx = 64, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};

  const double c = 1.0;
  const int mode = 1;                          // one wavelength across lx
  const double k = 2.0 * M_PI * mode / g.lx;   // wavenumber
  const double dx = g.dx();
  const double dt = 0.5 * dx / c;              // comfortably below CFL

  // Seed a forward-travelling Ez/By plane wave: Ez = sin(kx), By = -Ez/c so the
  // mode propagates in +x. (Colocated seed; the scheme is self-consistent.)
  auto& f = solver.fields();
  std::vector<quasar::Real> ez(f.ez.size(), 0.0), by(f.by.size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const double x = (i + 0.5) * dx;
      const double e = std::sin(k * x);
      ez[g.index(i, j)] = e;
      by[g.index(i, j)] = -e / c;
    }
  }
  f.ez.copy_from_host(ez.data(), ez.size());
  f.by.copy_from_host(by.data(), by.size());

  // Advance several steps and measure the phase shift of the dominant mode by
  // projecting Ez onto sin/cos at wavenumber k (row j=0).
  const int steps = 8;
  for (int s = 0; s < steps; ++s) solver.step(dt);

  std::vector<quasar::Real> ez_out(f.ez.size());
  f.ez.copy_to_host(ez_out.data(), ez_out.size());

  double proj_sin = 0.0, proj_cos = 0.0;
  for (int i = 0; i < g.nx; ++i) {
    const double x = (i + 0.5) * dx;
    const double v = ez_out[g.index(i, 0)];
    proj_sin += v * std::sin(k * x);
    proj_cos += v * std::cos(k * x);
  }
  // Phase of the mode: Ez ~ sin(kx - phase) => atan2 picks up -phase.
  const double phase = std::atan2(-proj_cos, proj_sin);
  const double measured_v = (phase / k) / (steps * dt);

  const double expected_v = yee_phase_velocity(k, dx, dt, c);

  // The measured phase velocity must match the Yee prediction (and be near c).
  EXPECT_NEAR(measured_v, expected_v, 0.02 * expected_v)
      << "measured v=" << measured_v << " expected=" << expected_v;
  EXPECT_GT(measured_v, 0.9 * c);
  EXPECT_LT(measured_v, 1.01 * c);
}
