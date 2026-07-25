// PEC plane-wave reflection.
//
// With the ghost-aware FDTD stencil, a PEC field wall imposes its boundary every
// step. A 1D plane wave (Ez, By) bouncing between two PEC x-walls (periodic in y)
// must:
//   * conserve total EM energy across a full reflect-and-return (a perfect
//     conductor does no work and the scheme is lossless), and
//   * keep the tangential electric field (Ez) at the conducting wall ~ 0.
// Run for both 2nd- and 4th-order curls.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

template <int Order>
void run_reflection(int nghost) {
  const int nx = 128, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, nghost};
  quasar::pic::EmPicConfig cfg{g, Order, "cic"};
  // PEC on the x walls, periodic on y.
  cfg.boundary.field[static_cast<int>(quasar::Side::x_lo)] =
      "pec";
  cfg.boundary.field[static_cast<int>(quasar::Side::x_hi)] =
      "pec";
  cfg.boundary.particle[static_cast<int>(quasar::Side::x_lo)] =
      "specular";
  cfg.boundary.particle[static_cast<int>(quasar::Side::x_hi)] =
      "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  // Seed a localized Ez/By pulse travelling +x.
  auto& F = solver.fields();
  std::vector<double> ez(g.storage_size(), 0.0), by(g.storage_size(), 0.0);
  const double x0 = 0.3, sigma = 0.05;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) / nx;
      const double e = std::exp(-((x - x0) * (x - x0)) / (2 * sigma * sigma));
      ez[g.index(i, j)] = e;
      by[g.index(i, j)] = -e;  // Ez = +By magnitude, +x propagation (c = 1)
    }
  }
  F.ez.copy_from_host(ez.data(), ez.size());
  F.by.copy_from_host(by.data(), by.size());

  const double dx = g.dx();
  const double dt = 0.5 * dx;  // c = 1
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());

  // Step long enough for the pulse to hit x_hi and return past center.
  const int steps = static_cast<int>(1.6 / dt);
  double max_wall_ez = 0.0;
  for (int s = 0; s < steps; ++s) {
    solver.step(dt);
    std::vector<double> cur(g.storage_size());
    F.ez.copy_to_host(cur.data(), cur.size());
    // Tangential E at the conducting wall (last interior column) must stay small.
    for (int j = 0; j < ny; ++j) {
      max_wall_ez = std::max(max_wall_ez, std::abs(cur[g.index(nx - 1, j)]));
    }
  }

  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  // Energy conserved through the reflection (no leak / no gain).
  EXPECT_NEAR(e1, e0, 0.02 * e0) << "PEC reflection changed EM energy: "
                                 << e0 << " -> " << e1;
  // The incident pulse peaks at ~1; at the conducting wall tangential E is held
  // near zero by the odd mirror.
  EXPECT_LT(max_wall_ez, 0.1) << "tangential E at PEC wall too large: " << max_wall_ez;
}

}  // namespace

TEST(PicPecPlaneWaveReflection, Order2ConservesEnergyAndZeroesTangentialE) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  run_reflection<2>(1);
}

TEST(PicPecPlaneWaveReflection, Order4ConservesEnergyAndZeroesTangentialE) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  run_reflection<4>(2);
}
