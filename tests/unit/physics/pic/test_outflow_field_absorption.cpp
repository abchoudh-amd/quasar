// First-order Mur outflow (open) field wall.
//
// An outflow wall lets an outgoing wave leave with little reflection. These
// cases use an outflow channel (outflow on both x walls, periodic in y) so a
// uniform-in-y pulse bleeds out the x ends without involving outflow-outflow
// corners (corner behavior is exercised separately by the mixed-BC test). A 1D
// Ez/By pulse must leave most of its energy, and a channel seeded with a pulse
// must bleed essentially all of it after a few light-crossing times.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

void seed_pulse(quasar::pic::EmPic2D3V& solver, const quasar::Grid2D& g,
                double x0, double sigma) {
  auto& F = solver.fields();
  std::vector<double> ez(g.storage_size(), 0.0), by(g.storage_size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const double x = (i + 0.5) / g.nx;
      const double e = std::exp(-((x - x0) * (x - x0)) / (2 * sigma * sigma));
      ez[g.index(i, j)] = e;
      by[g.index(i, j)] = -e;  // +x propagation, c = 1
    }
  }
  F.ez.copy_from_host(ez.data(), ez.size());
  F.by.copy_from_host(by.data(), by.size());
}

// Outflow on both x walls, periodic in y (a channel; no outflow-outflow corner).
quasar::pic::EmPic2D3V make_channel(const quasar::Grid2D& g, int order) {
  quasar::pic::EmPicConfig cfg{g, order, 1};
  cfg.boundary.field[static_cast<int>(quasar::Side::x_lo)] =
      quasar::boundary::FieldBoundaryKind::outflow;
  cfg.boundary.field[static_cast<int>(quasar::Side::x_hi)] =
      quasar::boundary::FieldBoundaryKind::outflow;
  return quasar::pic::EmPic2D3V{cfg};
}

}  // namespace

TEST(PicOutflowFieldAbsorption, PulseLeavesLowReflectedEnergyOrder2) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const int nx = 256, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, 1};
  auto solver = make_channel(g, 2);

  seed_pulse(solver, g, 0.5, 0.05);
  const double dt = 0.5 * g.dx();
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());

  // One light-crossing carries the pulse to a wall and out.
  const int steps = static_cast<int>(1.2 / dt);
  for (int s = 0; s < steps; ++s) solver.step(dt);

  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  EXPECT_LT(e1, 0.10 * e0) << "outflow reflected too much energy: "
                           << e0 << " -> " << e1;
}

TEST(PicOutflowFieldAbsorption, ChannelBleedsToZeroOrder2) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const int nx = 256, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, 1};
  auto solver = make_channel(g, 2);

  seed_pulse(solver, g, 0.5, 0.06);
  const double dt = 0.5 * g.dx();
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());

  // Several light-crossings: the split pulse exits both ends and any first-order
  // Mur residual decays.
  const int steps = static_cast<int>(4.0 / dt);
  for (int s = 0; s < steps; ++s) solver.step(dt);

  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  EXPECT_LT(e1, 0.05 * e0) << "outflow channel retained too much energy: "
                           << e0 << " -> " << e1;
}
