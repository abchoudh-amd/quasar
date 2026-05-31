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
  quasar::pic::EmPicConfig cfg{g, order, "cic"};
  cfg.boundary.field[static_cast<int>(quasar::Side::x_lo)] =
      quasar::boundary::FieldBoundaryKind::outflow;
  cfg.boundary.field[static_cast<int>(quasar::Side::x_hi)] =
      quasar::boundary::FieldBoundaryKind::outflow;
  return quasar::pic::EmPic2D3V{cfg};
}

void run_channel_bleed(int order, int nghost, double tol) {
  const int nx = 256, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, nghost};
  auto solver = make_channel(g, order);
  seed_pulse(solver, g, 0.5, 0.06);
  const double dt = 0.5 * g.dx();
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  const int steps = static_cast<int>(4.0 / dt);
  for (int s = 0; s < steps; ++s) solver.step(dt);
  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  EXPECT_LT(e1, tol * e0) << "outflow channel retained too much energy (order "
                          << order << "): " << e0 << " -> " << e1;
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

TEST(PicOutflowFieldAbsorption, PulseLeavesLowReflectedEnergyOrder4) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const int nx = 256, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, 2};
  auto solver = make_channel(g, 4);
  seed_pulse(solver, g, 0.5, 0.05);
  const double dt = 0.5 * g.dx();
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  const int steps = static_cast<int>(1.2 / dt);
  for (int s = 0; s < steps; ++s) solver.step(dt);
  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  EXPECT_LT(e1, 0.10 * e0) << "order-4 outflow reflected too much: "
                           << e0 << " -> " << e1;
}

TEST(PicOutflowFieldAbsorption, ChannelBleedsToZeroOrder2) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  run_channel_bleed(2, 1, 0.05);
}

TEST(PicOutflowFieldAbsorption, ChannelBleedsToZeroOrder4) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  run_channel_bleed(4, 2, 0.05);
}

// NOTE: an open box with outflow on ALL FOUR sides is intentionally not tested.
// First-order Mur is weakly unstable where two Mur walls meet at a corner (the
// two one-way-wave conditions couple through the shared corner node), so an
// all-outflow box can grow without a dedicated corner-extrapolation closure,
// which is out of scope here. The supported, stable configurations are an
// outflow channel (outflow on one axis, periodic/other on the rest) and outflow
// mixed with PEC walls (the PEC pin breaks the corner feedback) -- both covered
// above and below. See docs/CHANGELOG for the documented limitation.

// Mixed corner: PEC on the x walls, outflow on the y walls. The PEC corner must
// win for the doubly-tangential ez (pinned 0, not Mur'd), the box must never gain
// energy, and the outflow y walls must still remove some.
TEST(PicOutflowFieldAbsorption, PecXOutflowYCornerStable) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const int nx = 64, ny = 64;
  quasar::Grid2D g{nx, ny, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.boundary.field[static_cast<int>(quasar::Side::x_lo)] =
      quasar::boundary::FieldBoundaryKind::pec;
  cfg.boundary.field[static_cast<int>(quasar::Side::x_hi)] =
      quasar::boundary::FieldBoundaryKind::pec;
  cfg.boundary.field[static_cast<int>(quasar::Side::y_lo)] =
      quasar::boundary::FieldBoundaryKind::outflow;
  cfg.boundary.field[static_cast<int>(quasar::Side::y_hi)] =
      quasar::boundary::FieldBoundaryKind::outflow;
  quasar::pic::EmPic2D3V solver{cfg};

  // A pulse with a +y component so the outflow walls actually see flux.
  auto& F = solver.fields();
  std::vector<double> ez(g.storage_size(), 0.0), bx(g.storage_size(), 0.0);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) / nx, y = (j + 0.5) / ny;
      const double e = std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
                                / (2 * 0.06 * 0.06));
      ez[g.index(i, j)] = e;
      bx[g.index(i, j)] = e;  // Ez/Bx pair -> +y propagation component
    }
  }
  F.ez.copy_from_host(ez.data(), ez.size());
  F.bx.copy_from_host(bx.data(), bx.size());

  const double dt = 0.5 * g.dx();
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  double emax = e0;
  for (int s = 0; s < static_cast<int>(3.0 / dt); ++s) {
    solver.step(dt);
    emax = std::max(emax, quasar::pic::total_em_energy(solver.fields(), solver.grid()));
  }
  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  // Corner ez is shorted by the PEC pin (no Mur double-update), so no growth.
  EXPECT_LT(emax, 1.2 * e0) << "mixed PEC/outflow corner grew energy: " << emax;
  // The open y walls remove energy overall.
  EXPECT_LT(e1, 0.7 * e0) << "outflow y walls did not absorb: " << e0 << " -> " << e1;
}

// Long-run stability guard for the 4th-order one-sided boundary closure: an
// outflow channel must never gain energy over many thousands of steps (the
// reduced-order closure is energy-neutral; a sign error would grow unboundedly).
TEST(PicOutflowFieldAbsorption, Order4ChannelLongRunStable) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const int nx = 64, ny = 4;
  quasar::Grid2D g{nx, ny, 1.0, ny / static_cast<double>(nx), 0.0, 0.0, 2};
  auto solver = make_channel(g, 4);
  seed_pulse(solver, g, 0.5, 0.08);
  const double dt = 0.5 * g.dx();
  const double e0 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  for (int s = 0; s < 5000; ++s) {
    solver.step(dt);
    if (s % 500 == 0) {
      const double e = quasar::pic::total_em_energy(solver.fields(), solver.grid());
      ASSERT_LT(e, 1.5 * e0 + 1e-12) << "energy blew up at step " << s
                                     << ": " << e << " (e0=" << e0 << ")";
      ASSERT_FALSE(std::isnan(e)) << "energy NaN at step " << s;
    }
  }
  const double e1 = quasar::pic::total_em_energy(solver.fields(), solver.grid());
  EXPECT_LT(e1, 0.05 * e0) << "long-run outflow channel did not bleed: "
                           << e0 << " -> " << e1;
}
