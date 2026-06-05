// Cylindrical (r,z) diagnostics: total_em_energy must integrate the field energy
// over the axisymmetric ring volume cell_volume(i) = 2*pi*r*dr*dz, not the flat
// Cartesian area, and gauss_residual must use the axisymmetric divergence with
// the radial metric factor and the natural on-axis (r=0) closure. These pin the
// geometry-aware diagnostics so a cylindrical run reports physical numbers.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

quasar::pic::EmPicConfig cyl_cfg(const quasar::Grid2D& g) {
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  // A cylindrical run wants a non-periodic outer-radius wall; pec is fine here
  // (we set fields directly and never step). x_lo is auto-wired to the axis BC.
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  return cfg;
}

}  // namespace

TEST(PicCylDiagnostics, EmEnergyUsesRingVolume) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // r-domain [0, 1], z-domain [0, 1]; r=0 on the axis at i=0.
  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{cyl_cfg(g)};

  const quasar::Real E0 = 3.0, B0 = 2.0;
  auto& f = solver.fields();
  std::vector<quasar::Real> ez(f.ez.size(), 0.0), bz(f.bz.size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ez[g.index(i, j)] = E0;
      bz[g.index(i, j)] = B0;
    }
  }
  f.ez.copy_from_host(ez.data(), ez.size());
  f.bz.copy_from_host(bz.data(), bz.size());

  // u = 0.5*(E^2 + B^2) per cell, weighted by the ring volume cell_volume(i)
  // (which varies with radius), summed over the interior.
  quasar::Real expected = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      expected += 0.5 * (E0 * E0 + B0 * B0) * g.cell_volume(i);
    }
  }

  const quasar::Real got =
      quasar::pic::total_em_energy(solver.fields(), solver.grid(), /*cylindrical=*/true);
  EXPECT_NEAR(got, expected, 1e-9);

  // The flat-area Cartesian weighting would give a different (wrong) number for a
  // cylindrical run; confirm the ring weighting actually changed the result.
  const quasar::Real flat =
      quasar::pic::total_em_energy(solver.fields(), solver.grid(), /*cylindrical=*/false);
  EXPECT_GT(std::abs(got - flat), 1e-6);
}

TEST(PicCylDiagnostics, GaussResidualUniformAxialFieldIsZero) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{cyl_cfg(g)};

  // A uniform axial field E_z (= ey in the (r,z,phi) mapping) has zero
  // divergence: d E_z/dz = 0 and there is no radial component, so the
  // axisymmetric residual must be ~0.
  auto& f = solver.fields();
  std::vector<quasar::Real> ey(f.ey.size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) ey[g.index(i, j)] = 1.7;
  }
  f.ey.copy_from_host(ey.data(), ey.size());

  EXPECT_NEAR(
      quasar::pic::gauss_residual(solver.fields(), solver.current(), /*cylindrical=*/true),
      0.0, 1e-9);
}

TEST(PicCylDiagnostics, GaussResidualRadialFieldMatchesMetricDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 1.0, 1.0, 0.0, 0.0, 1};  // dr = dz = 0.25
  quasar::pic::EmPic2D3V solver{cyl_cfg(g)};

  // A single nonzero radial component E_r (= ex) at one interior node. The
  // cylindrical residual is the L2 norm of (1/r_c) d(r E_r)/dr, nonzero only at
  // that node and its +r neighbour. Compute the expectation from the same metric
  // the implementation uses.
  auto& f = solver.fields();
  std::vector<quasar::Real> ex(f.ex.size(), 0.0);
  const quasar::Real val = 2.0;
  const int ic = 2, jc = 1;  // interior node away from the axis
  ex[g.index(ic, jc)] = val;
  f.ex.copy_from_host(ex.data(), ex.size());

  // div at (ic,jc): (r_e(ic)*val - r_e(ic-1)*0)/(r_c(ic)*dr)
  // div at (ic+1,jc): (r_e(ic+1)*0 - r_e(ic)*val)/(r_c(ic+1)*dr)
  const quasar::Real dr = g.dx();
  const quasar::Real d_here = (g.r_at_edge(ic) * val) / (g.r_at_cell_center(ic) * dr);
  const quasar::Real d_next = (-g.r_at_edge(ic) * val) / (g.r_at_cell_center(ic + 1) * dr);
  const quasar::Real expected = std::sqrt(d_here * d_here + d_next * d_next);

  EXPECT_NEAR(
      quasar::pic::gauss_residual(solver.fields(), solver.current(), /*cylindrical=*/true),
      expected, 1e-9);
}
