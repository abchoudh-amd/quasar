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
#include <limits>
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
  std::vector<quasar::Real> ey(f.ey.size(), 0.0), by(f.by.size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ey[g.index(i, j)] = E0;
      by[g.index(i, j)] = B0;
    }
  }
  f.ey.copy_from_host(ey.data(), ey.size());
  f.by.copy_from_host(by.data(), by.size());

  // u = 0.5*(E^2 + B^2) per cell, weighted by the ring volume cell_volume(i)
  // (which varies with radius), summed over the interior.
  quasar::Real expected = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      expected += 0.5 * (E0 * E0 + B0 * B0) * g.cell_volume(i);
    }
  }

  const quasar::Real got = quasar::pic::total_em_energy(solver);
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

  EXPECT_NEAR(quasar::pic::electric_divergence_norm(
                  solver.fields(), 2, solver.config().boundary,
                  /*cylindrical=*/true),
              0.0, 1e-9);
}

TEST(PicCylDiagnostics, GaussResidualRadialFieldMatchesMetricDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 1.0, 1.0, 0.0, 0.0, 1};  // dr = dz = 0.25
  quasar::pic::EmPic2D3V solver{cyl_cfg(g)};

  // A single nonzero radial component E_r (= ex) at one interior node. The
  // cylindrical residual is the ring-volume-weighted RMS of
  // (1/r_c)d(r E_r)/dr, nonzero in the charge cells on either side of that face.
  auto& f = solver.fields();
  std::vector<quasar::Real> ex(f.ex.size(), 0.0);
  const quasar::Real val = 2.0;
  const int ic = 2, jc = 1;  // interior node away from the axis
  ex[g.index(ic, jc)] = val;
  f.ex.copy_from_host(ex.data(), ex.size());

  // div at (ic-1,jc): +r_e(ic)*val/(r_c(ic-1)*dr)
  // div at (ic,jc):   -r_e(ic)*val/(r_c(ic)*dr)
  const quasar::Real dr = g.dx();
  const quasar::Real d_here =
      (g.r_at_edge(ic) * val) / (g.r_at_cell_center(ic - 1) * dr);
  const quasar::Real d_next =
      (-g.r_at_edge(ic) * val) / (g.r_at_cell_center(ic) * dr);
  quasar::Real weighted = d_here * d_here * g.cell_volume(ic - 1)
                        + d_next * d_next * g.cell_volume(ic);
  quasar::Real volume = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) volume += g.cell_volume(i);
  }
  const quasar::Real expected = std::sqrt(weighted / volume);

  EXPECT_NEAR(quasar::pic::electric_divergence_norm(
                  solver.fields(), 2, solver.config().boundary,
                  /*cylindrical=*/true),
              expected, 1e-9);
}

TEST(PicCylDiagnostics, ThinLargeRadiusAnnulusRetainsEnergyAndZeroGaussNorm) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // r^2 overflows and 2*r0 overflows in binary64, while the factored annular
  // volume remains finite after multiplication by the very short z extent.
  constexpr double r0 = 1.0e308;
  constexpr double dr = 8.0e292;
  constexpr double lz = 1.0e-308;
  quasar::Grid2D g{1, 2, dr, lz, r0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[0] = "pec";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[0] = "specular";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  std::vector<double> ey(g.storage_size(), 0.0);
  for (int j = 0; j < g.ny; ++j) ey[g.index(0, j)] = 1.0;
  solver.fields().ey.copy_from_host(ey.data(), ey.size());

  const long double annular_volume =
      std::acos(-1.0L) * static_cast<long double>(dr)
      * (2.0L * static_cast<long double>(r0)
         + static_cast<long double>(dr))
      * static_cast<long double>(lz);
  ASSERT_TRUE(std::isfinite(annular_volume));
  const double expected = static_cast<double>(0.5L * annular_volume);
  ASSERT_TRUE(std::isfinite(expected));
  EXPECT_NEAR(quasar::pic::total_em_energy(solver), expected,
              std::abs(expected) * 2.0e-14);
  EXPECT_DOUBLE_EQ(quasar::pic::electric_divergence_norm(
                       solver.fields(), 2, solver.config().boundary,
                       /*cylindrical=*/true),
                   0.0);
}

TEST(PicCylDiagnostics, ExtremeRadialAxialChargeCancellationStaysFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{1, 1, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::YeeField2D<quasar::Real> fields{g};
  quasar::ScalarGrid2D<quasar::Real> charge{g};
  quasar::boundary::BoundarySpec boundary;
  boundary.field = {"axis", "pec", "pec", "pec"};

  const double a = 0.75 * std::numeric_limits<double>::max();
  std::vector<double> er(g.storage_size(), 0.0);
  std::vector<double> ez(g.storage_size(), 0.0);
  std::vector<double> rho(g.storage_size(), 0.0);
  er[g.index(0, 0)] = 0.0;
  er[g.index(1, 0)] = 0.5 * a;  // (1/r)d(r Er)/dr = a at r=dr/2
  ez[g.index(0, 0)] = -0.5 * a;
  ez[g.index(0, 1)] = 0.5 * a;  // dEz/dz = a
  rho[g.index(0, 0)] = a;
  fields.ex.copy_from_host(er.data(), er.size());
  fields.ey.copy_from_host(ez.data(), ez.size());
  charge.values.copy_from_host(rho.data(), rho.size());

  EXPECT_DOUBLE_EQ(quasar::pic::gauss_residual(
                       fields, charge, 2, boundary, true),
                   a);
}
