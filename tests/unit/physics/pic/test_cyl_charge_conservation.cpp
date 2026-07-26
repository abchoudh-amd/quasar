// Validates that the axisymmetric (r,z) Esirkepov deposit is charge-conserving:
// the deposited (Jr, Jz) must satisfy the discrete cylindrical continuity
//   (rho_new-rho_old)/dt
//     + (r_e(i+1)*Jr[i+1,j] - r_e(i)*Jr[i,j])/(r_c(i)*dr)
//     + (Jz[i,j+1]-Jz[i,j])/dz = 0
// cell-by-cell, where rho = q*w*S_r*S_z / cell_volume(i) (the ring volume). The
// forward ring-flux divergence is the cylindrical Ampere/Gauss operator, so
// satisfying it preserves Gauss's law on the axisymmetric grid. This is the
// cylindrical analogue of test_charge_conservation.cpp, which only covers the
// Cartesian deposit.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// rho = q*w*S_r*S_z / cell_volume(i), the same ring-volume normalization the
// cylindrical deposit uses, accumulated onto a host grid.
template <int ShapeOrder>
void accumulate_rho_cyl(std::vector<double>& rho, const quasar::Grid2D& g,
                        double q, double w, double r, double z) {
  const auto sw = quasar::numerics::shape_weights_2d<ShapeOrder>(r, z, g);
  for (int b = 0; b < sw.ny; ++b) {
    for (int a = 0; a < sw.nx; ++a) {
      int ii = sw.ix[a];
      if (ii < 0) ii = -ii - 1;
      if (ii < 0 || ii >= g.nx) continue;
      const double vol = g.cell_volume(ii);
      if (vol <= 0.0) continue;
      rho[g.index(ii, g.wrap_j(sw.iy[b]))] +=
          q * w * sw.wx[a] * sw.wy[b] / vol;
    }
  }
}

template <int ShapeOrder>
void accumulate_rho_cyl_reflect_outer(std::vector<double>& rho,
                                      const quasar::Grid2D& g, double q,
                                      double w, double r, double z) {
  const auto sw = quasar::numerics::shape_weights_2d<ShapeOrder>(r, z, g);
  for (int b = 0; b < sw.ny; ++b) {
    const int jj = g.wrap_j(sw.iy[b]);
    for (int a = 0; a < sw.nx; ++a) {
      int ii = sw.ix[a];
      if (ii < 0) ii = -ii - 1;
      if (ii >= g.nx) ii = 2 * g.nx - 1 - ii;
      if (ii < 0 || ii >= g.nx) continue;
      // Fold charge amount, then divide by the image ring's volume.  Directly
      // adding ghost density would be wrong because adjacent cylindrical rings
      // have different volumes.
      rho[g.index(ii, jj)] +=
          q * w * sw.wx[a] * sw.wy[b] / g.cell_volume(ii);
    }
  }
}

template <int ShapeOrder, int FdtdOrder = 2>
void run_cyl_continuity_case(double* out_resid, double* out_jmag,
                             bool near_axis = false) {
  const int halo = std::max(quasar::required_nghost(FdtdOrder),
                            ShapeOrder == 2 ? 2 : 1);
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0,
                   halo};
  quasar::pic::EmPicConfig cfg{g, FdtdOrder,
                               ShapeOrder == 1 ? "cic" : "tsc"};
  cfg.geometry = "cylindrical";
  // Outer radius must be a non-periodic wall; z stays periodic so its telescoping
  // divergence wraps cleanly. x_lo is auto-wired to the axis BC.
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  const double q = 1.0, m = 1.0, w = 1.0;
  const double r0 = near_axis ? 0.018 : 0.5;
  const double z0 = 0.5;
  // vphi = 0 so the cylindrical position advance is linear (no azimuthal sweep);
  // with zero initial fields the Boris rotation is the identity, so the post-move
  // position is exactly (r0 + dt*vr, z0 + dt*vz). vr = vx, vz(axial) = vy.
  const double vr = near_axis ? 0.21 : 0.37;
  const double vz_axial = -0.29, vphi = 0.0;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", q, m, 1}};
  sp.set_host_particles({r0}, {z0}, {vr}, {vz_axial}, {vphi}, {w});
  solver.add_species(std::move(sp));

  // Sub-CFL sub-cell displacement (dr = dz = 1/16, cyl CFL limit ~ 0.044).
  const double dt = FdtdOrder == 4 ? 0.02 : 0.03;
  solver.step(dt);

  const double r1 = r0 + dt * vr;
  const double z1 = z0 + dt * vz_axial;

  std::vector<double> rho_old(g.storage_size(), 0.0);
  std::vector<double> rho_new(g.storage_size(), 0.0);
  accumulate_rho_cyl<ShapeOrder>(rho_old, g, q, w, r0, z0);
  accumulate_rho_cyl<ShapeOrder>(rho_new, g, q, w, r1, z1);

  auto& J = solver.current();
  std::vector<double> jr(g.storage_size()), jz(g.storage_size());
  J.jx.copy_to_host(jr.data(), jr.size());
  J.jy.copy_to_host(jz.data(), jz.size());

  const double dr = g.dx();
  const double dz = g.dy();
  double max_resid = 0.0, max_jmag = 0.0;
  int max_i = -1, max_j = -1;
  double max_drho = 0.0, max_radial = 0.0, max_axial = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      const double drho = (rho_new[k] - rho_old[k]) / dt;
      double radial_flux;
      double axial_flux;
      if constexpr (FdtdOrder == 4) {
        const auto q = [&](int face) {
          int sample = face;
          if (sample < 0) sample = -sample;
          if (sample > g.nx) sample = 2 * g.nx - sample;
          // q=r*Jr is even at the axis because both r and Jr are odd.  The
          // stored positive-radius Jr therefore represents q(-r) with the
          // positive face radius, not the signed negative ghost coordinate.
          // At the outer PEC wall Jr itself continues evenly about the physical
          // face nx, while q uses the radius of the requested (possibly ghost)
          // face. The independent physical high face must not be clamped away.
          return g.r_at_edge(std::abs(face)) * jr[g.index(sample, j)];
        };
        radial_flux = (9.0 / 8.0) * (q(i + 1) - q(i))
                    - (1.0 / 24.0) * (q(i + 2) - q(i - 1));
        axial_flux =
            (9.0 / 8.0) * (jz[g.index(i, g.wrap_j(j + 1))] - jz[k])
          - (1.0 / 24.0) *
                (jz[g.index(i, g.wrap_j(j + 2))] -
                 jz[g.index(i, g.wrap_j(j - 1))]);
      } else {
        radial_flux = g.r_at_edge(i + 1) * jr[g.index(i + 1, j)]
                    - g.r_at_edge(i) * jr[k];
        axial_flux = jz[g.index(i, g.wrap_j(j + 1))] - jz[k];
      }
      const double radial = radial_flux / (g.r_at_cell_center(i) * dr);
      const double axial = axial_flux / dz;
      const double residual = std::abs(drho + radial + axial);
      if (residual > max_resid) {
        max_resid = residual;
        max_i = i;
        max_j = j;
        max_drho = drho;
        max_radial = radial;
        max_axial = axial;
      }
      max_jmag = std::max(max_jmag, std::max(std::abs(jr[k]), std::abs(jz[k])));
    }
  }
  if (max_resid > 1.0e-8) {
    std::cerr << "max cylindrical continuity residual at (" << max_i << ','
              << max_j << "): drho=" << max_drho
              << " radial=" << max_radial << " axial=" << max_axial << '\n';
  }
  *out_resid = max_resid;
  *out_jmag = max_jmag;
}

}  // namespace

TEST(PicCylChargeConservation, EsirkepovCylSatisfiesContinuityCIC) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_cyl_continuity_case<1>(&resid, &jmag);
  EXPECT_GT(jmag, 1.0e-6);  // non-trivial deposit
  EXPECT_LT(resid, 1.0e-9) << "max cyl continuity residual " << resid;
}

TEST(PicCylChargeConservation, EsirkepovCylSatisfiesContinuityTSC) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_cyl_continuity_case<2>(&resid, &jmag);
  EXPECT_GT(jmag, 1.0e-6);
  EXPECT_LT(resid, 1.0e-9) << "max cyl continuity residual " << resid;
}

TEST(PicCylChargeConservation, AxisCellSatisfiesContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_cyl_continuity_case<2>(&resid, &jmag, /*near_axis=*/true);
  EXPECT_GT(jmag, 1.0e-6);
  EXPECT_LT(resid, 2.0e-9) << "axis-cell continuity residual " << resid;
}

TEST(PicCylChargeConservation, OrderFourAxisCellSatisfiesContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_cyl_continuity_case<2, 4>(&resid, &jmag, /*near_axis=*/true);
  EXPECT_GT(jmag, 1.0e-6);
  EXPECT_LT(resid, 3.0e-8) << "order-four axis continuity residual " << resid;
}

TEST(PicCylChargeConservation, OuterSpecularWallSatisfiesContinuity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  constexpr double q = 1.0, w = 1.0, r0 = 0.96, z0 = 0.5;
  constexpr double vr = 0.99, dt = 0.044;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"q", q, 1.0, 1}};
  sp.set_host_particles({r0}, {z0}, {vr}, {0.0}, {0.0}, {w});
  solver.add_species(std::move(sp));
  solver.step(dt);
  const auto snap = solver.species()[0].to_host();

  std::vector<double> rho_old(g.storage_size(), 0.0);
  std::vector<double> rho_new(g.storage_size(), 0.0);
  accumulate_rho_cyl_reflect_outer<1>(rho_old, g, q, w, r0, z0);
  accumulate_rho_cyl_reflect_outer<1>(rho_new, g, q, w, snap.x[0],
                                      snap.y[0]);
  std::vector<double> jr(g.storage_size()), jz(g.storage_size());
  solver.current().jx.copy_to_host(jr.data(), jr.size());
  solver.current().jy.copy_to_host(jz.data(), jz.size());
  double max_resid = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const auto k = g.index(i, j);
      const double residual =
          (rho_new[k] - rho_old[k]) / dt
        + (g.r_at_edge(i + 1) * jr[g.index(i + 1, j)]
           - g.r_at_edge(i) * jr[k])
              / (g.r_at_cell_center(i) * g.dx())
        + (jz[g.index(i, g.wrap_j(j + 1))] - jz[k]) / g.dy();
      max_resid = std::max(max_resid, std::abs(residual));
    }
  }
  EXPECT_LT(max_resid, 3.0e-9)
      << "cylindrical reflecting-wall continuity residual " << max_resid;
}

TEST(PicCylChargeConservation, AxisJphiDepositIsAdjointToOddEphiGather) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 4, 1.0, 1.0, 0.0, 0.0, 2};
  constexpr double q = 1.3;
  constexpr double macro_weight = 0.7;
  constexpr double vphi = 0.2;
  const double r = 0.1 * g.dx();  // TSC support reaches radial face -1.
  const double z = 0.5;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"ring", q, 1.0, 1}};
  sp.set_host_particles(
      {r}, {z}, {0.0}, {0.0}, {vphi}, {macro_weight});
  quasar::JField2D<double> current{g};
  launch_pic_deposit_cyl_shape2(
      g, sp, current, 0.01, /*periodic_x=*/0, /*periodic_y=*/1,
      nullptr);

  // An odd linear Ephi profile on the signed radial extension. The physical
  // face values are enough for the grid-work sum; the gather oracle applies the
  // same axis parity as boundary_axis_fields.
  const auto ephi = [](int face) { return 0.7 * static_cast<double>(face); };
  const auto weights = quasar::numerics::shape_weights_2d_at_offset<2>(
      r, z, g, 0.0, 0.5);
  double gathered = 0.0;
  for (int b = 0; b < weights.ny; ++b) {
    for (int a = 0; a < weights.nx; ++a) {
      const int face = weights.ix[a];
      gathered += weights.wx[a] * weights.wy[b] * ephi(face);
    }
  }

  std::vector<double> jphi(g.storage_size(), 0.0);
  current.jz.copy_to_host(jphi.data(), jphi.size());
  double grid_work = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 1; i <= g.nx; ++i) {
      const double width = i == g.nx ? 0.5 * g.dx() : g.dx();
      const double midpoint =
          i == g.nx ? g.r_at_edge(i) - 0.5 * width : g.r_at_edge(i);
      const double dual_volume =
          quasar::pi * width * (2.0 * midpoint) * g.dy();
      grid_work += jphi[g.index(i, j)] * ephi(i) * dual_volume;
    }
  }
  const double particle_work = q * macro_weight * vphi * gathered;
  EXPECT_NEAR(grid_work, particle_work,
              3.0e-14 * std::max(1.0, std::abs(particle_work)));
}

TEST(PicCylChargeConservation,
     ScaledAxialDepositKeepsRepresentableNodeContributions) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // q/(dt*2*pi*r_c*dr) overflows for the inner radial cell, but multiplication
  // by the O(dt) axial shape delta brings every final Jz contribution back into
  // range. Materialising the common prefix before that delta loses a valid
  // conservative current.
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  const double largest = std::numeric_limits<double>::max();
  const double q = 0.75 * largest;
  constexpr double r = 1.0;
  constexpr double z0 = 1.0;
  constexpr double z1 = 1.0005;
  constexpr double dt = 0.001;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"scaled-ring", q, largest, 1}};
  sp.set_host_particles({r}, {z0}, {0.0}, {0.5}, {0.0}, {1.0});
  quasar::backend::device_memcpy_h2d(sp.y(), &z1, sizeof(z1));

  quasar::JField2D<double> current{g};
  launch_pic_deposit_cyl_shape1(
      g, sp, current, dt, /*periodic_x=*/0, /*periodic_y=*/0, nullptr);
  EXPECT_NO_THROW(launch_pic_deposit_overflow_check(sp, nullptr));

  std::vector<double> jz(g.storage_size(), 0.0);
  current.jy.copy_to_host(jz.data(), jz.size());
  EXPECT_TRUE(std::all_of(jz.begin(), jz.end(),
                          [](double v) { return std::isfinite(v); }));
  const double expected =
      q * (0.5 * ((z1 - z0) / dt) / quasar::pi);
  EXPECT_NEAR(jz[g.index(0, 1)], expected, 5.0e-15 * q);
}

TEST(PicCylChargeConservation,
     CollocatedRingCurrentAccumulationOverflowIsRejected) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Eight stationary rings deposit individually finite q*vphi/(2*pi) values
  // onto the same Jphi face. Their atomic sum exceeds DBL_MAX.
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  const double largest = std::numeric_limits<double>::max();
  constexpr std::size_t count = 8;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"collocated-ring-current",
                                 largest, largest, count}};
  sp.set_host_particles(std::vector<double>(count, 1.0),
                        std::vector<double>(count, 0.5),
                        std::vector<double>(count, 0.0),
                        std::vector<double>(count, 0.0),
                        std::vector<double>(count, 0.9),
                        std::vector<double>(count, 1.0));
  quasar::JField2D<double> current{g};
  launch_pic_deposit_cyl_shape1(
      g, sp, current, 0.1, /*periodic_x=*/0, /*periodic_y=*/0, nullptr);
  EXPECT_THROW(launch_pic_deposit_overflow_check(sp, nullptr),
               std::runtime_error);
}

TEST(PicCylChargeConservation,
     CollocatedRingChargeAccumulationOverflowIsRejected) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // At r=1 the CIC charge splits between the adjacent ring centres. Eight
  // individually finite image-volume contributions overflow the inner rho node.
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  const double largest = std::numeric_limits<double>::max();
  constexpr std::size_t count = 8;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"collocated-ring-charge",
                                 largest, largest, count}};
  sp.set_host_particles(std::vector<double>(count, 1.0),
                        std::vector<double>(count, 0.5),
                        std::vector<double>(count, 0.0),
                        std::vector<double>(count, 0.0),
                        std::vector<double>(count, 0.0),
                        std::vector<double>(count, 1.0));
  quasar::ScalarGrid2D<double> rho{g};
  launch_pic_charge_cyl_shape1(
      g, sp, rho, /*periodic_x=*/0, /*periodic_y=*/0, nullptr);
  EXPECT_THROW(launch_pic_deposit_overflow_check(sp, nullptr),
               std::runtime_error);
}
