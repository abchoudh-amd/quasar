// Validates that the axisymmetric (r,z) Esirkepov deposit is charge-conserving:
// the deposited (Jr, Jz) must satisfy the discrete cylindrical continuity
//   (rho_new-rho_old)/dt
//     + (r_e(i)*Jr[i,j] - r_e(i-1)*Jr[i-1,j])/(r_c(i)*dr)
//     + (Jz[i,j]-Jz[i,j-1])/dz = 0
// cell-by-cell, where rho = q*w*S_r*S_z / cell_volume(i) (the ring volume). The
// backward ring-flux divergence is the adjoint of the cylindrical FDTD curl, so
// satisfying it preserves Gauss's law on the axisymmetric grid. This is the
// cylindrical analogue of test_charge_conservation.cpp, which only covers the
// Cartesian deposit.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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
      const double vol = g.cell_volume(sw.ix[a]);
      if (vol <= 0.0) continue;
      rho[g.periodic_index(sw.ix[a], sw.iy[b])] +=
          q * w * sw.wx[a] * sw.wy[b] / vol;
    }
  }
}

template <int ShapeOrder>
void run_cyl_continuity_case(double* out_resid, double* out_jmag) {
  // r-domain [0,1], z-domain [0,1]; the particle sits well inside r>0 so its
  // shape support never reaches the axis (i=0) or the outer wall.
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, ShapeOrder == 1 ? "cic" : "tsc"};
  cfg.geometry = "cylindrical";
  // Outer radius must be a non-periodic wall; z stays periodic so its telescoping
  // divergence wraps cleanly. x_lo is auto-wired to the axis BC.
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  const double q = 1.0, m = 1.0, w = 1.0;
  const double r0 = 0.5, z0 = 0.5;
  // vphi = 0 so the cylindrical position advance is linear (no azimuthal sweep);
  // with zero initial fields the Boris rotation is the identity, so the post-move
  // position is exactly (r0 + dt*vr, z0 + dt*vz). vr = vx, vz(axial) = vy.
  const double vr = 0.37, vz_axial = -0.29, vphi = 0.0;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", q, m, 1}};
  sp.set_host_particles({r0}, {z0}, {vr}, {vz_axial}, {vphi}, {w});
  solver.add_species(std::move(sp));

  // Sub-CFL sub-cell displacement (dr = dz = 1/16, cyl CFL limit ~ 0.044).
  const double dt = 0.03;
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
  // Start the radial sweep at i=1: the i=0 node uses the on-axis closure (a
  // distinct form), and the particle deposits nothing there, so 0 = 0 trivially.
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 1; i < g.nx; ++i) {
      const std::size_t k = g.periodic_index(i, j);
      const std::size_t krm = g.periodic_index(i - 1, j);
      const std::size_t kzm = g.periodic_index(i, j - 1);
      const double drho = (rho_new[k] - rho_old[k]) / dt;
      const double radial =
          (g.r_at_edge(i) * jr[k] - g.r_at_edge(i - 1) * jr[krm]) /
          (g.r_at_cell_center(i) * dr);
      const double axial = (jz[k] - jz[kzm]) / dz;
      max_resid = std::max(max_resid, std::abs(drho + radial + axial));
      max_jmag = std::max(max_jmag, std::max(std::abs(jr[k]), std::abs(jz[k])));
    }
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
