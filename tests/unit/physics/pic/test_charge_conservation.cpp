// Validates that the Esirkepov current deposition is charge-conserving: the
// deposited in-plane current (Jx, Jy) must satisfy the discrete continuity
// relation
//   (rho_new - rho_old)/dt + (Jx[i,j]-Jx[i-1,j])/dx + (Jy[i,j]-Jy[i,j-1])/dy = 0
// cell-by-cell, where rho is the charge density formed from the same shape
// function. The backward-difference divergence here is the adjoint of the
// forward-difference curl used by the FDTD E-update, so satisfying it preserves
// Gauss's law under the field solve.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

// Accumulate rho = q*w*S(x) onto a host grid using the same shape order as the
// deposit under test.
template <int ShapeOrder>
void accumulate_rho(std::vector<double>& rho, const quasar::Grid2D& g,
                    double q, double w, double x, double y) {
  const auto sw = quasar::numerics::shape_weights_2d<ShapeOrder>(x, y, g);
  for (int b = 0; b < sw.ny; ++b) {
    for (int a = 0; a < sw.nx; ++a) {
      rho[g.periodic_index(sw.ix[a], sw.iy[b])] += q * w * sw.wx[a] * sw.wy[b];
    }
  }
}

// Runs one field-free step of a single drifting macro-particle and returns the
// worst-case discrete-continuity residual and the peak |J|. When `seam` is true
// the particle starts adjacent to the upper periodic boundary and drifts across
// it within the step, exercising the wrap BC's x_prev co-shift.
template <int ShapeOrder>
void run_continuity_case(double* out_resid, double* out_jmag, bool seam = false) {
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{
      quasar::pic::EmPicConfig{g, 2, ShapeOrder == 1 ? "cic" : "tsc"}};

  const double q = 1.0, m = 1.0, w = 1.0;
  const double x0 = seam ? 0.985 : 0.51;
  const double y0 = seam ? 0.985 : 0.52;
  const double vx = seam ? 0.8 : 0.37, vy = seam ? 0.8 : -0.29, vz = 0.13;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", q, m, 1}};
  sp.set_host_particles({x0}, {y0}, {vx}, {vy}, {vz}, {w});
  solver.add_species(std::move(sp));

  // Sub-cell displacement (dx = 1/16): the CFL regime the deposit targets.
  const double dt = 0.05;
  solver.step(dt);

  // Unwrapped post-move position; periodic_index folds it back onto the torus,
  // matching where the deposit lands after the wrap co-shift.
  const double x1 = x0 + dt * vx;
  const double y1 = y0 + dt * vy;

  std::vector<double> rho_old(g.storage_size(), 0.0);
  std::vector<double> rho_new(g.storage_size(), 0.0);
  accumulate_rho<ShapeOrder>(rho_old, g, q, w, x0, y0);
  accumulate_rho<ShapeOrder>(rho_new, g, q, w, x1, y1);

  auto& J = solver.current();
  std::vector<double> jx(g.storage_size()), jy(g.storage_size());
  J.jx.copy_to_host(jx.data(), jx.size());
  J.jy.copy_to_host(jy.data(), jy.size());

  double max_resid = 0.0, max_jmag = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k   = g.periodic_index(i, j);
      const std::size_t kxm = g.periodic_index(i - 1, j);
      const std::size_t kym = g.periodic_index(i, j - 1);
      const double drho = (rho_new[k] - rho_old[k]) / dt / (g.dx() * g.dy());
      const double divJ = (jx[k] - jx[kxm]) / g.dx() + (jy[k] - jy[kym]) / g.dy();
      max_resid = std::max(max_resid, std::abs(drho + divJ));
      max_jmag  = std::max(max_jmag, std::max(std::abs(jx[k]), std::abs(jy[k])));
    }
  }
  *out_resid = max_resid;
  *out_jmag = max_jmag;
}

}  // namespace

TEST(PicChargeConservation, EsirkepovSatisfiesContinuityCIC) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_continuity_case<1>(&resid, &jmag);
  EXPECT_GT(jmag, 1.0e-6);                 // non-trivial deposit
  EXPECT_LT(resid, 1.0e-9) << "max continuity residual " << resid;
}

TEST(PicChargeConservation, EsirkepovSatisfiesContinuityTSC) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_continuity_case<2>(&resid, &jmag);
  EXPECT_GT(jmag, 1.0e-6);
  EXPECT_LT(resid, 1.0e-9) << "max continuity residual " << resid;
}

TEST(PicChargeConservation, EsirkepovConservesAcrossPeriodicSeam) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  // Regression: a particle wrapping the periodic boundary in one step must keep
  // the deposit charge-conserving. Before the wrap kernel co-shifted x_prev,
  // the deposit saw a ~whole-domain displacement and the residual blew up.
  double resid = 0.0, jmag = 0.0;
  run_continuity_case<1>(&resid, &jmag, /*seam=*/true);
  EXPECT_GT(jmag, 1.0e-6);
  EXPECT_LT(resid, 1.0e-9) << "max continuity residual across seam " << resid;
}

TEST(PicChargeConservation, OversizedDisplacementFailsInsteadOfTruncatingCurrent) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", 1.0, 1.0, 1}};
  sp.set_host_particles({0.5}, {0.5}, {10.0}, {0.0}, {0.0}, {1.0});
  solver.add_species(std::move(sp));

  EXPECT_THROW(solver.step(0.2), std::runtime_error);
}

TEST(PicChargeConservation, DepositTypesExist) {
  quasar::numerics::Esirkepov2D<1> cic;
  quasar::numerics::Esirkepov2D<2> tsc;
  (void)cic;
  (void)tsc;
  SUCCEED();
}
