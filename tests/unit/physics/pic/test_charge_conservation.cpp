// Validates that the Esirkepov current deposition is charge-conserving: the
// deposited in-plane current (Jx, Jy) must satisfy the discrete continuity
// relation
//   (rho_new-rho_old)/dt + (Jx[i+1,j]-Jx[i,j])/dx
//                          + (Jy[i,j+1]-Jy[i,j])/dy = 0
// cell-by-cell, where rho is the charge density formed from the same shape
// function. This is the forward divergence of the stored lower-face currents,
// exactly matching Ampere's staggered curl and Gauss operator.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
template <int ShapeOrder, int FdtdOrder = 2>
void run_continuity_case(double* out_resid, double* out_jmag, bool seam = false) {
  const int halo = std::max(quasar::required_nghost(FdtdOrder),
                            ShapeOrder == 2 ? 2 : 1);
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0,
                   halo};
  quasar::pic::EmPicConfig cfg{
      g, FdtdOrder, ShapeOrder == 1 ? "cic" : "tsc"};
  cfg.neutralizing_background = true;
  quasar::pic::EmPic2D3V solver{cfg};

  const double q = 1.0, m = 1.0, w = 1.0;
  const double x0 = seam ? 0.985 : 0.51;
  const double y0 = seam ? 0.985 : 0.52;
  const double vx = seam ? 0.65 : 0.37;
  const double vy = seam ? 0.55 : -0.29;
  const double vz = 0.13;
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", q, m, 1}};
  sp.set_host_particles({x0}, {y0}, {vx}, {vy}, {vz}, {w});
  solver.add_species(std::move(sp));

  // Sub-cell displacement (dx = 1/16): the CFL regime the deposit targets.
  const double dt = FdtdOrder == 4 ? 0.02 : 0.03;
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
      const double drho = (rho_new[k] - rho_old[k]) / dt / (g.dx() * g.dy());
      double divJ;
      if constexpr (FdtdOrder == 4) {
        const double dxj =
            (9.0 / 8.0) *
                (jx[g.periodic_index(i + 1, j)] - jx[k])
          - (1.0 / 24.0) *
                (jx[g.periodic_index(i + 2, j)] -
                 jx[g.periodic_index(i - 1, j)]);
        const double dyj =
            (9.0 / 8.0) *
                (jy[g.periodic_index(i, j + 1)] - jy[k])
          - (1.0 / 24.0) *
                (jy[g.periodic_index(i, j + 2)] -
                 jy[g.periodic_index(i, j - 1)]);
        divJ = dxj / g.dx() + dyj / g.dy();
      } else {
        divJ = (jx[g.periodic_index(i + 1, j)] - jx[k]) / g.dx()
             + (jy[g.periodic_index(i, j + 1)] - jy[k]) / g.dy();
      }
      max_resid = std::max(max_resid, std::abs(drho + divJ));
      max_jmag  = std::max(max_jmag, std::max(std::abs(jx[k]), std::abs(jy[k])));
    }
  }
  *out_resid = max_resid;
  *out_jmag = max_jmag;
}

}  // namespace

TEST(PicChargeConservation,
     DoublyPeriodicDomainRejectsResolvedNetInitialCharge) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{4, 4, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{
      quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies positive{
      quasar::pic::SpeciesConfig{"positive", 1.0, 1.0, 1}};
  positive.set_host_particles(
      {0.25}, {0.25}, {0.0}, {0.0}, {0.0}, {2.0});
  solver.add_species(std::move(positive));

  EXPECT_THROW((void)solver.charge_density(), std::invalid_argument);

  // The rejected materialization has not advanced or staggered the solver.
  // Supplying the missing counter-charge must make the same object usable.
  quasar::pic::ParticleSpecies negative{
      quasar::pic::SpeciesConfig{"negative", -1.0, 1.0, 1}};
  negative.set_host_particles(
      {0.75}, {0.75}, {0.0}, {0.0}, {0.0}, {2.0});
  solver.add_species(std::move(negative));
  EXPECT_NO_THROW((void)solver.charge_density());
}

TEST(PicChargeConservation,
     ExplicitBackgroundPermitsPeriodicChargedSpecies) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{4, 4, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.neutralizing_background = true;
  quasar::pic::EmPic2D3V solver{cfg};
  quasar::pic::ParticleSpecies charged{
      quasar::pic::SpeciesConfig{"electron", -1.0, 1.0, 1}};
  charged.set_host_particles(
      {0.5}, {0.5}, {0.0}, {0.0}, {0.0}, {1.0});
  solver.add_species(std::move(charged));
  EXPECT_NO_THROW((void)solver.charge_density());
}

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

TEST(PicChargeConservation, OrderFourCurrentMatchesOrderFourGaussOperator) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  double resid = 0.0, jmag = 0.0;
  run_continuity_case<2, 4>(&resid, &jmag);
  EXPECT_GT(jmag, 1.0e-6);
  EXPECT_LT(resid, 2.0e-9) << "max order-four continuity residual " << resid;
}

TEST(PicChargeConservation, OrderFourCorrectionIncludesNonperiodicHighFaces) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const quasar::Grid2D g{8, 4, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::JField2D<double> current{g};
  std::vector<double> raw(g.storage_size(), 0.0);
  const int j = 2;
  for (int i = 0; i <= g.nx; ++i) {
    raw[g.index(i, j)] = 0.2 * i * i - 0.3 * i + 0.7;
  }
  current.jx.copy_from_host(raw.data(), raw.size());
  quasar::backend::DeviceBuffer<double> rhs_x{g.storage_size()};
  quasar::backend::DeviceBuffer<double> rhs_y{g.storage_size()};
  quasar::backend::DeviceBuffer<double> iter_x{g.storage_size()};
  quasar::backend::DeviceBuffer<double> iter_y{g.storage_size()};
  launch_pic_current_correct_order4(
      g, current, rhs_x.device_ptr(), rhs_y.device_ptr(), iter_x.device_ptr(),
      iter_y.device_ptr(), /*x_lo PEC-even=*/1, /*x_hi PEC-even=*/1,
      /*y_lo PEC-even=*/1, /*y_hi PEC-even=*/1, nullptr);

  std::vector<double> corrected(g.storage_size(), 0.0);
  current.jx.copy_to_host(corrected.data(), corrected.size());
  const auto face = [&](int i) {
    // Match the normal-E PEC continuation used by the field boundary: the
    // physical wall face is even, so its first ghost mirrors face 1/N-1.
    const int ii = i < 0 ? -i : (i > g.nx ? 2 * g.nx - i : i);
    return corrected[g.index(ii, j)];
  };
  double worst = 0.0;
  for (int i = 0; i < g.nx; ++i) {
    const double d4 = (9.0 / 8.0) * (face(i + 1) - face(i))
                    - (1.0 / 24.0) * (face(i + 2) - face(i - 1));
    const double d2 = raw[g.index(i + 1, j)] - raw[g.index(i, j)];
    worst = std::max(worst, std::abs(d4 - d2));
  }
  EXPECT_LT(worst, 2.0e-14);
  EXPECT_NE(corrected[g.index(g.nx, j)], raw[g.index(g.nx, j)]);
}

TEST(PicChargeConservation, OrderFourStepPreservesGaussFromNeutralInitialState) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{32, 32, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 4, "tsc"}};
  quasar::pic::ParticleSpecies positive{
      quasar::pic::SpeciesConfig{"positive", 1.0, 1.0, 1}};
  quasar::pic::ParticleSpecies negative{
      quasar::pic::SpeciesConfig{"negative", -1.0, 1.0, 1}};
  positive.set_host_particles({0.43}, {0.57}, {0.31}, {-0.17}, {0.0}, {1.0});
  negative.set_host_particles({0.43}, {0.57}, {-0.23}, {0.29}, {0.0}, {1.0});
  solver.add_species(std::move(positive));
  solver.add_species(std::move(negative));

  solver.step(0.01);
  EXPECT_LT(quasar::pic::gauss_residual(solver), 2.0e-8);
}

TEST(PicChargeConservation, OrderTwoPeriodicSeamClosesCurrentFieldAndGauss) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{32, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies positive{
      quasar::pic::SpeciesConfig{"positive", 1.0, 1.0, 1}};
  quasar::pic::ParticleSpecies negative{
      quasar::pic::SpeciesConfig{"negative", -1.0, 1.0, 1}};
  // Initially collocated opposite charges give rho=0 exactly. Their opposite
  // velocities make q*v add while the positive particle crosses the seam.
  positive.set_host_particles({0.995}, {0.47}, {0.30}, {0.0}, {0.0}, {1.0});
  negative.set_host_particles({0.995}, {0.47}, {-0.20}, {0.0}, {0.0}, {1.0});
  solver.add_species(std::move(positive));
  solver.add_species(std::move(negative));
  solver.step(0.02);

  std::vector<double> jx(g.storage_size()), ex(g.storage_size());
  solver.current().jx.copy_to_host(jx.data(), jx.size());
  solver.fields().ex.copy_to_host(ex.data(), ex.size());
  for (int j = 0; j < g.ny; ++j) {
    EXPECT_DOUBLE_EQ(jx[g.index(g.nx, j)], jx[g.index(0, j)]);
    EXPECT_DOUBLE_EQ(ex[g.index(g.nx, j)], ex[g.index(0, j)]);
  }
  EXPECT_LT(quasar::pic::gauss_residual(solver), 3.0e-10);
}

TEST(PicChargeConservation, RejectsInvalidTimeStepsAndMissingHalo) {
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  EXPECT_THROW(solver.step(0.0), std::invalid_argument);
  EXPECT_THROW(solver.step(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(solver.step(1.01 * solver.cfl_limit()), std::invalid_argument);

  quasar::Grid2D no_halo{16, 16, 1.0, 1.0, 0.0, 0.0, 0};
  EXPECT_THROW(
      quasar::pic::EmPic2D3V(quasar::pic::EmPicConfig{no_halo, 2, "cic"}),
      std::invalid_argument);
  EXPECT_THROW(quasar::required_nghost(3), std::invalid_argument);
}

TEST(PicChargeConservation, RejectsMalformedConfigBeforeFieldAllocation) {
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.0, 0.0, 1};

  const quasar::Grid2D one_x{1, 2, 1.0, 2.0, 0.0, 0.0, 2};
  EXPECT_THROW(
      quasar::pic::EmPic2D3V(
          quasar::pic::EmPicConfig{one_x, 4, "cic"}),
      std::invalid_argument);
  const quasar::Grid2D one_y{2, 1, 2.0, 1.0, 0.0, 0.0, 2};
  EXPECT_THROW(
      quasar::pic::EmPic2D3V(
          quasar::pic::EmPicConfig{one_y, 4, "cic"}),
      std::invalid_argument);

  auto bad_geometry = quasar::pic::EmPicConfig{g, 2, "cic"};
  bad_geometry.geometry = "spherical";
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{bad_geometry},
               std::invalid_argument);

  auto tsc_without_true_high_ghost = quasar::pic::EmPicConfig{g, 2, "tsc"};
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{tsc_without_true_high_ghost},
               std::invalid_argument);

  auto one_sided_periodic = quasar::pic::EmPicConfig{g, 2, "cic"};
  one_sided_periodic.boundary.particle[1] = "absorbing";
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{one_sided_periodic},
               std::invalid_argument);

  auto mismatched_topology = quasar::pic::EmPicConfig{g, 2, "cic"};
  mismatched_topology.boundary.particle = {
      "absorbing", "absorbing", "periodic", "periodic"};
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{mismatched_topology},
               std::invalid_argument);

  auto inconsistent_normalization = quasar::pic::EmPicConfig{g, 2, "cic"};
  inconsistent_normalization.normalization =
      quasar::Normalization::plasma(1.0e18, 1.602176634e-19,
                                    9.1093837015e-31);
  inconsistent_normalization.normalization.omega_p_ref *= 1.01;
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{inconsistent_normalization},
               std::invalid_argument);
}

TEST(PicChargeConservation, RejectsUnrepresentablePeriodicParticleImages) {
  const double maximum = std::numeric_limits<double>::max();
  const quasar::Grid2D translated{
      1, 1, maximum, 1.0, -maximum, 0.0, 1};
  EXPECT_THROW(
      quasar::pic::EmPic2D3V(
          quasar::pic::EmPicConfig{translated, 2, "cic"}),
      std::overflow_error);
}

TEST(PicChargeConservation, RejectsCapacityOutsideDeviceCounterRange) {
  if constexpr (std::numeric_limits<std::size_t>::max()
                > std::numeric_limits<unsigned int>::max()) {
    quasar::pic::SpeciesConfig cfg{"too_many", 1.0, 1.0,
        static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())
            + std::size_t{1}};
    EXPECT_THROW(quasar::pic::ParticleSpecies species{cfg},
                 std::invalid_argument);
  }
}

TEST(PicChargeConservation, RejectsNonrepresentableChargeToMassRatio) {
  const double largest = std::numeric_limits<double>::max();
  const double smallest = std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(
      quasar::pic::ParticleSpecies(
          quasar::pic::SpeciesConfig{"overflow", largest, smallest, 0}),
      std::invalid_argument);
  EXPECT_THROW(
      quasar::pic::ParticleSpecies(
          quasar::pic::SpeciesConfig{"underflow", smallest, largest, 0}),
      std::invalid_argument);
}

TEST(PicChargeConservation, SolverRejectsDuplicateNamesAndInvalidOwnedUploads) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  solver.add_species(quasar::pic::ParticleSpecies{
      quasar::pic::SpeciesConfig{"probe", 0.0, 1.0, 1}});
  EXPECT_THROW(
      solver.add_species(quasar::pic::ParticleSpecies{
          quasar::pic::SpeciesConfig{"probe", 0.0, 1.0, 0}}),
      std::invalid_argument);

  EXPECT_THROW(
      solver.set_species_particles(
          0, {-0.01}, {0.5}, {0.0}, {0.0}, {0.0}, {1.0}),
      std::invalid_argument);
  EXPECT_THROW(
      solver.set_species_particles(
          1, {0.5}, {0.5}, {0.0}, {0.0}, {0.0}, {1.0}),
      std::out_of_range);
  solver.set_species_particles(
      0, {0.5}, {0.5}, {0.0}, {0.0}, {0.0}, {0.0});
  ASSERT_EQ(solver.species()[0].size(), 1u);
  EXPECT_DOUBLE_EQ(solver.species()[0].to_host().x[0], 0.5);

  solver.step(0.01);
  EXPECT_THROW(
      solver.set_species_particles(
          0, {0.5}, {0.5}, {0.0}, {0.0}, {0.0}, {0.0}),
      std::logic_error);
  EXPECT_THROW(
      solver.add_species(quasar::pic::ParticleSpecies{
          quasar::pic::SpeciesConfig{"late", 0.0, 1.0, 0}}),
      std::logic_error);
}

TEST(PicChargeConservation, RejectsSuperluminalNonrelativisticParticles) {
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"q", 1.0, 1.0, 1}};
  EXPECT_THROW(
      sp.set_host_particles({0.5}, {0.5}, {1.0}, {0.0}, {0.0}, {1.0}),
      std::invalid_argument);
  EXPECT_THROW(
      sp.set_host_particles(
          {0.5}, {0.5}, {std::numeric_limits<double>::max()},
          {std::numeric_limits<double>::max()}, {0.0}, {1.0}),
      std::invalid_argument);
}

TEST(PicChargeConservation, AdvanceUsesShortenedFinalStep) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"neutral", 0.0, 1.0, 1}};
  sp.set_host_particles({1.0}, {1.0}, {0.25}, {0.0}, {0.0}, {1.0});
  solver.add_species(std::move(sp));
  solver.advance(0.10, 0.06);
  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.x[0], 1.025, 1.0e-12);
}

TEST(PicChargeConservation, ShortenedFinalStepUsesVariableLeapfrogCentering) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  std::vector<double> ex(g.storage_size(), 1.0);
  solver.external_fields().ex.copy_from_host(ex.data(), ex.size());
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  // Zero macro-weight isolates the pusher from its self-current while retaining
  // q/m=1. The uploaded velocity is the physical value at t=0.
  sp.set_host_particles({1.0}, {1.0}, {0.0}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));

  solver.advance(0.10, 0.06);  // position steps 0.06, 0.04
  const auto snap = solver.species()[0].to_host();
  // The startup force interval is 0.06/2=0.03, then the variable-step interval
  // is (0.06+0.04)/2=0.05. Thus v=0.08 and
  // x-x0=0.06*0.03 + 0.04*0.08 = 0.005 for constant unit acceleration.
  EXPECT_NEAR(snap.vx[0], 0.08, 2.0e-14);
  EXPECT_NEAR(snap.x[0], 1.005, 2.0e-14);
}

TEST(PicChargeConservation,
     AdjacentSubnormalStepsPreserveTheCenteredTimestep) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 4, 16.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  std::vector<double> ez(g.storage_size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ez[g.index(i, j)] = g.x_at_cell_center(i);
    }
  }
  solver.fields().ez.copy_from_host(ez.data(), ez.size());

  const double denorm = std::numeric_limits<double>::denorm_min();
  // This field-only case has no Boris update, so the unrepresentable first
  // particle half-step is irrelevant. Faraday can still advance by the full
  // 2*denorm interval and the following centred magnetic interval.
  solver.step(2.0 * denorm);
  solver.step(denorm);

  std::vector<double> by(g.storage_size(), 0.0);
  solver.fields().by.copy_to_host(by.data(), by.size());
  // The second magnetic interval is midpoint(2*denorm, denorm)=2*denorm.
  // Evaluating each half separately would lose half of the smaller operand and
  // leave only 3*denorm after the two updates instead of 4*denorm.
  EXPECT_EQ(by[g.index(8, 2)], 4.0 * denorm);

  // A neutral particle has no force interval to resolve. Its first Boris kick
  // is the identity even though dt/2 underflows, while the full-width position
  // drift and zero charge/current deposition remain well-defined.
  quasar::pic::EmPic2D3V neutral_solver{
      quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies neutral{
      quasar::pic::SpeciesConfig{"neutral", 0.0, 1.0, 1}};
  neutral.set_host_particles(
      {8.0}, {2.0}, {0.0}, {0.0}, {0.0}, {1.0});
  neutral_solver.add_species(std::move(neutral));
  EXPECT_NO_THROW(neutral_solver.step(denorm));
  const auto neutral_snapshot = neutral_solver.species()[0].to_host();
  EXPECT_DOUBLE_EQ(neutral_snapshot.x[0], 8.0);
  EXPECT_DOUBLE_EQ(neutral_snapshot.y[0], 2.0);

  quasar::pic::EmPic2D3V particle_solver{
      quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies species{
      quasar::pic::SpeciesConfig{"physical-v0", 1.0, 1.0, 1}};
  species.set_host_particles(
      {8.0}, {2.0}, {0.0}, {0.0}, {0.0}, {0.0});
  particle_solver.add_species(std::move(species));
  EXPECT_THROW(particle_solver.step(denorm), std::overflow_error);
}

TEST(PicChargeConservation, VariableStepWeightsMagneticHalfStepsAtForceTime) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{16, 4, 16.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  std::vector<double> ez(g.storage_size(), 0.0);
  const double particle_x = 8.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ez[g.index(i, j)] = g.x_at_cell_center(i) - particle_x;
    }
  }
  solver.fields().ez.copy_from_host(ez.data(), ez.size());

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  sp.set_host_particles({particle_x}, {2.0}, {0.1}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  solver.step(0.06);  // establishes B_y^{1/2}=0.06 near the particle
  const auto first_snap = solver.species()[0].to_host();
  std::vector<double> by_after_first(g.storage_size(), 0.0);
  solver.fields().by.copy_to_host(by_after_first.data(),
                                  by_after_first.size());
  EXPECT_NEAR(by_after_first[g.index(8, 2)], 0.06, 2.0e-14);
  const double first_tau = 0.5 * 0.03 * 0.03;
  EXPECT_NEAR(first_snap.vx[0],
              0.1 * (1.0 - first_tau * first_tau)
                  / (1.0 + first_tau * first_tau),
              2.0e-14);
  EXPECT_NEAR(first_snap.vz[0],
              0.1 * (2.0 * first_tau) / (1.0 + first_tau * first_tau),
              2.0e-14);

  // Keep the linear curl that advances By, but recenter Ez's constant offset on
  // the moved particle. Without this, Ez=x-particle_x is no longer zero after
  // the first position update and its real O(0.006*0.05) electric impulse
  // contaminates a regression intended to isolate magnetic time centering.
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      ez[g.index(i, j)] = g.x_at_cell_center(i) - first_snap.x[0];
    }
  }
  solver.fields().ez.copy_from_host(ez.data(), ez.size());
  solver.step(0.04);
  std::vector<double> by_after_second(g.storage_size(), 0.0);
  solver.fields().by.copy_to_host(by_after_second.data(),
                                  by_after_second.size());
  EXPECT_NEAR(by_after_second[g.index(8, 2)], 0.11, 2.0e-14);

  // The particle is present from t=0 because adding a new integer-time species
  // after evolution begins is intentionally forbidden. The first force interval
  // has width 0.03 and samples 0.5*(B^-1/2+B^1/2)=0.03. The second interval is
  // 0.05 with B_old=0.06, B_new=0.11 and the variable-step interpolation
  // 0.4*old+0.6*new=0.09. Compose the two Boris plane rotations.
  const auto rotate = [](double vx, double vz, double tau) {
    const double denom = 1.0 + tau * tau;
    const double cosine = (1.0 - tau * tau) / denom;
    const double sine = 2.0 * tau / denom;
    return std::pair<double, double>{
        cosine * vx - sine * vz, sine * vx + cosine * vz};
  };
  const auto after_first = rotate(0.1, 0.0, 0.5 * 0.03 * 0.03);
  const auto expected = rotate(
      after_first.first, after_first.second, 0.5 * 0.05 * 0.09);
  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.vx[0], expected.first, 2.0e-13);
  EXPECT_NEAR(snap.vz[0], expected.second, 2.0e-13);
}

TEST(PicChargeConservation, AdvanceDoesNotOvershootNearIntegerQuotient) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"neutral", 0.0, 1.0, 1}};
  sp.set_host_particles({1.0}, {1.0}, {0.25}, {0.0}, {0.0}, {1.0});
  solver.add_species(std::move(sp));
  const double t_end = std::nextafter(0.12, 0.0);
  solver.advance(t_end, 0.06);
  const auto snap = solver.species()[0].to_host();
  EXPECT_NEAR(snap.x[0], 1.0 + 0.25 * t_end, 2.0e-15);
}

TEST(PicChargeConservation, AdvanceRejectsUnrepresentableStepCount) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  EXPECT_THROW(solver.advance(std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::denorm_min()),
               std::overflow_error);
}

TEST(PicChargeConservation,
     ScaledDepositKeepsRepresentableNodeContributions) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // The unweighted prefactors q/(dy*dt) and q/(dx*dy) both exceed binary64,
  // while every shape-weighted current/charge contribution is comfortably
  // representable. A deposit must retain the prefactor in exponent-scaled form
  // until the small shape delta/weight has been applied.
  quasar::Grid2D g{4, 4, 4.0, 2.0, 0.0, 0.0, 1};
  const double largest = std::numeric_limits<double>::max();
  const double q = 0.75 * largest;
  constexpr double x0 = 1.0;
  constexpr double x1 = 1.125;
  constexpr double y = 0.5;
  constexpr double dt = 0.25;
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"scaled", q, largest, 1}};
  sp.set_host_particles({x0}, {y}, {0.5}, {0.0}, {0.0}, {1.0});
  quasar::backend::device_memcpy_h2d(sp.x(), &x1, sizeof(x1));

  quasar::JField2D<double> current{g};
  launch_pic_deposit_shape1(g, sp, current, dt, /*periodic_x=*/0,
                            /*periodic_y=*/0, nullptr);
  EXPECT_NO_THROW(launch_pic_deposit_overflow_check(sp, nullptr));

  quasar::ScalarGrid2D<double> rho{g};
  launch_pic_charge_shape1(g, sp, rho, /*periodic_x=*/0,
                           /*periodic_y=*/0, nullptr);
  EXPECT_NO_THROW(launch_pic_deposit_overflow_check(sp, nullptr));

  std::vector<double> jx(g.storage_size(), 0.0);
  std::vector<double> charge(g.storage_size(), 0.0);
  current.jx.copy_to_host(jx.data(), jx.size());
  rho.values.copy_to_host(charge.data(), charge.size());
  EXPECT_TRUE(std::all_of(jx.begin(), jx.end(),
                          [](double v) { return std::isfinite(v); }));
  EXPECT_TRUE(std::all_of(charge.begin(), charge.end(),
                          [](double v) { return std::isfinite(v); }));
  EXPECT_NEAR(jx[g.index(1, 0)], 0.5 * q, 4.0e-15 * q);
  EXPECT_NEAR(charge[g.index(1, 0)], 0.625 * q, 4.0e-15 * q);
}

TEST(PicChargeConservation,
     CollocatedParticleCurrentAccumulationOverflowIsRejected) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Each particle contributes 0.75*DBL_MAX to the same Jz node. Both
  // contributions are individually finite, but their atomic sum is not.
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  const double largest = std::numeric_limits<double>::max();
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"collocated-current", largest, largest, 2}};
  sp.set_host_particles({0.5, 0.5}, {0.5, 0.5},
                        {0.0, 0.0}, {0.0, 0.0}, {0.75, 0.75},
                        {1.0, 1.0});
  quasar::JField2D<double> current{g};
  launch_pic_deposit_shape1(g, sp, current, 0.1,
                            /*periodic_x=*/0, /*periodic_y=*/0, nullptr);
  EXPECT_THROW(launch_pic_deposit_overflow_check(sp, nullptr),
               std::runtime_error);
}

TEST(PicChargeConservation,
     CollocatedParticleChargeAccumulationOverflowIsRejected) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // At a cell centre each particle contributes DBL_MAX to the same rho node;
  // only the accumulated result overflows.
  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 1};
  const double largest = std::numeric_limits<double>::max();
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"collocated-charge", largest, largest, 2}};
  sp.set_host_particles({0.5, 0.5}, {0.5, 0.5},
                        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                        {1.0, 1.0});
  quasar::ScalarGrid2D<double> rho{g};
  launch_pic_charge_shape1(g, sp, rho,
                           /*periodic_x=*/0, /*periodic_y=*/0, nullptr);
  EXPECT_THROW(launch_pic_deposit_overflow_check(sp, nullptr),
               std::runtime_error);
}

TEST(PicChargeConservation,
     SpecularCurrentFoldbackOverflowIsRejectedAndFlagIsReusable) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 2};
  quasar::JField2D<double> current{g};
  std::vector<double> jy(g.storage_size(), 0.0);
  const double large = 0.75 * std::numeric_limits<double>::max();
  // The x-low specular image adds ghost (-1,0) to interior (0,0).
  // Both deposits are finite; their folded sum is not.
  jy[g.index(0, 0)] = large;
  jy[g.index(-1, 0)] = large;
  current.jy.copy_from_host(jy.data(), jy.size());
  launch_pic_boundary_specular_foldback(
      g, current, /*side=*/0, /*cylindrical=*/0, nullptr);

  quasar::backend::DeviceBuffer<unsigned int> error{1};
  EXPECT_THROW(launch_pic_validate_finite_sources(
                   g, &current, nullptr, error.device_ptr(), nullptr),
               std::overflow_error);

  // Every invocation clears the caller-owned sticky flag. A finite combined
  // current/charge scan must therefore pass even after the preceding throw.
  std::fill(jy.begin(), jy.end(), 0.0);
  current.jy.copy_from_host(jy.data(), jy.size());
  quasar::ScalarGrid2D<double> charge{g};
  EXPECT_NO_THROW(launch_pic_validate_finite_sources(
      g, &current, &charge, error.device_ptr(), nullptr));
}

TEST(PicChargeConservation, SpecularChargeFoldbackOverflowIsRejected) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{4, 4, 4.0, 4.0, 0.0, 0.0, 2};
  quasar::ScalarGrid2D<double> charge{g};
  std::vector<double> rho(g.storage_size(), 0.0);
  const double large = 0.75 * std::numeric_limits<double>::max();
  rho[g.index(0, 0)] = large;
  rho[g.index(-1, 0)] = large;
  charge.values.copy_from_host(rho.data(), rho.size());
  launch_pic_boundary_specular_foldback_charge(
      g, charge, /*side=*/0, /*cylindrical=*/0, nullptr);

  quasar::backend::DeviceBuffer<unsigned int> error{1};
  EXPECT_THROW(launch_pic_validate_finite_sources(
                   g, nullptr, &charge, error.device_ptr(), nullptr),
               std::overflow_error);
}

TEST(PicChargeConservation, CompensatedFilterOverflowIsRejected) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 4, 8.0, 4.0, 0.0, 0.0, 2};
  quasar::JField2D<double> current{g};
  std::vector<double> jz(g.storage_size(), 0.0);
  const double large = 0.9 * std::numeric_limits<double>::max();
  // A smoother followed by its one-pass compensator has the one-dimensional
  // five-point kernel [-1/16, 1/4, 5/8, 1/4, -1/16]. This finite sign pattern
  // therefore produces 1.25*large at i=3, which is genuinely unrepresentable.
  for (int j = 0; j < g.ny; ++j) {
    jz[g.index(1, j)] = -large;
    jz[g.index(2, j)] = large;
    jz[g.index(3, j)] = large;
    jz[g.index(4, j)] = large;
    jz[g.index(5, j)] = -large;
  }
  current.jz.copy_from_host(jz.data(), jz.size());
  quasar::backend::DeviceBuffer<double> scratch{3 * g.storage_size()};
  launch_pic_filter_compensated(
      g, current, scratch.device_ptr(), /*passes=*/1,
      /*periodic_x=*/0, /*periodic_y=*/0, /*cylindrical=*/0, nullptr);

  quasar::backend::DeviceBuffer<unsigned int> error{1};
  EXPECT_THROW(launch_pic_validate_finite_sources(
                   g, &current, nullptr, error.device_ptr(), nullptr),
               std::overflow_error);
}

TEST(PicChargeConservation, DepositTypesExist) {
  quasar::numerics::Esirkepov2D<1> cic;
  quasar::numerics::Esirkepov2D<2> tsc;
  (void)cic;
  (void)tsc;
  SUCCEED();
}
