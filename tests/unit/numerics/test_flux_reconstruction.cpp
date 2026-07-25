// Accuracy and invariants of the ideal-MHD flux reconstruction schemes.
//
// Targets the contract in include/quasar/numerics/flux_reconstruction.hpp:
//
//   class IFluxReconstruction {
//    public: virtual ~IFluxReconstruction() = default;
//     virtual int  required_nghost() const = 0;   // muscl_minmod=2, mp5=3, mp7=4
//     virtual bool is_characteristic() const = 0;
//     virtual void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
//                                    quasar::numerics::MhdInterfaceStates<Real>& out,
//                                    Real gamma) const = 0;
//   };
//
// Registry names: "muscl_minmod", "mp5", "mp7", obtained via
//   quasar::Registry<quasar::numerics::IFluxReconstruction>::instance().create(name).
//
// A field is seeded by writing host buffers of size grid.storage_size() into the
// MhdField2D<Real> DeviceBuffer members via copy_from_host (rho/mx/my/mz/energy,
// the staggered bx_face/by_face, and the cell-centered bz_cell). Reconstructed
// L/R interface states are read back on the host through the staging accessors
// MhdInterfaceStates::state_left(i,j) / state_right(i,j).
//
// For dir=0 the interface (i,j) (stored at grid.index(i,j)) is the face between
// cells (i-1,j) and (i,j): state_left is the right-biased extrapolation from the
// left cell, state_right the left-biased extrapolation from the right cell.
//
// The evolved state is a finite-volume cell average.  The accuracy probes seed
// exact analytic cell averages and check both reconstructed faces and the
// conservative face-flux difference; the latter prevents point-sample Lagrange
// coefficients from masquerading as a high-order finite-volume method.

#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/backend/device.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::IFluxReconstruction;
using quasar::numerics::MhdInterfaceStates;
using quasar::numerics::MhdPrim;
using quasar::numerics::MhdState;

constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
constexpr Real kGamma = Real{5} / Real{3};

std::unique_ptr<IFluxReconstruction> make_scheme(const std::string& name) {
  return quasar::Registry<IFluxReconstruction>::instance().create(name);
}

// A smooth entropy wave in x.  Velocity, pressure, and magnetic field are
// constant while density varies sinusoidally, so every conserved component is
// either constant or linear in rho. Exact finite-volume cell averages are then
// available analytically and the nonlinear primitive transform cannot pollute
// the measured spatial order.
constexpr Real kBxUniform = Real{0.7};
constexpr Real kByUniform = Real{0.2};
constexpr Real kBzUniform = Real{0.1};
constexpr Real kPressure = Real{3};
constexpr Real kVx = Real{0.35};

Real smooth_rho(Real x) {
  return Real{2} + Real{0.5} * std::sin(Real{2} * kPi * x);
}

Real smooth_rho_average(Real x_center, Real dx) {
  const Real q = kPi * dx;
  const Real sinc = (q == Real{0}) ? Real{1} : std::sin(q) / q;
  return Real{2} + Real{0.5} * sinc
      * std::sin(Real{2} * kPi * x_center);
}

void seed_smooth_field(quasar::mhd::MhdField2D<Real>& u, const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
  std::vector<Real> en(n, Real{0});
  std::vector<Real> bxf(n, kBxUniform), byf(n, Real{0}), bzc(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const Real x = g.x_at_cell_center(i);
      const Real r = smooth_rho_average(x, g.dx());
      const std::size_t k = g.index(i, j);
      rho[k] = r;
      mx[k] = r * kVx;
      my[k] = Real{0};
      mz[k] = Real{0};
      byf[k] = kByUniform;
      bzc[k] = kBzUniform;
      const Real mag = Real{0.5} * (kBxUniform * kBxUniform
                                    + kByUniform * kByUniform
                                    + kBzUniform * kBzUniform);
      en[k] = kPressure / (gamma - Real{1})
            + Real{0.5} * r * kVx * kVx + mag;
    }
  }
  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(mx.data(), mx.size());
  u.my.copy_from_host(my.data(), my.size());
  u.mz.copy_from_host(mz.data(), mz.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bx_face.copy_from_host(bxf.data(), bxf.size());
  u.by_face.copy_from_host(byf.data(), byf.size());
  u.bz_cell.copy_from_host(bzc.data(), bzc.size());
}

// L1 norm (per interior interface) of the reconstructed left-interface density
// error vs the exact face value, for a dir=0 reconstruction on `nx` cells. The
// left-interface value (state_left) of a smooth profile converges at the scheme's
// design order to the analytic value at the face location x = origin_x + i*dx.
Real interface_rho_l1(const std::string& name, int nx, Real gamma) {
  const int nghost = 4;  // enough for mp7
  Grid2D g{nx, 4, Real{1}, Real{1}, Real{0}, Real{0}, nghost};
  quasar::mhd::MhdField2D<Real> u{g};
  seed_smooth_field(u, g, gamma);

  auto scheme = make_scheme(name);
  MhdInterfaceStates<Real> out{g, /*dir=*/0};
  scheme->reconstruct_faces(u, /*dir=*/0, out, gamma);

  Real sum = Real{0};
  int count = 0;
  const int j = g.ny / 2;
  for (int i = 1; i < g.nx; ++i) {  // interior faces between cell i-1 and i
    const Real x_face = g.origin_x + static_cast<Real>(i) * g.dx();
    const Real ref = smooth_rho(x_face);
    const MhdState l = out.state_left(i, j);
    sum += std::abs(l.rho - ref);
    ++count;
  }
  return sum / static_cast<Real>(count);
}

// Observed convergence order from two grids via the log-log slope of the L1
// interface error. Returns -1 if either error is non-finite or non-positive.
Real observed_order(const std::string& name, int nx_coarse, int nx_fine, Real gamma) {
  const Real e_coarse = interface_rho_l1(name, nx_coarse, gamma);
  const Real e_fine = interface_rho_l1(name, nx_fine, gamma);
  if (!(std::isfinite(e_coarse) && std::isfinite(e_fine)) ||
      e_coarse <= Real{0} || e_fine <= Real{0}) {
    return Real{-1};
  }
  return std::log2(e_coarse / e_fine);
}

// L1 error of the actual finite-volume density residual for positive constant
// advection velocity.  HLLD resolves this entropy wave with the upwind left
// density, so -v*(rho^L_{i+1/2}-rho^L_{i-1/2})/dx is the numerical residual.
// The exact cell-average residual is the same face difference using analytic
// rho(x_face). Point-value coefficients fed these cell averages remain O(dx^2)
// here, even if a separate interface interpolation experiment looks high order.
Real residual_rho_l1(const std::string& name, int nx, Real gamma) {
  const int nghost = 4;
  Grid2D g{nx, 4, Real{1}, Real{1}, Real{0}, Real{0}, nghost};
  quasar::mhd::MhdField2D<Real> u{g};
  seed_smooth_field(u, g, gamma);

  auto scheme = make_scheme(name);
  MhdInterfaceStates<Real> out{g, /*dir=*/0};
  scheme->reconstruct_faces(u, /*dir=*/0, out, gamma);

  Real sum = Real{0};
  const int j = g.ny / 2;
  for (int i = 0; i < g.nx; ++i) {
    const Real numerical = -kVx
        * (out.state_left(i + 1, j).rho - out.state_left(i, j).rho) / g.dx();
    const Real x_lo = g.origin_x + static_cast<Real>(i) * g.dx();
    const Real x_hi = x_lo + g.dx();
    const Real exact = -kVx * (smooth_rho(x_hi) - smooth_rho(x_lo)) / g.dx();
    sum += std::abs(numerical - exact);
  }
  return sum / static_cast<Real>(g.nx);
}

Real observed_residual_order(const std::string& name, int nx_coarse, int nx_fine,
                             Real gamma) {
  const Real e_coarse = residual_rho_l1(name, nx_coarse, gamma);
  const Real e_fine = residual_rho_l1(name, nx_fine, gamma);
  if (!(std::isfinite(e_coarse) && std::isfinite(e_fine))
      || e_coarse <= Real{0} || e_fine <= Real{0}) {
    return Real{-1};
  }
  return std::log2(e_coarse / e_fine);
}

}  // namespace

// ---- pure-host probes (run unconditionally) -------------------------------

TEST(MhdFluxReconstruction, SchemesAreRegistered) {
  const auto& reg = quasar::Registry<IFluxReconstruction>::instance();
  EXPECT_TRUE(reg.contains("muscl_minmod"));
  EXPECT_TRUE(reg.contains("mp5"));
  EXPECT_TRUE(reg.contains("mp7"));
}

TEST(MhdFluxReconstruction, RequiredGhostMatchesDesignWidth) {
  EXPECT_EQ(make_scheme("muscl_minmod")->required_nghost(), 2);
  EXPECT_EQ(make_scheme("mp5")->required_nghost(), 3);
  EXPECT_EQ(make_scheme("mp7")->required_nghost(), 4);
}

TEST(MhdFluxReconstruction, HighOrderSchemesAreCharacteristic) {
  EXPECT_FALSE(make_scheme("muscl_minmod")->is_characteristic());
  EXPECT_TRUE(make_scheme("mp5")->is_characteristic());
  EXPECT_TRUE(make_scheme("mp7")->is_characteristic());
}

// ---- registry construct-and-run smoke (finite output) ---------------------

// Each registered scheme constructs and runs reconstruct_faces without error and
// produces finite interface states on a smooth field.
TEST(MhdFluxReconstruction, EverySchemeRunsAndProducesFiniteStates) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const std::string name : {"muscl_minmod", "mp5", "mp7"}) {
    Grid2D g{32, 4, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
    quasar::mhd::MhdField2D<Real> u{g};
    seed_smooth_field(u, g, kGamma);

    auto scheme = make_scheme(name);
    MhdInterfaceStates<Real> out{g, /*dir=*/0};
    ASSERT_NO_THROW(scheme->reconstruct_faces(u, /*dir=*/0, out, kGamma)) << name;

    const int j = g.ny / 2;
    for (int i = 1; i < g.nx; ++i) {
      const MhdState l = out.state_left(i, j);
      const MhdState r = out.state_right(i, j);
      for (const Real v : {l.rho, l.mx, l.my, l.mz, l.energy, l.bx, l.by, l.bz,
                           r.rho, r.mx, r.my, r.mz, r.energy, r.bx, r.by, r.bz}) {
        EXPECT_TRUE(std::isfinite(v)) << name << " face " << i;
      }
      EXPECT_GT(l.rho, Real{0}) << name << " face " << i;
      EXPECT_GT(r.rho, Real{0}) << name << " face " << i;
    }
  }
}

// A componentwise primitive MUSCL state is not automatically range-admissible.
// These three source cells are individually finite with positive pressure, but
// the central cell's left-face extrapolation combines rho ~= 0.5e100 with
// vx ~= -1.5*sqrt(3.3e208).  Its kinetic energy is about 1.856e308, just beyond
// binary64, even though the central cell's own kinetic energy is 1.65e308.
// Reconstruction must discard that overflowing face and return the exact
// adjacent piecewise-constant cell, rather than feeding Inf to HLLD.
TEST(MhdFluxReconstruction, MusclExtremeFaceFallsBackToAdjacentCell) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{8, 2, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/2};
  const std::size_t n = g.storage_size();
  constexpr Real internal_energy = Real{1e300};
  const Real p = (kGamma - Real{1}) * internal_energy;
  const Real velocity_scale = std::sqrt(Real{3.3e208});

  const MhdState left = quasar::numerics::to_conserved(
      MhdPrim{Real{1}, -Real{2} * velocity_scale, Real{0}, Real{0}, p,
              Real{0}, Real{0}, Real{0}}, kGamma);
  const MhdState center = quasar::numerics::to_conserved(
      MhdPrim{Real{1e100}, -velocity_scale, Real{0}, Real{0}, p,
              Real{0}, Real{0}, Real{0}}, kGamma);
  const MhdState right = quasar::numerics::to_conserved(
      MhdPrim{Real{2e100}, Real{0}, Real{0}, Real{0}, p,
              Real{0}, Real{0}, Real{0}}, kGamma);

  for (const MhdState* source : {&left, &center, &right}) {
    EXPECT_TRUE(std::isfinite(source->rho));
    EXPECT_TRUE(std::isfinite(source->mx));
    EXPECT_TRUE(std::isfinite(source->energy));
    EXPECT_GT(quasar::numerics::pressure(*source, kGamma), Real{0});
  }
  EXPECT_TRUE(std::isinf(quasar::numerics::kinetic_from_velocity(
      Real{0.5e100}, -Real{1.5} * velocity_scale, Real{0}, Real{0})))
      << "test setup no longer overflows the unlimited face kinetic energy";

  std::vector<Real> rho(n, right.rho), mx(n, right.mx), my(n, Real{0});
  std::vector<Real> mz(n, Real{0}), en(n, right.energy);
  std::vector<Real> bxf(n, Real{0}), byf(n, Real{0}), bzc(n, Real{0});
  const auto store = [&](int i, int j, const MhdState& s) {
    const std::size_t k = g.index(i, j);
    rho[k] = s.rho;
    mx[k] = s.mx;
    my[k] = s.my;
    mz[k] = s.mz;
    en[k] = s.energy;
  };
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    store(/*i=*/2, j, left);
    store(/*i=*/3, j, center);
    store(/*i=*/4, j, right);
  }

  quasar::mhd::MhdField2D<Real> u{g};
  u.rho.copy_from_host(rho.data(), n);
  u.mx.copy_from_host(mx.data(), n);
  u.my.copy_from_host(my.data(), n);
  u.mz.copy_from_host(mz.data(), n);
  u.energy.copy_from_host(en.data(), n);
  u.bx_face.copy_from_host(bxf.data(), n);
  u.by_face.copy_from_host(byf.data(), n);
  u.bz_cell.copy_from_host(bzc.data(), n);

  MhdInterfaceStates<Real> out{g, /*dir=*/0};
  make_scheme("muscl_minmod")->reconstruct_faces(u, /*dir=*/0, out, kGamma);

  // Face i=3 lies between the left cell i=2 and the central cell i=3; its
  // RIGHT state is the troubled extrapolation from the central cell.
  const MhdState recovered = out.state_right(/*i=*/3, /*j=*/0);
  EXPECT_EQ(recovered.rho, center.rho);
  EXPECT_EQ(recovered.mx, center.mx);
  EXPECT_EQ(recovered.my, center.my);
  EXPECT_EQ(recovered.mz, center.mz);
  EXPECT_EQ(recovered.energy, center.energy);
  EXPECT_EQ(recovered.bx, center.bx);
  EXPECT_EQ(recovered.by, center.by);
  EXPECT_EQ(recovered.bz, center.bz);
  EXPECT_GT(quasar::numerics::pressure(recovered, kGamma), Real{0});
}

// When even the adjacent-cell fallback cannot be normalized to the exact CT
// face in binary64, reconstruction must return its explicit rejection token.
// Here the two bounding faces cancel in each low-order cell average, so both
// source cells have Bx=0 and positive pressure, while the shared interface face
// itself is DBL_MAX and its magnetic energy is unrepresentable.
TEST(MhdFluxReconstruction, UnrepresentableCtFaceEmitsDeterministicSentinel) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{6, 2, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/2};
  const std::size_t n = g.storage_size();
  constexpr Real rho0 = Real{1};
  constexpr Real p0 = Real{1};
  constexpr Real energy0 = p0 / (kGamma - Real{1});
  const Real huge = std::numeric_limits<Real>::max();

  std::vector<Real> rho(n, rho0), zero(n, Real{0}), energy(n, energy0);
  std::vector<Real> bx(n, -huge);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    // Face i=2 is shared by cells 1 and 2.  Each cell's low-order average is
    // 0.5*(-huge)+0.5*huge or its exact reverse, hence zero.
    bx[g.index(/*i=*/2, j)] = huge;
  }

  quasar::mhd::MhdField2D<Real> u{g};
  u.rho.copy_from_host(rho.data(), n);
  u.mx.copy_from_host(zero.data(), n);
  u.my.copy_from_host(zero.data(), n);
  u.mz.copy_from_host(zero.data(), n);
  u.energy.copy_from_host(energy.data(), n);
  u.bx_face.copy_from_host(bx.data(), n);
  u.by_face.copy_from_host(zero.data(), n);
  u.bz_cell.copy_from_host(zero.data(), n);

  const quasar::mhd::MhdBackgroundField<Real> background{};
  const quasar::mhd::BoundaryFlags4 flags{};
  MhdInterfaceStates<Real> out{g, /*dir=*/0};
  quasar::mhd::launch_mhd_reconstruct(
      u, background, /*dir=*/0, out, /*scheme_order=*/1, flags, kGamma,
      /*stream=*/nullptr);
  quasar::backend::device_synchronize(nullptr);

  const MhdState left = out.state_left(/*i=*/2, /*j=*/0);
  const MhdState right = out.state_right(/*i=*/2, /*j=*/0);
  for (const MhdState* state : {&left, &right}) {
    for (const Real component : {state->rho, state->mx, state->my, state->mz,
                                 state->energy, state->bx, state->by, state->bz}) {
      EXPECT_TRUE(std::isnan(component));
    }
  }
}

// ---- convergence order on a smooth profile --------------------------------

// muscl_minmod is ~2nd order on a smooth profile.
TEST(MhdFluxReconstruction, MusclConvergesAtSecondOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_order("muscl_minmod", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{2.0} - Real{0.7}) << "observed order " << order << " (design 2)";
}

// MP5 reaches its fifth-order finite-volume face accuracy.
TEST(MhdFluxReconstruction, Mp5ConvergesAtFifthOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_order("mp5", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{5.0} - Real{0.7}) << "observed order " << order << " (design 5)";
}

// MP7 reaches its seventh-order finite-volume face accuracy.
TEST(MhdFluxReconstruction, Mp7ConvergesAtSeventhOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_order("mp7", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{7.0} - Real{0.7}) << "observed order " << order << " (design 7)";
}

TEST(MhdFluxReconstruction, Mp5ConservativeResidualConvergesAtFifthOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_residual_order("mp5", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{4.3}) << "observed conservative residual order " << order;
}

TEST(MhdFluxReconstruction, Mp7ConservativeResidualConvergesAtSeventhOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_residual_order("mp7", 24, 48, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{6.0}) << "observed conservative residual order " << order;
}

// ---- high-order device path is genuinely running --------------------------

// At a fixed resolution the interface states from an "mp7" deck must differ
// MEASURABLY from a "muscl_minmod" deck on the same smooth field. If the device
// silently downgrades mp7 to 2nd-order MUSCL, the two are identical and the
// max-abs difference collapses to ~0 -- this is the assertion that catches it.
TEST(MhdFluxReconstruction, HighOrderPathIsNotSilentlyDowngraded) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{64, 4, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};

  quasar::mhd::MhdField2D<Real> u_lo{g};
  quasar::mhd::MhdField2D<Real> u_hi{g};
  seed_smooth_field(u_lo, g, kGamma);
  seed_smooth_field(u_hi, g, kGamma);

  auto lo = make_scheme("muscl_minmod");
  auto hi = make_scheme("mp7");
  MhdInterfaceStates<Real> out_lo{g, /*dir=*/0};
  MhdInterfaceStates<Real> out_hi{g, /*dir=*/0};
  lo->reconstruct_faces(u_lo, /*dir=*/0, out_lo, kGamma);
  hi->reconstruct_faces(u_hi, /*dir=*/0, out_hi, kGamma);

  Real max_diff = Real{0};
  const int j = g.ny / 2;
  for (int i = 1; i < g.nx; ++i) {
    const MhdState a = out_lo.state_left(i, j);
    const MhdState b = out_hi.state_left(i, j);
    max_diff = std::max(max_diff, std::abs(a.rho - b.rho));
    max_diff = std::max(max_diff, std::abs(a.energy - b.energy));
    max_diff = std::max(max_diff, std::abs(a.mx - b.mx));
  }
  EXPECT_GT(max_diff, Real{1e-8})
      << "mp7 interface states match muscl_minmod -- high-order path is being "
         "silently downgraded to 2nd order (max abs diff " << max_diff << ")";
}

// ---- normal-B is the CT face value on BOTH reconstructed states -----------

// For dir=0 the reconstructed L.bx and R.bx must equal the interface bx_face
// value on both states for mp5 and mp7 (normal B never passes through the
// characteristic projection). With kBxUniform on the staggered storage the
// interface normal-B is exactly kBxUniform; pin to round-off.
TEST(MhdFluxReconstruction, NormalBxEqualsCtFaceValueDir0) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const std::string name : {"mp5", "mp7"}) {
    Grid2D g{48, 4, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
    quasar::mhd::MhdField2D<Real> u{g};
    seed_smooth_field(u, g, kGamma);

    auto scheme = make_scheme(name);
    MhdInterfaceStates<Real> out{g, /*dir=*/0};
    scheme->reconstruct_faces(u, /*dir=*/0, out, kGamma);

    const int j = g.ny / 2;
    for (int i = 2; i < g.nx - 1; ++i) {
      const MhdState l = out.state_left(i, j);
      const MhdState r = out.state_right(i, j);
      EXPECT_NEAR(l.bx, kBxUniform, Real{1e-12}) << name << " L face " << i;
      EXPECT_NEAR(r.bx, kBxUniform, Real{1e-12}) << name << " R face " << i;
    }
  }
}

// For dir=1 the reconstructed L.by and R.by must equal the interface by_face
// value on both states for mp5 and mp7. We seed by_face uniform here so the
// interface normal-B equals kBxUniform exactly along y as well.
TEST(MhdFluxReconstruction, NormalByEqualsCtFaceValueDir1) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const std::string name : {"mp5", "mp7"}) {
    Grid2D g{4, 48, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
    const std::size_t n = g.storage_size();

    // Smooth structure along y for the reconstructed variables; uniform normal
    // By on the staggered by_face storage so the interface normal-B is exact.
    std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
    std::vector<Real> en(n, Real{0});
    std::vector<Real> bxf(n, Real{0}), byf(n, kBxUniform), bzc(n, Real{0});
    for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
      for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
        const Real y = g.y_at_cell_center(j);
        const Real r = smooth_rho_average(y, g.dy());
        const Real p = kPressure;
        const Real vy = kVx;
        const Real bx = kByUniform;
        const Real bz = kBzUniform;
        const std::size_t k = g.index(i, j);
        rho[k] = r;
        my[k] = r * vy;
        bxf[k] = bx;
        bzc[k] = bz;
        const Real mag = Real{0.5} * (bx * bx + kBxUniform * kBxUniform + bz * bz);
        en[k] = p / (kGamma - Real{1}) + Real{0.5} * r * vy * vy + mag;
      }
    }
    quasar::mhd::MhdField2D<Real> u{g};
    u.rho.copy_from_host(rho.data(), n);
    u.mx.copy_from_host(mx.data(), n);
    u.my.copy_from_host(my.data(), n);
    u.mz.copy_from_host(mz.data(), n);
    u.energy.copy_from_host(en.data(), n);
    u.bx_face.copy_from_host(bxf.data(), n);
    u.by_face.copy_from_host(byf.data(), n);
    u.bz_cell.copy_from_host(bzc.data(), n);

    auto scheme = make_scheme(name);
    MhdInterfaceStates<Real> out{g, /*dir=*/1};
    scheme->reconstruct_faces(u, /*dir=*/1, out, kGamma);

    const int i = g.nx / 2;
    for (int j = 2; j < g.ny - 1; ++j) {
      const MhdState l = out.state_left(i, j);
      const MhdState r = out.state_right(i, j);
      EXPECT_NEAR(l.by, kBxUniform, Real{1e-12}) << name << " L face " << j;
      EXPECT_NEAR(r.by, kBxUniform, Real{1e-12}) << name << " R face " << j;
    }
  }
}
