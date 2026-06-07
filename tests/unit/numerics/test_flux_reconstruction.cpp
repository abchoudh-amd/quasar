// RED-phase tests for the high-order ideal-MHD flux reconstruction schemes.
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
// WHY THESE FAIL NOW (RED): the device reconstruction path currently runs only
// 2nd-order MUSCL and ignores the requested scheme order, so:
//   * the mp5/mp7 convergence-order probes do not reach 5th/7th order, and
//   * an "mp7" deck produces interface states bit-identical (to ~2nd order) to a
//     "muscl_minmod" deck, so the HighOrderPathIsNotSilentlyDowngraded assertion
//     (max |mp7 - muscl| > 1e-8) fails. That distinguishability check is the key
//     RED assertion: it proves the high-order path is genuinely running.

#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/backend/device.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::IFluxReconstruction;
using quasar::numerics::MhdInterfaceStates;
using quasar::numerics::MhdState;

constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
constexpr Real kGamma = Real{5} / Real{3};

std::unique_ptr<IFluxReconstruction> make_scheme(const std::string& name) {
  return quasar::Registry<IFluxReconstruction>::instance().create(name);
}

// Smooth analytic profiles in x. Density, pressure, the transverse/out-of-plane
// magnetic components, and the velocity are all nonzero and smooth so that the
// conserved-to-primitive map and the characteristic eigensystem are well defined.
Real smooth_rho(Real x) { return Real{2} + Real{0.5} * std::sin(Real{2} * kPi * x); }
Real smooth_p(Real x) { return Real{3} + Real{0.4} * std::cos(Real{2} * kPi * x); }
Real smooth_vx(Real x) { return Real{0.3} + Real{0.1} * std::sin(Real{2} * kPi * x); }
Real smooth_by(Real x) { return Real{0.2} + Real{0.15} * std::cos(Real{2} * kPi * x); }
Real smooth_bz(Real x) { return Real{0.1} + Real{0.05} * std::sin(Real{4} * kPi * x); }

// Seed a smooth conserved field whose structure varies only in x. The in-plane
// normal field Bx is seeded UNIFORM (kBxUniform) on the staggered bx_face storage
// so that the constrained-transport interface normal-B (the average of the two
// adjacent face samples) equals kBxUniform exactly at every interface; that lets
// the normal-B passthrough assertion pin to round-off independent of how the
// scheme samples the face. The transverse By and out-of-plane Bz carry smooth
// structure so they exercise the full reconstruction path.
constexpr Real kBxUniform = Real{0.7};

void seed_smooth_field(quasar::mhd::MhdField2D<Real>& u, const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
  std::vector<Real> en(n, Real{0});
  std::vector<Real> bxf(n, kBxUniform), byf(n, Real{0}), bzc(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const Real x = g.x_at_cell_center(i);
      const Real r = smooth_rho(x);
      const Real p = smooth_p(x);
      const Real vx = smooth_vx(x);
      const Real by = smooth_by(x);
      const Real bz = smooth_bz(x);
      const std::size_t k = g.index(i, j);
      rho[k] = r;
      mx[k] = r * vx;
      my[k] = Real{0};
      mz[k] = Real{0};
      byf[k] = by;
      bzc[k] = bz;
      const Real mag = Real{0.5} * (kBxUniform * kBxUniform + by * by + bz * bz);
      en[k] = p / (gamma - Real{1}) + Real{0.5} * r * vx * vx + mag;
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

// ---- convergence order on a smooth profile --------------------------------

// muscl_minmod is ~2nd order on a smooth profile.
TEST(MhdFluxReconstruction, MusclConvergesAtSecondOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_order("muscl_minmod", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{2.0} - Real{0.7}) << "observed order " << order << " (design 2)";
}

// mp5 reaches ~5th order. With the current 2nd-order-only device path this stays
// near 2 and fails.
TEST(MhdFluxReconstruction, Mp5ConvergesAtFifthOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_order("mp5", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{5.0} - Real{0.7}) << "observed order " << order << " (design 5)";
}

// mp7 reaches ~7th order. With the current 2nd-order-only device path this stays
// near 2 and fails.
TEST(MhdFluxReconstruction, Mp7ConvergesAtSeventhOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real order = observed_order("mp7", 32, 64, kGamma);
  ASSERT_GT(order, Real{0}) << "error sequence not usable (order=" << order << ")";
  EXPECT_GT(order, Real{7.0} - Real{0.7}) << "observed order " << order << " (design 7)";
}

// ---- KEY RED ASSERTION: high-order path is genuinely running ---------------

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
        const Real r = smooth_rho(y);
        const Real p = smooth_p(y);
        const Real vy = smooth_vx(y);
        const Real bx = smooth_by(y);
        const Real bz = smooth_bz(y);
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
