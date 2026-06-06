// RED-phase tests for the high-order ideal-MHD flux reconstruction schemes.
//
// Targets the blind contract in include/quasar/numerics/flux_reconstruction.hpp:
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
// ASSUMED ACCESSORS (documented so the blind implementer matches them):
//   * quasar::mhd::MhdField2D<Real>(Grid2D) ctor; cell-centered conserved
//     DeviceBuffers .rho/.mx/.my/.mz/.energy and cell-centered field component
//     .bz_cell, all DeviceBuffer<Real> of length grid.storage_size(). The
//     cell-reconstructed Bx/By that this scheme produces are carried in the
//     interface-state output (see below), NOT in the face-staggered storage.
//     We seed these via member.copy_from_host(host.data(), host.size()) exactly
//     as the PIC YeeField2D tests do.
//   * quasar::numerics::MhdInterfaceStates<Real> with:
//       - ctor MhdInterfaceStates(Grid2D, int dir)
//       - MhdState state_left(int i, int j) const   // reconstructed LEFT  conserved state of face (i,j)
//       - MhdState state_right(int i, int j) const   // reconstructed RIGHT conserved state of face (i,j)
//     where MhdState (quasar/numerics/mhd_state.hpp) is the 8-component conserved
//     struct {rho,mx,my,mz,energy,bx,by,bz} in that order. The face (i,j) for a
//     dir=0 (x) reconstruction is the interface between cells (i-1,j) and (i,j);
//     state_left is the state extrapolated from the left cell, state_right from the
//     right cell. The Shu-Osher conservative-FD reconstruction is performed
//     after reconstruct_faces(); the L/R interface values are read on the device
//     buffers and copied to host for the assertions below.
//
// Device-touching assertions are guarded with has_hip_runtime() / GTEST_SKIP,
// matching the project convention for tests that exercise DeviceBuffer storage.
// The pure-host probes (required_nghost / is_characteristic / registry presence)
// run unconditionally and fail by missing symbol until the scheme exists.

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

std::unique_ptr<IFluxReconstruction> make_scheme(const std::string& name) {
  return quasar::Registry<IFluxReconstruction>::instance().create(name);
}

// Smooth analytic density / pressure profile in x; the reconstruction error vs
// the analytic interface value is what we measure for the convergence test.
Real smooth_rho(Real x) { return Real{2} + Real{0.5} * std::sin(Real{2} * M_PI * x); }
Real smooth_p(Real x) { return Real{3} + Real{0.4} * std::cos(Real{2} * M_PI * x); }

// Seed a 1D-in-x smooth conserved field on a periodic grid (a uniform velocity
// and field so only rho/energy carry the structure). Returns nothing; mutates u.
void seed_smooth_field(quasar::mhd::MhdField2D<Real>& u, const Grid2D& g, Real gamma) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
  std::vector<Real> en(n, Real{0}), bz(n, Real{0});
  const Real vx = Real{0.2};
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const Real x = g.x_at_cell_center(i);
      const Real r = smooth_rho(x);
      const Real p = smooth_p(x);
      const std::size_t k = g.index(i, j);
      rho[k] = r;
      mx[k] = r * vx;
      my[k] = Real{0};
      mz[k] = Real{0};
      // total energy with zero B (Bx/By cell-reconstructed handled by scheme).
      en[k] = p / (gamma - Real{1}) + Real{0.5} * r * vx * vx;
      bz[k] = Real{0};
    }
  }
  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(mx.data(), mx.size());
  u.my.copy_from_host(my.data(), my.size());
  u.mz.copy_from_host(mz.data(), mz.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bz_cell.copy_from_host(bz.data(), bz.size());
}

// Max-norm of the reconstructed interior interface-density error vs the analytic
// value at the face location for a dir=0 (x) reconstruction. We compare the
// average of L/R (which both converge to the analytic interface value for a
// smooth profile) against rho(x_face).
Real interface_rho_error(const std::string& name, int nx, Real gamma) {
  const int nghost = 4;  // enough for mp7
  Grid2D g{nx, 4, Real{1}, Real{1}, Real{0}, Real{0}, nghost};
  quasar::mhd::MhdField2D<Real> u{g};
  seed_smooth_field(u, g, gamma);

  auto scheme = make_scheme(name);
  MhdInterfaceStates<Real> out{g, /*dir=*/0};
  scheme->reconstruct_faces(u, /*dir=*/0, out, gamma);

  Real max_err = Real{0};
  const int j = g.ny / 2;
  for (int i = 1; i < g.nx; ++i) {  // interior faces between cell i-1 and i
    const Real x_face = g.origin_x + static_cast<Real>(i) * g.dx();
    const Real ref = smooth_rho(x_face);
    const MhdState l = out.state_left(i, j);
    const MhdState r = out.state_right(i, j);
    const Real recon = Real{0.5} * (l.rho + r.rho);
    max_err = std::max(max_err, std::abs(recon - ref));
  }
  return max_err;
}

}  // namespace

// ---- pure-host probes (fail by missing symbol) ----------------------------

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
  EXPECT_TRUE(make_scheme("mp5")->is_characteristic());
  EXPECT_TRUE(make_scheme("mp7")->is_characteristic());
}

// ---- device-touching behavior --------------------------------------------

// A smooth profile is reconstructed at the scheme's design order: refining the
// grid reduces the interface-value error by ~2^p. We measure the observed order
// across two resolutions and require it to be near the design order (with a
// generous lower bound to absorb the asymptotic-regime slack of a coarse pair).
TEST(MhdFluxReconstruction, Mp5ConvergesAtFifthOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Real e_coarse = interface_rho_error("mp5", 32, gamma);
  const Real e_fine = interface_rho_error("mp5", 64, gamma);
  ASSERT_GT(e_coarse, Real{0});
  ASSERT_GT(e_fine, Real{0});
  const Real order = std::log2(e_coarse / e_fine);
  EXPECT_GT(order, Real{4.0}) << "observed order " << order << " (design 5)";
}

TEST(MhdFluxReconstruction, Mp7ConvergesAtSeventhOrderOnSmoothProfile) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Real e_coarse = interface_rho_error("mp7", 32, gamma);
  const Real e_fine = interface_rho_error("mp7", 64, gamma);
  ASSERT_GT(e_coarse, Real{0});
  ASSERT_GT(e_fine, Real{0});
  const Real order = std::log2(e_coarse / e_fine);
  EXPECT_GT(order, Real{6.0}) << "observed order " << order << " (design 7)";
}

// Reconstructing across a sharp jump stays monotone: the reconstructed L/R
// interface values must lie within the [min,max] envelope of the neighboring
// cell values (no new extrema / no overshoot beyond a tiny monotonicity bound).
TEST(MhdFluxReconstruction, Mp5IsMonotoneAcrossSharpJump) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const int nghost = 4;
  const int nx = 64;
  Grid2D g{nx, 4, Real{1}, Real{1}, Real{0}, Real{0}, nghost};
  quasar::mhd::MhdField2D<Real> u{g};

  // A step in density (and matching energy) at the mid-plane: a contact-like
  // discontinuity that must not produce ringing in a non-oscillatory scheme.
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), my(n, Real{0}), mz(n, Real{0});
  std::vector<Real> en(n, Real{0}), bz(n, Real{0});
  const Real rho_lo = Real{1}, rho_hi = Real{3}, p0 = Real{1};
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real r = (i < g.nx / 2) ? rho_lo : rho_hi;
      rho[k] = r;
      en[k] = p0 / (gamma - Real{1});  // at rest
    }
  }
  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(mx.data(), mx.size());
  u.my.copy_from_host(my.data(), my.size());
  u.mz.copy_from_host(mz.data(), mz.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bz_cell.copy_from_host(bz.data(), bz.size());

  auto scheme = make_scheme("mp5");
  MhdInterfaceStates<Real> out{g, /*dir=*/0};
  scheme->reconstruct_faces(u, /*dir=*/0, out, gamma);

  const Real tol = Real{1e-10};  // monotonicity bound: no overshoot beyond round-off
  const int j = g.ny / 2;
  for (int i = 1; i < g.nx; ++i) {
    const Real local_min = std::min(rho_lo, rho_hi) - tol;
    const Real local_max = std::max(rho_lo, rho_hi) + tol;
    const MhdState l = out.state_left(i, j);
    const MhdState r = out.state_right(i, j);
    EXPECT_GE(l.rho, local_min) << "L overshoot below min at face " << i;
    EXPECT_LE(l.rho, local_max) << "L overshoot above max at face " << i;
    EXPECT_GE(r.rho, local_min) << "R overshoot below min at face " << i;
    EXPECT_LE(r.rho, local_max) << "R overshoot above max at face " << i;
  }
}
