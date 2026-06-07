// RED-phase tests for the constrained-transport (CT) scheme.
//
// Targets the blind contract in include/quasar/numerics/ct_scheme.hpp:
//
//   class ICtScheme {
//    public: virtual ~ICtScheme() = default;
//     virtual void compute_emf(const quasar::mhd::MhdField2D<Real>& u,
//                              const quasar::numerics::MhdInterfaceStates<Real>& ifx,
//                              const quasar::numerics::MhdInterfaceStates<Real>& ify,
//                              quasar::mhd::EmfField2D<Real>& emf, Real gamma) const = 0;
//     virtual void update_face_b(quasar::mhd::MhdField2D<Real>& u,
//                                const quasar::mhd::EmfField2D<Real>& emf, Real dt) const = 0;
//     virtual Real divergence_b_linf(const quasar::mhd::MhdField2D<Real>& u) const = 0;
//   };
//
// Registry name "fd_ct_christlieb", obtained via
//   quasar::Registry<quasar::numerics::ICtScheme>::instance().create("fd_ct_christlieb").
//
// CONTRACT NOTE (post-port): ChristliebFdCt::compute_emf REUSES the existing
// device cell-centered EMF discretization, which differs NUMERICALLY from the
// old host interface-state-averaged compute_emf. The ONLY behavior these tests
// pin is the div(B) telescoping guarantee (the discrete curl of ANY corner Ez
// annihilates the cell-centered div(B) to round-off) plus the correctness of
// the divergence_b_linf device reduction. We therefore DO NOT assert any exact
// interface-state-averaged EMF value, nor any exact ez_edge number -- only that
// div(B) is preserved and that the L-inf reduction reproduces the host-computed
// max over interior cells.
//
// ASSUMED ACCESSORS (documented so the blind implementer matches them):
//   * quasar::mhd::MhdField2D<Real>(Grid2D) ctor; face-staggered field
//     DeviceBuffers .bx_face and .by_face, each DeviceBuffer<Real> of length
//     grid.storage_size(). The staggered convention this test seeds:
//       - bx_face stored at index(i,j) is Bx on the LEFT (x_lo) face of cell
//         (i,j), located at x = origin_x + i*dx, y = cell-center y.
//       - by_face stored at index(i,j) is By on the BOTTOM (y_lo) face of cell
//         (i,j), located at y = origin_y + j*dy, x = cell-center x.
//     The discrete cell divergence used by divergence_b_linf is therefore
//       divB(i,j) = (bx_face(i+1,j) - bx_face(i,j))/dx
//                 + (by_face(i,j+1) - by_face(i,j))/dy.
//     We seed bx_face/by_face from a discrete vector potential A_z so this
//     divergence is exactly zero to round-off (see seed_divergence_free_b).
//   * quasar::mhd::EmfField2D<Real>(Grid2D) ctor with edge-centered DeviceBuffers
//     .ez_edge / .ex_edge / .ey_edge (the CT corner/edge EMF the scheme fills).
//   * We seed via member.copy_from_host(host.data(), host.size()) as the PIC
//     YeeField2D tests do.
//
// Device-touching assertions are guarded with has_hip_runtime() / GTEST_SKIP.
// The registry-presence probe runs unconditionally and fails by missing symbol.

#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/backend/device.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::ICtScheme;

std::unique_ptr<ICtScheme> make_ct() {
  return quasar::Registry<ICtScheme>::instance().create("fd_ct_christlieb");
}

// Discrete vector potential A_z(x,y) at cell corners; B = curl(A_z z^) gives a
// face-staggered field with EXACTLY zero discrete divergence under the staggered
// convention documented above:
//   bx_face(i,j) =  (A(i, j+1) - A(i, j)) / dy     // d/dy of A at left x-face
//   by_face(i,j) = -(A(i+1, j) - A(i, j)) / dx     // -d/dx of A at bottom y-face
// where A(i,j) is the corner potential at (origin_x+i*dx, origin_y+j*dy).
Real corner_A(const Grid2D& g, int i, int j) {
  const Real x = g.origin_x + static_cast<Real>(i) * g.dx();
  const Real y = g.origin_y + static_cast<Real>(j) * g.dy();
  return std::sin(Real{2} * M_PI * x) * std::cos(Real{2} * M_PI * y);
}

// Fill `bx`/`by` host vectors (length g.storage_size()) with a discretely
// divergence-free face field derived from corner_A.
void fill_divergence_free_faces(const Grid2D& g, std::vector<Real>& bx,
                                std::vector<Real>& by) {
  const std::size_t n = g.storage_size();
  bx.assign(n, Real{0});
  by.assign(n, Real{0});
  const Real dx = g.dx(), dy = g.dy();
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      bx[k] = (corner_A(g, i, j + 1) - corner_A(g, i, j)) / dy;
      by[k] = -(corner_A(g, i + 1, j) - corner_A(g, i, j)) / dx;
    }
  }
}

// Host re-computation of the L-infinity norm of the cell-centered discrete
// div(B) over the interior cells [0,nx) x [0,ny), using exactly the staggered
// stencil documented in ct_scheme.hpp. Independent of the device reduction so it
// can validate it.
Real host_divergence_b_linf(const Grid2D& g, const std::vector<Real>& bx,
                            const std::vector<Real>& by) {
  const Real dx = g.dx(), dy = g.dy();
  Real m = Real{0};
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real divb =
          (bx[g.index(i + 1, j)] - bx[g.index(i, j)]) / dx +
          (by[g.index(i, j + 1)] - by[g.index(i, j)]) / dy;
      m = std::max(m, std::abs(divb));
    }
  }
  return m;
}

void seed_divergence_free_b(quasar::mhd::MhdField2D<Real>& u, const Grid2D& g) {
  std::vector<Real> bx, by;
  fill_divergence_free_faces(g, bx, by);
  u.bx_face.copy_from_host(bx.data(), bx.size());
  u.by_face.copy_from_host(by.data(), by.size());

  // A quiescent, positive fluid background so the scheme can form an EMF.
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, Real{1}), zero(n, Real{0}), en(n, Real{1});
  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(zero.data(), zero.size());
  u.my.copy_from_host(zero.data(), zero.size());
  u.mz.copy_from_host(zero.data(), zero.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bz_cell.copy_from_host(zero.data(), zero.size());
}

}  // namespace

TEST(MhdCtScheme, SchemeIsRegistered) {
  EXPECT_TRUE(quasar::Registry<ICtScheme>::instance().contains("fd_ct_christlieb"));
}

// Constructing the scheme by registry name succeeds and yields a usable object.
TEST(MhdCtScheme, ConstructByRegistryNameSucceeds) {
  auto ct = make_ct();
  ASSERT_NE(ct, nullptr);
}

// The divergence diagnostic reports ~0 on an analytically divergence-free seed.
TEST(MhdCtScheme, DivergenceFreeSeedHasZeroDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{32, 32, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  quasar::mhd::MhdField2D<Real> u{g};
  seed_divergence_free_b(u, g);

  auto ct = make_ct();
  const Real divb = ct->divergence_b_linf(u);
  // A small multiple of round-off scaled by 1/dx (the divergence has units 1/L).
  const Real eps = std::numeric_limits<Real>::epsilon();
  EXPECT_LT(divb, Real{1e3} * eps / g.dx())
      << "div(B) on divergence-free seed = " << divb;
}

// The device reduction divergence_b_linf reproduces the host-computed max-abs
// cell-centered div(B) over interior cells, on a divergence-free seed (~0).
// This pins the reduction's CORRECTNESS, not just its near-zero output.
TEST(MhdCtScheme, DivergenceLinfMatchesHostOnDivFreeField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{24, 40, Real{1}, Real{2}, Real{0}, Real{0}, 4};
  quasar::mhd::MhdField2D<Real> u{g};

  std::vector<Real> bx, by;
  fill_divergence_free_faces(g, bx, by);
  u.bx_face.copy_from_host(bx.data(), bx.size());
  u.by_face.copy_from_host(by.data(), by.size());

  const Real host_linf = host_divergence_b_linf(g, bx, by);
  auto ct = make_ct();
  const Real dev_linf = ct->divergence_b_linf(u);

  // The reduction must agree with the independent host computation. Allow a few
  // ULPs of |B|/dx slack for the differing summation/order on device.
  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real tol = Real{1e3} * eps / g.dx();
  EXPECT_NEAR(dev_linf, host_linf, tol)
      << "device L-inf div(B) = " << dev_linf << ", host = " << host_linf;
}

// The device reduction returns a KNOWN NONZERO maximum when the field carries a
// deliberate, localized divergence at exactly one interior cell. This is the
// strongest correctness check on the reduction: it must FIND the spike (not just
// report ~0), at the right magnitude. Most-likely RED: the device reduction does
// not exist yet.
TEST(MhdCtScheme, DivergenceLinfFindsKnownNonzeroSpike) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{16, 16, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  quasar::mhd::MhdField2D<Real> u{g};

  // Start from a divergence-free field, then perturb one face value so that
  // exactly one interior cell acquires a known nonzero divergence.
  std::vector<Real> bx, by;
  fill_divergence_free_faces(g, bx, by);

  // Choose an interior target cell well away from the boundary.
  const int ti = 7, tj = 9;
  // divB(ti,tj) uses bx_face(ti+1,tj) with weight +1/dx. Bumping that one face
  // value by `bump` raises ONLY divB(ti,tj) (and divB(ti+1,tj) via its own
  // -1/dx term). To make a single clean maximum, also bump bx_face(ti+2,tj) by
  // the same amount so divB(ti+1,tj) is left unchanged: its two bx terms shift
  // equally and cancel.
  const Real bump = Real{0.5};
  bx[g.index(ti + 1, tj)] += bump;
  bx[g.index(ti + 2, tj)] += bump;

  u.bx_face.copy_from_host(bx.data(), bx.size());
  u.by_face.copy_from_host(by.data(), by.size());

  // Independent host max-abs over interior cells, computed from the same stencil.
  const Real host_linf = host_divergence_b_linf(g, bx, by);

  // Sanity: the perturbation actually produced a clearly nonzero divergence.
  const Real expected_spike = bump / g.dx();  // the lone +1/dx face bump
  ASSERT_GT(host_linf, Real{0.5} * expected_spike)
      << "test setup failed to create a nonzero divergence";

  auto ct = make_ct();
  const Real dev_linf = ct->divergence_b_linf(u);

  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real tol = Real{1e3} * eps / g.dx();
  EXPECT_NEAR(dev_linf, host_linf, tol)
      << "device L-inf div(B) = " << dev_linf << ", host = " << host_linf;
  // And it must be the genuinely nonzero spike, not a residual near zero.
  EXPECT_GT(dev_linf, Real{0.5} * expected_spike)
      << "reduction failed to surface the known divergence spike";
}

// After a CT update step (compute_emf + update_face_b) the discretely
// divergence-free property is preserved to machine epsilon: the CT update curl
// of ANY corner Ez telescopes the cell-centered div(B) change to identically
// zero. This must hold regardless of the EMF magnitude / discretization.
TEST(MhdCtScheme, UpdatePreservesDivergenceFreeToMachineEps) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{32, 32, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const Real gamma = Real{5} / Real{3};

  quasar::mhd::MhdField2D<Real> u{g};
  seed_divergence_free_b(u, g);

  // Interface states feeding the EMF reconstruction. They carry the cell-
  // reconstructed B components; for this CT test their exact content only needs
  // to be a valid (allocated, finite) input -- the CT update curl is what must
  // preserve div(B)=0 regardless of the EMF magnitude.
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  quasar::mhd::EmfField2D<Real> emf{g};

  auto ct = make_ct();
  const Real divb0 = ct->divergence_b_linf(u);

  ct->compute_emf(u, ifx, ify, emf, gamma);
  ct->update_face_b(u, emf, /*dt=*/Real{1e-3});

  const Real divb1 = ct->divergence_b_linf(u);
  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real bound = Real{1e3} * eps / g.dx();
  EXPECT_LT(divb1, bound) << "div(B) after CT step = " << divb1;
  // The CT step must not manufacture a divergence larger than the seed's.
  EXPECT_LT(divb1, divb0 + bound);
}

// The telescoping guarantee is independent of dt: a much larger step must still
// preserve div(B) to round-off. Pins that the preservation is structural (curl
// of a discrete potential), not a small-dt artifact.
TEST(MhdCtScheme, UpdatePreservesDivergenceFreeForLargeDt) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{32, 32, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const Real gamma = Real{5} / Real{3};

  quasar::mhd::MhdField2D<Real> u{g};
  seed_divergence_free_b(u, g);

  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  quasar::mhd::EmfField2D<Real> emf{g};

  auto ct = make_ct();
  const Real divb0 = ct->divergence_b_linf(u);

  ct->compute_emf(u, ifx, ify, emf, gamma);
  ct->update_face_b(u, emf, /*dt=*/Real{0.25});  // large step

  const Real divb1 = ct->divergence_b_linf(u);
  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real bound = Real{1e3} * eps / g.dx();
  EXPECT_LT(divb1, bound) << "div(B) after large-dt CT step = " << divb1;
  EXPECT_LT(divb1, divb0 + bound);
}
