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

void seed_divergence_free_b(quasar::mhd::MhdField2D<Real>& u, const Grid2D& g) {
  const std::size_t n = g.storage_size();
  std::vector<Real> bx(n, Real{0}), by(n, Real{0});
  const Real dx = g.dx(), dy = g.dy();
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      bx[k] = (corner_A(g, i, j + 1) - corner_A(g, i, j)) / dy;
      by[k] = -(corner_A(g, i + 1, j) - corner_A(g, i, j)) / dx;
    }
  }
  u.bx_face.copy_from_host(bx.data(), bx.size());
  u.by_face.copy_from_host(by.data(), by.size());

  // A quiescent, positive fluid background so the scheme can form an EMF.
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

// After a CT update step (compute_emf + update_face_b) the discretely
// divergence-free property is preserved to machine epsilon: CT updates the face
// fields with edge EMFs whose curl is discretely divergence-free by construction.
TEST(MhdCtScheme, UpdatePreservesDivergenceFreeToMachineEps) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  Grid2D g{32, 32, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const Real gamma = Real{5} / Real{3};

  quasar::mhd::MhdField2D<Real> u{g};
  seed_divergence_free_b(u, g);

  // Interface states feeding the EMF reconstruction. They carry the cell-
  // reconstructed B components; for this CT test their exact content only needs
  // to be a valid (allocated, finite) input — the CT update curl is what must
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
