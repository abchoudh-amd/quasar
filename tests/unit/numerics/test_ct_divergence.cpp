// Tests for the constrained-transport (CT) scheme.
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
// ChristliebFdCt constructs corner EMFs from the directional Godunov interface
// fluxes.  The tests below pin both the range safety of its MP5/MP7 transverse
// interpolation and the defining CT invariant: the discrete divergence of a
// corner-EMF curl vanishes to round-off.  Cartesian runs use ordinary face
// differences; cylindrical (r,z) runs use radial face areas and annular cell
// volumes.
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
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
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
using quasar::numerics::MhdPrim;
using quasar::numerics::MhdState;

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

Real host_cylindrical_divergence_b_linf(
    const Grid2D& g, const std::vector<Real>& br,
    const std::vector<Real>& bz) {
  const Real dr = g.dx(), dz = g.dy();
  Real m = Real{0};
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real r_lo = g.r_at_edge(i);
      const Real r_hi = g.r_at_edge(i + 1);
      const Real r_center = g.r_at_cell_center(i);
      const Real divb =
          (r_hi * br[g.index(i + 1, j)] -
           r_lo * br[g.index(i, j)]) / (r_center * dr) +
          (bz[g.index(i, j + 1)] - bz[g.index(i, j)]) / dz;
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

void seed_uniform_interface(quasar::numerics::MhdInterfaceStates<Real>& iface,
                            const MhdState& state) {
  const std::size_t n = iface.grid.storage_size();
  std::vector<Real> values(n);
  const auto fill = [&](auto& buffer, Real value) {
    std::fill(values.begin(), values.end(), value);
    buffer.copy_from_host(values.data(), values.size());
  };
  fill(iface.Lrho, state.rho); fill(iface.Rrho, state.rho);
  fill(iface.Lmx, state.mx); fill(iface.Rmx, state.mx);
  fill(iface.Lmy, state.my); fill(iface.Rmy, state.my);
  fill(iface.Lmz, state.mz); fill(iface.Rmz, state.mz);
  fill(iface.Lenergy, state.energy); fill(iface.Renergy, state.energy);
  fill(iface.Lbx, state.bx); fill(iface.Rbx, state.bx);
  fill(iface.Lby, state.by); fill(iface.Rby, state.by);
  fill(iface.Lbz, state.bz); fill(iface.Rbz, state.bz);
}

void copy_interface_states(quasar::numerics::MhdInterfaceStates<Real>& iface,
                           const std::vector<MhdState>& state) {
  const std::size_t n = iface.grid.storage_size();
  ASSERT_EQ(state.size(), n);
  std::vector<Real> values(n);
  const auto fill = [&](auto& left, auto& right, Real MhdState::*member) {
    for (std::size_t k = 0; k < n; ++k) values[k] = state[k].*member;
    left.copy_from_host(values.data(), values.size());
    right.copy_from_host(values.data(), values.size());
  };
  fill(iface.Lrho, iface.Rrho, &MhdState::rho);
  fill(iface.Lmx, iface.Rmx, &MhdState::mx);
  fill(iface.Lmy, iface.Rmy, &MhdState::my);
  fill(iface.Lmz, iface.Rmz, &MhdState::mz);
  fill(iface.Lenergy, iface.Renergy, &MhdState::energy);
  fill(iface.Lbx, iface.Rbx, &MhdState::bx);
  fill(iface.Lby, iface.Rby, &MhdState::by);
  fill(iface.Lbz, iface.Rbz, &MhdState::bz);
}

void seed_smooth_interface(quasar::numerics::MhdInterfaceStates<Real>& iface,
                           Real gamma) {
  const Grid2D& g = iface.grid;
  const std::size_t n = g.storage_size();
  std::vector<MhdState> state(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const MhdPrim primitive{
          Real{1} + Real{0.002} * i,
          Real{0.1} + Real{0.003} * j,
          Real{-0.2} + Real{0.004} * i,
          Real{0.02} - Real{0.001} * j,
          Real{1} + Real{0.002} * j,
          Real{0.3} + Real{0.002} * j,
          Real{-0.1} + Real{0.003} * i,
          Real{0.05} + Real{0.001} * (i - j)};
      state[g.index(i, j)] = quasar::numerics::to_conserved(primitive, gamma);
    }
  }
  copy_interface_states(iface, state);
}

void seed_directional_emf_pattern(
    quasar::numerics::MhdInterfaceStates<Real>& iface,
    const std::vector<Real>& emf_pattern, Real gamma) {
  const Grid2D& g = iface.grid;
  const std::size_t n = g.storage_size();
  std::vector<MhdState> state(n);
  constexpr Real rho = Real{1e-308};
  constexpr Real internal_energy = Real{1e300};
  const Real p = (gamma - Real{1}) * internal_energy;
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const int q = iface.dir == 0 ? j : i;
      const Real ez = q >= 0 && q < static_cast<int>(emf_pattern.size())
          ? emf_pattern[static_cast<std::size_t>(q)]
          : Real{0};
      const MhdPrim primitive = iface.dir == 0
          ? MhdPrim{rho, Real{0}, ez, Real{0}, p,
                    Real{1}, Real{0}, Real{0}}
          : MhdPrim{rho, -ez, Real{0}, Real{0}, p,
                    Real{0}, Real{1}, Real{0}};
      state[g.index(i, j)] = quasar::numerics::to_conserved(primitive, gamma);
    }
  }
  copy_interface_states(iface, state);
}

void expect_physical_corner_emf_finite(const quasar::mhd::EmfField2D<Real>& emf,
                                       const Grid2D& g) {
  std::vector<Real> ez(g.storage_size());
  emf.ez_edge.copy_to_host(ez.data(), ez.size());
  for (int j = 0; j <= g.ny; ++j) {
    for (int i = 0; i <= g.nx; ++i) {
      EXPECT_TRUE(std::isfinite(ez[g.index(i, j)]))
          << "non-finite corner EMF at (" << i << "," << j << ")";
    }
  }
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

// MP5/MP7 face-to-corner coefficients reach 150/1225.  Multiplying those by a
// constant near DBL_MAX before the final /256 or /2048 produces Inf even though
// the interpolation and the final x/y average are exactly that constant.  Use
// separate x/y states whose directional ideal-MHD electric field is the same
// enormous, representable tangential velocity.
TEST(MhdCtScheme, HighOrderConstantNearMaxEmfRemainsFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  const Real emf_value = std::numeric_limits<Real>::max() / Real{4};
  constexpr Real rho = Real{1e-308};
  constexpr Real internal_energy = Real{1e300};
  constexpr Real p = (gamma - Real{1}) * internal_energy;

  // x face: Ez = Bx*vy - By*vx = emf_value, with vx=By=0.
  const MhdState xstate = quasar::numerics::to_conserved(
      quasar::numerics::MhdPrim{rho, Real{0}, emf_value, Real{0}, p,
                                Real{1}, Real{0}, Real{0}}, gamma);
  // y face: the same Ez, now from -By*vx with vy=Bx=0.
  const MhdState ystate = quasar::numerics::to_conserved(
      quasar::numerics::MhdPrim{rho, -emf_value, Real{0}, Real{0}, p,
                                Real{0}, Real{1}, Real{0}}, gamma);
  ASSERT_TRUE(std::isfinite(xstate.energy));
  ASSERT_TRUE(std::isfinite(ystate.energy));
  ASSERT_GT(quasar::numerics::pressure(xstate, gamma), Real{0});
  ASSERT_GT(quasar::numerics::pressure(ystate, gamma), Real{0});

  quasar::mhd::MhdField2D<Real> u{g};
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  seed_uniform_interface(ifx, xstate);
  seed_uniform_interface(ify, ystate);
  const quasar::mhd::MhdBackgroundField<Real> background{};
  const quasar::mhd::BoundaryFlags4 flags{};

  for (const int order : {5, 7}) {
    quasar::mhd::EmfField2D<Real> emf{g};
    quasar::mhd::launch_mhd_ct_emf(
        u, background, ifx, ify, flags, emf, gamma, /*stream=*/nullptr,
        order, /*cylindrical=*/false, /*hll_only=*/true);
    quasar::backend::device_synchronize(nullptr);
    std::vector<Real> ez(g.storage_size());
    emf.ez_edge.copy_to_host(ez.data(), ez.size());
    const Real actual = ez[g.index(/*i=*/3, /*j=*/3)];
    EXPECT_TRUE(std::isfinite(actual)) << "order=" << order;
    EXPECT_NEAR(actual, emf_value, Real{2e-13} * emf_value)
        << "order=" << order;
  }
}

TEST(MhdCtScheme, Mp7EmfRetainsSmallSurvivorAfterGiantCancellation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{12, 12, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real survivor = Real{21};
  const Real large = std::numeric_limits<Real>::max() / Real{4};
  std::vector<Real> pattern(8, Real{0});
  // MP7 weights are [-5,49,-245,1225,1225,-245,49,-5]/2048.
  // The two central contributions cancel exactly; sample zero carries 21.
  pattern[0] = survivor * Real{2048} / Real{-5};
  pattern[3] = large;
  pattern[4] = -large;

  quasar::mhd::MhdField2D<Real> u{g};
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  seed_directional_emf_pattern(ifx, pattern, gamma);
  seed_directional_emf_pattern(ify, pattern, gamma);
  const quasar::mhd::MhdBackgroundField<Real> background{};
  const quasar::mhd::BoundaryFlags4 flags{};
  quasar::mhd::EmfField2D<Real> emf{g};
  quasar::mhd::launch_mhd_ct_emf(
      u, background, ifx, ify, flags, emf, gamma, /*stream=*/nullptr,
      /*scheme_order=*/7, /*cylindrical=*/false, /*hll_only=*/true);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> ez(g.storage_size());
  emf.ez_edge.copy_to_host(ez.data(), ez.size());
  const Real actual = ez[g.index(/*i=*/4, /*j=*/4)];
  ASSERT_TRUE(std::isfinite(actual));
  EXPECT_NEAR(actual, survivor, Real{2e-12});
}

// The retained direct update API must advance both physical high-face layers,
// not just the nx-by-ny low-face rectangle.  Omitting Bx(nx,j) or By(i,ny)
// breaks the curl/divergence telescoping in the last cell row/column.
TEST(MhdCtScheme, DirectFaceUpdateAdvancesHighFacesAndPreservesDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{6, 5, Real{2}, Real{3}, Real{0}, Real{0}, /*nghost=*/2};
  const std::size_t n = g.storage_size();
  constexpr Real dt = Real{0.125};
  std::vector<Real> zero(n, Real{0});
  std::vector<Real> ez(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      ez[g.index(i, j)] =
          Real{0.25} * i * i - Real{0.5} * j * j + Real{0.125} * i * j;
    }
  }

  quasar::mhd::MhdField2D<Real> u{g};
  u.bx_face.copy_from_host(zero.data(), zero.size());
  u.by_face.copy_from_host(zero.data(), zero.size());
  quasar::mhd::EmfField2D<Real> emf{g};
  emf.ez_edge.copy_from_host(ez.data(), ez.size());

  quasar::mhd::launch_mhd_face_b_update(
      u, emf, dt, /*stream=*/nullptr, /*cylindrical=*/false);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> bx(n), by(n);
  u.bx_face.copy_to_host(bx.data(), bx.size());
  u.by_face.copy_to_host(by.data(), by.size());
  for (int j = 0; j < g.ny; ++j) {
    const Real expected = -dt *
        (ez[g.index(g.nx, j + 1)] - ez[g.index(g.nx, j)]) / g.dy();
    EXPECT_NEAR(bx[g.index(g.nx, j)], expected, Real{2e-14});
  }
  for (int i = 0; i < g.nx; ++i) {
    const Real expected = dt *
        (ez[g.index(i + 1, g.ny)] - ez[g.index(i, g.ny)]) / g.dx();
    EXPECT_NEAR(by[g.index(i, g.ny)], expected, Real{2e-14});
  }
  EXPECT_LT(host_divergence_b_linf(g, bx, by), Real{2e-13});
}

// In axisymmetric (r,z) geometry the same corner curl must annihilate
// (1/r)d(r Br)/dr + dBz/dz, not the Cartesian divergence.  Exercise the axis
// cell (r_lo=0), the interior annuli, and both physical high-face layers.
TEST(MhdCtScheme, CylindricalFaceUpdatePreservesAnnularDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{6, 5, Real{3}, Real{4}, Real{0}, Real{-2}, /*nghost=*/2};
  const std::size_t n = g.storage_size();
  constexpr Real dt = Real{0.125};
  std::vector<Real> zero(n, Real{0});
  std::vector<Real> ez(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      ez[g.index(i, j)] =
          Real{0.25} * i * i - Real{0.5} * j * j + Real{0.125} * i * j;
    }
  }

  quasar::mhd::MhdField2D<Real> u{g};
  u.bx_face.copy_from_host(zero.data(), zero.size());
  u.by_face.copy_from_host(zero.data(), zero.size());
  quasar::mhd::EmfField2D<Real> emf{g};
  emf.ez_edge.copy_from_host(ez.data(), ez.size());
  quasar::mhd::launch_mhd_face_b_update(
      u, emf, dt, /*stream=*/nullptr, /*cylindrical=*/true);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> br(n), bz(n);
  u.bx_face.copy_to_host(br.data(), br.size());
  u.by_face.copy_to_host(bz.data(), bz.size());
  for (int j = 0; j < g.ny; ++j) {
    const Real expected = -dt *
        (ez[g.index(g.nx, j + 1)] - ez[g.index(g.nx, j)]) / g.dy();
    EXPECT_NEAR(br[g.index(g.nx, j)], expected, Real{2e-14});
  }
  for (int i = 0; i < g.nx; ++i) {
    const Real ez_lo = ez[g.index(i, g.ny)];
    const Real ez_hi = ez[g.index(i + 1, g.ny)];
    const Real expected = dt *
        ((ez_hi - ez_lo) / g.dx() +
         (ez_hi + ez_lo) / (Real{2} * g.r_at_cell_center(i)));
    EXPECT_NEAR(bz[g.index(i, g.ny)], expected, Real{2e-14});
  }
  EXPECT_LT(host_cylindrical_divergence_b_linf(g, br, bz), Real{3e-13});
  EXPECT_GT(host_divergence_b_linf(g, br, bz), Real{1e-3});
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

  // Smooth, positive interface states produce a finite, spatially varying EMF;
  // invalid zero-density defaults could otherwise turn the whole update into
  // NaNs that a max reduction is permitted to ignore.
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  seed_smooth_interface(ifx, gamma);
  seed_smooth_interface(ify, gamma);
  quasar::mhd::EmfField2D<Real> emf{g};

  auto ct = make_ct();
  const Real divb0 = ct->divergence_b_linf(u);

  ct->compute_emf(u, ifx, ify, emf, gamma);
  expect_physical_corner_emf_finite(emf, g);
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
  seed_smooth_interface(ifx, gamma);
  seed_smooth_interface(ify, gamma);
  quasar::mhd::EmfField2D<Real> emf{g};

  auto ct = make_ct();
  const Real divb0 = ct->divergence_b_linf(u);

  ct->compute_emf(u, ifx, ify, emf, gamma);
  expect_physical_corner_emf_finite(emf, g);
  ct->update_face_b(u, emf, /*dt=*/Real{0.25});  // large step

  const Real divb1 = ct->divergence_b_linf(u);
  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real bound = Real{1e3} * eps / g.dx();
  EXPECT_LT(divb1, bound) << "div(B) after large-dt CT step = " << divb1;
  EXPECT_LT(divb1, divb0 + bound);
}
