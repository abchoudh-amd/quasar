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
#include "quasar/numerics/radial_tables.hpp"
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
#include <utility>
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

// Exact ring average of sin(k r) over radial cell i.  The long-double
// antiderivative keeps the analytic seed substantially more accurate than the
// MP7 interpolation being measured below.
Real ring_average_sine(const Grid2D& g, int i, Real wavenumber) {
  const long double a = static_cast<long double>(g.r_at_edge(i));
  const long double b = static_cast<long double>(g.r_at_edge(i + 1));
  const long double k = static_cast<long double>(wavenumber);
  const auto primitive = [k](long double r) {
    return std::sin(k * r) / (k * k) - r * std::cos(k * r) / k;
  };
  const long double measure = (b * b - a * a) / 2.0L;
  return static_cast<Real>((primitive(b) - primitive(a)) / measure);
}

// Isolate the CT finish phase's radial face-average-to-corner interpolation.
// x-face and cell-centred contributions are exactly zero, while y-face entries
// are ring averages of a smooth radial profile.  With both one-dimensional
// proof tables false, the corner result is therefore precisely the y pass.
void seed_radial_face_emf(quasar::mhd::EmfField2D<Real>& emf,
                          Real wavenumber) {
  const Grid2D& g = emf.grid;
  const std::size_t n = g.storage_size();
  std::vector<Real> zero(n, Real{0});
  std::vector<Real> yface(n, Real{0});
  std::vector<int> no_jump(n, 0);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      yface[g.index(i, j)] = ring_average_sine(g, i, wavenumber);
    }
  }
  emf.cell_ez_average.copy_from_host(zero.data(), n);
  emf.xface_ez.copy_from_host(zero.data(), n);
  emf.yface_ez.copy_from_host(yface.data(), n);
  emf.xface_no_jump.copy_from_host(no_jump.data(), n);
  emf.yface_no_jump.copy_from_host(no_jump.data(), n);
}

void zero_field(quasar::mhd::MhdField2D<Real>& field) {
  const std::size_t n = field.grid.storage_size();
  const std::vector<Real> zero(n, Real{0});
  field.rho.copy_from_host(zero.data(), n);
  field.mx.copy_from_host(zero.data(), n);
  field.my.copy_from_host(zero.data(), n);
  field.mz.copy_from_host(zero.data(), n);
  field.energy.copy_from_host(zero.data(), n);
  field.bx_face.copy_from_host(zero.data(), n);
  field.by_face.copy_from_host(zero.data(), n);
  field.bz_cell.copy_from_host(zero.data(), n);
}

struct CylindricalCtRun {
  Real worst_divb{0};
  Real largest_b{0};
  Real axis_emf{0};
};

CylindricalCtRun run_cylindrical_mp7_ct(Real origin_r) {
  constexpr int kSteps = 20;
  constexpr Real kDt = Real{2e-3};
  constexpr Real kWavenumber = Real{4.25};
  const Grid2D g{20, 14, Real{1}, Real{0.8}, origin_r, Real{-0.4},
                 /*nghost=*/4};
  quasar::numerics::RadialTables radial_tables{g, /*scheme_order=*/7};
  EXPECT_EQ(radial_tables.view().active, 1);
  EXPECT_EQ(radial_tables.view().scheme_order, 7);

  quasar::mhd::EmfField2D<Real> emf{g};
  seed_radial_face_emf(emf, kWavenumber);
  quasar::mhd::MhdField2D<Real> state{g};
  quasar::mhd::MhdField2D<Real> next{g};
  quasar::mhd::MhdField2D<Real> rate{g};
  zero_field(state);
  zero_field(next);
  zero_field(rate);

  quasar::mhd::BoundaryFlags4 flags{{1, 1, 1, 1}};
  if (origin_r == Real{0}) flags.side[0] = 3;

  CylindricalCtRun result;
  std::vector<Real> br(g.storage_size()), bz(g.storage_size());
  for (int step = 0; step < kSteps; ++step) {
    // This is the same CT path used by the assembled solver: MP7 corner EMF,
    // annular curl rate, then a stage update.  Rebuilding the corner table on
    // every step also keeps the active RadialTablesView in the exercised path.
    quasar::mhd::launch_mhd_ct_emf_finish(
        flags, emf, /*stream=*/nullptr, /*scheme_order=*/7,
        /*cylindrical=*/true, radial_tables.view());
    quasar::mhd::launch_mhd_emf_curl_rate(
        emf, rate, g, /*stream=*/nullptr, /*cylindrical=*/true);
    quasar::mhd::launch_mhd_rk_stage(
        next, state, state, rate, Real{1}, Real{0}, kDt,
        /*stream=*/nullptr);
    quasar::backend::device_synchronize(nullptr);

    next.bx_face.copy_to_host(br.data(), br.size());
    next.by_face.copy_to_host(bz.data(), bz.size());
    result.worst_divb = std::max(
        result.worst_divb, host_cylindrical_divergence_b_linf(g, br, bz));
    for (int j = 0; j < g.ny; ++j) {
      for (int i = 0; i <= g.nx; ++i) {
        result.largest_b = std::max(
            result.largest_b, std::abs(br[g.index(i, j)]));
      }
    }
    for (int j = 0; j <= g.ny; ++j) {
      for (int i = 0; i < g.nx; ++i) {
        result.largest_b = std::max(
            result.largest_b, std::abs(bz[g.index(i, j)]));
      }
    }
    std::swap(state, next);
  }

  std::vector<Real> edge(g.storage_size());
  emf.ez_edge.copy_to_host(edge.data(), edge.size());
  result.axis_emf = edge[g.index(0, g.ny / 2)];
  return result;
}

struct CornerEmfErrors {
  Real weighted{0};
  Real cartesian_control{0};
};

CornerEmfErrors cylindrical_corner_emf_errors(int nx) {
  constexpr Real kWavenumber = Real{7.4};
  const Grid2D g{nx, 12, Real{1}, Real{1}, Real{0.7}, Real{0},
                 /*nghost=*/4};
  quasar::numerics::RadialTables radial_tables{g, /*scheme_order=*/7};
  const int corner_i = nx / 2;
  const int corner_j = g.ny / 2;
  const Real exact = std::sin(kWavenumber * g.r_at_edge(corner_i));
  const quasar::mhd::BoundaryFlags4 outflow{{1, 1, 1, 1}};

  const auto error = [&](quasar::numerics::RadialTablesView view) {
    quasar::mhd::EmfField2D<Real> emf{g};
    seed_radial_face_emf(emf, kWavenumber);
    quasar::mhd::launch_mhd_ct_emf_finish(
        outflow, emf, /*stream=*/nullptr, /*scheme_order=*/7,
        /*cylindrical=*/true, view);
    quasar::backend::device_synchronize(nullptr);
    std::vector<Real> edge(g.storage_size());
    emf.ez_edge.copy_to_host(edge.data(), edge.size());
    return std::abs(edge[g.index(corner_i, corner_j)] - exact);
  };

  return CornerEmfErrors{
      error(radial_tables.view()),
      error(quasar::numerics::RadialTablesView{})};
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

void seed_uniform_field(quasar::mhd::MhdField2D<Real>& field,
                        const MhdState& state) {
  const std::size_t n = field.grid.storage_size();
  std::vector<Real> values(n);
  const auto fill = [&](auto& buffer, Real value) {
    std::fill(values.begin(), values.end(), value);
    buffer.copy_from_host(values.data(), values.size());
  };
  fill(field.rho, state.rho); fill(field.mx, state.mx);
  fill(field.my, state.my); fill(field.mz, state.mz);
  fill(field.energy, state.energy); fill(field.bx_face, state.bx);
  fill(field.by_face, state.by); fill(field.bz_cell, state.bz);
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

void copy_field_states(quasar::mhd::MhdField2D<Real>& field,
                       const std::vector<MhdState>& state) {
  const std::size_t n = field.grid.storage_size();
  ASSERT_EQ(state.size(), n);
  std::vector<Real> values(n);
  const auto fill = [&](auto& buffer, Real MhdState::*member) {
    for (std::size_t k = 0; k < n; ++k) values[k] = state[k].*member;
    buffer.copy_from_host(values.data(), values.size());
  };
  fill(field.rho, &MhdState::rho); fill(field.mx, &MhdState::mx);
  fill(field.my, &MhdState::my); fill(field.mz, &MhdState::mz);
  fill(field.energy, &MhdState::energy); fill(field.bx_face, &MhdState::bx);
  fill(field.by_face, &MhdState::by); fill(field.bz_cell, &MhdState::bz);
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

// MP5/MP7 face-average-to-corner coefficients reach 37/533. Multiplying those by
// a constant near DBL_MAX before the final /60 or /840 produces Inf even though
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
  seed_uniform_field(u, xstate);
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

TEST(MhdCtScheme, Mp5CommonDenominatorCancellationIsExact) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{16, 16, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  const Real scale = std::scalbn(Real{1}, 900);
  const Real face_emf[2][6] = {
      {Real{60} * scale,
       Real{3.30859375} * scale,
       Real{-0.765625} * scale,
       Real{-0.140625} * scale,
       Real{0}, Real{0}},
      {Real{1.9171051776448014} * scale,
       Real{1.8461202561526626} * scale,
       Real{-1.6574366374011578} * scale,
       Real{1.568143878640971} * scale,
       Real{-1.8120937183404688} * scale,
       Real{1.6589391989796616} * scale}};
  // With MP5 face-average-to-edge weights [1,-8,37,37,-8,1]/60,
  // both integer-coefficient numerators are exactly zero. The first catches a
  // denominator distributed over the terms; the second catches the roundoff
  // in coefficient*sample itself and requires an error-free FMA TwoProd.

  const MhdState quiet = quasar::numerics::to_conserved(
      MhdPrim{Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
              Real{1}, Real{0}, Real{0}}, gamma);
  quasar::mhd::MhdField2D<Real> u{g};
  seed_uniform_field(u, quiet);
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  seed_uniform_interface(ify, quiet);

  const std::size_t n = g.storage_size();
  std::vector<MhdState> xstate(n);
  const Real rho = std::scalbn(Real{1}, -1020);
  const Real bz_magnitude = std::scalbn(Real{1}, 400);
  const Real gas_pressure = std::scalbn(Real{1}, 780);
  const int corner_j = 6;
  for (int regression = 0; regression < 2; ++regression) {
    for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
      Real ez = Real{0};
      const int sample = j - (corner_j - 3);
      if (sample >= 0 && sample < 6) ez = face_emf[regression][sample];
      const Real bz = ((j & 1) == 0 ? Real{1} : Real{-1}) * bz_magnitude;
      const Real my = rho * ez;
      const Real kinetic = Real{0.5} * my * ez;
      const Real magnetic = Real{0.5} * bz * bz + Real{0.5};
      const MhdState state{
          rho, Real{0}, my, Real{0},
          gas_pressure / (gamma - Real{1}) + kinetic + magnetic,
          Real{1}, Real{0}, bz};
      ASSERT_TRUE(std::isfinite(state.energy));
      ASSERT_GT(quasar::numerics::pressure(state, gamma), Real{0});
      for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
        xstate[g.index(i, j)] = state;
      }
    }
    // Alternating Bz makes the MP5 midpoint recovery overshoot |Bz| while the
    // face-average states remain admissible. Every nonlinear face solve in the
    // six-point corner stencil therefore takes its exact base-face fallback,
    // isolating the CT rational interpolation exercised by this regression.
    copy_interface_states(ifx, xstate);

    const quasar::mhd::MhdBackgroundField<Real> background{};
    const quasar::mhd::BoundaryFlags4 flags{};
    quasar::mhd::EmfField2D<Real> emf{g};
    quasar::mhd::launch_mhd_ct_emf(
        u, background, ifx, ify, flags, emf, gamma, /*stream=*/nullptr,
        /*scheme_order=*/5, /*cylindrical=*/false, /*hll_only=*/true);
    quasar::backend::device_synchronize(nullptr);

    std::vector<Real> ez(n);
    emf.ez_edge.copy_to_host(ez.data(), ez.size());
    EXPECT_EQ(ez[g.index(/*i=*/6, corner_j)], Real{0})
        << "regression=" << regression;
  }
}

TEST(MhdCtScheme, Mp7EmfRetainsSmallSurvivorAfterGiantCancellation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  Grid2D g{12, 12, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real survivor = Real{21};
  const Real large = std::numeric_limits<Real>::max() / Real{4};
  std::vector<Real> pattern(static_cast<std::size_t>(g.ny), Real{0});
  // Finite-volume average-to-edge weights are
  // [-3,29,-139,533,533,-139,29,-3]/840.
  // The two central contributions cancel exactly; sample zero carries 21.
  pattern[0] = survivor * Real{840} / Real{-3};
  pattern[3] = large;
  pattern[4] = -large;

  quasar::mhd::MhdField2D<Real> u{g};
  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  const std::size_t n = g.storage_size();
  std::vector<MhdState> cell_state(n), yface_state(n);
  constexpr Real rho = Real{1e-308};
  constexpr Real internal_energy = Real{1e300};
  const Real p = (gamma - Real{1}) * internal_energy;
  const auto wrap = [](int q, int extent) {
    int value = q % extent;
    return value < 0 ? value + extent : value;
  };
  const auto edge_value = [&](int edge) {
    constexpr Real weight[8] = {
        Real{-3}, Real{29}, Real{-139}, Real{533},
        Real{533}, Real{-139}, Real{29}, Real{-3}};
    quasar::numerics::ScaledProductQuotientAccumulator<8> sum;
    for (int q = 0; q < 8; ++q) {
      quasar::numerics::append_scaled_product_quotient(
          sum, weight[q],
          pattern[static_cast<std::size_t>(wrap(edge - 4 + q, g.ny))],
          Real{840}, Real{1});
    }
    return quasar::numerics::finish_scaled_product_quotient_sum(sum);
  };
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    const Real cell_emf = pattern[static_cast<std::size_t>(wrap(j, g.ny))];
    const Real face_emf = edge_value(j);
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      cell_state[g.index(i, j)] = quasar::numerics::to_conserved(
          MhdPrim{rho, Real{0}, cell_emf, Real{0}, p,
                  Real{1}, Real{0}, Real{0}}, gamma);
      yface_state[g.index(i, j)] = quasar::numerics::to_conserved(
          MhdPrim{rho, Real{0}, face_emf, Real{0}, p,
                  Real{1}, Real{0}, Real{0}}, gamma);
    }
  }
  copy_field_states(u, cell_state);
  copy_interface_states(ifx, cell_state);
  copy_interface_states(ify, yface_state);
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

// Exercise the production MP7 CT seam with an active radial coefficient table,
// the annular emf_curl_rate kernel, and the RK stage update.  The corner EMF is
// non-uniform, so the magnetic field changes substantially; nevertheless the
// annular divergence must remain at round-off for every one of twenty updates.
TEST(MhdCtScheme, CylindricalCtPreservesAnnularDivB) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  for (const Real origin_r : {Real{0}, Real{0.75}}) {
    SCOPED_TRACE(::testing::Message{}
                 << (origin_r == Real{0} ? "axis" : "annulus"));
    const CylindricalCtRun run = run_cylindrical_mp7_ct(origin_r);
    ASSERT_GT(run.largest_b, Real{1e-3})
        << "the repeated CT updates did not produce a meaningful field";
    const Real scale = std::max(Real{1}, run.largest_b);
    const Real dr = Real{1} / Real{20};
    const Real roundoff_bound =
        Real{2e4} * std::numeric_limits<Real>::epsilon() * scale / dr;
    EXPECT_LT(run.worst_divb, roundoff_bound);
    if (origin_r == Real{0}) {
      EXPECT_EQ(run.axis_emf, Real{0})
          << "the coordinate-axis corner EMF must be pinned exactly";
    }
  }
}

// The y-face CT table varies in i, so it is the radial interpolation.  Isolate
// that pass with smooth exact ring averages and verify MP7 convergence.  The
// inactive-table run is a deliberate direction-swap control: radializing the
// x pass instead cannot affect this data and therefore leaves this Cartesian
// (uniform-measure) error in place.
TEST(MhdCtScheme, CylindricalCornerEmfIsHighOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const CornerEmfErrors coarse = cylindrical_corner_emf_errors(/*nx=*/12);
  const CornerEmfErrors fine = cylindrical_corner_emf_errors(/*nx=*/24);
  ASSERT_GT(coarse.weighted, Real{0});
  ASSERT_GT(fine.weighted, Real{0});
  ASSERT_GT(coarse.cartesian_control, Real{0});
  ASSERT_GT(fine.cartesian_control, Real{0});

  const Real weighted_order = std::log2(coarse.weighted / fine.weighted);
  const Real control_order =
      std::log2(coarse.cartesian_control / fine.cartesian_control);
  EXPECT_GT(weighted_order, Real{6.4})
      << "coarse=" << coarse.weighted << " fine=" << fine.weighted;
  EXPECT_LT(control_order, Real{3})
      << "the Cartesian control unexpectedly behaves like radial MP7";
  EXPECT_LT(fine.weighted, fine.cartesian_control * Real{1e-3})
      << "active radial tables did not materially improve the corner EMF";
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

namespace {

struct OneDimensionalEmfResult {
  Real corner{};
  Real godunov{};
};

OneDimensionalEmfResult one_dimensional_emf(
    int order, bool rotate_xy, bool active_background) {
  Grid2D g{24, 24, Real{1}, Real{1}, Real{0}, Real{0}, /*nghost=*/4};
  const std::size_t n = g.storage_size();
  constexpr Real gamma = Real{5} / Real{3};
  const MhdPrim left{Real{1.0}, Real{0.45}, Real{0.31}, Real{0.07},
                     Real{1.0}, Real{0.5}, Real{0.82}, Real{-0.12}};
  const MhdPrim right{Real{0.72}, Real{-0.24}, Real{-0.13}, Real{0.02},
                      Real{0.63}, Real{0.5}, Real{-0.37}, Real{0.18}};
  const auto rotated = [](MhdPrim w) {
    std::swap(w.vx, w.vy);
    std::swap(w.bx, w.by);
    return w;
  };

  std::vector<Real> rho(n), mx(n), my(n), mz(n), energy(n);
  std::vector<Real> bx(n), by(n), bz(n);
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const int q = rotate_xy ? ((j % g.ny) + g.ny) % g.ny
                              : ((i % g.nx) + g.nx) % g.nx;
      MhdPrim w = q < g.nx / 2 ? left : right;
      if (rotate_xy) w = rotated(w);
      const MhdState s = quasar::numerics::to_conserved(w, gamma);
      const std::size_t k = g.index(i, j);
      rho[k] = s.rho; mx[k] = s.mx; my[k] = s.my; mz[k] = s.mz;
      energy[k] = s.energy; bx[k] = s.bx; by[k] = s.by; bz[k] = s.bz;
    }
  }

  quasar::mhd::MhdField2D<Real> u{g};
  u.rho.copy_from_host(rho.data(), n);
  u.mx.copy_from_host(mx.data(), n);
  u.my.copy_from_host(my.data(), n);
  u.mz.copy_from_host(mz.data(), n);
  u.energy.copy_from_host(energy.data(), n);
  u.bx_face.copy_from_host(bx.data(), n);
  u.by_face.copy_from_host(by.data(), n);
  u.bz_cell.copy_from_host(bz.data(), n);

  quasar::mhd::MhdBackgroundField<Real> background{g};
  if (active_background) {
    background.active = true;
    const Real b0x = rotate_xy ? Real{-0.19} : Real{0.27};
    const Real b0y = rotate_xy ? Real{0.27} : Real{-0.19};
    std::vector<Real> b0x_values(n, b0x), b0y_values(n, b0y);
    std::vector<Real> b0z_values(n, Real{0.11});
    background.b0x_face.copy_from_host(b0x_values.data(), n);
    background.b0y_face.copy_from_host(b0y_values.data(), n);
    background.b0z_cell.copy_from_host(b0z_values.data(), n);
  }

  quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
  quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
  const quasar::mhd::BoundaryFlags4 periodic{};
  quasar::mhd::launch_mhd_reconstruct(
      u, background, 0, ifx, order, periodic, gamma, nullptr);
  quasar::mhd::launch_mhd_reconstruct(
      u, background, 1, ify, order, periodic, gamma, nullptr);

  quasar::mhd::MhdField2D<Real> directional_flux{g};
  const int dir = rotate_xy ? 1 : 0;
  quasar::mhd::launch_mhd_hlld_flux(
      rotate_xy ? ify : ifx, background, dir, directional_flux, periodic,
      gamma, nullptr, /*hll_only=*/false, /*momentum_parts=*/nullptr, order);
  quasar::mhd::EmfField2D<Real> emf{g};
  quasar::mhd::launch_mhd_ct_emf(
      u, background, ifx, ify, periodic, emf, gamma, nullptr, order,
      /*cylindrical=*/false, /*hll_only=*/false);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> edge(n), magnetic_flux(n);
  emf.ez_edge.copy_to_host(edge.data(), n);
  if (rotate_xy) {
    directional_flux.bx_face.copy_to_host(magnetic_flux.data(), n);
  } else {
    directional_flux.by_face.copy_to_host(magnetic_flux.data(), n);
  }
  const int fixed = 7;
  const int face = g.nx / 2;
  const std::size_t k = rotate_xy ? g.index(fixed, face)
                                  : g.index(face, fixed);
  return OneDimensionalEmfResult{
      edge[k], rotate_xy ? magnetic_flux[k] : -magnetic_flux[k]};
}

}  // namespace

// A flux-CT arithmetic average retains only half of the normal Godunov
// dissipation in this discontinuous one-dimensional problem.  Upwind CT must
// instead collapse to the complete x-face EMF, and after x/y rotation to the
// complete y-face EMF with the pseudovector sign reversal.  Exercise both the
// ordinary and active-background HLLD paths.
TEST(MhdCtScheme, OneDimensionalLimitIsExactAndRotationCovariant) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const int order : {2, 5, 7}) {
    for (const bool active_background : {false, true}) {
      const OneDimensionalEmfResult x =
          one_dimensional_emf(order, /*rotate_xy=*/false, active_background);
      const OneDimensionalEmfResult y =
          one_dimensional_emf(order, /*rotate_xy=*/true, active_background);
      EXPECT_EQ(x.corner, x.godunov)
          << "x-only order=" << order << " background=" << active_background;
      EXPECT_EQ(y.corner, y.godunov)
          << "y-only order=" << order << " background=" << active_background;
      EXPECT_EQ(x.corner, -y.corner)
          << "rotation order=" << order << " background=" << active_background;
    }
  }
}

TEST(MhdCtScheme,
     HighOrderOneDimensionalProofIncludesEachFaceRecoveryHalo) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // At the tested corner, MP5/MP7 interpolate y-face EMFs over one window, and
  // each of those faces recovers Gauss-point states from another two/three face
  // averages on either side. A jump just beyond the first window must therefore
  // invalidate the exact-1D shortcut even though every central interface in
  // that window has L==R.
  const Grid2D g{24, 24, Real{1}, Real{1}, Real{0}, Real{0},
                 /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  constexpr int corner_i = 10;
  constexpr int corner_j = 10;
  const std::size_t n = g.storage_size();
  const MhdState rest{Real{1}, Real{0}, Real{0}, Real{0}, Real{20},
                      Real{0}, Real{0}, Real{0}};

  for (const int order : {5, 7}) {
    SCOPED_TRACE(::testing::Message{} << "order=" << order);
    const int interpolation_lo = order == 5 ? -3 : -4;
    const int recovery_half = order == 5 ? 2 : 3;

    quasar::mhd::MhdField2D<Real> u{g};
    seed_uniform_field(u, rest);
    quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
    quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
    seed_uniform_interface(ifx, rest);
    seed_uniform_interface(ify, rest);

    // Keep the x-face induction flux exactly zero while making the orthogonal
    // no-jump proof false, so only the x-only shortcut can be selected.
    std::vector<Real> left_rho(n, rest.rho);
    std::vector<Real> right_rho(n, rest.rho);
    right_rho[g.index(corner_i, corner_j)] = Real{1.5};
    ifx.Lrho.copy_from_host(left_rho.data(), n);
    ifx.Rrho.copy_from_host(right_rho.data(), n);

    // This jump is outside the face-to-edge interpolation window, but inside
    // the transverse recovery stencil of its first y face.
    std::vector<Real> left_bx(n, rest.bx);
    std::vector<Real> right_bx(n, rest.bx);
    const std::size_t remote = g.index(
        corner_i + interpolation_lo - recovery_half, corner_j);
    left_bx[remote] = Real{2};
    right_bx[remote] = Real{-2};
    ify.Lbx.copy_from_host(left_bx.data(), n);
    ify.Rbx.copy_from_host(right_bx.data(), n);

    quasar::mhd::MhdBackgroundField<Real> background{g};
    quasar::mhd::EmfField2D<Real> emf{g};
    const quasar::mhd::BoundaryFlags4 periodic{};
    quasar::mhd::launch_mhd_ct_emf_prepare(
        u, background, ifx, ify, periodic, emf, gamma, nullptr, order,
        /*hll_only=*/false);
    quasar::backend::device_synchronize(nullptr);

    std::vector<int> yface_no_jump(n);
    emf.yface_no_jump.copy_to_host(yface_no_jump.data(), n);
    const std::size_t first_face =
        g.index(corner_i + interpolation_lo, corner_j);
    EXPECT_EQ(yface_no_jump[first_face], 0)
        << "the per-face proof ignored its transverse recovery halo";

    // Isolate the shortcut decision from HLLD details. The synthetic first tap
    // contributes exactly one to either MP5 or MP7 interpolation. The old
    // central-interface-only summary falsely selected the zero x-face EMF.
    std::vector<Real> zero(n, Real{0});
    std::vector<Real> synthetic_yface(n, Real{0});
    synthetic_yface[first_face] = order == 5 ? Real{60} : Real{-280};
    emf.xface_ez.copy_from_host(zero.data(), n);
    emf.yface_ez.copy_from_host(synthetic_yface.data(), n);
    emf.cell_ez_average.copy_from_host(zero.data(), n);
    quasar::mhd::launch_mhd_ct_emf_finish(
        periodic, emf, nullptr, order, /*cylindrical=*/false);
    quasar::backend::device_synchronize(nullptr);

    std::vector<Real> edge(n);
    emf.ez_edge.copy_to_host(edge.data(), n);
    EXPECT_EQ(edge[g.index(corner_i, corner_j)], Real{1});
  }
}

TEST(MhdCtScheme,
     HighOrderOneDimensionalProofPreservesBoundaryCoordinateMapping) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g{16, 16, Real{1}, Real{1}, Real{0}, Real{0},
                 /*nghost=*/4};
  constexpr Real gamma = Real{5} / Real{3};
  constexpr int face_j = 8;
  const std::size_t n = g.storage_size();
  const MhdState rest{Real{1}, Real{0}, Real{0}, Real{0}, Real{10},
                      Real{0}, Real{0}, Real{0}};

  for (const int order : {5, 7}) {
    SCOPED_TRACE(::testing::Message{} << "order=" << order);
    const int recovery_half = order == 5 ? 2 : 3;
    quasar::mhd::MhdField2D<Real> u{g};
    seed_uniform_field(u, rest);
    quasar::numerics::MhdInterfaceStates<Real> ifx{g, /*dir=*/0};
    quasar::numerics::MhdInterfaceStates<Real> ify{g, /*dir=*/1};
    seed_uniform_interface(ifx, rest);
    seed_uniform_interface(ify, rest);

    // Make only the raw low-x recovery guard unequal. Periodic and physical
    // proofs map this coordinate to their serial wrap/closure; an internal tile
    // must instead consume and validate the exchanged guard itself.
    std::vector<Real> right_rho(n, rest.rho);
    right_rho[g.index(-recovery_half, face_j)] = Real{1.5};
    ify.Rrho.copy_from_host(right_rho.data(), n);

    const auto proof_at_low_x = [&](quasar::mhd::BoundaryFlags4 flags) {
      quasar::mhd::MhdBackgroundField<Real> background{g};
      quasar::mhd::EmfField2D<Real> emf{g};
      quasar::mhd::launch_mhd_ct_emf_prepare(
          u, background, ifx, ify, flags, emf, gamma, nullptr, order,
          /*hll_only=*/false);
      quasar::backend::device_synchronize(nullptr);
      std::vector<int> proof(n);
      emf.yface_no_jump.copy_to_host(proof.data(), n);
      return proof[g.index(0, face_j)];
    };

    EXPECT_EQ(proof_at_low_x(quasar::mhd::BoundaryFlags4{}), 1)
        << "periodic recovery coordinates must wrap";
    EXPECT_EQ(proof_at_low_x(quasar::mhd::BoundaryFlags4{{1, 1, 1, 1}}), 1)
        << "physical recovery coordinates must use the one-sided closure";
    EXPECT_EQ(proof_at_low_x(quasar::mhd::BoundaryFlags4{{4, 4, 4, 4}}), 0)
        << "internal recovery coordinates must retain exchanged guards";
  }
}
