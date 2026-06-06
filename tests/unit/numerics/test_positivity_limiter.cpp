// RED-phase tests for the positivity (troubled-cell) limiter.
//
// Targets the blind contract in include/quasar/numerics/positivity_limiter.hpp:
//
//   class IPositivityLimiter {
//    public: virtual ~IPositivityLimiter() = default;
//     virtual void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor,
//                        Real p_floor, Real gamma) const = 0;
//   };
//
// Registry name "troubled_cell", obtained via
//   quasar::Registry<quasar::numerics::IPositivityLimiter>::instance().create("troubled_cell").
//
// ASSUMED ACCESSORS (documented so the blind implementer matches them):
//   * quasar::mhd::MhdField2D<Real>(Grid2D) ctor; cell-centered conserved
//     DeviceBuffers .rho/.mx/.my/.mz/.energy and cell-centered .bz_cell, plus
//     face-staggered .bx_face/.by_face, each DeviceBuffer<Real> of length
//     grid.storage_size(). We seed and read them back via copy_from_host /
//     copy_to_host as the PIC YeeField2D tests do.
//   * The cell pressure follows the total-energy gamma law used by
//     quasar/numerics/mhd_state.hpp::pressure(MhdState, gamma):
//       p = (gamma-1) * (E - 0.5*|m|^2/rho - 0.5*|B|^2).
//     For the limiter's positivity check the per-cell B is taken to be the
//     cell-centered triple (bx_face(i,j), by_face(i,j), bz_cell(i,j)); this test
//     keeps B = 0 so the check reduces to the hydrodynamic pressure and the
//     limiter target is unambiguous.
//
// Device-touching assertions are guarded with has_hip_runtime() / GTEST_SKIP.
// The registry-presence probe runs unconditionally and fails by missing symbol.

#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/backend/device.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::IPositivityLimiter;
using quasar::numerics::MhdState;

std::unique_ptr<IPositivityLimiter> make_limiter() {
  return quasar::Registry<IPositivityLimiter>::instance().create("troubled_cell");
}

// Host mirror of a field's conserved cell components, for building MhdState per
// cell and checking positivity after the limiter runs.
struct HostState {
  std::vector<Real> rho, mx, my, mz, en, bx, by, bz;
};

HostState read_host(const quasar::mhd::MhdField2D<Real>& u, std::size_t n) {
  HostState h;
  h.rho.resize(n);
  h.mx.resize(n);
  h.my.resize(n);
  h.mz.resize(n);
  h.en.resize(n);
  h.bx.resize(n);
  h.by.resize(n);
  h.bz.resize(n);
  u.rho.copy_to_host(h.rho.data(), n);
  u.mx.copy_to_host(h.mx.data(), n);
  u.my.copy_to_host(h.my.data(), n);
  u.mz.copy_to_host(h.mz.data(), n);
  u.energy.copy_to_host(h.en.data(), n);
  u.bx_face.copy_to_host(h.bx.data(), n);
  u.by_face.copy_to_host(h.by.data(), n);
  u.bz_cell.copy_to_host(h.bz.data(), n);
  return h;
}

MhdState state_at(const HostState& h, std::size_t k) {
  MhdState s{};
  s.rho = h.rho[k];
  s.mx = h.mx[k];
  s.my = h.my[k];
  s.mz = h.mz[k];
  s.energy = h.en[k];
  s.bx = h.bx[k];
  s.by = h.by[k];
  s.bz = h.bz[k];
  return s;
}

}  // namespace

TEST(MhdPositivityLimiter, IsRegistered) {
  EXPECT_TRUE(
      quasar::Registry<IPositivityLimiter>::instance().contains("troubled_cell"));
}

// A cell whose energy is set so the (B=0) pressure is negative is restored to a
// physical state: density >= rho_floor and pressure >= p_floor everywhere.
TEST(MhdPositivityLimiter, RestoresNegativePressureCell) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Real rho_floor = Real{1e-8}, p_floor = Real{1e-9};
  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const std::size_t n = g.storage_size();

  quasar::mhd::MhdField2D<Real> u{g};
  std::vector<Real> rho(n, Real{1}), zero(n, Real{0}), en(n, Real{0});
  // Healthy positive-pressure background: at rest, p = (gamma-1)*E.
  const Real p_bg = Real{1.0};
  for (std::size_t k = 0; k < n; ++k) en[k] = p_bg / (gamma - Real{1});

  // Forced bad cell: a moving cell with energy below its kinetic energy so the
  // thermal pressure p = (gamma-1)*(E - 0.5*|m|^2/rho) is strictly negative.
  const int bi = 4, bj = 4;
  const std::size_t bk = g.index(bi, bj);
  const Real rho_b = Real{1.0}, vx_b = Real{2.0};
  std::vector<Real> mx(n, Real{0});
  rho[bk] = rho_b;
  mx[bk] = rho_b * vx_b;
  en[bk] = Real{0.5} * rho_b * vx_b * vx_b - Real{0.5};  // kinetic minus a bit => p<0

  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(mx.data(), mx.size());
  u.my.copy_from_host(zero.data(), zero.size());
  u.mz.copy_from_host(zero.data(), zero.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bx_face.copy_from_host(zero.data(), zero.size());
  u.by_face.copy_from_host(zero.data(), zero.size());
  u.bz_cell.copy_from_host(zero.data(), zero.size());

  // Sanity: the seeded bad cell really has negative pressure before limiting.
  {
    const HostState before = read_host(u, n);
    EXPECT_LT(quasar::numerics::pressure(state_at(before, bk), gamma), Real{0})
        << "test setup failed to force a negative-pressure cell";
  }

  make_limiter()->apply(u, rho_floor, p_floor, gamma);

  const HostState after = read_host(u, n);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      const MhdState s = state_at(after, k);
      EXPECT_GE(s.rho, rho_floor) << "density below floor at (" << i << "," << j << ")";
      EXPECT_GE(quasar::numerics::pressure(s, gamma), p_floor)
          << "pressure below floor at (" << i << "," << j << ")";
    }
  }
}

// An already-positive field is left unchanged to round-off: the limiter only
// touches troubled cells.
TEST(MhdPositivityLimiter, LeavesPositiveFieldUnchanged) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Real rho_floor = Real{1e-8}, p_floor = Real{1e-9};
  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const std::size_t n = g.storage_size();

  quasar::mhd::MhdField2D<Real> u{g};
  std::vector<Real> rho(n, Real{0}), mx(n, Real{0}), zero(n, Real{0}), en(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      const Real x = g.x_at_cell_center(i);
      const Real r = Real{2} + Real{0.3} * std::sin(Real{2} * M_PI * x);
      const Real p = Real{3} + Real{0.2} * std::cos(Real{2} * M_PI * x);
      const Real vx = Real{0.1};
      rho[k] = r;
      mx[k] = r * vx;
      en[k] = p / (gamma - Real{1}) + Real{0.5} * r * vx * vx;
    }
  }
  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(mx.data(), mx.size());
  u.my.copy_from_host(zero.data(), zero.size());
  u.mz.copy_from_host(zero.data(), zero.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bx_face.copy_from_host(zero.data(), zero.size());
  u.by_face.copy_from_host(zero.data(), zero.size());
  u.bz_cell.copy_from_host(zero.data(), zero.size());

  const HostState before = read_host(u, n);
  make_limiter()->apply(u, rho_floor, p_floor, gamma);
  const HostState after = read_host(u, n);

  const Real tol = Real{1e-12};
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      EXPECT_NEAR(after.rho[k], before.rho[k], tol);
      EXPECT_NEAR(after.mx[k], before.mx[k], tol);
      EXPECT_NEAR(after.en[k], before.en[k], tol);
    }
  }
}
