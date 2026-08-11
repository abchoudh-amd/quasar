// Regression tests for the positivity (troubled-cell) limiter.
//
// "troubled_cell" launches the explicit device-floor utility from apply() and
// evaluates conservative retry bounds through admissible_fraction(). These
// tests pin both contracts, including geometry-matched magnetic collocation.
//
// Targets the contract in include/quasar/numerics/positivity_limiter.hpp:
//
//   class IPositivityLimiter {
//    public: virtual ~IPositivityLimiter() = default;
//     virtual void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor,
//                        Real p_floor, Real gamma,
//                        int collocation_order = 0,
//                        RadialTablesView radial_tables = {}) const = 0;
//   };
//
// Registry name "troubled_cell", obtained via
//   quasar::Registry<quasar::numerics::IPositivityLimiter>::instance().create("troubled_cell").
//
// FLOOR SEMANTICS PINNED:
//   * gas pressure follows the total-energy gamma law in
//       quasar/numerics/mhd_state.hpp::pressure(MhdState, gamma):
//         p = (gamma-1)*(E - 0.5*|m|^2/rho - 0.5*|B|^2).
//     For the limiter's positivity check the transverse magnetic components
//     are centered from their staggered faces. We keep B = 0 in most cells so
//     the check reduces to the hydrodynamic pressure.
//   * a sub-floor-density cell (rho < rho_floor) has rho raised to EXACTLY
//     rho_floor.
//   * a sub-floor-pressure cell (p < p_floor) has gas pressure raised to EXACTLY
//     p_floor by re-deriving total energy while holding momentum and B FIXED
//     (only rho and energy may change).
//   * an already-physical cell is left UNCHANGED to round-off across ALL eight
//     conserved components.
//
// Device-touching assertions are guarded with has_hip_runtime() / GTEST_SKIP.
// The registry-presence probe runs unconditionally and fails by missing symbol.

#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/backend/device.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

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
// cell and checking positivity / invariance after the limiter runs.
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

MhdState state_at(const HostState& h, const Grid2D& g, int i, int j) {
  const std::size_t k = g.index(i, j);
  MhdState s{};
  s.rho = h.rho[k];
  s.mx = h.mx[k];
  s.my = h.my[k];
  s.mz = h.mz[k];
  s.energy = h.en[k];
  // Mirror the same order-aware finite-volume face-to-cell collocation used by
  // load_cell_state() in the repair kernel. A two-face arithmetic mean is not
  // the solver EOS on MP5/MP7 grids and can misdiagnose the repaired pressure.
  s.bx = quasar::mhd::cell_bx(g, h.bx.data(), i, j);
  s.by = quasar::mhd::cell_by(g, h.by.data(), i, j);
  s.bz = h.bz[k];
  return s;
}

}  // namespace

// Constructing the limiter by registry name "troubled_cell" succeeds (the factory
// is registered); this probe runs even without a HIP runtime.
TEST(MhdPositivityLimiter, IsRegistered) {
  EXPECT_TRUE(
      quasar::Registry<IPositivityLimiter>::instance().contains("troubled_cell"));
  EXPECT_NO_THROW({ auto p = make_limiter(); EXPECT_NE(p, nullptr); });
}

// apply() runs without error on a full MhdField2D constructed via the registry.
TEST(MhdPositivityLimiter, ApplyRunsOnFullField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Real rho_floor = Real{1e-8}, p_floor = Real{1e-9};
  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const std::size_t n = g.storage_size();

  quasar::mhd::MhdField2D<Real> u{g};
  std::vector<Real> rho(n, Real{1}), zero(n, Real{0}), en(n, Real{1});
  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(zero.data(), zero.size());
  u.my.copy_from_host(zero.data(), zero.size());
  u.mz.copy_from_host(zero.data(), zero.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bx_face.copy_from_host(zero.data(), zero.size());
  u.by_face.copy_from_host(zero.data(), zero.size());
  u.bz_cell.copy_from_host(zero.data(), zero.size());

  EXPECT_NO_THROW(make_limiter()->apply(u, rho_floor, p_floor, gamma));
}

// The conservative predicate must use the same scale-safe quadratic forms as
// the EOS. Here Bx*Bx overflows although |Bx|^2/2 and the full total energy are
// representable. Identical positive base/candidate states therefore have an
// admissible fraction of exactly one.
TEST(MhdPositivityLimiter, AdmissibleFractionAcceptsExtremeRepresentableField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  constexpr Real gamma = Real{5} / Real{3};
  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const std::size_t n = g.storage_size();

  const Real bx_value = Real{1.5e154};
  const Real magnetic =
      quasar::numerics::half_squared_norm3(bx_value, Real{0}, Real{0});
  ASSERT_TRUE(std::isfinite(magnetic));
  ASSERT_FALSE(std::isfinite(bx_value * bx_value));
  const Real energy = magnetic + Real{1e300} / (gamma - Real{1});

  quasar::mhd::MhdField2D<Real> base{g};
  quasar::mhd::MhdField2D<Real> candidate{g};
  const std::vector<Real> rho(n, Real{1});
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> bx(n, bx_value);
  const std::vector<Real> en(n, energy);
  auto seed = [&](quasar::mhd::MhdField2D<Real>& u) {
    u.rho.copy_from_host(rho.data(), n);
    u.mx.copy_from_host(zero.data(), n);
    u.my.copy_from_host(zero.data(), n);
    u.mz.copy_from_host(zero.data(), n);
    u.energy.copy_from_host(en.data(), n);
    u.bx_face.copy_from_host(bx.data(), n);
    u.by_face.copy_from_host(zero.data(), n);
    u.bz_cell.copy_from_host(zero.data(), n);
  };
  seed(base);
  seed(candidate);

  const Real theta = make_limiter()->admissible_fraction(
      base, candidate, Real{0}, Real{0}, gamma);
  EXPECT_EQ(theta, Real{1});
}

TEST(MhdPositivityLimiter,
     AdmissibleFractionUsesActiveCylindricalRadialCollocation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  constexpr Real gamma = Real{5} / Real{3};
  const Grid2D grid = Grid2D::from_cell_spacing(
      8, 8, Real{1}, Real{1}, Real{4}, Real{0}, /*halo=*/4);
  const std::size_t storage_size = grid.storage_size();
  constexpr int target_radial_cell = 2;
  constexpr int target_axial_cell = 3;
  const std::size_t target =
      grid.index(target_radial_cell, target_axial_cell);

  std::vector<Real> density(storage_size, Real{1});
  std::vector<Real> zero(storage_size, Real{0});
  std::vector<Real> radial_face_field(storage_size, Real{0});
  std::vector<Real> base_energy(storage_size, Real{1e3});
  std::vector<Real> candidate_energy(storage_size, Real{1e3});
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int face = -grid.nghost; face < grid.nx + grid.nghost; ++face) {
      radial_face_field[grid.index(face, j)] = grid.r_at_edge(face);
    }
  }

  const Real cell_center_radius = grid.r_at_cell_center(target_radial_cell);
  const Real half_width = Real{0.5} * grid.dx();
  const Real cartesian_cell_field = cell_center_radius;
  const Real cylindrical_cell_field =
      cell_center_radius +
      half_width * half_width / (Real{3} * cell_center_radius);
  const Real cartesian_magnetic_energy =
      Real{0.5} * cartesian_cell_field * cartesian_cell_field;
  const Real cylindrical_magnetic_energy =
      Real{0.5} * cylindrical_cell_field * cylindrical_cell_field;
  candidate_energy[target] =
      Real{0.5} * (cartesian_magnetic_energy +
                   cylindrical_magnetic_energy);
  base_energy[target] = cylindrical_magnetic_energy + Real{1};

  ASSERT_NEAR(
      quasar::mhd::cell_bx(grid, radial_face_field.data(),
                           target_radial_cell, target_axial_cell),
      cartesian_cell_field, Real{1e-14});
  ASSERT_LT(cartesian_magnetic_energy, candidate_energy[target]);
  ASSERT_LT(candidate_energy[target], cylindrical_magnetic_energy);

  quasar::mhd::MhdField2D<Real> base{grid};
  quasar::mhd::MhdField2D<Real> candidate{grid};
  auto seed = [&](quasar::mhd::MhdField2D<Real>& field,
                  const std::vector<Real>& energy) {
    field.rho.copy_from_host(density.data(), storage_size);
    field.mx.copy_from_host(zero.data(), storage_size);
    field.my.copy_from_host(zero.data(), storage_size);
    field.mz.copy_from_host(zero.data(), storage_size);
    field.energy.copy_from_host(energy.data(), storage_size);
    field.bx_face.copy_from_host(radial_face_field.data(), storage_size);
    field.by_face.copy_from_host(zero.data(), storage_size);
    field.bz_cell.copy_from_host(zero.data(), storage_size);
  };
  seed(base, base_energy);
  seed(candidate, candidate_energy);

  auto limiter = make_limiter();
  const Real inactive_fraction = limiter->admissible_fraction(
      base, candidate, Real{0}, Real{0}, gamma,
      /*collocation_order=*/7, quasar::numerics::RadialTablesView{});
  const quasar::numerics::RadialTables radial_tables{grid, /*scheme_order=*/7};
  const Real active_fraction = limiter->admissible_fraction(
      base, candidate, Real{0}, Real{0}, gamma,
      /*collocation_order=*/7, radial_tables.view());

  const Real expected_active_fraction =
      (base_energy[target] - cylindrical_magnetic_energy) /
      (base_energy[target] - candidate_energy[target]);
  EXPECT_EQ(inactive_fraction, Real{1});
  EXPECT_GT(active_fraction, Real{0});
  EXPECT_LT(active_fraction, Real{1});
  EXPECT_NEAR(active_fraction, expected_active_fraction, Real{1e-13});
}

TEST(MhdPositivityLimiter, ApplyUsesActiveCylindricalRadialCollocation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real rho_floor = Real{1e-8};
  constexpr Real p_floor = Real{1e-9};
  const Grid2D grid = Grid2D::from_cell_spacing(
      8, 8, Real{1}, Real{1}, Real{4}, Real{0}, /*halo=*/4);
  const std::size_t storage_size = grid.storage_size();
  constexpr int target_radial_cell = 2;
  constexpr int target_axial_cell = 3;
  const std::size_t target =
      grid.index(target_radial_cell, target_axial_cell);

  std::vector<Real> density(storage_size, Real{1});
  std::vector<Real> zero(storage_size, Real{0});
  std::vector<Real> radial_face_field(storage_size, Real{0});
  std::vector<Real> energy(storage_size, Real{1e3});
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int face = -grid.nghost; face < grid.nx + grid.nghost; ++face) {
      radial_face_field[grid.index(face, j)] = grid.r_at_edge(face);
    }
  }

  const quasar::numerics::RadialTables radial_tables{
      grid, /*scheme_order=*/7};
  const Real cartesian_cell_field = quasar::mhd::cell_bx(
      grid, radial_face_field.data(), target_radial_cell, target_axial_cell);
  const Real cylindrical_cell_field = quasar::mhd::cell_bx(
      grid, radial_face_field.data(), target_radial_cell, target_axial_cell,
      radial_tables.host_view());
  const Real cartesian_magnetic_energy = quasar::numerics::half_squared_norm3(
      cartesian_cell_field, Real{0}, Real{0});
  const Real cylindrical_magnetic_energy = quasar::numerics::half_squared_norm3(
      cylindrical_cell_field, Real{0}, Real{0});
  ASSERT_LT(cartesian_magnetic_energy, cylindrical_magnetic_energy);
  energy[target] =
      Real{0.5} * (cartesian_magnetic_energy + cylindrical_magnetic_energy);
  ASSERT_GT((gamma - Real{1}) *
                (energy[target] - cartesian_magnetic_energy),
            p_floor);
  ASSERT_LT((gamma - Real{1}) *
                (energy[target] - cylindrical_magnetic_energy),
            p_floor);

  quasar::mhd::MhdField2D<Real> inactive_field{grid};
  quasar::mhd::MhdField2D<Real> active_field{grid};
  const auto seed = [&](quasar::mhd::MhdField2D<Real>& field) {
    field.rho.copy_from_host(density.data(), storage_size);
    field.mx.copy_from_host(zero.data(), storage_size);
    field.my.copy_from_host(zero.data(), storage_size);
    field.mz.copy_from_host(zero.data(), storage_size);
    field.energy.copy_from_host(energy.data(), storage_size);
    field.bx_face.copy_from_host(radial_face_field.data(), storage_size);
    field.by_face.copy_from_host(zero.data(), storage_size);
    field.bz_cell.copy_from_host(zero.data(), storage_size);
  };
  seed(inactive_field);
  seed(active_field);

  auto limiter = make_limiter();
  limiter->apply(inactive_field, rho_floor, p_floor, gamma,
                 /*collocation_order=*/7,
                 quasar::numerics::RadialTablesView{});
  limiter->apply(active_field, rho_floor, p_floor, gamma,
                 /*collocation_order=*/7, radial_tables.view());

  const HostState inactive = read_host(inactive_field, storage_size);
  const HostState active = read_host(active_field, storage_size);
  const Real expected_active_energy =
      p_floor / (gamma - Real{1}) + cylindrical_magnetic_energy;
  EXPECT_EQ(inactive.en[target], energy[target]);
  EXPECT_NEAR(active.en[target], expected_active_energy,
              Real{1e-13} * expected_active_energy);
  EXPECT_GT(active.en[target], inactive.en[target]);
  EXPECT_EQ(active.bx, radial_face_field);
}

// A cell with sub-floor density has its density raised to EXACTLY rho_floor, and
// a cell with sub-floor (negative) pressure has its gas pressure raised to
// EXACTLY p_floor with MOMENTUM and B held byte-unchanged. Already-physical
// cells are untouched to round-off in every component.
TEST(MhdPositivityLimiter, FloorsDensityAndPressureExactlyHoldingMomentumAndB) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const Real gamma = Real{5} / Real{3};
  const Real rho_floor = Real{1e-8}, p_floor = Real{1e-9};
  Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, 4};
  const std::size_t n = g.storage_size();

  quasar::mhd::MhdField2D<Real> u{g};

  // Physical background: rho = 1, some momentum and B, healthy positive pressure.
  std::vector<Real> rho(n, Real{1});
  std::vector<Real> mx(n, Real{0.3}), my(n, Real{-0.2}), mz(n, Real{0.1});
  std::vector<Real> bx(n, Real{0.4}), by(n, Real{0.5}), bz(n, Real{-0.6});
  std::vector<Real> en(n, Real{0});
  const Real p_bg = Real{2.0};
  for (std::size_t k = 0; k < n; ++k) {
    const Real kinetic = Real{0.5} *
        (mx[k] * mx[k] + my[k] * my[k] + mz[k] * mz[k]) / rho[k];
    const Real magnetic = Real{0.5} *
        (bx[k] * bx[k] + by[k] * by[k] + bz[k] * bz[k]);
    en[k] = p_bg / (gamma - Real{1}) + kinetic + magnetic;
  }

  // Sub-floor-density cell: rho below the floor (B = 0 there so the pressure
  // check is unambiguous; energy left small but positive-pressure).
  const std::size_t kd = g.index(2, 3);
  rho[kd] = rho_floor / Real{4};   // strictly below the floor
  mx[kd] = Real{0}; my[kd] = Real{0}; mz[kd] = Real{0};
  bx[kd] = Real{0}; by[kd] = Real{0}; bz[kd] = Real{0};
  bx[g.index(3, 3)] = Real{0};
  by[g.index(2, 4)] = Real{0};
  en[kd] = Real{1.0};              // p = (gamma-1)*E > 0, only density is bad

  // Sub-floor-pressure cell: at rest with B = 0 and a strictly-negative total
  // energy, so the thermal pressure p = (gamma-1)*E < 0. Choosing zero momentum
  // AND zero B is deliberate: after the floor sets E = p_floor/(gamma-1), the
  // recomputed pressure is (gamma-1)*E = p_floor with NO kinetic/magnetic terms
  // to subtract, so there is no catastrophic cancellation and the exact
  // p_floor target holds to a tight 1e-13-relative tolerance. The byte-unchanged
  // momentum/B checks below still hold (they are zero before and after). The
  // physical background cells carry nonzero momentum/B and exercise the
  // "already-physical cell untouched" criterion.
  const std::size_t kp = g.index(5, 6);
  const Real rho_p = Real{1.0};
  rho[kp] = rho_p;
  mx[kp] = Real{0}; my[kp] = Real{0}; mz[kp] = Real{0};
  // MP7 collocates each in-plane component from eight face averages. Zero the
  // complete two stencils, not merely the adjacent low/high faces, so the
  // cell-centred B used by the EOS is exactly zero and the exact-floor assertion
  // below is not measuring cancellation against a residual magnetic energy.
  for (int q = 2; q <= 9; ++q) bx[g.index(q, 6)] = Real{0};
  for (int q = 3; q <= 10; ++q) by[g.index(5, q)] = Real{0};
  bz[kp] = Real{0};
  en[kp] = Real{-0.5};  // p = (gamma-1)*E < 0, no kinetic/magnetic terms

  u.rho.copy_from_host(rho.data(), rho.size());
  u.mx.copy_from_host(mx.data(), mx.size());
  u.my.copy_from_host(my.data(), my.size());
  u.mz.copy_from_host(mz.data(), mz.size());
  u.energy.copy_from_host(en.data(), en.size());
  u.bx_face.copy_from_host(bx.data(), bx.size());
  u.by_face.copy_from_host(by.data(), by.size());
  u.bz_cell.copy_from_host(bz.data(), bz.size());

  const HostState before = read_host(u, n);

  // Sanity: the seeded cells really are sub-floor before limiting.
  EXPECT_LT(before.rho[kd], rho_floor) << "test setup failed: density not sub-floor";
  EXPECT_LT(quasar::numerics::pressure(state_at(before, g, 5, 6), gamma), p_floor)
      << "test setup failed to force a sub-floor-pressure cell";

  make_limiter()->apply(u, rho_floor, p_floor, gamma);

  const HostState after = read_host(u, n);

  // --- Sub-floor-density cell: rho raised to EXACTLY rho_floor. ---
  EXPECT_NEAR(after.rho[kd], rho_floor, std::abs(rho_floor) * Real{1e-13})
      << "sub-floor density not raised to exactly rho_floor";

  // --- Sub-floor-pressure cell: gas pressure raised to EXACTLY p_floor, with
  //     momentum and B BYTE-UNCHANGED (only rho and energy may change). ---
  const MhdState sp = state_at(after, g, 5, 6);
  EXPECT_NEAR(quasar::numerics::pressure(sp, gamma), p_floor,
              std::abs(p_floor) * Real{1e-13})
      << "sub-floor pressure not raised to exactly p_floor";
  // Momentum and B held FIXED (bit-for-bit).
  EXPECT_EQ(after.mx[kp], before.mx[kp]);
  EXPECT_EQ(after.my[kp], before.my[kp]);
  EXPECT_EQ(after.mz[kp], before.mz[kp]);
  EXPECT_EQ(after.bx[kp], before.bx[kp]);
  EXPECT_EQ(after.by[kp], before.by[kp]);
  EXPECT_EQ(after.bz[kp], before.bz[kp]);

  // --- Every other interior cell is already physical and left UNCHANGED to
  //     round-off across all eight conserved components. ---
  const Real rel = Real{1e-13};
  auto unchanged = [&](Real a, Real b) {
    EXPECT_NEAR(a, b, std::abs(b) * rel + rel);
  };
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      if (k == kd || k == kp) continue;
      unchanged(after.rho[k], before.rho[k]);
      unchanged(after.mx[k], before.mx[k]);
      unchanged(after.my[k], before.my[k]);
      unchanged(after.mz[k], before.mz[k]);
      unchanged(after.en[k], before.en[k]);
      unchanged(after.bx[k], before.bx[k]);
      unchanged(after.by[k], before.by[k]);
      unchanged(after.bz[k], before.bz[k]);
    }
  }
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
    EXPECT_LT(quasar::numerics::pressure(state_at(before, g, bi, bj), gamma),
              Real{0})
        << "test setup failed to force a negative-pressure cell";
  }

  make_limiter()->apply(u, rho_floor, p_floor, gamma);

  const HostState after = read_host(u, n);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      const MhdState s = state_at(after, g, i, j);
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
