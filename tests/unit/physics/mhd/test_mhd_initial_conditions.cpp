// Device construction of the seeded ideal-MHD conserved state.
//
// The six generators' physics is exercised from Python against closed-form
// references (tests/python/test_mhd_io.py), which is where the analytic
// profiles belong. This file covers the two things that live below the deck:
//
//   * the invariant the two-pass assembly exists for -- that the magnetic
//     energy in the seeded state is the one implied by the CELL-collocated
//     field, so recovering the pressure with the solver's own EOS returns the
//     pressure the deck asked for (see the comment on the first case for why
//     that is a no-op for all six built-in generators, and what does exercise
//     it);
//   * the structural refusals, which a deck can only reach by way of the
//     Python validator and which must not depend on it.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/initial_conditions.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::mhd::MhdField2D;
using quasar::mhd::MhdInitialConditionKind;
using quasar::mhd::MhdInitialConditionSpec;

bool device_available() { return quasar::backend::device_count() > 0; }

Grid2D cartesian_grid() {
  // Unequal, non-power-of-two extents so a transposed index cannot survive.
  return Grid2D{14, 10, Real{1}, Real{1}, Real{0}, Real{0}, /*halo=*/3};
}

MhdInitialConditionSpec orszag_tang_spec() {
  MhdInitialConditionSpec spec;
  spec.kind = MhdInitialConditionKind::orszag_tang;
  spec.grid = cartesian_grid();
  spec.gamma = Real{5} / Real{3};
  spec.scheme_order = 5;
  spec.b_uniform[0] = Real{1} / std::sqrt(Real{4} * quasar::pi_v<Real>);
  return spec;
}

std::vector<Real> download(const quasar::backend::DeviceBuffer<Real>& buffer) {
  std::vector<Real> host(buffer.size());
  buffer.copy_to_host(host.data(), host.size());
  quasar::backend::device_synchronize(nullptr);
  return host;
}

TEST(MhdInitialConditions, RegisteredNamesRoundTripThroughTheirEnumerators) {
  const auto names = quasar::mhd::registered_mhd_initial_conditions();
  ASSERT_EQ(names.size(), 6u);
  for (std::size_t k = 0; k < names.size(); ++k) {
    EXPECT_EQ(static_cast<int>(quasar::mhd::initial_condition_kind(names[k])),
              static_cast<int>(k))
        << names[k];
  }
  EXPECT_THROW(quasar::mhd::initial_condition_kind("not_a_generator"),
               std::invalid_argument);
}

// Recover the gas pressure the way the solver's EOS does, from the conserved
// state and the CELL-collocated in-plane field. `face_collocated` selects the
// wrong-but-plausible alternative: reading the raw staggered samples instead.
Real worst_pressure_error(const MhdInitialConditionSpec& spec,
                          const MhdField2D<Real>& state, Real expected_pressure,
                          bool face_collocated) {
  const std::vector<Real> rho = download(state.rho);
  const std::vector<Real> mx = download(state.mx);
  const std::vector<Real> my = download(state.my);
  const std::vector<Real> mz = download(state.mz);
  const std::vector<Real> energy = download(state.energy);
  const std::vector<Real> bx_face = download(state.bx_face);
  const std::vector<Real> by_face = download(state.by_face);
  const std::vector<Real> bz_cell = download(state.bz_cell);

  const Grid2D& g = spec.grid;
  Real worst = Real{0};
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const std::size_t k = g.index(i, j);
      quasar::numerics::MhdState u;
      u.rho = rho[k];
      u.mx = mx[k];
      u.my = my[k];
      u.mz = mz[k];
      u.energy = energy[k];
      u.bz = bz_cell[k];
      u.bx = face_collocated ? bx_face[k]
                             : quasar::mhd::cell_bx(g, bx_face.data(), i, j);
      u.by = face_collocated ? by_face[k]
                             : quasar::mhd::cell_by(g, by_face.data(), i, j);
      const Real p = quasar::numerics::pressure(u, spec.gamma);
      worst = std::fmax(worst,
                        std::fabs(p - expected_pressure) / expected_pressure);
    }
  }
  return worst;
}

// The invariant the second pass exists for: the magnetic energy in the seeded
// state is the one implied by the cell-collocated field, so the solver's EOS
// returns the deck's pressure.
//
// Worth knowing before touching this: for all six built-in generators the
// collocation is a NO-OP to round-off, because each in-plane component is
// constant along its own staggering axis (Bx depends only on y, By only on x),
// which is precisely why their staggered seeds are divergence-free. So this
// case checks that the two passes compose to the identity where they must; the
// case below is the one that shows the pass does work when the field does vary
// along its own axis. Deleting the pass as "dead" would pass this test and fail
// that one.
TEST(MhdInitialConditions, SeededEnergyIsConsistentWithTheCollocatedEos) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  const MhdInitialConditionSpec spec = orszag_tang_spec();
  MhdField2D<Real> state{spec.grid};
  quasar::mhd::build_initial_state(spec, state);
  EXPECT_LE(worst_pressure_error(spec, state, spec.gamma, false), Real{1e-14});
}

// A Brio-Wu state whose NORMAL field jumps across the interface. The deck
// validator refuses this (a discontinuous Bx is not divergence-free), and that
// is the point: it is the only way to give the x-collocation an x-varying Bx
// and so the only way to show the second pass changes the answer. The state is
// still admissible as an assembly -- uniform density and pressure, zero
// velocity -- so nothing else in the two passes is under test here.
TEST(MhdInitialConditions, CollocationChangesTheEnergyWhenTheNormalFieldVaries) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  MhdInitialConditionSpec spec;
  spec.kind = MhdInitialConditionKind::brio_wu;
  spec.grid = cartesian_grid();
  spec.gamma = Real{5} / Real{3};
  spec.scheme_order = 5;
  spec.interface = Real{0.5};
  const Real pressure = Real{1};
  const Real left[8] = {Real{1}, pressure, Real{0},    Real{0},
                        Real{0}, Real{0.75}, Real{0}, Real{0}};
  const Real right[8] = {Real{1}, pressure, Real{0},     Real{0},
                         Real{0}, Real{-0.75}, Real{0}, Real{0}};
  for (int c = 0; c < 8; ++c) {
    spec.left[c] = left[c];
    spec.right[c] = right[c];
  }
  MhdField2D<Real> state{spec.grid};
  quasar::mhd::build_initial_state(spec, state);

  EXPECT_LE(worst_pressure_error(spec, state, pressure, false), Real{1e-14});
  // Reading the raw faces instead is wrong by the collocation overshoot of a
  // step, which is nothing like round-off.
  EXPECT_GT(worst_pressure_error(spec, state, pressure, true), Real{1e-3});
}

TEST(MhdInitialConditions, SeedIsBitwiseReproducible) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  const MhdInitialConditionSpec spec = orszag_tang_spec();
  MhdField2D<Real> first{spec.grid};
  MhdField2D<Real> second{spec.grid};
  quasar::mhd::build_initial_state(spec, first);
  quasar::mhd::build_initial_state(spec, second);
  const std::vector<Real> a = download(first.energy);
  const std::vector<Real> b = download(second.energy);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t k = 0; k < a.size(); ++k) EXPECT_EQ(a[k], b[k]) << "cell " << k;
}

TEST(MhdInitialConditions, RejectsAnInadmissibleSeedWithoutPublishingIt) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  MhdInitialConditionSpec spec;
  spec.kind = MhdInitialConditionKind::brio_wu;
  spec.grid = cartesian_grid();
  spec.gamma = Real{5} / Real{3};
  spec.scheme_order = 5;
  spec.interface = Real{0.5};
  const Real left[8] = {Real{1}, Real{1}, Real{0}, Real{0},
                        Real{0}, Real{0}, Real{0}, Real{0}};
  const Real right[8] = {Real{1}, Real{-1}, Real{0}, Real{0},
                         Real{0}, Real{0},  Real{0}, Real{0}};
  for (int c = 0; c < 8; ++c) {
    spec.left[c] = left[c];
    spec.right[c] = right[c];
  }
  MhdField2D<Real> state{spec.grid};
  // Caught in the host precondition, before a kernel runs: a negative deck
  // pressure names the offending state, where a per-cell status bit could only
  // say "somewhere".
  EXPECT_THROW(quasar::mhd::build_initial_state(spec, state),
               std::invalid_argument);
}

TEST(MhdInitialConditions, RejectsStructurallyInvalidGeneratorParameters) {
  MhdInitialConditionSpec spec;
  spec.grid = cartesian_grid();
  spec.gamma = Real{5} / Real{3};
  spec.scheme_order = 5;
  MhdField2D<Real> state{spec.grid};

  // A collapsed rotor taper divides by zero in every cell of the transition.
  spec.kind = MhdInitialConditionKind::rotor;
  spec.r0 = Real{0.1};
  spec.r1 = Real{0.1};
  spec.rho_in = Real{10};
  spec.rho_out = Real{1};
  spec.u0 = Real{2};
  spec.p_ambient = Real{1};
  EXPECT_THROW(quasar::mhd::build_initial_state(spec, state),
               std::invalid_argument);

  // copysign(x, 0.0) is +x, so a zero reference field would silently pick the
  // +x-propagating mode instead of reporting that the mode is undefined.
  spec = MhdInitialConditionSpec{};
  spec.kind = MhdInitialConditionKind::alfven_wave;
  spec.grid = cartesian_grid();
  spec.gamma = Real{5} / Real{3};
  spec.scheme_order = 5;
  spec.total_b0 = Real{0};
  EXPECT_THROW(quasar::mhd::build_initial_state(spec, state),
               std::invalid_argument);
}

TEST(MhdInitialConditions, RejectsAnAnnularHaloThatReachesTheAxis) {
  MhdInitialConditionSpec spec = orszag_tang_spec();
  spec.cylindrical = 1;
  // origin_x - nghost*dr <= 0: the radial collocation moments are undefined on
  // a cell straddling the axis, and an axis-starting grid gets the parity
  // closure instead, which is a property of the grid rather than of this seed.
  spec.grid = Grid2D{14, 10, Real{1}, Real{1}, Real{0.05}, Real{0}, /*halo=*/3};
  MhdField2D<Real> state{spec.grid};
  EXPECT_THROW(quasar::mhd::build_initial_state(spec, state),
               std::invalid_argument);
}

}  // namespace
