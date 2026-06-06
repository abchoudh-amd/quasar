// RED-phase tests for the ideal-MHD conserved/primitive state utilities.
//
// Targets the blind contract in include/quasar/numerics/mhd_state.hpp:
//   struct MhdState { Real rho, mx, my, mz, energy, bx, by, bz; };   // conserved
//   struct MhdPrim  { Real rho, vx, vy, vz, p, bx, by, bz; };
//   MhdPrim  to_primitive(const MhdState&, Real gamma);
//   MhdState to_conserved(const MhdPrim&,  Real gamma);
//   Real     pressure(const MhdState&, Real gamma);
//   Real     fast_magnetosonic_speed(const MhdState&, int dir, Real gamma);
//
// Energy is total: E = rho*e_internal + 0.5*rho*|v|^2 + 0.5*|B|^2,
// with p = (gamma-1)*rho*e_internal.

#include "quasar/numerics/mhd_state.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using quasar::Real;
using quasar::numerics::MhdPrim;
using quasar::numerics::MhdState;

// Round-off tolerance for conserved<->primitive round trips.
constexpr Real kRoundTripTol = 1e-12;
// Tolerance for closed-form speed identities.
constexpr Real kSpeedTol = 1e-10;

// Build a conserved state from primitive quantities using the total-energy
// definition pinned by the contract (local helper, not a library call).
MhdState conserved_from_prim(Real rho, Real vx, Real vy, Real vz, Real p, Real bx,
                             Real by, Real bz, Real gamma) {
  const Real kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
  const Real magnetic = 0.5 * (bx * bx + by * by + bz * bz);
  const Real e_int = p / ((gamma - 1.0) * rho);  // internal energy per unit mass
  MhdState s{};
  s.rho = rho;
  s.mx = rho * vx;
  s.my = rho * vy;
  s.mz = rho * vz;
  s.energy = rho * e_int + kinetic + magnetic;
  s.bx = bx;
  s.by = by;
  s.bz = bz;
  return s;
}

void expect_state_near(const MhdState& a, const MhdState& b, Real tol) {
  EXPECT_NEAR(a.rho, b.rho, tol);
  EXPECT_NEAR(a.mx, b.mx, tol);
  EXPECT_NEAR(a.my, b.my, tol);
  EXPECT_NEAR(a.mz, b.mz, tol);
  EXPECT_NEAR(a.energy, b.energy, tol);
  EXPECT_NEAR(a.bx, b.bx, tol);
  EXPECT_NEAR(a.by, b.by, tol);
  EXPECT_NEAR(a.bz, b.bz, tol);
}

void expect_prim_near(const MhdPrim& a, const MhdPrim& b, Real tol) {
  EXPECT_NEAR(a.rho, b.rho, tol);
  EXPECT_NEAR(a.vx, b.vx, tol);
  EXPECT_NEAR(a.vy, b.vy, tol);
  EXPECT_NEAR(a.vz, b.vz, tol);
  EXPECT_NEAR(a.p, b.p, tol);
  EXPECT_NEAR(a.bx, b.bx, tol);
  EXPECT_NEAR(a.by, b.by, tol);
  EXPECT_NEAR(a.bz, b.bz, tol);
}

}  // namespace

// conserved -> primitive -> conserved recovers the original for a fully
// magnetized state with all three velocity components.
TEST(MhdState, RoundTripConservedMagnetizedWithVelocity) {
  const Real gamma = 5.0 / 3.0;
  const MhdState s =
      conserved_from_prim(1.3, 0.4, -0.2, 0.7, 2.5, 0.6, -0.3, 0.9, gamma);

  const MhdPrim w = quasar::numerics::to_primitive(s, gamma);
  const MhdState s2 = quasar::numerics::to_conserved(w, gamma);

  expect_state_near(s, s2, kRoundTripTol);
}

// Unmagnetized hydro state still round-trips exactly.
TEST(MhdState, RoundTripConservedNoField) {
  const Real gamma = 1.4;
  const MhdState s =
      conserved_from_prim(0.8, -0.5, 0.25, 0.0, 1.1, 0.0, 0.0, 0.0, gamma);

  const MhdPrim w = quasar::numerics::to_primitive(s, gamma);
  const MhdState s2 = quasar::numerics::to_conserved(w, gamma);

  expect_state_near(s, s2, kRoundTripTol);
  EXPECT_NEAR(w.bx, 0.0, kRoundTripTol);
  EXPECT_NEAR(w.by, 0.0, kRoundTripTol);
  EXPECT_NEAR(w.bz, 0.0, kRoundTripTol);
}

// primitive -> conserved -> primitive recovers the original.
TEST(MhdState, RoundTripPrimitive) {
  const Real gamma = 2.0;
  MhdPrim w{};
  w.rho = 2.1;
  w.vx = 0.3;
  w.vy = 0.9;
  w.vz = -0.4;
  w.p = 3.3;
  w.bx = 0.75;
  w.by = 1.0;
  w.bz = -0.5;

  const MhdState s = quasar::numerics::to_conserved(w, gamma);
  const MhdPrim w2 = quasar::numerics::to_primitive(s, gamma);

  expect_prim_near(w, w2, kRoundTripTol);
}

// pressure() matches the gamma-law p = (gamma-1)*rho*e_internal for a known state.
TEST(MhdState, PressureMatchesGammaLaw) {
  const Real gamma = 5.0 / 3.0;
  const Real rho = 1.7;
  const Real p_expected = 4.2;
  const MhdState s =
      conserved_from_prim(rho, 0.6, -0.1, 0.2, p_expected, 0.5, 0.4, -0.3, gamma);

  EXPECT_NEAR(quasar::numerics::pressure(s, gamma), p_expected, kRoundTripTol);
}

// to_primitive recovers the pressure consistently with pressure().
TEST(MhdState, PrimitivePressureConsistentWithPressureHelper) {
  const Real gamma = 1.4;
  const MhdState s =
      conserved_from_prim(0.9, 0.2, 0.5, -0.3, 1.8, 0.3, -0.6, 0.1, gamma);

  const MhdPrim w = quasar::numerics::to_primitive(s, gamma);
  EXPECT_NEAR(w.p, quasar::numerics::pressure(s, gamma), kRoundTripTol);
}

// With B == 0 the fast magnetosonic speed reduces to the hydrodynamic sound
// speed c_s = sqrt(gamma*p/rho), in both directions.
TEST(MhdState, FastSpeedReducesToSoundSpeedWhenUnmagnetized) {
  const Real gamma = 5.0 / 3.0;
  const Real rho = 1.25;
  const Real p = 2.0;
  const MhdState s =
      conserved_from_prim(rho, 0.3, -0.2, 0.4, p, 0.0, 0.0, 0.0, gamma);

  const Real cs = std::sqrt(gamma * p / rho);
  EXPECT_NEAR(quasar::numerics::fast_magnetosonic_speed(s, 0, gamma), cs, kSpeedTol);
  EXPECT_NEAR(quasar::numerics::fast_magnetosonic_speed(s, 1, gamma), cs, kSpeedTol);
}

// For a magnetized state the fast magnetosonic speed bounds both the sound
// speed and the directional Alfven speed from above, in each direction.
TEST(MhdState, FastSpeedBoundsSoundAndAlfvenForMagnetizedState) {
  const Real gamma = 5.0 / 3.0;
  const Real rho = 1.1;
  const Real p = 1.6;
  const Real bx = 0.7;
  const Real by = 1.2;
  const Real bz = -0.5;
  const MhdState s = conserved_from_prim(rho, 0.0, 0.0, 0.0, p, bx, by, bz, gamma);

  const Real cs = std::sqrt(gamma * p / rho);

  // dir 0 (x): the Alfven speed along the wave-normal uses Bx.
  const Real ca_x = std::abs(bx) / std::sqrt(rho);
  const Real cf_x = quasar::numerics::fast_magnetosonic_speed(s, 0, gamma);
  EXPECT_GE(cf_x, cs - kSpeedTol);
  EXPECT_GE(cf_x, ca_x - kSpeedTol);

  // dir 1 (y): the Alfven speed along the wave-normal uses By.
  const Real ca_y = std::abs(by) / std::sqrt(rho);
  const Real cf_y = quasar::numerics::fast_magnetosonic_speed(s, 1, gamma);
  EXPECT_GE(cf_y, cs - kSpeedTol);
  EXPECT_GE(cf_y, ca_y - kSpeedTol);
}

// The fast magnetosonic speed is strictly positive for a physical state.
TEST(MhdState, FastSpeedPositive) {
  const Real gamma = 2.0;
  const MhdState s =
      conserved_from_prim(0.5, 0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, gamma);
  EXPECT_GT(quasar::numerics::fast_magnetosonic_speed(s, 0, gamma), 0.0);
  EXPECT_GT(quasar::numerics::fast_magnetosonic_speed(s, 1, gamma), 0.0);
}
