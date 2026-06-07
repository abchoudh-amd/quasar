// RED-phase tests for the post-port characteristic-projection surface.
//
// Targets the committed contract in
// include/quasar/numerics/characteristic_projection.hpp AFTER the device port:
//   struct CharVec7 { Real w[7]; };                       // POD, exactly 7 DOF
//   static CharVec7 to_char(const MhdState& delta, const MhdEigensystem& eig);
//   static MhdState from_char(const CharVec7& w, const MhdEigensystem& eig);
// (Both to_char/from_char become QUASAR_HOST_DEVICE; device-callability is
// exercised indirectly by the reconstruction path, not asserted here.)
//
// This file WILL FAIL TO COMPILE until CharVec7 exists -- that is the intended
// RED state (failing for the right reason: the CharVec7 symbol is not yet
// defined). The std::array<Real,7> surface it replaces is deliberately NOT used.
//
// OBSERVABLE invariants pinned here:
//   (1) Round-trip identity: from_char(to_char(delta)) == delta to round-off
//       for a conserved delta whose NORMAL-B perturbation is zero, for BOTH
//       dir=0 (bx=0) and dir=1 (by=0), about a representative reference
//       eigensystem. The projection excludes the normal-B DOF, so the round
//       trip is exact only when that component is zero.
//   (2) CharVec7 carries exactly 7 entries (w[0..6]) and the round-trip is
//       exercised through it.

#include "quasar/numerics/characteristic_projection.hpp"
#include "quasar/numerics/mhd_eigensystem.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using quasar::Real;
using quasar::numerics::CharacteristicProjector;
using quasar::numerics::CharVec7;
using quasar::numerics::MhdEigensystem;
using quasar::numerics::MhdState;

// Round-off tolerance for the eigenvector round-trip (L*R = I to machine
// precision, so to_char/from_char compose to the identity on the 7-wave
// subspace up to a handful of ulp).
constexpr Real kRoundTripTol = 1e-12;

// A representative, well-separated magnetized state used to build the reference
// eigensystem the projection is taken about (no degeneracies).
MhdState make_reference_state(Real gamma) {
  const Real rho = 1.1;
  const Real vx = 0.2, vy = -0.15, vz = 0.3;
  const Real p = 1.4;
  const Real bx = 0.55, by = 0.8, bz = -0.4;
  const Real e_int = p / ((gamma - 1.0) * rho);
  MhdState s{};
  s.rho = rho;
  s.mx = rho * vx;
  s.my = rho * vy;
  s.mz = rho * vz;
  s.energy = rho * e_int + 0.5 * rho * (vx * vx + vy * vy + vz * vz) +
             0.5 * (bx * bx + by * by + bz * bz);
  s.bx = bx;
  s.by = by;
  s.bz = bz;
  return s;
}

}  // namespace

// CharVec7 is a POD with exactly 7 scalar entries addressable as w[0..6].
TEST(CharVec7Type, HasExactlySevenEntries) {
  CharVec7 v{};
  for (int k = 0; k < 7; ++k) v.w[k] = static_cast<Real>(k + 1);
  for (int k = 0; k < 7; ++k) {
    EXPECT_EQ(v.w[k], static_cast<Real>(k + 1));
  }
  // The storage is exactly 7 contiguous Reals; no more, no less.
  EXPECT_EQ(sizeof(CharVec7), 7 * sizeof(Real));
}

// (1, dir=0) from_char(to_char(delta)) == delta for a delta with zero normal-B
// (bx = 0), about the x-normal eigensystem. Exact round trip to round-off.
TEST(CharacteristicProjection, RoundTripIdentityDirX) {
  const Real gamma = 5.0 / 3.0;
  const MhdState ref = make_reference_state(gamma);

  MhdEigensystem eig;
  eig.build(ref, /*dir=*/0, gamma);

  // Small conserved delta; NORMAL-B (bx) is zero for dir=0.
  MhdState delta{};
  delta.rho = 1e-3;
  delta.mx = -2e-3;
  delta.my = 3e-3;
  delta.mz = 1.5e-3;
  delta.energy = 4e-3;
  delta.bx = 0.0;  // normal-B excluded from the 7-wave subspace
  delta.by = -1e-3;
  delta.bz = 2e-3;

  const CharVec7 w = CharacteristicProjector::to_char(delta, eig);
  const MhdState back = CharacteristicProjector::from_char(w, eig);

  EXPECT_NEAR(back.rho, delta.rho, kRoundTripTol);
  EXPECT_NEAR(back.mx, delta.mx, kRoundTripTol);
  EXPECT_NEAR(back.my, delta.my, kRoundTripTol);
  EXPECT_NEAR(back.mz, delta.mz, kRoundTripTol);
  EXPECT_NEAR(back.energy, delta.energy, kRoundTripTol);
  EXPECT_NEAR(back.bx, delta.bx, kRoundTripTol);  // normal-B: 0 in, 0 out
  EXPECT_NEAR(back.by, delta.by, kRoundTripTol);
  EXPECT_NEAR(back.bz, delta.bz, kRoundTripTol);
}

// (1, dir=1) Same round-trip identity about the y-normal eigensystem with the
// normal-B (by) perturbation set to zero.
TEST(CharacteristicProjection, RoundTripIdentityDirY) {
  const Real gamma = 5.0 / 3.0;
  const MhdState ref = make_reference_state(gamma);

  MhdEigensystem eig;
  eig.build(ref, /*dir=*/1, gamma);

  // Small conserved delta; NORMAL-B (by) is zero for dir=1.
  MhdState delta{};
  delta.rho = 1e-3;
  delta.mx = 2.5e-3;
  delta.my = -1e-3;
  delta.mz = 3e-3;
  delta.energy = 4e-3;
  delta.bx = -1.5e-3;
  delta.by = 0.0;  // normal-B excluded from the 7-wave subspace
  delta.bz = 2e-3;

  const CharVec7 w = CharacteristicProjector::to_char(delta, eig);
  const MhdState back = CharacteristicProjector::from_char(w, eig);

  EXPECT_NEAR(back.rho, delta.rho, kRoundTripTol);
  EXPECT_NEAR(back.mx, delta.mx, kRoundTripTol);
  EXPECT_NEAR(back.my, delta.my, kRoundTripTol);
  EXPECT_NEAR(back.mz, delta.mz, kRoundTripTol);
  EXPECT_NEAR(back.energy, delta.energy, kRoundTripTol);
  EXPECT_NEAR(back.bx, delta.bx, kRoundTripTol);
  EXPECT_NEAR(back.by, delta.by, kRoundTripTol);  // normal-B: 0 in, 0 out
  EXPECT_NEAR(back.bz, delta.bz, kRoundTripTol);
}
