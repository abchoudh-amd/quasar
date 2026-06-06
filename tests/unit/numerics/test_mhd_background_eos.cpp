// RED-phase tests for the split-aware ideal-MHD EOS helpers (static background
// field B0 vs evolved perturbation field b).
//
// Targets the blind contract additions in include/quasar/numerics/mhd_state.hpp:
//   struct MhdBackground { Real b0x{}, b0y{}, b0z{}; };                       // NEW
//   Real pressure(const MhdState&, const MhdBackground&, Real gamma);         // NEW
//   Real fast_magnetosonic_speed(const MhdState&, const MhdBackground&,
//                                int dir, Real gamma);                        // NEW
//
// Split EOS contract (single source of truth):
//   The conserved state stores PERTURBATION-ONLY magnetic energy:
//     E = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|b|^2,
//   where b = (bx,by,bz) is the EVOLVED (perturbation) field and B0 is a
//   separate static background.
//   - GAS pressure = (gamma-1)*(E - 0.5*rho|v|^2 - 0.5|b|^2) is INDEPENDENT of
//     B0 (the stored energy already excludes B0).
//   - The fast magnetosonic speed uses the TOTAL field B0+b:
//       ca^2 = |B0+b|^2/rho, normal Alfven term uses (b_dir + B0_dir),
//       cs^2 = gamma*p_gas/rho with p_gas the (B0-independent) gas pressure.

#include "quasar/numerics/mhd_state.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;

// Round-off tolerance for exact-equality identities.
constexpr Real kRoundTripTol = 1e-12;
// Tolerance for closed-form speed identities.
constexpr Real kSpeedTol = 1e-10;

// Build a conserved state from primitive quantities using the PERTURBATION-ONLY
// energy convention pinned by the split contract (local helper, not a library
// call). b = (bx,by,bz) is the stored perturbation field; B0 is NOT included
// in the stored energy.
MhdState conserved_from_prim(Real rho, Real vx, Real vy, Real vz, Real p, Real bx,
                             Real by, Real bz, Real gamma) {
  const Real kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
  const Real magnetic = 0.5 * (bx * bx + by * by + bz * bz);
  const Real internal = p / (gamma - 1.0);
  MhdState s{};
  s.rho = rho;
  s.mx = rho * vx;
  s.my = rho * vy;
  s.mz = rho * vz;
  s.energy = internal + kinetic + magnetic;
  s.bx = bx;
  s.by = by;
  s.bz = bz;
  return s;
}

}  // namespace

// pressure(u, {0,0,0}, gamma) must equal the existing zero-B0 overload
// pressure(u, gamma) EXACTLY (bitwise/round-off), for several states.
TEST(MhdBackgroundEos, PressureZeroBackgroundMatchesScalarOverload) {
  const Real gamma = 5.0 / 3.0;
  const MhdBackground zero{0.0, 0.0, 0.0};

  const MhdState a =
      conserved_from_prim(1.3, 0.4, -0.2, 0.7, 2.5, 0.6, -0.3, 0.9, gamma);
  const MhdState b =
      conserved_from_prim(0.8, -0.5, 0.25, 0.0, 1.1, 0.0, 0.0, 0.0, gamma);
  const MhdState c =
      conserved_from_prim(2.1, 0.3, 0.9, -0.4, 3.3, 0.75, 1.0, -0.5, gamma);

  EXPECT_NEAR(quasar::numerics::pressure(a, zero, gamma),
              quasar::numerics::pressure(a, gamma), kRoundTripTol);
  EXPECT_NEAR(quasar::numerics::pressure(b, zero, gamma),
              quasar::numerics::pressure(b, gamma), kRoundTripTol);
  EXPECT_NEAR(quasar::numerics::pressure(c, zero, gamma),
              quasar::numerics::pressure(c, gamma), kRoundTripTol);
}

// fast_magnetosonic_speed(u, {0,0,0}, dir, gamma) must equal the existing
// zero-B0 overload fast_magnetosonic_speed(u, dir, gamma) EXACTLY, both dirs.
TEST(MhdBackgroundEos, FastSpeedZeroBackgroundMatchesScalarOverload) {
  const Real gamma = 5.0 / 3.0;
  const MhdBackground zero{0.0, 0.0, 0.0};

  const MhdState s =
      conserved_from_prim(1.1, 0.0, 0.0, 0.0, 1.6, 0.7, 1.2, -0.5, gamma);

  EXPECT_NEAR(quasar::numerics::fast_magnetosonic_speed(s, zero, 0, gamma),
              quasar::numerics::fast_magnetosonic_speed(s, 0, gamma), kSpeedTol);
  EXPECT_NEAR(quasar::numerics::fast_magnetosonic_speed(s, zero, 1, gamma),
              quasar::numerics::fast_magnetosonic_speed(s, 1, gamma), kSpeedTol);
}

// Gas pressure is INDEPENDENT of B0: for a fixed stored state u, adding any
// nonzero background must NOT change the returned gas pressure.
TEST(MhdBackgroundEos, GasPressureIndependentOfBackground) {
  const Real gamma = 1.4;
  const MhdState u =
      conserved_from_prim(0.9, 0.2, 0.5, -0.3, 1.8, 0.3, -0.6, 0.1, gamma);

  const MhdBackground zero{0.0, 0.0, 0.0};
  const Real p_ref = quasar::numerics::pressure(u, zero, gamma);

  const MhdBackground b0_a{1.5, -2.0, 0.75};
  const MhdBackground b0_b{-3.3, 0.0, 4.1};
  const MhdBackground b0_c{10.0, 10.0, -10.0};

  EXPECT_NEAR(quasar::numerics::pressure(u, b0_a, gamma), p_ref, kRoundTripTol);
  EXPECT_NEAR(quasar::numerics::pressure(u, b0_b, gamma), p_ref, kRoundTripTol);
  EXPECT_NEAR(quasar::numerics::pressure(u, b0_c, gamma), p_ref, kRoundTripTol);
}

// A nonzero background raises the fast magnetosonic speed: with the same stored
// state the total field magnitude grows, so cf must strictly increase in each
// direction.
TEST(MhdBackgroundEos, NonzeroBackgroundRaisesFastSpeed) {
  const Real gamma = 5.0 / 3.0;
  const MhdBackground zero{0.0, 0.0, 0.0};
  // Background aligned with the stored field so the total magnitude grows.
  const MhdBackground b0{0.5, 0.8, 0.3};

  const MhdState s =
      conserved_from_prim(1.25, 0.0, 0.0, 0.0, 2.0, 0.4, 0.6, 0.2, gamma);

  const Real cf0_x = quasar::numerics::fast_magnetosonic_speed(s, zero, 0, gamma);
  const Real cf0_y = quasar::numerics::fast_magnetosonic_speed(s, zero, 1, gamma);
  const Real cfb_x = quasar::numerics::fast_magnetosonic_speed(s, b0, 0, gamma);
  const Real cfb_y = quasar::numerics::fast_magnetosonic_speed(s, b0, 1, gamma);

  EXPECT_GT(cfb_x, cf0_x);
  EXPECT_GT(cfb_y, cf0_y);
}

// Field-split consistency: an unsplit state carrying the full field B with
// zero background gives the same fast speed as the equivalent split where the
// stored field is b = B - B0 and the background is B0 (same total field). Both
// directions, to round-off.
TEST(MhdBackgroundEos, SplitConsistencyMatchesUnsplit) {
  const Real gamma = 5.0 / 3.0;
  const Real rho = 1.1;
  const Real p = 1.6;
  const Real vx = 0.3, vy = -0.2, vz = 0.4;

  // Total in-plane field B (plus out-of-plane bz) and a uniform background B0.
  const Real Bx = 1.4, By = -0.9, Bz = 0.6;
  const MhdBackground b0{0.5, -0.4, 0.2};

  // Unsplit: full field stored, zero background.
  const MhdState u_total =
      conserved_from_prim(rho, vx, vy, vz, p, Bx, By, Bz, gamma);
  const MhdBackground zero{0.0, 0.0, 0.0};

  // Split: stored perturbation b = B - B0, background = B0. NOTE the stored
  // energy differs (perturbation-only), so the state must be rebuilt with the
  // reduced stored field for the energy bookkeeping to stay consistent.
  const MhdState u_split = conserved_from_prim(
      rho, vx, vy, vz, p, Bx - b0.b0x, By - b0.b0y, Bz - b0.b0z, gamma);

  for (int dir = 0; dir < 2; ++dir) {
    const Real cf_unsplit =
        quasar::numerics::fast_magnetosonic_speed(u_total, zero, dir, gamma);
    const Real cf_split =
        quasar::numerics::fast_magnetosonic_speed(u_split, b0, dir, gamma);
    EXPECT_NEAR(cf_split, cf_unsplit, kSpeedTol) << "dir=" << dir;
  }
}
