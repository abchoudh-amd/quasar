// Tests for the ideal-MHD conserved/primitive state utilities.
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
#include "quasar/core/grid.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

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

TEST(MhdState, PressureAvoidsFalseOverflowInQuadraticEnergies) {
  const Real gamma = Real{5} / Real{3};
  MhdState s{};
  s.rho = Real{1e100};
  s.mx = Real{1e200};
  s.my = Real{-2e200};
  s.mz = Real{0.5e200};
  s.bx = Real{1e150};
  s.by = Real{-1e150};
  s.bz = Real{0.5e150};
  const Real kinetic = Real{2.625e300};
  const Real magnetic = Real{1.125e300};
  const Real internal = Real{0.5e300};
  s.energy = kinetic + magnetic + internal;

  const Real p = quasar::numerics::pressure(s, gamma);
  ASSERT_TRUE(std::isfinite(p));
  EXPECT_NEAR(p / ((gamma - Real{1}) * internal), Real{1}, Real{2e-15});
  const MhdPrim w = quasar::numerics::to_primitive(s, gamma);
  EXPECT_TRUE(std::isfinite(w.vx));
  EXPECT_TRUE(std::isfinite(w.vy));
  EXPECT_TRUE(std::isfinite(w.vz));
  EXPECT_TRUE(std::isfinite(w.p));
}

TEST(MhdState, FastSpeedAvoidsDiscriminantOverflow) {
  const Real gamma = Real{5} / Real{3};
  MhdPrim w{};
  w.rho = Real{1};
  w.p = Real{2e299};
  w.bx = Real{8e149};
  w.by = Real{-6e149};
  w.bz = Real{4e149};
  const MhdState s = quasar::numerics::to_conserved(w, gamma);
  ASSERT_TRUE(std::isfinite(s.energy));

  const Real cf = quasar::numerics::fast_magnetosonic_speed(s, 0, gamma);
  const Real sound = std::sqrt(gamma) * std::sqrt(w.p);
  const Real normal_alfven = std::abs(w.bx);
  EXPECT_TRUE(std::isfinite(cf));
  EXPECT_GE(cf, sound * (Real{1} - Real{2e-15}));
  EXPECT_GE(cf, normal_alfven * (Real{1} - Real{2e-15}));
}

TEST(MhdState, FastSpeedRetainsUnderflowedMagneticSquare) {
  // bx^2 underflows to zero in binary64, but bx/sqrt(rho) is representable.
  MhdState s{};
  s.rho = Real{1e-300};
  s.bx = Real{1e-170};
  const Real cf = quasar::numerics::fast_magnetosonic_speed(
      s, /*dir=*/0, Real{5} / Real{3});
  EXPECT_TRUE(std::isfinite(cf));
  EXPECT_NEAR(cf / Real{1e-20}, Real{1}, Real{2e-15});
}

TEST(MhdState, ProductSumPreservesSmallTermAfterOverflowingCancellation) {
  const Real large = Real{1e200};
  ASSERT_FALSE(std::isfinite(large * large));
  const Real result = quasar::numerics::product_sum3(
      large, large, large, -large, Real{3}, Real{7});
  EXPECT_EQ(result, Real{21});
}

TEST(MhdState, ScaledReducersCancelOppositeGiantsBeforeSameSignAddition) {
  const Real a0 = Real{1e200};
  const Real a1 = std::nextafter(a0, std::numeric_limits<Real>::infinity());
  const Real a[5] = {a0, a1, a0, a1, Real{1}};
  const Real sign[5] = {Real{1}, Real{1}, Real{-1}, Real{-1}, Real{1}};
  const Real one[5] = {Real{1}, Real{1}, Real{1}, Real{1}, Real{1}};

  // Adding the two positive O(1e200) terms before seeing their negatives can
  // leave an O(ulp(1e200)) rounding residue and erase the exact unit survivor.
  EXPECT_EQ(quasar::numerics::scaled_product_sum(a, sign, 5), Real{1});
  EXPECT_EQ(quasar::numerics::scaled_quaternary_product_sum(
                a, sign, one, one, 5),
            Real{1});
  EXPECT_EQ(quasar::numerics::scaled_product_quotient_sum(
                a, sign, one, one, 5),
            Real{1});
}

TEST(MhdState, StreamingAccumulatorIsBoundedAndPreservesAllTerms) {
  static_assert(sizeof(quasar::numerics::ScaledQuaternaryAccumulator) <= 320,
                "hot-path scaled accumulator must stay below 320 bytes");
  const Real a0 = Real{1e200};
  const Real a1 = std::nextafter(a0, std::numeric_limits<Real>::infinity());
  quasar::numerics::ScaledQuaternaryAccumulator sum;
  quasar::numerics::append_scaled_quaternary_product(
      sum, a0, Real{1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_quaternary_product(
      sum, a1, Real{1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_quaternary_product(
      sum, a0, Real{-1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_quaternary_product(
      sum, a1, Real{-1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_quaternary_product(
      sum, Real{1}, Real{1}, Real{1}, Real{1});
  for (int k = 5; k < 24; ++k) {
    quasar::numerics::append_scaled_quaternary_product(
        sum, Real{0}, Real{1}, Real{1}, Real{1});
  }
  EXPECT_EQ(sum.count, 24);
  EXPECT_EQ(quasar::numerics::finish_scaled_quaternary_sum(sum), Real{1});

  quasar::numerics::ScaledQuaternaryAccumulator overflow_cancel;
  quasar::numerics::append_scaled_quaternary_product(
      overflow_cancel, a0, a0, Real{1}, Real{1});
  quasar::numerics::append_scaled_quaternary_product(
      overflow_cancel, a0, -a0, Real{1}, Real{1});
  quasar::numerics::append_scaled_quaternary_product(
      overflow_cancel, Real{3}, Real{7}, Real{1}, Real{1});
  EXPECT_EQ(quasar::numerics::finish_scaled_quaternary_sum(overflow_cancel),
            Real{21});
}

TEST(MhdState, StreamingQuotientAccumulatorSupportsFusedStressCapacity) {
  using StressAccumulator =
      quasar::numerics::ScaledProductQuotientAccumulator<20>;
  using EnergyAccumulator =
      quasar::numerics::ScaledProductQuotientAccumulator<27>;
  static_assert(sizeof(StressAccumulator) <= 256,
                "20-term quotient accumulator must stay bounded");
  static_assert(sizeof(EnergyAccumulator) <= 344,
                "27-term quotient accumulator must stay bounded");
  const Real a0 = Real{1e200};
  const Real a1 = std::nextafter(a0, std::numeric_limits<Real>::infinity());
  StressAccumulator sum;
  quasar::numerics::append_scaled_product_quotient(
      sum, a0, Real{1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, a1, Real{1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, a0, Real{-1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, a1, Real{-1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, Real{1}, Real{1}, Real{1}, Real{1});
  for (int k = 5; k < 17; ++k) {
    quasar::numerics::append_scaled_product_quotient(
        sum, Real{0}, Real{1}, Real{1}, Real{1});
  }
  EXPECT_EQ(sum.count, 17);
  EXPECT_EQ(quasar::numerics::finish_scaled_product_quotient_sum(sum),
            Real{1});

  EnergyAccumulator energy_sum;
  for (int k = 0; k < 27; ++k) {
    quasar::numerics::append_scaled_product_quotient(
        energy_sum, Real{0}, Real{1}, Real{1}, Real{1});
  }
  EXPECT_EQ(energy_sum.count, 27);
  EXPECT_EQ(quasar::numerics::finish_scaled_product_quotient_sum(energy_sum),
            Real{0});

  Real a[17]{}, b[17]{}, d0[17]{}, d1[17]{};
  for (int k = 0; k < 17; ++k) {
    d0[k] = Real{1};
    d1[k] = Real{1};
  }
  a[0] = Real{3};
  b[0] = Real{7};
  EXPECT_EQ(quasar::numerics::scaled_product_quotient_sum_extended(
                a, b, d0, d1, 17),
            Real{21});
}

TEST(MhdState, ScaledReducersCombineExplicitInfinityWithFiniteOverflow) {
  const Real inf = std::numeric_limits<Real>::infinity();
  const Real a2[2] = {inf, Real{-1e300}};
  const Real b2[2] = {Real{1}, Real{1e100}};
  const Real one2[2] = {Real{1}, Real{1}};
  EXPECT_TRUE(std::isnan(quasar::numerics::scaled_product_sum(a2, b2, 2)));
  EXPECT_TRUE(std::isnan(quasar::numerics::scaled_quaternary_product_sum(
      a2, b2, one2, one2, 2)));
  EXPECT_TRUE(std::isnan(quasar::numerics::scaled_product_quotient_sum(
      a2, b2, one2, one2, 2)));
}

TEST(MhdState, ProductSumPreservesIeeeNonfinitePropagation) {
  const Real inf = std::numeric_limits<Real>::infinity();
  EXPECT_TRUE(std::isnan(quasar::numerics::product_sum2(
      Real{0}, inf, Real{1}, Real{2})));
  EXPECT_TRUE(std::isnan(quasar::numerics::product_sum2(
      Real{1}, inf, Real{-1}, inf)));
}

TEST(MhdState, ProductQuotientSumAvoidsIntermediateRangeFailures) {
  const Real a[3] = {Real{1e300}, Real{1e300}, Real{3}};
  const Real b[3] = {Real{1e300}, Real{-1e300}, Real{7}};
  const Real d0[3] = {Real{1e200}, Real{1e200}, Real{1}};
  const Real d1[3] = {Real{1e200}, Real{1e200}, Real{1}};
  ASSERT_FALSE(std::isfinite(a[0] * b[0]));
  ASSERT_FALSE(std::isfinite(d0[0] * d1[0]));
  EXPECT_EQ(quasar::numerics::scaled_product_quotient_sum(
                a, b, d0, d1, 3),
            Real{21});

  // (max - (-max))/max is exactly two, although forming the numerator first
  // overflows.  Face-difference stencils use precisely this signed-quotient
  // structure.
  const Real max = std::numeric_limits<Real>::max();
  const Real diff_a[2] = {max, max};
  const Real diff_b[2] = {Real{1}, Real{1}};
  const Real diff_d0[2] = {max, max};
  const Real diff_d1[2] = {Real{1}, Real{1}};
  EXPECT_EQ(quasar::numerics::scaled_product_quotient_sum(
                diff_a, diff_b, diff_d0, diff_d1, 2),
            Real{2});
}

TEST(MhdState, ProductQuotientSumRoundsOnlyAfterSubnormalAccumulation) {
  const Real denorm = std::numeric_limits<Real>::denorm_min();
  const Real a[2] = {denorm, denorm};
  const Real half[2] = {Real{0.5}, Real{0.5}};
  const Real one[2] = {Real{1}, Real{1}};
  ASSERT_EQ(denorm * Real{0.5}, Real{0});
  EXPECT_EQ(quasar::numerics::scaled_product_quotient_sum(
                a, half, one, one, 2),
            denorm);
}

TEST(MhdState, ProductQuotientSumPreservesIeeeInvalidPropagation) {
  const Real inf = std::numeric_limits<Real>::infinity();
  const Real a[2] = {Real{0}, Real{1}};
  const Real b[2] = {inf, Real{2}};
  const Real one[2] = {Real{1}, Real{1}};
  EXPECT_TRUE(std::isnan(quasar::numerics::scaled_product_quotient_sum(
      a, b, one, one, 2)));
}

TEST(MhdState, HalfSquaredNormRetainsRepresentableSubnormalSum) {
  // Each half-square is exactly 0.5*denorm_min and rounds to zero in isolation,
  // but the three-component half norm is 1.5*denorm_min and rounds to the even
  // representable value 2*denorm_min. The normalized sum must be applied before
  // the final product rounds.
  const Real x = std::scalbn(Real{1}, -537);
  const Real denorm = std::numeric_limits<Real>::denorm_min();
  EXPECT_EQ(quasar::numerics::half_squared_norm3(x, x, x), Real{2} * denorm);
}

TEST(MhdStaggering, FaceSamplesToCellAverageIsDegreeSevenExact) {
  const quasar::Grid2D g{12, 10, Real{12}, Real{10}, Real{0}, Real{0}, 4};
  std::vector<Real> bx(g.storage_size()), by(g.storage_size());
  for (int degree = 0; degree <= 7; ++degree) {
    for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
      for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
        bx[g.index(i, j)] = std::pow(static_cast<Real>(i), degree);
        by[g.index(i, j)] = std::pow(static_cast<Real>(j), degree);
      }
    }
    // Include the outermost high ghost cell: it has no stored upper face and
    // therefore exercises the explicit one-sided polynomial-integral closure.
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const Real expected =
          (std::pow(static_cast<Real>(i + 1), degree + 1) -
           std::pow(static_cast<Real>(i), degree + 1)) /
          static_cast<Real>(degree + 1);
      const Real actual = quasar::mhd::cell_bx(g, bx.data(), i, 0);
      EXPECT_NEAR(actual, expected, Real{2e-11} * std::max(Real{1}, std::abs(expected)))
          << "degree=" << degree << " i=" << i;
    }
    for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
      const Real expected =
          (std::pow(static_cast<Real>(j + 1), degree + 1) -
           std::pow(static_cast<Real>(j), degree + 1)) /
          static_cast<Real>(degree + 1);
      const Real actual = quasar::mhd::cell_by(g, by.data(), 0, j);
      EXPECT_NEAR(actual, expected, Real{2e-11} * std::max(Real{1}, std::abs(expected)))
          << "degree=" << degree << " j=" << j;
    }
  }
}

TEST(MhdStaggering, CellAveragesToFaceIsDegreeSevenExact) {
  const quasar::Grid2D g{12, 10, Real{12}, Real{10}, Real{0}, Real{0}, 4};
  for (int degree = 0; degree <= 7; ++degree) {
    std::vector<Real> average(g.nx + 2 * g.nghost);
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      average[static_cast<std::size_t>(i + g.nghost)] =
          (std::pow(static_cast<Real>(i + 1), degree + 1) -
           std::pow(static_cast<Real>(i), degree + 1)) /
          static_cast<Real>(degree + 1);
    }
    for (int face = 0; face <= g.nx; ++face) {
      const Real actual = quasar::mhd::cell_averages_to_face(
          g, 0, face, [&](int i) {
            return average[static_cast<std::size_t>(i + g.nghost)];
          });
      const Real expected = std::pow(static_cast<Real>(face), degree);
      EXPECT_NEAR(actual, expected, Real{2e-12} * std::max(Real{1}, std::abs(expected)))
          << "degree=" << degree << " face=" << face;
    }
  }
}

TEST(MhdStaggering, CenteredStencilsAreRangeSafeAtEveryHighOrderWidth) {
  struct Stencil {
    int nghost;
    int width;
    const int* face_to_cell;
    Real face_to_cell_denominator;
    const int* cell_to_face;
    Real cell_to_face_denominator;
  };
  constexpr int f2c4[4] = {-1, 13, 13, -1};
  constexpr int f2c6[6] = {11, -93, 802, 802, -93, 11};
  constexpr int f2c8[8] = {
      -191, 1879, -9531, 68323, 68323, -9531, 1879, -191};
  constexpr int c2f4[4] = {-1, 7, 7, -1};
  constexpr int c2f6[6] = {1, -8, 37, 37, -8, 1};
  constexpr int c2f8[8] = {-3, 29, -139, 533, 533, -139, 29, -3};
  const Stencil stencils[3] = {
      {2, 4, f2c4, Real{24}, c2f4, Real{12}},
      {3, 6, f2c6, Real{1440}, c2f6, Real{60}},
      {4, 8, f2c8, Real{120960}, c2f8, Real{840}}};

  const Real near_max = std::numeric_limits<Real>::max() / Real{4};
  const Real denorm = std::numeric_limits<Real>::denorm_min();
  constexpr Real survivor = Real{21};

  for (const Stencil& stencil : stencils) {
    const quasar::Grid2D g{12, 4, Real{12}, Real{4}, Real{0}, Real{0},
                           stencil.nghost};
    const int cell = 4;
    const int face = 5;
    const int face_start = cell - (stencil.width / 2 - 1);
    const int average_start = face - stencil.width / 2;
    std::vector<Real> samples(g.storage_size(), near_max);
    std::vector<Real> averages(
        static_cast<std::size_t>(g.nx + 2 * g.nghost), near_max);

    const Real f2c_constant = quasar::mhd::cell_bx(
        g, samples.data(), cell, /*j=*/0);
    const Real c2f_constant = quasar::mhd::cell_averages_to_face(
        g, /*axis=*/0, face, [&](int i) {
          return averages[static_cast<std::size_t>(i + g.nghost)];
        });
    const Real constant_tol =
        Real{64} * std::numeric_limits<Real>::epsilon() * near_max;
    EXPECT_NEAR(f2c_constant, near_max, constant_tol)
        << "face-to-cell width=" << stencil.width;
    EXPECT_NEAR(c2f_constant, near_max, constant_tol)
        << "cell-to-face width=" << stencil.width;

    // Equal central coefficients multiply opposite near-maximum samples.  A
    // third, moderate term is the exact survivor.  Pre-forming either central
    // product overflows for every width.
    std::fill(samples.begin(), samples.end(), Real{0});
    samples[g.index(face_start + stencil.width / 2 - 1, 0)] = near_max;
    samples[g.index(face_start + stencil.width / 2, 0)] = -near_max;
    samples[g.index(face_start, 0)] =
        survivor * stencil.face_to_cell_denominator /
        static_cast<Real>(stencil.face_to_cell[0]);
    EXPECT_NEAR(quasar::mhd::cell_bx(g, samples.data(), cell, 0),
                survivor, Real{2e-13})
        << "face-to-cell cancellation width=" << stencil.width;

    std::fill(averages.begin(), averages.end(), Real{0});
    averages[static_cast<std::size_t>(average_start + stencil.width / 2 - 1
                                      + g.nghost)] = near_max;
    averages[static_cast<std::size_t>(average_start + stencil.width / 2
                                      + g.nghost)] = -near_max;
    averages[static_cast<std::size_t>(average_start + g.nghost)] =
        survivor * stencil.cell_to_face_denominator /
        static_cast<Real>(stencil.cell_to_face[0]);
    EXPECT_NEAR(quasar::mhd::cell_averages_to_face(
                    g, 0, face, [&](int i) {
                      return averages[static_cast<std::size_t>(i + g.nghost)];
                    }),
                survivor, Real{2e-13})
        << "cell-to-face cancellation width=" << stencil.width;

    std::fill(samples.begin(), samples.end(), denorm);
    std::fill(averages.begin(), averages.end(), denorm);
    EXPECT_EQ(quasar::mhd::cell_bx(g, samples.data(), cell, 0), denorm)
        << "face-to-cell subnormal width=" << stencil.width;
    EXPECT_EQ(quasar::mhd::cell_averages_to_face(
                  g, 0, face, [&](int i) {
                    return averages[static_cast<std::size_t>(i + g.nghost)];
                  }),
              denorm)
        << "cell-to-face subnormal width=" << stencil.width;
  }
}

TEST(MhdStaggering, OneSidedClosuresAreRangeSafeAtEveryHighOrderWidth) {
  const Real near_max = std::numeric_limits<Real>::max() / Real{4};
  const Real denorm = std::numeric_limits<Real>::denorm_min();
  for (const int nghost : {2, 3, 4}) {
    const int width = 2 * nghost;
    const quasar::Grid2D g{12, 4, Real{12}, Real{4}, Real{0}, Real{0}, nghost};
    std::vector<Real> face(g.storage_size(), near_max);
    const int low_outer = -g.nghost;
    const int high_outer = g.nx + g.nghost - 1;
    const Real tolerance =
        Real{2e-12} * near_max;

    // Both outermost ghost cells force the one-sided barycentric/Gauss path;
    // the high side additionally extrapolates one cell beyond the last sample.
    EXPECT_NEAR(quasar::mhd::cell_bx(g, face.data(), low_outer, 0),
                near_max, tolerance)
        << "low one-sided constant width=" << width;
    EXPECT_NEAR(quasar::mhd::cell_bx(g, face.data(), high_outer, 0),
                near_max, tolerance)
        << "high one-sided constant width=" << width;

    // A linear polynomial odd about the integrated cell midpoint has an exact
    // zero average.  Its stored samples are O(DBL_MAX) and drive the old
    // barycentric numerator through overflowing, alternating products.  Test
    // both interpolation and one-cell extrapolation closures.
    const Real slope =
        std::numeric_limits<Real>::max() / (Real{4} * width);
    std::fill(face.begin(), face.end(), Real{0});
    for (int k = 0; k < width; ++k) {
      face[g.index(low_outer + k, 0)] =
          slope * (static_cast<Real>(k) - Real{0.5});
    }
    const Real low_cancel =
        quasar::mhd::cell_bx(g, face.data(), low_outer, 0);
    EXPECT_TRUE(std::isfinite(low_cancel));
    EXPECT_NEAR(low_cancel, Real{0}, Real{2e-11} * slope)
        << "low one-sided cancellation width=" << width;

    std::fill(face.begin(), face.end(), Real{0});
    const int high_start = high_outer - width + 1;
    for (int k = 0; k < width; ++k) {
      face[g.index(high_start + k, 0)] =
          slope * (static_cast<Real>(k) -
                   (static_cast<Real>(width) - Real{0.5}));
    }
    const Real high_cancel =
        quasar::mhd::cell_bx(g, face.data(), high_outer, 0);
    EXPECT_TRUE(std::isfinite(high_cancel));
    EXPECT_NEAR(high_cancel, Real{0}, Real{2e-10} * slope)
        << "high one-sided cancellation width=" << width;

    std::fill(face.begin(), face.end(), denorm);
    EXPECT_EQ(quasar::mhd::cell_bx(g, face.data(), low_outer, 0), denorm)
        << "low one-sided subnormal width=" << width;
    EXPECT_EQ(quasar::mhd::cell_bx(g, face.data(), high_outer, 0), denorm)
        << "high one-sided subnormal width=" << width;
  }
}
