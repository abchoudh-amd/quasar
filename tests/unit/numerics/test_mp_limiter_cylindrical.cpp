#include "quasar/numerics/mp_limiter.hpp"
#include "quasar/numerics/radial_moments.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace {

using quasar::Real;
using quasar::numerics::RadialMomentTarget;
using quasar::numerics::mp_interp_weighted;
using quasar::numerics::mp_limit;
using quasar::numerics::solve_radial_row;

long double sine_antiderivative(long double radius) {
  constexpr long double angular_frequency =
      2.0L * std::numbers::pi_v<long double>;
  return std::sin(angular_frequency * radius) /
             (angular_frequency * angular_frequency) -
         radius * std::cos(angular_frequency * radius) /
             angular_frequency;
}

Real ring_average(long double reduced_center, long double cell_width) {
  const long double radius_low = (reduced_center - 0.5L) * cell_width;
  const long double radius_high = (reduced_center + 0.5L) * cell_width;
  const long double volume =
      0.5L * (radius_high * std::abs(radius_high) -
              radius_low * std::abs(radius_low));

  long double weighted_sine = 0.0L;
  if (radius_low < 0.0L) {
    const long double negative_high = std::min(radius_high, 0.0L);
    weighted_sine -= sine_antiderivative(negative_high) -
                     sine_antiderivative(radius_low);
  }
  if (radius_high > 0.0L) {
    const long double positive_low = std::max(radius_low, 0.0L);
    weighted_sine += sine_antiderivative(radius_high) -
                     sine_antiderivative(positive_low);
  }

  return static_cast<Real>(2.0L + 0.5L * weighted_sine / volume);
}

Real left_face_extrapolation(long double reduced_radius) {
  const long double center =
      quasar::numerics::normalized_cell_moment(reduced_radius, 1);
  const long double neighbor =
      quasar::numerics::normalized_cell_moment(reduced_radius - 1.0L, 1);
  return static_cast<Real>(
      (reduced_radius + 0.5L - center) / (center - neighbor));
}

TEST(MpLimiterCylindrical, LimiterDoesNotActivateOnSmoothAxisData) {
  struct SmoothCase {
    long double reduced_radius;
    long double cell_width;
  };
  // Each radius uses a different resolved curvature scale so the same smooth
  // profile probes the narrow part of that row's limiter boundary.
  constexpr std::array<SmoothCase, 3> cases{{
      {0.5L, 0.28057L},
      {1.5L, 0.37723L},
      {2.5L, 0.26751L},
  }};

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message{}
                 << "rho=" << static_cast<double>(test_case.reduced_radius)
                 << " dr=" << static_cast<double>(test_case.cell_width));
    const auto interpolation = solve_radial_row(
        test_case.reduced_radius, 5, -2,
        RadialMomentTarget::point_value, 0.5L);
    const auto face_pair = solve_radial_row(
        test_case.reduced_radius, 2, 0,
        RadialMomentTarget::point_value, 0.5L);

    std::array<Real, 5> values{};
    for (int k = 0; k < 5; ++k) {
      values[static_cast<std::size_t>(k)] = ring_average(
          test_case.reduced_radius - 2.0L + static_cast<long double>(k),
          test_case.cell_width);
    }

    const Real candidate =
        mp_interp_weighted(interpolation.c, values.data(), 5);
    const Real cylindrical = mp_limit(
        candidate, values[0], values[1], values[2], values[3], values[4],
        face_pair.c[0], face_pair.c[1],
        left_face_extrapolation(test_case.reduced_radius));
    const Real cartesian = mp_limit(
        candidate, values[0], values[1], values[2], values[3], values[4],
        Real{0.5}, Real{0.5}, Real{0.5});

    EXPECT_EQ(cylindrical, candidate);
    EXPECT_NE(cartesian, candidate);
    EXPECT_GT(std::abs(cartesian - candidate), Real{1e-3});
  }
}

TEST(MpLimiterCylindrical, LimiterStillBoundsADiscontinuity) {
  constexpr long double reduced_radius = 0.5L;
  const auto interpolation = solve_radial_row(
      reduced_radius, 5, -2, RadialMomentTarget::point_value, 0.5L);
  const auto face_pair = solve_radial_row(
      reduced_radius, 2, 0, RadialMomentTarget::point_value, 0.5L);
  constexpr std::array<std::array<Real, 5>, 2> steps{{
      {Real{0}, Real{0}, Real{1}, Real{1}, Real{1}},
      {Real{0}, Real{0}, Real{0}, Real{0}, Real{1}},
  }};

  for (const auto& values : steps) {
    const Real candidate =
        mp_interp_weighted(interpolation.c, values.data(), 5);
    const Real limited = mp_limit(
        candidate, values[0], values[1], values[2], values[3], values[4],
        face_pair.c[0], face_pair.c[1]);
    const auto [minimum, maximum] =
        std::minmax_element(values.begin(), values.end());

    EXPECT_TRUE(candidate < *minimum || candidate > *maximum);
    EXPECT_GE(limited, *minimum);
    EXPECT_LE(limited, *maximum);
  }
}

TEST(MpLimiterCylindrical, OneSidedLargeCurvatureBoundUsesRadialFactor) {
  // This profile reaches the v_lc side of the MP interval.  At rho=0.5 the
  // left state's two-cell linear extrapolation factor is 1/4, not Cartesian
  // 1/2; the bracketing face pair is independently (5/8,3/8).
  constexpr Real candidate = Real{-4};
  constexpr Real pair_left = Real{5} / Real{8};
  constexpr Real pair_right = Real{3} / Real{8};
  const Real cartesian = mp_limit(
      candidate, Real{-3}, Real{-1}, Real{0}, Real{-2}, Real{-3},
      pair_left, pair_right, Real{0.5});
  const Real cylindrical = mp_limit(
      candidate, Real{-3}, Real{-1}, Real{0}, Real{-2}, Real{-3},
      pair_left, pair_right, Real{0.25});

  EXPECT_NEAR(cartesian, -Real{5} / Real{6}, Real{1e-15});
  EXPECT_NEAR(cylindrical, -Real{13} / Real{12}, Real{1e-15});
  EXPECT_NE(cartesian, cylindrical);
}

TEST(MpLimiterCylindrical,
     RightOrientedLargeCurvatureBoundUsesItsOwnRadialFactor) {
  // Production reverses the five-point limiter window for the right state and
  // reverses the bracketing R6 pair. At the axis-adjacent face the outward
  // slope extrapolation is 25/44, distinct from both the left state's 1/4 and
  // the Cartesian 1/2. This profile reaches that v_lc branch.
  constexpr Real candidate = Real{-6};
  constexpr Real pair_left = Real{3} / Real{8};
  constexpr Real pair_right = Real{5} / Real{8};
  const Real cartesian_factor = mp_limit(
      candidate, Real{-4}, Real{-2}, Real{-1}, Real{-3}, Real{-4},
      pair_left, pair_right, Real{0.5});
  const Real radial_factor = mp_limit(
      candidate, Real{-4}, Real{-2}, Real{-1}, Real{-3}, Real{-4},
      pair_left, pair_right, Real{25} / Real{44});

  EXPECT_NEAR(cartesian_factor, -Real{11} / Real{6}, Real{1e-15});
  EXPECT_NEAR(radial_factor, -Real{233} / Real{132}, Real{1e-15});
  EXPECT_NE(cartesian_factor, radial_factor);
}

TEST(MpLimiterCylindrical, CartesianWeightsAreBitIdentical) {
  struct LimiterInput {
    Real candidate;
    std::array<Real, 5> values;
  };
  constexpr std::array<LimiterInput, 4> inputs{{
      {Real{1.13}, {Real{0}, Real{0}, Real{1}, Real{1}, Real{1}}},
      {Real{-0.04}, {Real{0}, Real{0}, Real{0}, Real{0}, Real{1}}},
      {Real{1.75}, {Real{1}, Real{1.25}, Real{1.5}, Real{2}, Real{2.75}}},
      {Real{-2.5}, {Real{-3}, Real{-2.75}, Real{-2}, Real{-1}, Real{0.5}}},
  }};

  for (const auto& input : inputs) {
    const auto& values = input.values;
    const Real default_weights = mp_limit(
        input.candidate, values[0], values[1], values[2], values[3],
        values[4]);
    const Real explicit_weights = mp_limit(
        input.candidate, values[0], values[1], values[2], values[3],
        values[4], Real{0.5}, Real{0.5}, Real{0.5});

    EXPECT_EQ(std::bit_cast<std::uint64_t>(default_weights),
              std::bit_cast<std::uint64_t>(explicit_weights));
  }
}

}  // namespace
