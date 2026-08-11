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
using quasar::numerics::RadialCellMeasure;
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

Real face_extrapolation(long double reduced_radius, int neighbor_offset,
                        long double face_offset,
                        RadialCellMeasure measure) {
  const long double center = quasar::numerics::normalized_cell_moment(
      reduced_radius, 1, measure);
  const long double neighbor = quasar::numerics::normalized_cell_moment(
      reduced_radius + static_cast<long double>(neighbor_offset), 1, measure);
  return static_cast<Real>(
      (reduced_radius + face_offset - center) / (center - neighbor));
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

TEST(MpLimiterCylindrical,
     Mp5NativeCandidatesUseTheirOwnLimiterGeometry) {
  constexpr long double rho = 0.5L;
  const auto annular_r1 = solve_radial_row(
      rho, 5, -2, RadialMomentTarget::point_value,
      RadialCellMeasure::annular, 0.5L);
  const auto angular_r1 = solve_radial_row(
      rho, 5, -2, RadialMomentTarget::point_value,
      RadialCellMeasure::angular_momentum, 0.5L);
  const auto uniform_r1 = solve_radial_row(
      rho, 5, -2, RadialMomentTarget::point_value,
      RadialCellMeasure::uniform, 0.5L);
  const auto annular_r6 = solve_radial_row(
      rho, 2, 0, RadialMomentTarget::point_value,
      RadialCellMeasure::annular, 0.5L);
  const auto angular_r6 = solve_radial_row(
      rho, 2, 0, RadialMomentTarget::point_value,
      RadialCellMeasure::angular_momentum, 0.5L);
  const auto uniform_r6 = solve_radial_row(
      rho, 2, 0, RadialMomentTarget::point_value,
      RadialCellMeasure::uniform, 0.5L);

  constexpr std::array<Real, 5> bphi{-3, -2, 2, 3, 3};
  const Real bphi_annular =
      mp_interp_weighted(annular_r1.c, bphi.data(), 5);
  const Real bphi_native =
      mp_interp_weighted(uniform_r1.c, bphi.data(), 5);
  EXPECT_NEAR(bphi_annular, Real{2.65}, Real{1e-14});
  EXPECT_EQ(mp_limit(
                bphi_annular, bphi[0], bphi[1], bphi[2], bphi[3], bphi[4],
                annular_r6.c[0], annular_r6.c[1],
                face_extrapolation(
                    rho, -1, 0.5L, RadialCellMeasure::annular)),
            bphi_annular);
  EXPECT_NEAR(bphi_native, Real{3.1}, Real{1e-14});
  EXPECT_EQ(mp_limit(
                bphi_native, bphi[0], bphi[1], bphi[2], bphi[3], bphi[4],
                uniform_r6.c[0], uniform_r6.c[1],
                face_extrapolation(
                    rho, -1, 0.5L, RadialCellMeasure::uniform)),
            Real{3});

  constexpr std::array<Real, 5> mphi{-1, -2, 2, 1, 3};
  const Real mphi_annular =
      mp_interp_weighted(annular_r1.c, mphi.data(), 5);
  const Real mphi_native =
      mp_interp_weighted(angular_r1.c, mphi.data(), 5);
  EXPECT_NEAR(mphi_annular, Real{2}, Real{1e-14});
  EXPECT_EQ(mp_limit(
                mphi_annular, mphi[0], mphi[1], mphi[2], mphi[3], mphi[4],
                annular_r6.c[0], annular_r6.c[1],
                face_extrapolation(
                    rho, -1, 0.5L, RadialCellMeasure::annular)),
            mphi_annular);
  EXPECT_NEAR(mphi_native, Real{1.95}, Real{1e-14});
  EXPECT_EQ(mp_limit(
                mphi_native, mphi[0], mphi[1], mphi[2], mphi[3], mphi[4],
                angular_r6.c[0], angular_r6.c[1],
                face_extrapolation(
                    rho, -1, 0.5L,
                    RadialCellMeasure::angular_momentum)),
            Real{2});
}

TEST(MpLimiterCylindrical,
     Mp7NativeLimiterBoundsBothInterfaceOrientations) {
  constexpr long double left_rho = 5.5L;
  constexpr long double right_rho = 6.5L;
  constexpr std::array<Real, 8> values{
      1000000.0, -2.0, -1.0, 0.0, 1.0, 2.0,
      787812.0342935974, 6550699.945732342};
  const auto annular_left = solve_radial_row(
      left_rho, 7, -3, RadialMomentTarget::point_value,
      RadialCellMeasure::annular, 0.5L);
  const auto annular_right = solve_radial_row(
      right_rho, 7, -3, RadialMomentTarget::point_value,
      RadialCellMeasure::annular, -0.5L);
  const auto angular_left = solve_radial_row(
      left_rho, 7, -3, RadialMomentTarget::point_value,
      RadialCellMeasure::angular_momentum, 0.5L);
  const auto angular_right = solve_radial_row(
      right_rho, 7, -3, RadialMomentTarget::point_value,
      RadialCellMeasure::angular_momentum, -0.5L);
  const auto uniform_left = solve_radial_row(
      left_rho, 7, -3, RadialMomentTarget::point_value,
      RadialCellMeasure::uniform, 0.5L);
  const auto uniform_right = solve_radial_row(
      right_rho, 7, -3, RadialMomentTarget::point_value,
      RadialCellMeasure::uniform, -0.5L);

  const auto check_measure = [&](const auto& left_r1, const auto& right_r1,
                                 RadialCellMeasure measure,
                                 Real expected_left, Real expected_right) {
    const auto r6 = solve_radial_row(
        left_rho, 2, 0, RadialMomentTarget::point_value, measure, 0.5L);
    const Real left_candidate =
        mp_interp_weighted(left_r1.c, values.data(), 7);
    const Real right_candidate =
        mp_interp_weighted(right_r1.c, values.data() + 1, 7);
    const Real left_limited = mp_limit(
        left_candidate, values[1], values[2], values[3], values[4], values[5],
        r6.c[0], r6.c[1],
        face_extrapolation(left_rho, -1, 0.5L, measure));
    const Real right_limited = mp_limit(
        right_candidate, values[6], values[5], values[4], values[3], values[2],
        r6.c[1], r6.c[0],
        face_extrapolation(right_rho, 1, -0.5L, measure));
    if (measure == RadialCellMeasure::annular) {
      EXPECT_EQ(left_limited, left_candidate);
      EXPECT_EQ(right_limited, right_candidate);
    } else {
      EXPECT_EQ(left_limited, expected_left);
      EXPECT_EQ(right_limited, expected_right);
    }
    return std::array<Real, 2>{left_candidate, right_candidate};
  };

  const auto annular = check_measure(
      annular_left, annular_right, RadialCellMeasure::annular,
      /*expected_left=*/Real{0.5}, /*expected_right=*/Real{0.5});
  EXPECT_NEAR(annular[0], Real{0.499999999864}, Real{2e-10});
  EXPECT_NEAR(annular[1], Real{0.500000000065}, Real{2e-10});

  const auto uniform = check_measure(
      uniform_left, uniform_right, RadialCellMeasure::uniform,
      /*expected_left=*/Real{1}, /*expected_right=*/Real{1});
  EXPECT_NEAR(uniform[0], Real{360.56461232}, Real{2e-8});
  EXPECT_NEAR(uniform[1], Real{103.20957177}, Real{2e-8});

  const auto angular = check_measure(
      angular_left, angular_right, RadialCellMeasure::angular_momentum,
      /*expected_left=*/Real{0}, /*expected_right=*/Real{0});
  EXPECT_NEAR(angular[0], Real{-353.42301381}, Real{2e-8});
  EXPECT_NEAR(angular[1], Real{-95.361551705}, Real{2e-8});
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
