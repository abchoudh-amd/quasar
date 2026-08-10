#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/radial_moments.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

using quasar::Real;
using quasar::numerics::RadialMomentTarget;
using quasar::numerics::RadialStencilRow;
using quasar::numerics::normalized_cell_moment;
using quasar::numerics::radial_gauss_weights;
using quasar::numerics::solve_radial_row;

long double integer_power(long double value, int exponent) {
  long double result = 1.0L;
  for (int k = 0; k < exponent; ++k) result *= value;
  return result;
}

long double uniform_unit_cell_moment(int degree) {
  if ((degree & 1) != 0) return 0.0L;
  return 1.0L
         / (integer_power(2.0L, degree)
            * static_cast<long double>(degree + 1));
}

long double positive_radial_local_moment(long double rho, int degree) {
  return uniform_unit_cell_moment(degree)
         + uniform_unit_cell_moment(degree + 1) / rho;
}

Real in_order_sum(const RadialStencilRow& row) {
  Real sum = 0.0;
  for (int k = 0; k < row.width; ++k) sum += row.c[k];
  return sum;
}

template <std::size_t Width, class Expected>
void expect_row_near(const RadialStencilRow& row, const Expected& expected,
                     Real denominator, Real tolerance) {
  ASSERT_EQ(row.width, static_cast<int>(Width));
  for (std::size_t k = 0; k < Width; ++k) {
    EXPECT_NEAR(row.c[k], static_cast<Real>(expected[k]) / denominator,
                tolerance)
        << "coefficient " << k;
  }
  EXPECT_EQ(in_order_sum(row), Real{1});
}

template <std::size_t Width>
Real maximum_difference(const RadialStencilRow& row,
                        const std::array<Real, Width>& reference) {
  Real difference = 0.0;
  for (std::size_t k = 0; k < Width; ++k) {
    difference = std::max(
        difference, std::abs(row.c[k] - reference[k]));
  }
  return difference;
}

TEST(RadialMoments, NormalizedCellMomentUsesAbsoluteRadialMeasure) {
  for (const long double rho : {-10.0L, -0.5L, 0.0L, 0.5L, 10.0L}) {
    EXPECT_EQ(normalized_cell_moment(rho, 0), 1.0L);
  }

  constexpr long double rho = 2.0L;
  EXPECT_NEAR(
      static_cast<Real>(normalized_cell_moment(rho, 1)),
      static_cast<Real>(rho + 1.0L / (12.0L * rho)), Real{2e-16});
  for (int degree = 0; degree <= 10; ++degree) {
    const long double positive = normalized_cell_moment(rho, degree);
    const long double reflected = normalized_cell_moment(-rho, degree);
    EXPECT_NEAR(
        static_cast<Real>(reflected),
        static_cast<Real>((degree & 1) != 0 ? -positive : positive),
        Real{2e-14});
  }
}

TEST(RadialMoments, CartesianLimitMatchesExistingCoefficientFamilies) {
  // The leading cylindrical correction is O(1/rho).  rho=1e9 makes that
  // correction smaller than the requested 1e-9 Cartesian regression anchor
  // while remaining far below the range where global-power subtraction would
  // be accurate.  The engine solves in a shifted basis instead.
  constexpr long double rho = 1.0e9L;
  constexpr Real tolerance = 1.0e-9;

  constexpr std::array<int, 5> mp5_left{2, -13, 47, 27, -3};
  expect_row_near<5>(
      solve_radial_row(rho, 5, -2, RadialMomentTarget::point_value, 0.5L),
      mp5_left, Real{60}, tolerance);
  constexpr std::array<int, 5> mp5_right{-3, 27, 47, -13, 2};
  expect_row_near<5>(
      solve_radial_row(rho, 5, -2, RadialMomentTarget::point_value, -0.5L),
      mp5_right, Real{60}, tolerance);

  constexpr std::array<int, 7> mp7_left{-3, 25, -101, 319, 214, -38, 4};
  expect_row_near<7>(
      solve_radial_row(rho, 7, -3, RadialMomentTarget::point_value, 0.5L),
      mp7_left, Real{420}, tolerance);
  constexpr std::array<int, 7> mp7_right{4, -38, 214, 319, -101, 25, -3};
  expect_row_near<7>(
      solve_radial_row(rho, 7, -3, RadialMomentTarget::point_value, -0.5L),
      mp7_right, Real{420}, tolerance);

  for (int q = 0; q < 3; ++q) {
    const auto row = solve_radial_row(
        rho, 5, -2, RadialMomentTarget::point_value,
        static_cast<long double>(quasar::numerics::kMp5TransverseNodes[q]));
    expect_row_near<5>(
        row, quasar::numerics::kMp5TransversePointWeights[q], Real{1},
        tolerance);
  }
  for (int q = 0; q < 4; ++q) {
    const auto row = solve_radial_row(
        rho, 7, -3, RadialMomentTarget::point_value,
        static_cast<long double>(quasar::numerics::kMp7TransverseNodes[q]));
    expect_row_near<7>(
        row, quasar::numerics::kMp7TransversePointWeights[q], Real{1},
        tolerance);
  }

  expect_row_near<4>(
      solve_radial_row(rho, 4, -2, RadialMomentTarget::cell_average, 0.5L),
      quasar::mhd::detail::kFaceToCell4, Real{24}, tolerance);
  expect_row_near<6>(
      solve_radial_row(rho, 6, -3, RadialMomentTarget::cell_average, 0.5L),
      quasar::mhd::detail::kFaceToCell6, Real{1440}, tolerance);
  expect_row_near<8>(
      solve_radial_row(rho, 8, -4, RadialMomentTarget::cell_average, 0.5L),
      quasar::mhd::detail::kFaceToCell8, Real{120960}, tolerance);

  expect_row_near<4>(
      solve_radial_row(rho, 4, -1, RadialMomentTarget::point_value, 0.5L),
      quasar::mhd::detail::kCellToFace4, Real{12}, tolerance);
  expect_row_near<6>(
      solve_radial_row(rho, 6, -2, RadialMomentTarget::point_value, 0.5L),
      quasar::mhd::detail::kCellToFace6, Real{60}, tolerance);
  expect_row_near<8>(
      solve_radial_row(rho, 8, -3, RadialMomentTarget::point_value, 0.5L),
      quasar::mhd::detail::kCellToFace8, Real{840}, tolerance);

  constexpr std::array<int, 2> average{1, 1};
  expect_row_near<2>(
      solve_radial_row(rho, 2, 0, RadialMomentTarget::point_value, 0.5L),
      average, Real{2}, tolerance);

  expect_row_near<3>(
      radial_gauss_weights(
          rho, 3, quasar::numerics::kMp5TransverseNodes,
          quasar::numerics::kMp5TransverseGaussWeights),
      quasar::numerics::kMp5TransverseGaussWeights, Real{1}, tolerance);
  expect_row_near<4>(
      radial_gauss_weights(
          rho, 4, quasar::numerics::kMp7TransverseNodes,
          quasar::numerics::kMp7TransverseGaussWeights),
      quasar::numerics::kMp7TransverseGaussWeights, Real{1}, tolerance);
}

TEST(RadialMoments, CartesianLimitConvergesAtInverseRadius) {
  constexpr std::array<Real, 5> cartesian{
      Real{2} / Real{60}, Real{-13} / Real{60}, Real{47} / Real{60},
      Real{27} / Real{60}, Real{-3} / Real{60}};
  const auto coarse = solve_radial_row(
      1.0e5L, 5, -2, RadialMomentTarget::point_value, 0.5L);
  const auto fine = solve_radial_row(
      1.0e6L, 5, -2, RadialMomentTarget::point_value, 0.5L);
  const Real coarse_error = maximum_difference(coarse, cartesian);
  const Real fine_error = maximum_difference(fine, cartesian);
  EXPECT_GT(fine_error, Real{1e-9});
  EXPECT_NEAR(coarse_error / fine_error, Real{10}, Real{2e-4});
}

TEST(RadialMoments, MomentExactnessResidual) {
  constexpr std::array<long double, 5> radii{
      0.5L, 1.5L, 2.5L, 10.0L, 100.0L};
  constexpr std::array<int, 6> widths{2, 4, 5, 6, 7, 8};
  for (const long double rho : radii) {
    for (const int width : widths) {
      SCOPED_TRACE(::testing::Message{} << "rho=" << static_cast<double>(rho)
                                        << " width=" << width);
      const int offset = (width & 1) != 0 ? -width / 2 : 1 - width / 2;
      const auto point = solve_radial_row(
          rho, width, offset, RadialMomentTarget::point_value, 0.5L);
      EXPECT_LT(point.residual, Real{1e-11});
      EXPECT_EQ(in_order_sum(point), Real{1});

      if ((width & 1) == 0) {
        const auto average = solve_radial_row(
            rho, width, -width / 2,
            RadialMomentTarget::cell_average, 0.5L);
        EXPECT_LT(average.residual, Real{1e-11});
        EXPECT_EQ(in_order_sum(average), Real{1});
      }
    }
  }
}

TEST(RadialMoments, ExactOnAnalyticPolynomialProfiles) {
  constexpr long double rho = 2.5L;
  constexpr int width = 7;
  constexpr int offset = -3;
  constexpr long double node = 0.5L;
  const auto to_point = solve_radial_row(
      rho, width, offset, RadialMomentTarget::point_value, node);
  for (int degree = 0; degree < width; ++degree) {
    long double actual = 0.0L;
    for (int k = 0; k < width; ++k) {
      actual += static_cast<long double>(to_point.c[k])
                * normalized_cell_moment(rho + offset + k, degree);
    }
    const long double expected = integer_power(rho + node, degree);
    EXPECT_NEAR(static_cast<Real>(actual), static_cast<Real>(expected),
                Real{3e-11} * std::max(Real{1}, std::abs(static_cast<Real>(expected))))
        << "degree=" << degree;
  }

  constexpr int face_width = 8;
  constexpr int face_offset = -4;
  const auto to_average = solve_radial_row(
      rho, face_width, face_offset, RadialMomentTarget::cell_average, node);
  for (int degree = 0; degree < face_width; ++degree) {
    long double actual = 0.0L;
    for (int k = 0; k < face_width; ++k) {
      actual += static_cast<long double>(to_average.c[k])
                * integer_power(rho + face_offset + k + node, degree);
    }
    const long double expected = normalized_cell_moment(rho, degree);
    EXPECT_NEAR(static_cast<Real>(actual), static_cast<Real>(expected),
                Real{3e-11} * std::max(Real{1}, std::abs(static_cast<Real>(expected))))
        << "degree=" << degree;
  }
}

TEST(RadialMoments, AxisStencilIsFinite) {
  for (const long double rho : {0.0L, 0.5L}) {
    for (const int width : {2, 4, 5, 6, 7, 8}) {
      const int offset = (width & 1) != 0 ? -width / 2 : 1 - width / 2;
      const auto row = solve_radial_row(
          rho, width, offset, RadialMomentTarget::point_value, 0.5L);
      EXPECT_TRUE(std::isfinite(row.residual));
      EXPECT_EQ(in_order_sum(row), Real{1});
      for (int k = 0; k < width; ++k) {
        EXPECT_TRUE(std::isfinite(row.c[k]));
      }
    }
  }
}

template <int Count>
void expect_radial_gauss_degree(
    long double rho, const Real (&nodes)[Count],
    const Real (&cartesian_weights)[Count]) {
  const auto row = radial_gauss_weights(
      rho, Count, nodes, cartesian_weights);
  ASSERT_EQ(row.width, Count);
  EXPECT_EQ(in_order_sum(row), Real{1});
  EXPECT_LT(row.residual, Real{3e-14});

  for (int degree = 0; degree <= 2 * Count - 2; ++degree) {
    long double actual = 0.0L;
    for (int q = 0; q < Count; ++q) {
      actual += static_cast<long double>(row.c[q])
                * integer_power(static_cast<long double>(nodes[q]), degree);
    }
    EXPECT_NEAR(
        static_cast<Real>(actual),
        static_cast<Real>(positive_radial_local_moment(rho, degree)),
        Real{3e-14})
        << "degree=" << degree;
  }

  constexpr int first_inexact_degree = 2 * Count - 1;
  long double first_inexact = 0.0L;
  for (int q = 0; q < Count; ++q) {
    first_inexact += static_cast<long double>(row.c[q])
                     * integer_power(
                         static_cast<long double>(nodes[q]),
                         first_inexact_degree);
  }
  EXPECT_GT(
      std::fabs(first_inexact
                - positive_radial_local_moment(rho, first_inexact_degree)),
      1.0e-10L);
}

TEST(RadialMoments, GaussWeightsExactToDegree) {
  constexpr long double rho = 0.5L;
  expect_radial_gauss_degree(
      rho, quasar::numerics::kMp5TransverseNodes,
      quasar::numerics::kMp5TransverseGaussWeights);
  expect_radial_gauss_degree(
      rho, quasar::numerics::kMp7TransverseNodes,
      quasar::numerics::kMp7TransverseGaussWeights);
}

TEST(RadialMoments, RejectsInvalidRequests) {
  EXPECT_THROW(
      solve_radial_row(
          0.5L, 0, 0, RadialMomentTarget::point_value, 0.0L),
      std::invalid_argument);
  EXPECT_THROW(normalized_cell_moment(0.5L, -1), std::invalid_argument);
  EXPECT_THROW(radial_gauss_weights(0.5L, 3, nullptr, nullptr),
               std::invalid_argument);
}

}  // namespace
