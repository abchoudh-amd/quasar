// Guards for the Biot-Savart evaluator's degenerate inputs:
//   * an observation point lying ON a filament segment (a true ideal-filament
//     singularity, reported explicitly rather than replaced by zero), and
//   * empty conductor systems / empty point clouds (the N==0 || M==0 early
//     return that must still produce a correctly-sized all-zero field).
// Empty inputs still yield correctly-sized zeros.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "host_evaluate.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::BiotSavartEvaluatorF;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::circular_loop;

namespace {

// A single straight filament from (-1,0,0) to (1,0,0) carrying 1 A.
ConductorSystem make_straight_wire() {
  ConductorSystem cs;
  Filament f;
  f.name = "wire";
  f.current_A = Real{1};
  f.points = {Vec3{-1, 0, 0}, Vec3{1, 0, 0}};
  cs.add(f);
  return cs;
}

ConductorSystem repeated_segment(Vec3 a, Vec3 b, Real current,
                                 int positive_count, int negative_count) {
  ConductorSystem cs;
  for (int i = 0; i < positive_count; ++i) {
    cs.add(Filament{"positive", current, {a, b}});
  }
  for (int i = 0; i < negative_count; ++i) {
    cs.add(Filament{"negative", -current, {a, b}});
  }
  return cs;
}

ConductorSystem ordered_segment_currents(
    Vec3 a, Vec3 b, std::initializer_list<Real> currents) {
  ConductorSystem cs;
  int index = 0;
  for (const Real current : currents) {
    cs.add(Filament{"ordered_" + std::to_string(index++), current, {a, b}});
  }
  return cs;
}

}  // namespace

TEST(BiotSavartSingular, OnSegmentPointsThrowDomainError) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const BiotSavartEvaluator eval;
  const auto cs = make_straight_wire();

  PointCloud pc;
  pc.add(Vec3{-1, 0, 0});  // segment start vertex
  pc.add(Vec3{0, 0, 0});   // segment midpoint
  pc.add(Vec3{1, 0, 0});   // segment end vertex

  EXPECT_THROW((void)quasar::test::host_evaluate_B(eval, cs, pc), std::domain_error);
  EXPECT_THROW((void)quasar::test::host_evaluate_grad_B(eval, cs, pc), std::domain_error);
  EXPECT_THROW((void)quasar::test::host_evaluate_A(eval, cs, pc), std::domain_error);
}

TEST(BiotSavartSingular, DiagonalSegmentMidpointThrowsInBothPrecisions) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;
  cs.add(Filament{"diagonal", Real{1},
                  {Vec3{0, 0, 0}, Vec3{2, 4, 6}}});
  PointCloud pc;
  pc.add(Vec3{1, 2, 3});

  const auto expect_singular = [&](const auto& eval) {
    EXPECT_THROW((void)quasar::test::host_evaluate_B(eval, cs, pc), std::domain_error);
    EXPECT_THROW((void)quasar::test::host_evaluate_A(eval, cs, pc), std::domain_error);
    EXPECT_THROW((void)quasar::test::host_evaluate_grad_B(eval, cs, pc), std::domain_error);
  };
  expect_singular(BiotSavartEvaluator{});
  expect_singular(BiotSavartEvaluatorF{});
}

TEST(BiotSavartSingular, NonRepresentableSegmentFractionStillThrows) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  // (27,63) is exactly 9/11 of the way from (0,0) to (33,77), but 9/11 is
  // not representable.  Reconstructing the dominant coordinate with the
  // rounded quotient produces 63.00000000000001 in fp64 and misses the true
  // ideal-filament singularity.
  ConductorSystem cs;
  cs.add(Filament{"fractional_diagonal", Real{1},
                  {Vec3{0, 0, 0}, Vec3{33, 77, 0}}});
  PointCloud pc;
  pc.add(Vec3{27, 63, 0});

  const auto expect_singular = [&](const auto& eval) {
    EXPECT_THROW((void)quasar::test::host_evaluate_B(eval, cs, pc), std::domain_error);
    EXPECT_THROW((void)quasar::test::host_evaluate_A(eval, cs, pc), std::domain_error);
    EXPECT_THROW((void)quasar::test::host_evaluate_grad_B(eval, cs, pc), std::domain_error);
  };
  expect_singular(BiotSavartEvaluator{});
  expect_singular(BiotSavartEvaluatorF{});
}

TEST(BiotSavartSingular, DistinctNearDiagonalPointRemainsFinite) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;
  cs.add(Filament{"diagonal", Real{1},
                  {Vec3{0, 0, 0}, Vec3{2, 4, 6}}});
  PointCloud pc;
  pc.add(Vec3{1, 2, Real{3.0001}});

  const auto expect_finite = [&](const auto& eval) {
    const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
    const auto potential = quasar::test::host_evaluate_A(eval, cs, pc);
    const auto gradient = quasar::test::host_evaluate_grad_B(eval, cs, pc);
    EXPECT_TRUE(std::isfinite(field[0].x));
    EXPECT_TRUE(std::isfinite(field[0].y));
    EXPECT_TRUE(std::isfinite(field[0].z));
    EXPECT_TRUE(std::isfinite(potential[0].x));
    EXPECT_TRUE(std::isfinite(potential[0].y));
    EXPECT_TRUE(std::isfinite(potential[0].z));
    EXPECT_TRUE(std::isfinite(gradient[0].r0.x));
    EXPECT_TRUE(std::isfinite(gradient[0].r1.y));
    EXPECT_TRUE(std::isfinite(gradient[0].r2.z));
  };
  expect_finite(BiotSavartEvaluator{});
  expect_finite(BiotSavartEvaluatorF{});
}

TEST(BiotSavartSingular, ZeroCurrentHasNoFilamentSingularity) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;
  cs.add(Filament{"zero", Real{0}, {Vec3{-1, 0, 0}, Vec3{1, 0, 0}}});
  PointCloud pc;
  pc.add(Vec3{0, 0, 0});
  const auto B = quasar::test::host_evaluate_B(BiotSavartEvaluator{}, cs, pc);
  ASSERT_EQ(B.size(), 1u);
  EXPECT_EQ(B[0].x, Real{0});
  EXPECT_EQ(B[0].y, Real{0});
  EXPECT_EQ(B[0].z, Real{0});
}

TEST(BiotSavartNumerics, UnrepresentableResultThrowsInsteadOfReturningInfinity) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;
  cs.add(Filament{"extreme", std::numeric_limits<Real>::max(),
                  {Vec3{-1, 0, 0}, Vec3{1, 0, 0}}});
  PointCloud pc;
  pc.add(Vec3{0, Real{1e-100}, 0});
  const BiotSavartEvaluator eval;
  EXPECT_THROW((void)quasar::test::host_evaluate_B(eval, cs, pc), std::overflow_error);
  EXPECT_THROW((void)quasar::test::host_evaluate_grad_B(eval, cs, pc), std::overflow_error);
}

TEST(BiotSavartNumerics, OppositeSignObservationDifferenceDoesNotFalseSingular) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const Real largest = std::numeric_limits<Real>::max();
  ConductorSystem cs;
  cs.add(Filament{"extreme_coordinates", Real{1},
                  {Vec3{-largest, 0, 0}, Vec3{0, 0, 0}}});
  PointCloud pc;
  pc.add(Vec3{largest, 0, 0});  // collinear but strictly beyond the endpoint

  const BiotSavartEvaluator eval;
  const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
  const auto potential = quasar::test::host_evaluate_A(eval, cs, pc);
  ASSERT_EQ(field.size(), 1u);
  ASSERT_EQ(potential.size(), 1u);
  EXPECT_EQ(field[0].x, Real{0});
  EXPECT_EQ(field[0].y, Real{0});
  EXPECT_EQ(field[0].z, Real{0});
  EXPECT_NEAR(potential[0].x, quasar::mu0_over_4pi * std::log(Real{2}),
              Real{2e-15} * quasar::mu0_over_4pi);
  EXPECT_EQ(potential[0].y, Real{0});
  EXPECT_EQ(potential[0].z, Real{0});
}

TEST(BiotSavartNumerics,
     NextRepresentablePointPastEndpointIsFiniteInEitherOrientation) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const auto expect_finite_exterior = [](Vec3 a, Vec3 b, Real point_x) {
    ConductorSystem conductors;
    conductors.add(Filament{"endpoint_exterior", Real{1}, {a, b}});
    PointCloud point;
    point.add(Vec3{point_x, 0, 0});

    const BiotSavartEvaluator eval;
    const auto field = quasar::test::host_evaluate_B(eval, conductors, point);
    const auto potential = quasar::test::host_evaluate_A(eval, conductors, point);
    const auto gradient = quasar::test::host_evaluate_grad_B(eval, conductors, point);
    ASSERT_EQ(field.size(), 1u);
    ASSERT_EQ(potential.size(), 1u);
    ASSERT_EQ(gradient.size(), 1u);

    // A point on the exterior extension has exactly zero B.  A and grad(B)
    // become large this close to the endpoint, but remain mathematically finite.
    EXPECT_EQ(field[0].x, Real{0});
    EXPECT_EQ(field[0].y, Real{0});
    EXPECT_EQ(field[0].z, Real{0});
    for (const Real value : {
             potential[0].x, potential[0].y, potential[0].z,
             gradient[0].r0.x, gradient[0].r0.y, gradient[0].r0.z,
             gradient[0].r1.x, gradient[0].r1.y, gradient[0].r1.z,
             gradient[0].r2.x, gradient[0].r2.y, gradient[0].r2.z}) {
      EXPECT_TRUE(std::isfinite(value));
    }
  };

  const Real infinity = std::numeric_limits<Real>::infinity();
  expect_finite_exterior(
      Vec3{-1, 0, 0}, Vec3{1, 0, 0}, std::nextafter(Real{1}, infinity));
  expect_finite_exterior(
      Vec3{1, 0, 0}, Vec3{-1, 0, 0}, std::nextafter(Real{-1}, -infinity));
}

TEST(BiotSavartNumerics, UnderflowedTransverseOffsetIsNotFalseSingular) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Real half_length = Real{5e307};
  ConductorSystem cs;
  cs.add(Filament{"extreme_scale", std::numeric_limits<Real>::min(),
                  {Vec3{-half_length, 0, 0}, Vec3{half_length, 0, 0}}});
  PointCloud pc;
  pc.add(Vec3{0, std::numeric_limits<Real>::denorm_min(), 0});

  // Normalizing this geometry makes the nonzero y offset underflow.  It is an
  // unresolved finite-precision geometry, not a point on the ideal filament.
  const BiotSavartEvaluator eval;
  EXPECT_THROW((void)quasar::test::host_evaluate_B(eval, cs, pc), std::overflow_error);
  EXPECT_THROW((void)quasar::test::host_evaluate_A(eval, cs, pc), std::overflow_error);
  EXPECT_THROW((void)quasar::test::host_evaluate_grad_B(eval, cs, pc), std::overflow_error);
}

TEST(BiotSavartNumerics, FarFieldProductsDoNotOverflowBeforeFiniteFp64Result) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Real distance = Real{1e200};
  const Real current = std::numeric_limits<Real>::max();
  ConductorSystem cs;
  cs.add(Filament{"far_field", current,
                  {Vec3{0, 0, -1}, Vec3{0, 0, 1}}});
  PointCloud pc;
  pc.add(Vec3{distance, 0, 0});

  const BiotSavartEvaluator eval;
  const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
  const auto potential = quasar::test::host_evaluate_A(eval, cs, pc);
  const auto gradient = quasar::test::host_evaluate_grad_B(eval, cs, pc);

  const Real inv_distance = Real{1} / distance;
  const Real inv_radius = inv_distance
      / std::sqrt(Real{1} + inv_distance * inv_distance);
  const Real current_scale = quasar::mu0_over_4pi * current;
  const Real expected_b = ((Real{2} * current_scale) * inv_distance) * inv_radius;
  const Real expected_a = (Real{2} * current_scale) * std::asinh(inv_distance);
  const Real expected_db_dx = -expected_b * inv_distance
      * (Real{1} + Real{1} / (Real{1} + inv_distance * inv_distance));

  ASSERT_EQ(field.size(), 1u);
  ASSERT_EQ(potential.size(), 1u);
  ASSERT_EQ(gradient.size(), 1u);
  EXPECT_NEAR(field[0].y, expected_b, std::abs(expected_b) * Real{2e-14});
  EXPECT_NEAR(potential[0].z, expected_a, std::abs(expected_a) * Real{2e-14});
  EXPECT_NEAR(gradient[0].r1.x, expected_db_dx,
              std::abs(expected_db_dx) * Real{5e-13});
}

TEST(BiotSavartNumerics, TinySegmentFarObserverKeepsFiniteCompensatedResults) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Real length = Real{1e-210};
  constexpr Real half_length = length / Real{2};
  constexpr Real distance = Real{1e100};
  const Real current = std::numeric_limits<Real>::max();
  ConductorSystem cs;
  cs.add(Filament{"tiny_far", current,
                  {Vec3{0, 0, -half_length}, Vec3{0, 0, half_length}}});
  PointCloud pc;
  pc.add(Vec3{distance, 0, 0});

  const BiotSavartEvaluator eval;
  const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
  const auto potential = quasar::test::host_evaluate_A(eval, cs, pc);
  const auto gradient = quasar::test::host_evaluate_grad_B(eval, cs, pc);

  const Real inv_distance = Real{1} / distance;
  const Real half_ratio = half_length * inv_distance;
  const Real inv_radius = inv_distance
      / std::sqrt(Real{1} + half_ratio * half_ratio);
  const Real current_scale = quasar::mu0_over_4pi * current;
  const Real expected_b = ((current_scale * length) * inv_distance) * inv_radius;
  const Real expected_a = (Real{2} * current_scale) * std::asinh(half_ratio);
  const Real expected_db_dx = -expected_b * inv_distance
      * (Real{1} + Real{1} / (Real{1} + half_ratio * half_ratio));

  ASSERT_GT(expected_b, Real{0});
  ASSERT_GT(expected_a, Real{0});
  ASSERT_LT(expected_db_dx, Real{0});
  EXPECT_NEAR(field[0].y, expected_b, std::abs(expected_b) * Real{3e-14});
  EXPECT_NEAR(potential[0].z, expected_a, std::abs(expected_a) * Real{3e-14});
  EXPECT_NEAR(gradient[0].r1.x, expected_db_dx,
              std::abs(expected_db_dx) * Real{1e-12});
}

TEST(BiotSavartNumerics, FarFieldProductsDoNotOverflowBeforeFiniteFp32Result) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Real distance = Real{1e20};
  const Real current = static_cast<Real>(std::numeric_limits<float>::max());
  ConductorSystem cs;
  cs.add(Filament{"far_field_f32", current,
                  {Vec3{0, 0, -1}, Vec3{0, 0, 1}}});
  PointCloud pc;
  pc.add(Vec3{distance, 0, 0});

  const BiotSavartEvaluatorF eval;
  const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
  const auto potential = quasar::test::host_evaluate_A(eval, cs, pc);
  const auto gradient = quasar::test::host_evaluate_grad_B(eval, cs, pc);

  const Real inv_distance = Real{1} / distance;
  const Real inv_radius = inv_distance
      / std::sqrt(Real{1} + inv_distance * inv_distance);
  const Real current_scale = quasar::mu0_over_4pi * current;
  const Real expected_b = ((Real{2} * current_scale) * inv_distance) * inv_radius;
  const Real expected_a = (Real{2} * current_scale) * std::asinh(inv_distance);
  const Real expected_db_dx = -expected_b * inv_distance
      * (Real{1} + Real{1} / (Real{1} + inv_distance * inv_distance));

  ASSERT_EQ(field.size(), 1u);
  ASSERT_EQ(potential.size(), 1u);
  ASSERT_EQ(gradient.size(), 1u);
  EXPECT_NEAR(static_cast<Real>(field[0].y), expected_b,
              std::abs(expected_b) * Real{5e-5});
  EXPECT_NEAR(static_cast<Real>(potential[0].z), expected_a,
              std::abs(expected_a) * Real{5e-5});
  EXPECT_NEAR(static_cast<Real>(gradient[0].r1.x), expected_db_dx,
              std::abs(expected_db_dx) * Real{1e-4});
}

TEST(BiotSavartNumerics, OutOfRangePartialSumsMayCancelToFiniteResults) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Vec3 a{0, 0, -1};
  constexpr Vec3 b{0, 0, 1};
  const Real maximum = std::numeric_limits<Real>::max();
  const BiotSavartEvaluator eval;

  // B_y/I = 2 K/(r sqrt(1+r^2)).  Each of the first two contributions is
  // 0.6*DBL_MAX, so their ordinary partial sum overflows before the third one
  // cancels it back to the exactly representable single-segment result.
  constexpr Real radius_b = Real{1e-8};
  const Real b_per_amp = Real{2} * quasar::mu0_over_4pi
      / (radius_b * std::sqrt(Real{1} + radius_b * radius_b));
  const Real current_b = (Real{0.6} * maximum) / b_per_amp;
  PointCloud point_b;
  point_b.add(Vec3{radius_b, 0, 0});
  const auto expected_b = quasar::test::host_evaluate_B(eval, 
      repeated_segment(a, b, current_b, 1, 0), point_b);
  const auto cancelled_b = quasar::test::host_evaluate_B(eval, 
      repeated_segment(a, b, current_b, 2, 1), point_b);
  ASSERT_TRUE(std::isfinite(cancelled_b[0].y));
  EXPECT_NEAR(cancelled_b[0].y / expected_b[0].y, Real{1}, Real{2e-14});

  // The same construction for one gradient component.
  constexpr Real radius_g = Real{1e-8};
  const Real field_per_amp = Real{2} * quasar::mu0_over_4pi
      / (radius_g * std::sqrt(Real{1} + radius_g * radius_g));
  const Real grad_per_amp = field_per_amp
      * (Real{1} / radius_g
         + radius_g / (Real{1} + radius_g * radius_g));
  const Real current_g = (Real{0.6} * maximum) / grad_per_amp;
  PointCloud point_g;
  point_g.add(Vec3{radius_g, 0, 0});
  const auto expected_g = quasar::test::host_evaluate_grad_B(eval, 
      repeated_segment(a, b, current_g, 1, 0), point_g);
  const auto cancelled_g = quasar::test::host_evaluate_grad_B(eval, 
      repeated_segment(a, b, current_g, 2, 1), point_g);
  ASSERT_TRUE(std::isfinite(cancelled_g[0].r1.x));
  EXPECT_NEAR(cancelled_g[0].r1.x / expected_g[0].r1.x,
              Real{1}, Real{2e-14});

  // A single finite vector-potential contribution cannot approach DBL_MAX: its
  // geometric factor grows only logarithmically.  Repeating 8192 near-filament
  // segments nevertheless drives the positive partial sum above DBL_MAX; 8191
  // opposite currents then leave one finite contribution.
  constexpr Real radius_a = Real{1e-300};
  PointCloud point_a;
  point_a.add(Vec3{radius_a, 0, 0});
  const auto expected_a = quasar::test::host_evaluate_A(eval, 
      repeated_segment(a, b, maximum, 1, 0), point_a);
  const auto cancelled_a = quasar::test::host_evaluate_A(eval, 
      repeated_segment(a, b, maximum, 8192, 8191), point_a);
  ASSERT_TRUE(std::isfinite(cancelled_a[0].z));
  EXPECT_NEAR(cancelled_a[0].z / expected_a[0].z, Real{1}, Real{2e-10});
}

TEST(BiotSavartNumerics, ScaledReductionPreservesSourceLinearityAtUnderflow) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Vec3 a{0, 0, -1};
  constexpr Vec3 b{0, 0, 1};
  constexpr Real distance = Real{3e158};
  PointCloud point;
  point.add(Vec3{distance, 0, 0});
  const BiotSavartEvaluator eval;

  // Each 1 A contribution is below half of DBL_TRUE_MIN, while their exact
  // sum rounds to DBL_TRUE_MIN.  Rounding each segment before reduction would
  // therefore make two coincident 1 A sources disagree with one 2 A source.
  const auto separate = quasar::test::host_evaluate_B(eval, 
      repeated_segment(a, b, Real{1}, 2, 0), point);
  const auto combined = quasar::test::host_evaluate_B(eval, 
      repeated_segment(a, b, Real{2}, 1, 0), point);
  ASSERT_EQ(separate.size(), 1u);
  ASSERT_EQ(combined.size(), 1u);
  EXPECT_EQ(combined[0].y, std::numeric_limits<Real>::denorm_min());
  EXPECT_EQ(separate[0].y, combined[0].y);
}

TEST(BiotSavartNumerics, ScaledReductionRetainsSmallTermBetweenLargeCancellation) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Vec3 a{0, 0, -1};
  constexpr Vec3 b{0, 0, 1};
  constexpr Real radius = Real{1e-8};
  const Real field_per_amp = Real{2} * quasar::mu0_over_4pi
      / (radius * std::sqrt(Real{1} + radius * radius));
  const Real large_current = (Real{0.5} * std::numeric_limits<Real>::max())
                           / field_per_amp;
  PointCloud point;
  point.add(Vec3{radius, 0, 0});
  const BiotSavartEvaluator eval;

  const auto expected = quasar::test::host_evaluate_B(eval, 
      repeated_segment(a, b, Real{1}, 1, 0), point);
  const auto cancelled = quasar::test::host_evaluate_B(eval, 
      ordered_segment_currents(
          a, b, {large_current, Real{1}, -large_current}), point);
  ASSERT_EQ(cancelled.size(), 1u);
  EXPECT_EQ(cancelled[0].y, expected[0].y);
}

TEST(BiotSavartNumerics, LargeToroidReductionDoesNotExhaustExpansion) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  // Four 256-filament current sheets, each loop discretized into 256 straight
  // segments: 262,144 finite contributions at one observation point.  The
  // former unbounded-exact accumulator filled its twelve storage limbs only
  // after the first 131,072 segments and incorrectly reported status bit 2,
  // even though either half and their sum were comfortably finite.
  constexpr int sheet_filaments = 256;
  constexpr int loop_segments = 256;
  constexpr Real inner_radius = Real{0.08};
  constexpr Real outer_radius = Real{0.12};
  constexpr Real half_height = Real{0.02};
  constexpr Real spacing = (outer_radius - inner_radius) / sheet_filaments;
  constexpr Real current = Real{1500} / sheet_filaments;

  ConductorSystem conductors;
  for (int k = 0; k < sheet_filaments; ++k) {
    const Real radius = inner_radius + (static_cast<Real>(k) + Real{0.5})
                                      * spacing;
    conductors.add(circular_loop(
        Vec3{0, 0, half_height}, Vec3{0, 0, 1}, radius, loop_segments,
        current, "top_" + std::to_string(k)));
  }
  for (int k = 0; k < sheet_filaments; ++k) {
    const Real radius = inner_radius + (static_cast<Real>(k) + Real{0.5})
                                      * spacing;
    conductors.add(circular_loop(
        Vec3{0, 0, -half_height}, Vec3{0, 0, 1}, radius, loop_segments,
        current, "bottom_" + std::to_string(k)));
  }
  for (int k = 0; k < sheet_filaments; ++k) {
    const Real height = -half_height + (static_cast<Real>(k) + Real{0.5})
                                      * spacing;
    conductors.add(circular_loop(
        Vec3{0, 0, height}, Vec3{0, 0, 1}, inner_radius, loop_segments,
        -current, "inner_" + std::to_string(k)));
  }
  for (int k = 0; k < sheet_filaments; ++k) {
    const Real height = -half_height + (static_cast<Real>(k) + Real{0.5})
                                      * spacing;
    conductors.add(circular_loop(
        Vec3{0, 0, height}, Vec3{0, 0, 1}, outer_radius, loop_segments,
        -current, "outer_" + std::to_string(k)));
  }
  ASSERT_EQ(conductors.segments_soa().n_segments(), 262144u);

  PointCloud point;
  point.add(Vec3{Real{0.085}, 0, Real{-0.015}});
  const auto field = quasar::test::host_evaluate_B(BiotSavartEvaluator{}, conductors, point);

  ASSERT_EQ(field.size(), 1u);
  EXPECT_TRUE(std::isfinite(field[0].x));
  EXPECT_TRUE(std::isfinite(field[0].y));
  EXPECT_TRUE(std::isfinite(field[0].z));
  EXPECT_NEAR(field[0].x, Real{0.0240132}, Real{5e-6});
  EXPECT_NEAR(field[0].z, Real{0.0238816}, Real{5e-6});
}

TEST(BiotSavartNumerics, ExteriorCollinearDiagonalHasExactlyZeroField) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem conductors;
  conductors.add(Filament{"diagonal", Real{1e300},
                          {Vec3{0, 0, 0}, Vec3{1, 5, 0}}});
  PointCloud point;
  point.add(Vec3{2, 10, 0});

  const auto field = quasar::test::host_evaluate_B(BiotSavartEvaluator{}, conductors, point);
  ASSERT_EQ(field.size(), 1u);
  EXPECT_EQ(field[0].x, Real{0});
  EXPECT_EQ(field[0].y, Real{0});
  EXPECT_EQ(field[0].z, Real{0});
}

TEST(BiotSavartNumerics, ExtremeFiniteSegmentGradientAvoidsReciprocalOverflow) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  constexpr Real length = std::numeric_limits<Real>::max();
  constexpr Real distance = Real{1e-15};
  constexpr Real current = Real{1e-300};
  ConductorSystem conductors;
  conductors.add(Filament{"extreme", current,
                          {Vec3{0, 0, 0}, Vec3{length, 0, 0}}});
  PointCloud point;
  point.add(Vec3{0, distance, 0});

  const auto gradient =
      quasar::test::host_evaluate_grad_B(BiotSavartEvaluator{}, conductors, point);
  ASSERT_EQ(gradient.size(), 1u);
  const Real entries[] = {
      gradient[0].r0.x, gradient[0].r0.y, gradient[0].r0.z,
      gradient[0].r1.x, gradient[0].r1.y, gradient[0].r1.z,
      gradient[0].r2.x, gradient[0].r2.y, gradient[0].r2.z};
  for (const Real entry : entries) EXPECT_TRUE(std::isfinite(entry));

  // At this scale the far endpoint is indistinguishable from infinity.  The
  // endpoint derivatives are +/- K I / d^2 to far beyond binary64 relative
  // precision; compare the two nonzero Jacobian entries to that stable form.
  const Real reference = (quasar::mu0_over_4pi * current)
                       / (distance * distance);
  EXPECT_NEAR(gradient[0].r2.x / reference, Real{1}, Real{2e-14});
  EXPECT_NEAR(gradient[0].r2.y / reference, Real{-1}, Real{2e-14});
}

TEST(BiotSavartEmpty, EmptyConductorSystemYieldsZeroField) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const BiotSavartEvaluator eval;
  ConductorSystem empty_cs;

  PointCloud pc;
  pc.add(Vec3{0, 0, 1});
  pc.add(Vec3{1, 2, 3});

  const auto B = quasar::test::host_evaluate_B(eval, empty_cs, pc);
  ASSERT_EQ(B.size(), 2u);
  for (const auto& b : B) {
    EXPECT_EQ(b.x, Real{0});
    EXPECT_EQ(b.y, Real{0});
    EXPECT_EQ(b.z, Real{0});
  }
}

TEST(BiotSavartEmpty, EmptyPointCloudYieldsEmptyField) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const BiotSavartEvaluator eval;
  const auto cs = make_straight_wire();

  PointCloud empty_pc;
  const auto B = quasar::test::host_evaluate_B(eval, cs, empty_pc);
  EXPECT_EQ(B.size(), 0u);
}
