#include "quasar/physics/analytic_fields/gradient.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

TEST(GradientEvaluator, LinearFieldMatchesAnalytic) {
  // A physical linear magnetic field must be trace-free. Use diag(2,3,-5),
  // giving div(B)=0 while each component still has a distinct slope.
  const quasar::Vec3 b0{1.0, -1.0, 0.5};
  const quasar::Mat3x3 grad{quasar::Vec3{2, 0, 0},
                            quasar::Vec3{0, 3, 0},
                            quasar::Vec3{0, 0, -5}};
  quasar::analytic_fields::GradientEvaluator eval{b0, grad};

  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 0});
  pts.add(quasar::Vec3{1, 2, 3});

  auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 2u);
  // At origin -> b0.
  EXPECT_DOUBLE_EQ(b[0].x, 1.0);
  EXPECT_DOUBLE_EQ(b[0].y, -1.0);
  EXPECT_DOUBLE_EQ(b[0].z, 0.5);
  // At (1,2,3): Bx = 1+2*1, By = -1+3*2, Bz = 0.5-5*3.
  EXPECT_DOUBLE_EQ(b[1].x, 3.0);
  EXPECT_DOUBLE_EQ(b[1].y, 5.0);
  EXPECT_DOUBLE_EQ(b[1].z, -14.5);
}

TEST(GradientEvaluator, GradBIsConstantAndRegistered) {
  const quasar::Mat3x3 grad{quasar::Vec3{2, 0, 0},
                            quasar::Vec3{0, 3, 0},
                            quasar::Vec3{0, 0, -5}};
  quasar::analytic_fields::GradientEvaluator eval{quasar::Vec3{0, 0, 0}, grad};

  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{5, 6, 7});

  auto gb = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(gb.size(), 1u);
  EXPECT_DOUBLE_EQ(gb[0].r0.x, 2.0);
  EXPECT_DOUBLE_EQ(gb[0].r1.y, 3.0);
  EXPECT_DOUBLE_EQ(gb[0].r2.z, -5.0);

  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance()
                  .contains("gradient"));
}

TEST(GradientEvaluator, RejectsMagneticMonopoleGradient) {
  const quasar::Mat3x3 non_solenoidal{
      quasar::Vec3{2, 0, 0}, quasar::Vec3{0, 3, 0}, quasar::Vec3{0, 0, 4}};
  EXPECT_THROW((quasar::analytic_fields::GradientEvaluator{
                   quasar::Vec3{0, 0, 0}, non_solenoidal}),
               std::invalid_argument);

  quasar::analytic_fields::GradientEvaluator eval;
  quasar::numerics::EvaluatorParams p{{"grad", {2, 0, 0, 0, 3, 0, 0, 0, 4}}};
  EXPECT_THROW(eval.configure(p), std::invalid_argument);
}

TEST(GradientEvaluator, OffDiagonalScaleCannotHideNonzeroTrace) {
  const auto largest = std::numeric_limits<quasar::Real>::max();
  const quasar::Mat3x3 non_solenoidal{
      quasar::Vec3{1, largest, 0}, quasar::Vec3{}, quasar::Vec3{}};
  EXPECT_THROW((quasar::analytic_fields::GradientEvaluator{
                   quasar::Vec3{}, non_solenoidal}),
               std::invalid_argument);
}

TEST(GradientEvaluator, ExtremeFiniteCoordinateCancellationDoesNotBecomeNaN) {
  const auto largest = std::numeric_limits<quasar::Real>::max();
  const quasar::Mat3x3 grad{
      quasar::Vec3{1, -1, 0}, quasar::Vec3{}, quasar::Vec3{0, 0, -1}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{}, grad, quasar::Vec3{largest, largest, 0}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{-largest, -largest, 0});

  const auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_TRUE(std::isfinite(b[0].x));
  EXPECT_EQ(b[0].x, 0.0);
  EXPECT_EQ(b[0].y, 0.0);
  EXPECT_EQ(b[0].z, 0.0);
}

TEST(GradientEvaluator, PreservesSmallDisplacementAtLargeTranslatedOrigin) {
  const quasar::Real origin = 1.0e16;
  const quasar::Real point = std::nextafter(
      origin, std::numeric_limits<quasar::Real>::infinity());
  const quasar::Mat3x3 grad{
      quasar::Vec3{1, 0, 0}, quasar::Vec3{0, -1, 0}, quasar::Vec3{}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{}, grad, quasar::Vec3{origin, 0, 0}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{point, 0, 0});

  const auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_EQ(point - origin, quasar::Real{2});
  EXPECT_EQ(b[0].x, point - origin);
}

TEST(GradientEvaluator, LargeCancellationPreservesTinyIndependentTerm) {
  const auto largest = std::numeric_limits<quasar::Real>::max();
  const auto tiniest = std::numeric_limits<quasar::Real>::denorm_min();
  const quasar::Mat3x3 grad{
      quasar::Vec3{1, -1, 0}, quasar::Vec3{}, quasar::Vec3{0, 0, -1}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{tiniest, 0, 0}, grad,
      quasar::Vec3{largest, largest, 0}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{-largest, -largest, 0});

  const auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_EQ(b[0].x, tiniest);
}

TEST(GradientEvaluator, ReportsUnrepresentableFieldInsteadOfReturningInfinity) {
  const auto largest = std::numeric_limits<quasar::Real>::max();
  const quasar::Mat3x3 grad{
      quasar::Vec3{1, 0, 0}, quasar::Vec3{0, -1, 0}, quasar::Vec3{}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{}, grad, quasar::Vec3{-largest, 0, 0}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{largest, 0, 0});
  EXPECT_THROW((void)eval.evaluate_B(cs, pts), std::overflow_error);
}
