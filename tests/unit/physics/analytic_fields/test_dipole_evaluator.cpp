#include "quasar/physics/analytic_fields/dipole.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

TEST(DipoleEvaluator, OnAxisFieldIsFinite) {
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, 1}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 1});
  auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 1);
  EXPECT_GT(b[0].z, 0.0);
  const auto g = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(g.size(), 1u);
  const auto C = quasar::mu0_over_4pi;
  EXPECT_NEAR(g[0].r0.x,  3.0 * C, 1e-14 * C);
  EXPECT_NEAR(g[0].r1.y,  3.0 * C, 1e-14 * C);
  EXPECT_NEAR(g[0].r2.z, -6.0 * C, 1e-14 * C);
  EXPECT_NEAR(g[0].r0.x + g[0].r1.y + g[0].r2.z, 0.0, 1e-14 * C);
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance().contains("dipole"));
}

TEST(DipoleEvaluator, OriginSingularityIsReportedWithoutAZeroFieldCore) {
  const quasar::Vec3 origin{0.3, -0.4, 0.5};
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, 1}, origin};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(origin);
  EXPECT_THROW((void)eval.evaluate_B(cs, pts), std::domain_error);
  EXPECT_THROW((void)eval.evaluate_grad_B(cs, pts), std::domain_error);

  quasar::magnetostatics::PointCloud near;
  near.add(origin + quasar::Vec3{1e-16, 0, 0});
  const auto b = eval.evaluate_B(cs, near);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_TRUE(std::isfinite(b[0].x));
  EXPECT_TRUE(std::isfinite(b[0].y));
  EXPECT_TRUE(std::isfinite(b[0].z));
  EXPECT_NE(b[0].z, 0.0);
}

TEST(DipoleEvaluator, ZeroMomentHasNoSourceSingularity) {
  const quasar::Vec3 origin{0.3, -0.4, 0.5};
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{}, origin};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(origin);

  const auto b = eval.evaluate_B(cs, pts);
  const auto g = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  ASSERT_EQ(g.size(), 1u);
  EXPECT_EQ(b[0].x, 0.0);
  EXPECT_EQ(b[0].y, 0.0);
  EXPECT_EQ(b[0].z, 0.0);
  EXPECT_EQ(g[0].r0.x, 0.0);
  EXPECT_EQ(g[0].r1.y, 0.0);
  EXPECT_EQ(g[0].r2.z, 0.0);
}

TEST(DipoleEvaluator, ReportsUnrepresentableFieldInsteadOfReturningInfinity) {
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, 1}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{std::numeric_limits<quasar::Real>::denorm_min(), 0, 0});
  EXPECT_THROW((void)eval.evaluate_B(cs, pts), std::overflow_error);
  EXPECT_THROW((void)eval.evaluate_grad_B(cs, pts), std::overflow_error);
}

TEST(DipoleEvaluator, ExtremeFiniteCoordinateDifferenceDoesNotBecomeNaN) {
  const auto largest = std::numeric_limits<quasar::Real>::max();
  quasar::analytic_fields::DipoleEvaluator eval{
      quasar::Vec3{0, 0, 1}, quasar::Vec3{largest, 0, 0}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{-largest, 0, 0});
  const auto b = eval.evaluate_B(cs, pts);
  const auto g = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  ASSERT_EQ(g.size(), 1u);
  EXPECT_TRUE(std::isfinite(b[0].x));
  EXPECT_TRUE(std::isfinite(b[0].y));
  EXPECT_TRUE(std::isfinite(b[0].z));
  EXPECT_TRUE(std::isfinite(g[0].r0.x));
  EXPECT_TRUE(std::isfinite(g[0].r1.y));
  EXPECT_TRUE(std::isfinite(g[0].r2.z));
}

TEST(DipoleEvaluator, LargeMomentAndDistanceDoNotFalseUnderflowCoefficient) {
  const auto largest = std::numeric_limits<quasar::Real>::max();
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, largest}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 1.0e106});
  const auto b = eval.evaluate_B(cs, pts);
  const auto g = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  ASSERT_EQ(g.size(), 1u);
  EXPECT_TRUE(std::isfinite(b[0].z));
  EXPECT_GT(b[0].z, 0.0);
  EXPECT_TRUE(std::isfinite(g[0].r2.z));
  EXPECT_LT(g[0].r2.z, 0.0);
}

TEST(DipoleEvaluator, TinyMomentAndDistanceDoNotFalseUnderflowCoefficient) {
  // moment*mu0_over_4pi underflows when formed first in double, but the later
  // inverse-distance powers bring both B and grad(B) back into range.
  constexpr quasar::Real moment = 1.0e-320;
  constexpr quasar::Real distance = 1.0e-80;
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, moment}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, distance});

  const auto b = eval.evaluate_B(cs, pts);
  const auto g = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(b.size(), 1u);
  ASSERT_EQ(g.size(), 1u);
  EXPECT_TRUE(std::isfinite(b[0].z));
  EXPECT_GT(b[0].z, 0.0);
  EXPECT_TRUE(std::isfinite(g[0].r2.z));
  EXPECT_LT(g[0].r2.z, 0.0);

  // A logarithmic reference avoids reproducing the very underflow this test
  // targets on platforms where long double has the same range as double.
  const double log_m = std::log(moment);
  const double log_r = std::log(distance);
  const double expected_b = std::exp(std::log(2.0e-7) + log_m - 3.0 * log_r);
  const double expected_grad = -std::exp(
      std::log(6.0e-7) + log_m - 4.0 * log_r);
  EXPECT_NEAR(b[0].z, expected_b, 2.0e-12 * std::abs(expected_b));
  EXPECT_NEAR(g[0].r2.z, expected_grad,
              2.0e-12 * std::abs(expected_grad));
}
