#include "quasar/physics/analytic_fields/uniform.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "host_evaluate.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

TEST(UniformEvaluator, ReturnsConstantField) {
  quasar::analytic_fields::UniformEvaluator eval{quasar::Vec3{1, 2, 3}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 0});
  pts.add(quasar::Vec3{1, 0, 0});
  auto b = quasar::test::host_evaluate_B(eval, cs, pts);
  ASSERT_EQ(b.size(), 2);
  EXPECT_DOUBLE_EQ(b[0].x, 1.0);
  EXPECT_DOUBLE_EQ(b[1].z, 3.0);
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance().contains("uniform"));
}

TEST(UniformEvaluator, RejectsNonFiniteFieldsAtomically) {
  const auto nan = std::numeric_limits<quasar::Real>::quiet_NaN();
  EXPECT_THROW((quasar::analytic_fields::UniformEvaluator{
                   quasar::Vec3{nan, 0, 0}}),
               std::invalid_argument);

  quasar::analytic_fields::UniformEvaluator eval{quasar::Vec3{1, 2, 3}};
  quasar::numerics::EvaluatorParams params{{"b0", {4, 5, 6}},
                                            {"e0", {0, nan, 0}}};
  EXPECT_THROW(eval.configure(params), std::invalid_argument);

  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{});
  const auto b = quasar::test::host_evaluate_B(eval, cs, pts);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_EQ(b[0].x, 1.0);
  EXPECT_EQ(b[0].y, 2.0);
  EXPECT_EQ(b[0].z, 3.0);
}

TEST(UniformEvaluator, RejectsUnknownConfigurationKeys) {
  quasar::analytic_fields::UniformEvaluator eval;
  EXPECT_THROW(eval.configure({{"typo", {1, 2, 3}}}), std::invalid_argument);
}
