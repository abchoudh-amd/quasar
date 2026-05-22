#include "quasar/physics/analytic_fields/uniform.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

TEST(UniformEvaluator, ReturnsConstantField) {
  quasar::analytic_fields::UniformEvaluator eval{quasar::Vec3{1, 2, 3}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 0});
  pts.add(quasar::Vec3{1, 0, 0});
  auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 2);
  EXPECT_DOUBLE_EQ(b[0].x, 1.0);
  EXPECT_DOUBLE_EQ(b[1].z, 3.0);
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance().contains("uniform"));
}
