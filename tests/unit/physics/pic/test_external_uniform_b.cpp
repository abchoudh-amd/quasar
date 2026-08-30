#include "quasar/physics/analytic_fields/uniform.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "host_evaluate.hpp"

#include <gtest/gtest.h>

TEST(PicExternalField, UniformEvaluatorSuppliesB) {
  quasar::analytic_fields::UniformEvaluator eval{quasar::Vec3{0, 0, 2}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 0});
  auto b = quasar::test::host_evaluate_B(eval, cs, pts);
  EXPECT_DOUBLE_EQ(b[0].z, 2.0);
}
