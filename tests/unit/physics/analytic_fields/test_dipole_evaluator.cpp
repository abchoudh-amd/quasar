#include "quasar/physics/analytic_fields/dipole.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

TEST(DipoleEvaluator, OnAxisFieldIsFinite) {
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, 1}};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{0, 0, 1});
  auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 1);
  EXPECT_GT(b[0].z, 0.0);
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance().contains("dipole"));
}
