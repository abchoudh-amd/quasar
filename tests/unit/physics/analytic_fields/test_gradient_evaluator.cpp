#include "quasar/physics/analytic_fields/gradient.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

TEST(GradientEvaluator, LinearFieldMatchesAnalytic) {
  // B(r) = b0 + grad * (r - origin). Use a diagonal gradient so each component
  // scales its own coordinate: Bx = 1 + 2*x, By = -1 + 3*y, Bz = 0.5 + 4*z.
  const quasar::Vec3 b0{1.0, -1.0, 0.5};
  const quasar::Mat3x3 grad{quasar::Vec3{2, 0, 0},
                            quasar::Vec3{0, 3, 0},
                            quasar::Vec3{0, 0, 4}};
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
  // At (1,2,3): Bx = 1+2*1, By = -1+3*2, Bz = 0.5+4*3.
  EXPECT_DOUBLE_EQ(b[1].x, 3.0);
  EXPECT_DOUBLE_EQ(b[1].y, 5.0);
  EXPECT_DOUBLE_EQ(b[1].z, 12.5);
}

TEST(GradientEvaluator, GradBIsConstantAndRegistered) {
  const quasar::Mat3x3 grad{quasar::Vec3{2, 0, 0},
                            quasar::Vec3{0, 3, 0},
                            quasar::Vec3{0, 0, 4}};
  quasar::analytic_fields::GradientEvaluator eval{quasar::Vec3{0, 0, 0}, grad};

  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{5, 6, 7});

  auto gb = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(gb.size(), 1u);
  EXPECT_DOUBLE_EQ(gb[0].r0.x, 2.0);
  EXPECT_DOUBLE_EQ(gb[0].r1.y, 3.0);
  EXPECT_DOUBLE_EQ(gb[0].r2.z, 4.0);

  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance()
                  .contains("gradient"));
}
