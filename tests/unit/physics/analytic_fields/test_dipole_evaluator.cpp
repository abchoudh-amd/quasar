#include "quasar/physics/analytic_fields/dipole.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>

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

TEST(DipoleEvaluator, OriginSingularityIsSuppressed) {
  // The 1/r^5 dipole field diverges at the moment's origin; evaluate_B guards the
  // singularity by returning exactly zero within kEps of the origin (rather than
  // NaN/Inf). Place a point at the origin and one within the guard radius.
  const quasar::Vec3 origin{0.3, -0.4, 0.5};
  quasar::analytic_fields::DipoleEvaluator eval{quasar::Vec3{0, 0, 1}, origin};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(origin);                                              // exactly at origin
  pts.add(origin + quasar::Vec3{1e-16, 0, 0});                  // within sqrt(kEps)
  const auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 2u);
  for (const auto& v : b) {
    EXPECT_TRUE(std::isfinite(v.x));
    EXPECT_TRUE(std::isfinite(v.y));
    EXPECT_TRUE(std::isfinite(v.z));
    EXPECT_EQ(v.x, 0.0);
    EXPECT_EQ(v.y, 0.0);
    EXPECT_EQ(v.z, 0.0);
  }
}
