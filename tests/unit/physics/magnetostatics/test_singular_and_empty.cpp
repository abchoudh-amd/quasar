// Guards for the Biot-Savart evaluator's degenerate inputs:
//   * an observation point lying ON a filament segment (the 1/denominator
//     singularity that segment_B / segment_gradB drop to zero), and
//   * empty conductor systems / empty point clouds (the N==0 || M==0 early
//     return that must still produce a correctly-sized all-zero field).
// All cases must yield finite, exactly-zero results rather than NaN/Inf.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::PointCloud;

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

bool is_finite(const Vec3& v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}  // namespace

TEST(BiotSavartSingular, OnSegmentPointsAreFiniteZero) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const BiotSavartEvaluator eval;
  const auto cs = make_straight_wire();

  PointCloud pc;
  pc.add(Vec3{-1, 0, 0});  // segment start vertex
  pc.add(Vec3{0, 0, 0});   // segment midpoint
  pc.add(Vec3{1, 0, 0});   // segment end vertex

  const auto B = eval.evaluate_B(cs, pc);
  ASSERT_EQ(B.size(), 3u);
  for (const auto& b : B) {
    EXPECT_TRUE(is_finite(b)) << "on-segment B must be finite, not NaN/Inf";
    EXPECT_EQ(b.x, Real{0});
    EXPECT_EQ(b.y, Real{0});
    EXPECT_EQ(b.z, Real{0});
  }

  const auto G = eval.evaluate_grad_B(cs, pc);
  ASSERT_EQ(G.size(), 3u);
  for (const auto& g : G) {
    EXPECT_TRUE(is_finite(g.r0));
    EXPECT_TRUE(is_finite(g.r1));
    EXPECT_TRUE(is_finite(g.r2));
  }
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

  const auto B = eval.evaluate_B(empty_cs, pc);
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
  const auto B = eval.evaluate_B(cs, empty_pc);
  EXPECT_EQ(B.size(), 0u);
}
