// Analytical correctness checks for segment_gradB.
//
// Strategy: for a small system of conductors observed at a handful of
// random-but-fixed off-segment points, compare the analytic
// BiotSavartEvaluator::evaluate_grad_B against central-difference of
// evaluate_B in each of the three Cartesian directions.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::generic_polyline;
using ::quasar::magnetostatics::PointCloud;

namespace {

PointCloud shifted_cloud(const std::vector<Vec3>& base, int dir, Real delta) {
  PointCloud pc;
  for (Vec3 p : base) {
    if (dir == 0) p.x += delta;
    if (dir == 1) p.y += delta;
    if (dir == 2) p.z += delta;
    pc.add(p);
  }
  return pc;
}

}  // namespace

TEST(SegmentGradient, MatchesFiniteDifferenceAgainstBOnMixedSystem) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  // Conductors: a straight diagonal segment + a circular loop somewhere else.
  ConductorSystem cs;
  cs.add(generic_polyline(
      {Vec3{-0.05, 0.02, -0.10}, Vec3{0.07, -0.03, 0.08}},
      /*current_A=*/2.5, "diag"));
  cs.add(circular_loop(
      Vec3{0.10, 0.10, 0.0}, Vec3{0, 0, 1}, /*R=*/0.04, /*N=*/32,
      /*I=*/-1.2, "side_loop"));

  // Observation points that are comfortably off the conductors so the
  // gradient is well-conditioned for finite differences.
  const std::vector<Vec3> base_pts = {
      Vec3{0.00, 0.00, 0.05},
      Vec3{0.01, -0.02, 0.10},
      Vec3{-0.03, 0.04, -0.07},
      Vec3{0.05, 0.05, 0.02},
      Vec3{0.08, -0.06, 0.00},
  };
  PointCloud pc;
  for (Vec3 p : base_pts) pc.add(p);

  const BiotSavartEvaluator eval;

  // Analytic Jacobians.
  const auto gradB = eval.evaluate_grad_B(cs, pc);
  ASSERT_EQ(gradB.size(), base_pts.size());

  // Central difference: 2 evaluations per direction. h chosen so that
  // truncation O(h^2) and roundoff O(eps/h) balance near sqrt(eps) ~ 1e-8;
  // 1e-5 is well within the comfortable interior for double precision.
  constexpr Real h = Real{1e-5};

  std::array<::quasar::Field<Vec3>, 3> B_plus{
      eval.evaluate_B(cs, shifted_cloud(base_pts, 0, +h)),
      eval.evaluate_B(cs, shifted_cloud(base_pts, 1, +h)),
      eval.evaluate_B(cs, shifted_cloud(base_pts, 2, +h)),
  };
  std::array<::quasar::Field<Vec3>, 3> B_minus{
      eval.evaluate_B(cs, shifted_cloud(base_pts, 0, -h)),
      eval.evaluate_B(cs, shifted_cloud(base_pts, 1, -h)),
      eval.evaluate_B(cs, shifted_cloud(base_pts, 2, -h)),
  };

  for (std::size_t k = 0; k < base_pts.size(); ++k) {
    const auto g = gradB[k];
    // Build a reference Jacobian via central differences.
    // ref[i][j] = (B_i(p + h e_j) - B_i(p - h e_j)) / (2 h).
    const Real ref[3][3] = {
        // dB_x/d(p_x, p_y, p_z)
        {(B_plus[0][k].x - B_minus[0][k].x) / (Real{2} * h),
         (B_plus[1][k].x - B_minus[1][k].x) / (Real{2} * h),
         (B_plus[2][k].x - B_minus[2][k].x) / (Real{2} * h)},
        // dB_y/d(...)
        {(B_plus[0][k].y - B_minus[0][k].y) / (Real{2} * h),
         (B_plus[1][k].y - B_minus[1][k].y) / (Real{2} * h),
         (B_plus[2][k].y - B_minus[2][k].y) / (Real{2} * h)},
        // dB_z/d(...)
        {(B_plus[0][k].z - B_minus[0][k].z) / (Real{2} * h),
         (B_plus[1][k].z - B_minus[1][k].z) / (Real{2} * h),
         (B_plus[2][k].z - B_minus[2][k].z) / (Real{2} * h)},
    };
    const Real analytic[3][3] = {
        {g.r0.x, g.r0.y, g.r0.z},
        {g.r1.x, g.r1.y, g.r1.z},
        {g.r2.x, g.r2.y, g.r2.z},
    };

    // Frobenius norm of the reference is the natural scale here.
    Real frob_ref_sq = Real{0};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) frob_ref_sq += ref[i][j] * ref[i][j];
    const Real frob_ref = std::sqrt(frob_ref_sq);
    ASSERT_GT(frob_ref, Real{0}) << "central-difference reference is zero at point " << k;

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        // FD truncation O(h^2) ~ 1e-10 relative; allow 1e-6 absolute
        // (relative to Frobenius norm) as a tight but safe bound.
        EXPECT_LE(std::abs(analytic[i][j] - ref[i][j]), Real{1e-6} * frob_ref)
            << "mismatch at point " << k << " entry (" << i << "," << j << ") "
            << "analytic=" << analytic[i][j] << " fd=" << ref[i][j];
      }
    }
  }
}

TEST(SegmentGradient, IsExactlyZeroForEmptySystem) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;
  cs.add(generic_polyline(
      {Vec3{-1, 0, 0}, Vec3{1, 0, 0}}, /*I=*/0.0, "zero_current"));
  PointCloud pc;
  pc.add(Vec3{0, 0, 1});

  const BiotSavartEvaluator eval;
  const auto g = eval.evaluate_grad_B(cs, pc);
  ASSERT_EQ(g.size(), 1u);
  // Every entry should be exactly zero (coefficient is 0 from zero current).
  EXPECT_EQ(g[0].r0.x, Real{0}); EXPECT_EQ(g[0].r0.y, Real{0}); EXPECT_EQ(g[0].r0.z, Real{0});
  EXPECT_EQ(g[0].r1.x, Real{0}); EXPECT_EQ(g[0].r1.y, Real{0}); EXPECT_EQ(g[0].r1.z, Real{0});
  EXPECT_EQ(g[0].r2.x, Real{0}); EXPECT_EQ(g[0].r2.y, Real{0}); EXPECT_EQ(g[0].r2.z, Real{0});
}

TEST(SegmentGradient, JacobianRowSatisfiesDivBEqualsZero) {
  // Maxwell's div B = 0 means the trace of the Jacobian must vanish at any
  // observation point outside the wire. Tests are a useful cross-check on
  // the analytic Jacobian.
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  ConductorSystem cs;
  cs.add(circular_loop(Vec3{0, 0, 0}, Vec3{0, 0, 1}, /*R=*/0.07,
                       /*N=*/128, /*I=*/1.0));
  PointCloud pc;
  pc.add(Vec3{0.02, -0.01, 0.03});
  pc.add(Vec3{0.05, 0.02, -0.04});
  pc.add(Vec3{0.00, 0.00, 0.05});

  const auto gradB = BiotSavartEvaluator{}.evaluate_grad_B(cs, pc);
  ASSERT_EQ(gradB.size(), 3u);

  for (std::size_t k = 0; k < gradB.size(); ++k) {
    const Real trace = gradB[k].r0.x + gradB[k].r1.y + gradB[k].r2.z;
    Real frob_sq = 0;
    for (Vec3 row : {gradB[k].r0, gradB[k].r1, gradB[k].r2}) {
      frob_sq += row.x * row.x + row.y * row.y + row.z * row.z;
    }
    const Real scale = std::sqrt(frob_sq);
    EXPECT_LE(std::abs(trace), Real{1e-6} * scale)
        << "div B at point " << k << " was " << trace << " (scale=" << scale << ")";
  }
}
