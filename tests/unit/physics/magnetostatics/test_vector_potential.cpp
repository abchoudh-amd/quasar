// Correctness check for BiotSavartEvaluator::evaluate_A.
//
// Strategy: the Coulomb-gauge vector potential A satisfies B = curl A. For a
// small system of conductors observed at a handful of off-segment points,
// central-difference evaluate_A in each Cartesian direction, assemble the
// discrete curl, and compare against the analytic evaluate_B.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
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

TEST(VectorPotential, CurlMatchesBOnMixedSystem) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  // Conductors: a straight diagonal segment + a circular loop elsewhere.
  ConductorSystem cs;
  cs.add(generic_polyline(
      {Vec3{-0.05, 0.02, -0.10}, Vec3{0.07, -0.03, 0.08}},
      /*current_A=*/2.5, "diag"));
  cs.add(circular_loop(
      Vec3{0.10, 0.10, 0.0}, Vec3{0, 0, 1}, /*R=*/0.04, /*N=*/32,
      /*I=*/-1.2, "side_loop"));

  // Observation points comfortably off the conductors so the curl is
  // well-conditioned for finite differences.
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

  const auto B = eval.evaluate_B(cs, pc);
  ASSERT_EQ(B.size(), base_pts.size());

  // Central difference of A: 2 evaluations per direction. h ~ 1e-5 balances
  // O(h^2) truncation against O(eps/h) roundoff for double precision.
  constexpr Real h = Real{1e-5};
  const std::array<::quasar::Field<Vec3>, 3> A_plus{
      eval.evaluate_A(cs, shifted_cloud(base_pts, 0, +h)),
      eval.evaluate_A(cs, shifted_cloud(base_pts, 1, +h)),
      eval.evaluate_A(cs, shifted_cloud(base_pts, 2, +h)),
  };
  const std::array<::quasar::Field<Vec3>, 3> A_minus{
      eval.evaluate_A(cs, shifted_cloud(base_pts, 0, -h)),
      eval.evaluate_A(cs, shifted_cloud(base_pts, 1, -h)),
      eval.evaluate_A(cs, shifted_cloud(base_pts, 2, -h)),
  };

  for (std::size_t k = 0; k < base_pts.size(); ++k) {
    // dA_i/dp_j via central difference.
    auto dA = [&](int i, int j) -> Real {
      const Vec3 plus = A_plus[j][k];
      const Vec3 minus = A_minus[j][k];
      const Real pi = (i == 0) ? plus.x : (i == 1) ? plus.y : plus.z;
      const Real mi = (i == 0) ? minus.x : (i == 1) ? minus.y : minus.z;
      return (pi - mi) / (Real{2} * h);
    };
    // curl A = (dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy).
    const Vec3 curl{
        dA(2, 1) - dA(1, 2),
        dA(0, 2) - dA(2, 0),
        dA(1, 0) - dA(0, 1),
    };
    const Vec3 b = B[k];
    const Real scale = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
    ASSERT_GT(scale, Real{0}) << "B is zero at point " << k;
    EXPECT_LE(std::abs(curl.x - b.x), Real{1e-6} * scale)
        << "curl_x mismatch at point " << k;
    EXPECT_LE(std::abs(curl.y - b.y), Real{1e-6} * scale)
        << "curl_y mismatch at point " << k;
    EXPECT_LE(std::abs(curl.z - b.z), Real{1e-6} * scale)
        << "curl_z mismatch at point " << k;
  }
}

TEST(VectorPotential, IsExactlyZeroForZeroCurrent) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;
  cs.add(generic_polyline(
      {Vec3{-1, 0, 0}, Vec3{1, 0, 0}}, /*I=*/0.0, "zero_current"));
  PointCloud pc;
  pc.add(Vec3{0, 0, 1});

  const auto a = BiotSavartEvaluator{}.evaluate_A(cs, pc);
  ASSERT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0].x, Real{0});
  EXPECT_EQ(a[0].y, Real{0});
  EXPECT_EQ(a[0].z, Real{0});
}
