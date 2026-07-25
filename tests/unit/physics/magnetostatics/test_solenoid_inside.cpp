// A long thin solenoid produces a nearly uniform axial field
// inside (B_z ~ mu0 * n * I) and a field that drops as 1/r^3 far outside.
//
// For a finite solenoid of length L and radius R (with R << L) modelled as
// a continuous surface current K = nI, the on-axis B at the midpoint is
//
//   B_z(0) = mu0 * n * I / sqrt(1 + (2R/L)^2)
//
// which is ~mu0*n*I as 2R/L -> 0. Our solenoid() generator produces a
// discrete N-turn helix; for many turns (N = 200 here) the leading
// discretization correction is small (well below our 2% tolerance).

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::mu0;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::solenoid;

namespace {

constexpr Real kRadius   = Real{0.02};
constexpr Real kLength   = Real{0.5};
constexpr int  kTurns    = 200;
constexpr int  kPerTurn  = 24;
constexpr Real kCurrent  = Real{1.0};

ConductorSystem make_long_solenoid() {
  ConductorSystem cs;
  cs.add(solenoid(/*center=*/Vec3{0, 0, 0},
                  /*axis=*/Vec3{0, 0, 1},
                  /*radius_m=*/kRadius, /*length_m=*/kLength,
                  /*n_turns=*/kTurns, /*n_segments_per_turn=*/kPerTurn,
                  /*current_A=*/kCurrent,
                  /*name=*/"long_solenoid"));
  return cs;
}

}  // namespace

TEST(SolenoidInside, MidpointFieldMatchesMu0nI) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;
  const auto cs = make_long_solenoid();

  PointCloud pc;
  pc.add(Vec3{0, 0, 0});

  const auto B = eval.evaluate_B(cs, pc);
  ASSERT_EQ(B.size(), 1u);

  // Surface-current reference for a finite solenoid of length L, radius R.
  const Real n = static_cast<Real>(kTurns) / kLength;  // turns / metre
  const Real two_R_over_L = Real{2} * kRadius / kLength;
  const Real B_z_ref = mu0 * n * kCurrent
                       / std::sqrt(Real{1} + two_R_over_L * two_R_over_L);

  // Tolerance: 2% allows for both the finite-aspect correction (already in
  // B_z_ref) and the discrete-helix discretization.
  EXPECT_NEAR(B[0].z, B_z_ref, Real{2e-2} * B_z_ref)
      << "midpoint B_z=" << B[0].z << " vs ref " << B_z_ref;

  // Off-axis components on the symmetry axis must vanish.
  EXPECT_LT(std::abs(B[0].x), Real{1e-3} * std::abs(B_z_ref));
  EXPECT_LT(std::abs(B[0].y), Real{1e-3} * std::abs(B_z_ref));
}

TEST(SolenoidInside, FarOutsideFieldIsMuchWeakerThanInside) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;
  const auto cs = make_long_solenoid();

  PointCloud pc;
  // 5x the solenoid length above the upper end -> well into the dipole tail.
  pc.add(Vec3{0, 0, kLength * Real{2.5}});

  const auto B = eval.evaluate_B(cs, pc);
  ASSERT_EQ(B.size(), 1u);

  const Real n         = static_cast<Real>(kTurns) / kLength;
  const Real B_inside  = mu0 * n * kCurrent;
  const Real B_far_mag = std::sqrt(B[0].x * B[0].x
                                   + B[0].y * B[0].y
                                   + B[0].z * B[0].z);

  // At z = 2.5 L the magnetic-dipole tail is below 1% of mu0*n*I.
  EXPECT_LT(B_far_mag, Real{1e-2} * B_inside)
      << "|B| at z=2.5L is " << B_far_mag
      << ", inside-field reference " << B_inside;
}
