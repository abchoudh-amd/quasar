// A pair of coaxial circular loops separated by exactly one radius
// is the Helmholtz configuration. On the symmetry axis at the midpoint:
//
//   B_z(0)        =  (4/5)^(3/2) * mu0 * I / R
//   dB_z/dz       =  0          (linear z-flatness)
//   d^2 B_z/dz^2  =  0          (quadratic z-flatness)
//
// The first derivative is read directly from the analytic Jacobian; the
// second derivative is taken as a finite-difference of B_z (3-point stencil)
// to keep this test independent from any second-order gradient routine.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "host_evaluate.hpp"

#include <gtest/gtest.h>

#include <cmath>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::mu0;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::PointCloud;

namespace {

constexpr Real kR = Real{0.1};
constexpr Real kI = Real{1.0};
constexpr int  kN = 256;

ConductorSystem make_helmholtz_pair() {
  ConductorSystem cs;
  cs.add(circular_loop(/*center=*/Vec3{0, 0, -kR / Real{2}},
                       /*axis=*/Vec3{0, 0, 1},
                       /*radius_m=*/kR, /*n_segments=*/kN, /*current_A=*/kI,
                       /*name=*/"lower"));
  cs.add(circular_loop(/*center=*/Vec3{0, 0, +kR / Real{2}},
                       /*axis=*/Vec3{0, 0, 1},
                       /*radius_m=*/kR, /*n_segments=*/kN, /*current_A=*/kI,
                       /*name=*/"upper"));
  return cs;
}

Real B_z_at(const BiotSavartEvaluator& eval, const ConductorSystem& cs,
            Real z) {
  PointCloud pc;
  pc.add(Vec3{0, 0, z});
  return quasar::test::host_evaluate_B(eval, cs, pc)[0].z;
}

}  // namespace

TEST(HelmholtzPair, FirstAndSecondDerivativesOfBzVanishAtMidpoint) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;
  const auto cs = make_helmholtz_pair();

  // (a) Analytic value at the midpoint.
  const Real B_z_ref = mu0 * kI / kR * std::pow(Real{4} / Real{5}, Real{1.5});

  PointCloud pc_mid;
  pc_mid.add(Vec3{0, 0, 0});

  const auto B    = quasar::test::host_evaluate_B(eval, cs, pc_mid);
  const auto gradB = quasar::test::host_evaluate_grad_B(eval, cs, pc_mid);
  ASSERT_EQ(B.size(), 1u);
  ASSERT_EQ(gradB.size(), 1u);

  EXPECT_NEAR(B[0].z, B_z_ref, Real{1e-4} * B_z_ref)
      << "midpoint B_z=" << B[0].z << " vs ref " << B_z_ref;

  // (b) dB_z/dz at the midpoint comes from row 2, col 2 of the Jacobian.
  // Normalize by the natural scale B_z(0) / R so the bound is dimensionless.
  const Real natural_scale = B_z_ref / kR;
  EXPECT_LT(std::abs(gradB[0].r2.z), Real{1e-6} * natural_scale)
      << "dB_z/dz at midpoint = " << gradB[0].r2.z;

  // x- and y-derivatives of B_z must also vanish on axis by symmetry.
  EXPECT_LT(std::abs(gradB[0].r2.x), Real{1e-6} * natural_scale);
  EXPECT_LT(std::abs(gradB[0].r2.y), Real{1e-6} * natural_scale);

  // (c) d^2 B_z / dz^2 at the midpoint via 3-point finite difference on B_z.
  // Step h: 0.001 m is well into the quartic-flatness region.
  const Real h    = Real{1e-3};
  const Real Bzm  = B_z_at(eval, cs, -h);
  const Real Bz0  = B[0].z;
  const Real Bzp  = B_z_at(eval, cs, +h);
  const Real d2Bz = (Bzp - Real{2} * Bz0 + Bzm) / (h * h);

  // Helmholtz makes d^2 B_z/dz^2 vanish; the residual is the FD truncation
  // (h^2/12) * B_z''''(0), bounded by B_z(0)/R^2 in scale.
  const Real second_scale = B_z_ref / (kR * kR);
  EXPECT_LT(std::abs(d2Bz), Real{1e-3} * second_scale)
      << "d^2 B_z / dz^2 at midpoint = " << d2Bz
      << " (scale " << second_scale << ")";
}
