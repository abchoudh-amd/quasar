// The fp32 instantiation of the Biot-Savart kernels must agree with
// the fp64 instantiation at single-precision tolerance on the analytical
// test set. Both kernels share the same closed-form (Hanson-Hirshman) C++
// template, so any disagreement above ~few * single-precision rounding
// reflects per-segment fp32 evaluation/representation error or a real numeric
// bug. B, A, and all nine Jacobian components use exponent-scaled expansion
// accumulation.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <utility>

using ::quasar::Mat3x3;
using ::quasar::Mat3x3f;
using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::Vec3f;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::BiotSavartEvaluatorF;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::generic_polyline;
using ::quasar::magnetostatics::helix;
using ::quasar::magnetostatics::PointCloud;

namespace {

// Tolerance covers single-precision rounding in each segment's geometry and
// closed-form field, plus final fp32 representation. Segment sums themselves
// use a non-overlapping scaled expansion, so the bound does not assume an
// ordinary O(N eps) reduction. The Jacobian formula has more per-segment
// operations and therefore uses the wider of the two empirical bounds.
constexpr Real kAbsToB     = Real{5e-6};
constexpr Real kAbsToGradB = Real{5e-5};

Real frob_sq(const Mat3x3& m) noexcept {
  return m.r0.x * m.r0.x + m.r0.y * m.r0.y + m.r0.z * m.r0.z
       + m.r1.x * m.r1.x + m.r1.y * m.r1.y + m.r1.z * m.r1.z
       + m.r2.x * m.r2.x + m.r2.y * m.r2.y + m.r2.z * m.r2.z;
}

Real frob_sq_diff(const Mat3x3& a, const Mat3x3f& b) noexcept {
  Mat3x3 d;
  d.r0 = Vec3{a.r0.x - static_cast<Real>(b.r0.x),
              a.r0.y - static_cast<Real>(b.r0.y),
              a.r0.z - static_cast<Real>(b.r0.z)};
  d.r1 = Vec3{a.r1.x - static_cast<Real>(b.r1.x),
              a.r1.y - static_cast<Real>(b.r1.y),
              a.r1.z - static_cast<Real>(b.r1.z)};
  d.r2 = Vec3{a.r2.x - static_cast<Real>(b.r2.x),
              a.r2.y - static_cast<Real>(b.r2.y),
              a.r2.z - static_cast<Real>(b.r2.z)};
  return frob_sq(d);
}

ConductorSystem make_mixed_system() {
  ConductorSystem cs;
  cs.add(circular_loop(/*center=*/Vec3{0, 0, 0},
                       /*axis=*/Vec3{0, 0, 1},
                       /*radius_m=*/Real{0.05}, /*n_segments=*/128,
                       /*current_A=*/Real{1.0}, /*name=*/"loop_a"));
  cs.add(circular_loop(/*center=*/Vec3{0.10, 0, 0.02},
                       /*axis=*/Vec3{0, 0, 1},
                       /*radius_m=*/Real{0.03}, /*n_segments=*/64,
                       /*current_A=*/Real{-0.5}, /*name=*/"loop_b"));
  cs.add(helix(/*center=*/Vec3{-0.10, 0, 0},
               /*axis=*/Vec3{0, 0, 1},
               /*radius_m=*/Real{0.02}, /*pitch_m=*/Real{0.01},
               /*n_turns=*/3, /*n_segments_per_turn=*/24,
               /*current_A=*/Real{0.25}, /*name=*/"helix_c"));
  cs.add(generic_polyline(
      {Vec3{0.05, 0.02, -0.05}, Vec3{0.06, -0.01, 0.04}},
      /*current_A=*/Real{0.75}, /*name=*/"diag"));
  return cs;
}

PointCloud make_observation_cloud() {
  PointCloud pc;
  pc.add(Vec3{Real{0.00},  Real{0.00},  Real{0.00}});
  pc.add(Vec3{Real{0.03},  Real{0.00},  Real{0.04}});
  pc.add(Vec3{Real{-0.02}, Real{0.02},  Real{0.03}});
  pc.add(Vec3{Real{0.04},  Real{0.05},  Real{-0.02}});
  pc.add(Vec3{Real{0.07},  Real{0.00},  Real{0.00}});
  pc.add(Vec3{Real{0.00},  Real{-0.06}, Real{0.05}});
  pc.add(Vec3{Real{-0.04}, Real{0.00},  Real{0.00}});
  pc.add(Vec3{Real{0.00},  Real{0.00},  Real{-0.05}});
  return pc;
}

}  // namespace

TEST(Fp32VsFp64, B_FieldAgreesAtSinglePrecisionTolerance) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const ConductorSystem cs = make_mixed_system();
  const PointCloud pc       = make_observation_cloud();

  const auto B_d = BiotSavartEvaluator{}.evaluate_B(cs, pc);
  const auto B_f = BiotSavartEvaluatorF{}.evaluate_B(cs, pc);
  ASSERT_EQ(B_d.size(), B_f.size());
  ASSERT_EQ(B_d.size(), pc.size());

  for (std::size_t k = 0; k < B_d.size(); ++k) {
    const Real ref = std::sqrt(B_d[k].x * B_d[k].x
                                + B_d[k].y * B_d[k].y
                                + B_d[k].z * B_d[k].z);
    ASSERT_GT(ref, Real{0}) << "fp64 |B| at point " << k << " is zero";

    const Real dx = B_d[k].x - static_cast<Real>(B_f[k].x);
    const Real dy = B_d[k].y - static_cast<Real>(B_f[k].y);
    const Real dz = B_d[k].z - static_cast<Real>(B_f[k].z);
    const Real diff = std::sqrt(dx * dx + dy * dy + dz * dz);

    EXPECT_LE(diff / ref, kAbsToB)
        << "point " << k << " disagreement = " << diff
        << ", |B_fp64| = " << ref
        << ", relative = " << (diff / ref);
  }
}

TEST(Fp32VsFp64, GradBAgreesAtSinglePrecisionTolerance) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const ConductorSystem cs = make_mixed_system();
  const PointCloud pc       = make_observation_cloud();

  const auto G_d = BiotSavartEvaluator{}.evaluate_grad_B(cs, pc);
  const auto G_f = BiotSavartEvaluatorF{}.evaluate_grad_B(cs, pc);
  ASSERT_EQ(G_d.size(), G_f.size());
  ASSERT_EQ(G_d.size(), pc.size());

  for (std::size_t k = 0; k < G_d.size(); ++k) {
    const Real ref_sq  = frob_sq(G_d[k]);
    const Real diff_sq = frob_sq_diff(G_d[k], G_f[k]);
    ASSERT_GT(ref_sq, Real{0}) << "fp64 gradB Frobenius norm at point "
                                << k << " is zero";

    const Real ref_norm  = std::sqrt(ref_sq);
    const Real diff_norm = std::sqrt(diff_sq);

    EXPECT_LE(diff_norm / ref_norm, kAbsToGradB)
        << "point " << k << " Frobenius disagreement = " << diff_norm
        << ", |G_fp64|_F = " << ref_norm
        << ", relative = " << (diff_norm / ref_norm);
  }
}

TEST(Fp32VsFp64, EmptySystemReturnsZeroForBothPrecisions) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  ConductorSystem cs;  // empty
  PointCloud pc;
  pc.add(Vec3{0, 0, 0});

  const auto B_d = BiotSavartEvaluator{}.evaluate_B(cs, pc);
  const auto B_f = BiotSavartEvaluatorF{}.evaluate_B(cs, pc);

  ASSERT_EQ(B_d.size(), 1u);
  ASSERT_EQ(B_f.size(), 1u);
  EXPECT_EQ(B_d[0].x, Real{0}); EXPECT_EQ(B_d[0].y, Real{0}); EXPECT_EQ(B_d[0].z, Real{0});
  EXPECT_EQ(B_f[0].x, 0.0f);     EXPECT_EQ(B_f[0].y, 0.0f);     EXPECT_EQ(B_f[0].z, 0.0f);
}

TEST(Fp32VsFp64, GradientRemainsFiniteAcrossSimilarityScales) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  for (const Real s : {Real{1e-7}, Real{1e-6}, Real{1e5}}) {
    ConductorSystem cs;
    cs.add(generic_polyline({Vec3{-s, 0, 0}, Vec3{s, 0, 0}}, Real{1}, "scaled"));
    PointCloud pc;
    pc.add(Vec3{0, s, 0});

    const auto gd = BiotSavartEvaluator{}.evaluate_grad_B(cs, pc);
    const auto gf = BiotSavartEvaluatorF{}.evaluate_grad_B(cs, pc);
    ASSERT_EQ(gd.size(), 1u);
    ASSERT_EQ(gf.size(), 1u);
    const Real ref = std::sqrt(frob_sq(gd[0]));
    const Real diff = std::sqrt(frob_sq_diff(gd[0], gf[0]));
    EXPECT_TRUE(std::isfinite(gf[0].r0.x));
    EXPECT_TRUE(std::isfinite(gf[0].r0.y));
    EXPECT_TRUE(std::isfinite(gf[0].r0.z));
    EXPECT_TRUE(std::isfinite(gf[0].r1.x));
    EXPECT_TRUE(std::isfinite(gf[0].r1.y));
    EXPECT_TRUE(std::isfinite(gf[0].r1.z));
    EXPECT_TRUE(std::isfinite(gf[0].r2.x));
    EXPECT_TRUE(std::isfinite(gf[0].r2.y));
    EXPECT_TRUE(std::isfinite(gf[0].r2.z));
    EXPECT_LE(diff / ref, Real{5e-5}) << "similarity scale=" << s;
  }
}

TEST(Fp32VsFp64, CommonOriginPreservesRigidTranslation) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  auto evaluate = [](Real offset) {
    ConductorSystem cs;
    cs.add(generic_polyline(
        {Vec3{offset - 1, 0, 0}, Vec3{offset + 1, 0, 0}}, Real{1}, "translated"));
    PointCloud pc;
    pc.add(Vec3{offset, 1, 0});
    BiotSavartEvaluatorF eval;
    return std::pair{eval.evaluate_B(cs, pc), eval.evaluate_grad_B(cs, pc)};
  };

  const auto [b0, g0] = evaluate(Real{0});
  const auto [bt, gt] = evaluate(Real{1e8});
  ASSERT_EQ(b0.size(), 1u);
  ASSERT_EQ(bt.size(), 1u);
  EXPECT_FLOAT_EQ(bt[0].x, b0[0].x);
  EXPECT_FLOAT_EQ(bt[0].y, b0[0].y);
  EXPECT_FLOAT_EQ(bt[0].z, b0[0].z);
  EXPECT_FLOAT_EQ(gt[0].r0.x, g0[0].r0.x);
  EXPECT_FLOAT_EQ(gt[0].r0.y, g0[0].r0.y);
  EXPECT_FLOAT_EQ(gt[0].r0.z, g0[0].r0.z);
  EXPECT_FLOAT_EQ(gt[0].r1.x, g0[0].r1.x);
  EXPECT_FLOAT_EQ(gt[0].r1.y, g0[0].r1.y);
  EXPECT_FLOAT_EQ(gt[0].r1.z, g0[0].r1.z);
  EXPECT_FLOAT_EQ(gt[0].r2.x, g0[0].r2.x);
  EXPECT_FLOAT_EQ(gt[0].r2.y, g0[0].r2.y);
  EXPECT_FLOAT_EQ(gt[0].r2.z, g0[0].r2.z);
}
