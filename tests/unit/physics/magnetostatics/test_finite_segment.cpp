#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::mu0_over_4pi;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::BiotSavartEvaluatorF;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::IFieldEvaluator;
using ::quasar::magnetostatics::PointCloud;

namespace {

// Single straight segment of length L centered on the origin, oriented along
// +z, carrying current I.
ConductorSystem make_finite_segment_system(Real L, Real I) {
  ConductorSystem cs;
  cs.add({/*name=*/"segment",
          /*current_A=*/I,
          /*points=*/{Vec3{Real{0}, Real{0}, -L / Real{2}},
                      Vec3{Real{0}, Real{0}, +L / Real{2}}}});
  return cs;
}

// Closed-form B_y at observation point (d, 0, 0) for a symmetric straight
// segment of half-length H = L/2 oriented along +z, current I:
//
//   B_y = (mu0 I / 4 pi) * 2 H / ( d * sqrt(d^2 + H^2) )
//
// All other components of B are zero by symmetry.
Real finite_segment_By_ref(Real d, Real H, Real I) {
  return mu0_over_4pi * I * Real{2} * H / (d * std::sqrt(d * d + H * H));
}

constexpr Real kLength  = Real{2.0};
constexpr Real kCurrent = Real{1.0};
constexpr Real kHalfLen = kLength / Real{2};
constexpr Real kRelTol  = Real{1e-10};
constexpr Real kAbsTol  = Real{1e-14};

}  // namespace

TEST(FiniteSegment, NearWireOffFilamentIsNotSuppressed) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const ConductorSystem cs = make_finite_segment_system(kLength, kCurrent);
  constexpr Real d = Real{3e-8};
  PointCloud pc;
  pc.add(Vec3{d, 0, 0});
  const Real ref = finite_segment_By_ref(d, kHalfLen, kCurrent);

  const auto Bd = BiotSavartEvaluator{}.evaluate_B(cs, pc);
  const auto Bf = BiotSavartEvaluatorF{}.evaluate_B(cs, pc);
  ASSERT_EQ(Bd.size(), 1u);
  ASSERT_EQ(Bf.size(), 1u);
  EXPECT_NEAR(Bd[0].y, ref, Real{2e-12} * std::abs(ref));
  EXPECT_NEAR(static_cast<Real>(Bf[0].y), ref, Real{2e-5} * std::abs(ref));
  EXPECT_NE(Bd[0].y, Real{0});
  EXPECT_NE(Bf[0].y, 0.0f);
}

TEST(FiniteSegment, MatchesClosedFormViaDirectCtor) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const ConductorSystem cs = make_finite_segment_system(kLength, kCurrent);
  const BiotSavartEvaluator eval;

  for (const Real d : {Real{0.1}, Real{1.0}, Real{10.0}}) {
    PointCloud pc;
    pc.add(Vec3{d, Real{0}, Real{0}});

    const auto field = eval.evaluate_B(cs, pc);
    ASSERT_EQ(field.size(), 1u);

    const Real B_y_ref = finite_segment_By_ref(d, kHalfLen, kCurrent);

    EXPECT_LT(std::abs(field[0].x), kAbsTol)            << "d=" << d;
    EXPECT_LT(std::abs(field[0].z), kAbsTol)            << "d=" << d;
    EXPECT_NEAR(field[0].y, B_y_ref, kRelTol * std::abs(B_y_ref))
        << "d=" << d << " ref=" << B_y_ref << " got=" << field[0].y;
  }
}

TEST(FiniteSegment, MatchesClosedFormViaRegistry) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  auto& reg = ::quasar::Registry<IFieldEvaluator>::instance();
  ASSERT_TRUE(reg.contains("biot_savart"));

  auto eval = reg.create("biot_savart");
  ASSERT_NE(eval, nullptr);

  const ConductorSystem cs = make_finite_segment_system(kLength, kCurrent);

  for (const Real d : {Real{0.1}, Real{1.0}, Real{10.0}}) {
    PointCloud pc;
    pc.add(Vec3{d, Real{0}, Real{0}});

    const auto field = eval->evaluate_B(cs, pc);
    ASSERT_EQ(field.size(), 1u);

    const Real B_y_ref = finite_segment_By_ref(d, kHalfLen, kCurrent);

    EXPECT_LT(std::abs(field[0].x), kAbsTol)            << "d=" << d;
    EXPECT_LT(std::abs(field[0].z), kAbsTol)            << "d=" << d;
    EXPECT_NEAR(field[0].y, B_y_ref, kRelTol * std::abs(B_y_ref))
        << "d=" << d << " ref=" << B_y_ref << " got=" << field[0].y;
  }
}
