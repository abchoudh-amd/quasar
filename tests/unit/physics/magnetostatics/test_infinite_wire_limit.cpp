// A single straight current-carrying segment of length L approaches
// the infinite-wire azimuthal field
//
//     B_phi(d) = mu0 * I / (2 * pi * d)
//
// at fixed perpendicular distance d as L/d -> infinity.
//
// For a symmetric segment along +z of half-length H observed at (d, 0, 0)
// the closed-form B_y is mu0*I/(4 pi d) * 2 H / sqrt(d^2 + H^2). The
// fractional gap to the infinite-wire limit is
//
//     1 - B_y(H, d) / B_phi_inf(d)  =  1 - H / sqrt(d^2 + H^2)
//                                   ~  d^2 / (2 H^2)   for H >> d.
//
// The test checks both that the kernel matches the finite-segment closed form
// (bit-exact within 1e-10 rel) and that the gap to mu0*I/(2 pi d) decreases
// quadratically with H/d.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::mu0;
using ::quasar::mu0_over_4pi;
using ::quasar::pi;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::PointCloud;

namespace {

ConductorSystem make_symmetric_segment(Real half_length, Real current_A) {
  ConductorSystem cs;
  cs.add({/*name=*/"long_wire",
          /*current_A=*/current_A,
          /*points=*/{Vec3{0, 0, -half_length}, Vec3{0, 0, +half_length}}});
  return cs;
}

}  // namespace

TEST(InfiniteWireLimit, ApproachesMu0IOver2PiDAsLengthGrows) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;
  const Real d = Real{1.0};
  const Real I = Real{1.0};
  const Real B_inf = mu0 * I / (Real{2} * pi * d);

  PointCloud pc;
  pc.add(Vec3{d, 0, 0});

  // Increasing H -> the residual gap to the infinite-wire limit must shrink
  // and the y-component must dominate (other components vanish by symmetry).
  const std::array<Real, 4> ratios = {Real{1}, Real{10}, Real{100}, Real{1000}};
  std::array<Real, 4> rel_gaps{};

  for (std::size_t k = 0; k < ratios.size(); ++k) {
    const Real H = ratios[k] * d;
    const auto cs = make_symmetric_segment(H, I);
    const auto B = eval.evaluate_B(cs, pc);
    ASSERT_EQ(B.size(), 1u);

    // Closed form for the finite segment: B_y = mu0 I / (4 pi d) * 2 H / sqrt(d^2 + H^2).
    const Real B_y_finite = mu0_over_4pi * I * Real{2} * H
                            / (d * std::sqrt(d * d + H * H));
    EXPECT_NEAR(B[0].y, B_y_finite, Real{1e-10} * std::abs(B_y_finite))
        << "ratio=" << ratios[k];
    EXPECT_LT(std::abs(B[0].x), Real{1e-14}) << "ratio=" << ratios[k];
    EXPECT_LT(std::abs(B[0].z), Real{1e-14}) << "ratio=" << ratios[k];

    rel_gaps[k] = std::abs(B[0].y - B_inf) / B_inf;
  }

  // Each tenfold increase in H/d should shrink the gap roughly 100x
  // (gap ~ 1/2 * (d/H)^2). Allow some slack for the leading prefactor.
  for (std::size_t k = 0; k + 1 < ratios.size(); ++k) {
    const Real shrink = rel_gaps[k] / rel_gaps[k + 1];
    EXPECT_GT(shrink, Real{50.0})
        << "from ratio=" << ratios[k] << " to ratio=" << ratios[k + 1]
        << " gap shrank by " << shrink;
  }

  // At H/d = 1000, the residual gap should be below 1e-6 in absolute terms.
  EXPECT_LT(rel_gaps[ratios.size() - 1], Real{1e-6})
      << "final rel gap=" << rel_gaps[ratios.size() - 1];
}
