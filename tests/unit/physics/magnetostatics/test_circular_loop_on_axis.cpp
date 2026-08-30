#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "host_evaluate.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::pi;
using ::quasar::mu0;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::PointCloud;

namespace {

// Build a closed regular-N-polygon loop of radius R in the z=0 plane, current
// I, traversed in the +phi direction so that the on-axis field at +z is in +z.
Filament make_polygon_loop(Real R, int N, Real I) {
  Filament f;
  f.name      = "loop";
  f.current_A = I;
  f.points.reserve(static_cast<std::size_t>(N) + 1u);
  for (int k = 0; k <= N; ++k) {
    const Real theta = (Real{2} * pi * static_cast<Real>(k)) / static_cast<Real>(N);
    f.points.push_back(Vec3{R * std::cos(theta), R * std::sin(theta), Real{0}});
  }
  return f;
}

// On-axis B_z for an exact circular loop of radius R, current I, observed at
// z on the loop axis (loop in z=0 plane):
//
//   B_z = mu0 * I * R^2 / (2 * (R^2 + z^2)^(3/2))
Real circular_loop_axis_Bz_ref(Real R, Real z, Real I) {
  const Real denom = std::pow(R * R + z * z, Real{1.5});
  return mu0 * I * R * R / (Real{2} * denom);
}

// Linear regression slope of (log(err) vs log(N)) using simple least squares.
Real loglog_slope(const std::vector<int>& Ns, const std::vector<Real>& errs) {
  const std::size_t n = Ns.size();
  Real sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Real x = std::log(static_cast<Real>(Ns[i]));
    const Real y = std::log(errs[i]);
    sx  += x;
    sy  += y;
    sxx += x * x;
    sxy += x * y;
  }
  const Real fn = static_cast<Real>(n);
  return (fn * sxy - sx * sy) / (fn * sxx - sx * sx);
}

constexpr Real kRadius  = Real{0.1};
constexpr Real kCurrent = Real{1.0};
constexpr Real kRelTolAtN256 = Real{1e-4};
constexpr Real kSlopeMaxNeg  = Real{-1.8};  // theoretical -2, with safety margin.

const std::array<Real, 3> kZ      = {Real{0.0}, Real{0.05}, Real{0.2}};
const std::array<int,  4> kNs     = {32, 64, 128, 256};

}  // namespace

TEST(CircularLoopOnAxis, MatchesClosedFormAtN256) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;
  ConductorSystem cs;
  cs.add(make_polygon_loop(kRadius, /*N=*/256, kCurrent));

  PointCloud pc;
  for (Real z : kZ) pc.add(Vec3{Real{0}, Real{0}, z});

  const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
  ASSERT_EQ(field.size(), kZ.size());

  for (std::size_t i = 0; i < kZ.size(); ++i) {
    const Real ref = circular_loop_axis_Bz_ref(kRadius, kZ[i], kCurrent);
    EXPECT_NEAR(field[i].z, ref, kRelTolAtN256 * std::abs(ref))
        << "z=" << kZ[i] << " ref=" << ref << " got=" << field[i].z;
    // Off-axis components must vanish on the symmetry axis.
    EXPECT_LT(std::abs(field[i].x), Real{1e-12} * std::abs(ref) + Real{1e-18})
        << "z=" << kZ[i];
    EXPECT_LT(std::abs(field[i].y), Real{1e-12} * std::abs(ref) + Real{1e-18})
        << "z=" << kZ[i];
  }
}

TEST(CircularLoopOnAxis, ErrorConvergesQuadraticallyInN) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;

  for (Real z : kZ) {
    std::vector<Real> errs;
    errs.reserve(kNs.size());

    for (int N : kNs) {
      ConductorSystem cs;
      cs.add(make_polygon_loop(kRadius, N, kCurrent));

      PointCloud pc;
      pc.add(Vec3{Real{0}, Real{0}, z});

      const auto field = quasar::test::host_evaluate_B(eval, cs, pc);
      ASSERT_EQ(field.size(), 1u);

      const Real ref = circular_loop_axis_Bz_ref(kRadius, z, kCurrent);
      errs.push_back(std::abs(field[0].z - ref));
    }

    std::vector<int> Ns_v(kNs.begin(), kNs.end());
    const Real slope = loglog_slope(Ns_v, errs);
    EXPECT_LE(slope, kSlopeMaxNeg)
        << "z=" << z << " slope=" << slope
        << " errs=" << errs[0] << "," << errs[1] << ","
        << errs[2] << "," << errs[3];
  }
}
