// Polygon approximations of a smooth current loop converge to the
// exact field at O(1/N^2). We verify this at off-axis points where the
// closed-form B involves complete elliptic integrals - so instead of using a
// closed form, we treat the N=4096 polygon as the reference and check that
// successive doublings of N shrink the error by ~4x (log-log slope <= -1.8).

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::PointCloud;

namespace {

constexpr Real kRadius  = Real{0.1};
constexpr Real kCurrent = Real{1.0};
constexpr int  kRefN    = 4096;

const std::array<int, 5> kNs = {16, 32, 64, 128, 256};

// Off-axis observation points: a mix of in-loop-plane and tilted positions
// that stress the polygon approximation off-symmetry-axis.
PointCloud make_observation_cloud() {
  PointCloud pc;
  pc.add(Vec3{Real{0.05},  Real{0.00}, Real{0.02}});  // inside the loop
  pc.add(Vec3{Real{0.15},  Real{0.00}, Real{0.05}});  // outside the loop
  pc.add(Vec3{Real{0.07},  Real{0.07}, Real{0.10}});  // off-axis above
  pc.add(Vec3{Real{0.00},  Real{0.00}, Real{0.20}});  // far on-axis
  return pc;
}

ConductorSystem make_loop(int N) {
  ConductorSystem cs;
  cs.add(circular_loop(/*center=*/Vec3{0, 0, 0}, /*axis=*/Vec3{0, 0, 1},
                       /*radius_m=*/kRadius, /*n_segments=*/N,
                       /*current_A=*/kCurrent, /*name=*/"loop"));
  return cs;
}

Real frob(Vec3 v) noexcept {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Real loglog_slope(const std::vector<Real>& xs, const std::vector<Real>& ys) {
  const std::size_t n = xs.size();
  Real sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Real x = std::log(xs[i]);
    const Real y = std::log(ys[i]);
    sx  += x;
    sy  += y;
    sxx += x * x;
    sxy += x * y;
  }
  const Real fn = static_cast<Real>(n);
  return (fn * sxy - sx * sy) / (fn * sxx - sx * sx);
}

}  // namespace

TEST(PolylineConvergence, OffAxisErrorDecaysQuadraticallyInN) {
  if (!::quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }

  const BiotSavartEvaluator eval;
  const PointCloud pc = make_observation_cloud();
  const std::size_t M = pc.size();

  // Reference: the highest-resolution polygon.
  const auto cs_ref = make_loop(kRefN);
  const auto B_ref  = eval.evaluate_B(cs_ref, pc);
  ASSERT_EQ(B_ref.size(), M);

  // Per-observation error series across N.
  std::vector<std::vector<Real>> errors(M, std::vector<Real>{});
  for (auto& v : errors) v.reserve(kNs.size());

  for (int N : kNs) {
    const auto cs = make_loop(N);
    const auto B  = eval.evaluate_B(cs, pc);
    ASSERT_EQ(B.size(), M);
    for (std::size_t k = 0; k < M; ++k) {
      const Vec3 diff = B[k] - B_ref[k];
      errors[k].push_back(frob(diff));
    }
  }

  // For every observation point: log-log slope vs N should be <= -1.8.
  const std::vector<Real> log_Ns_x(kNs.begin(), kNs.end());
  for (std::size_t k = 0; k < M; ++k) {
    // Reject points where the error series ever underflows to zero - that
    // means the polygon already happens to agree with the reference (rare;
    // and the test point is useless then).
    bool any_zero = false;
    for (Real e : errors[k]) any_zero |= (e == Real{0});
    if (any_zero) {
      ADD_FAILURE() << "observation " << k << " produced zero error at some N";
      continue;
    }
    const Real slope = loglog_slope(log_Ns_x, errors[k]);
    EXPECT_LE(slope, Real{-1.8})
        << "obs " << k << " slope=" << slope
        << " errors=[" << errors[k][0] << "," << errors[k][1] << ","
        << errors[k][2] << "," << errors[k][3] << "," << errors[k][4] << "]";
  }
}
