// The device expansion of a structured observation set must agree with the
// host `point_at` accessor bit for bit.
//
// Not "to within a tolerance". These are the coordinates at which a field is
// sampled, so a one-ulp disagreement between the accessor a deck echoes and the
// points the evaluator actually used is a silent inconsistency, not a rounding
// detail. It matters most for LineProbe, whose expansion goes through std::lerp
// -- specified to be exact at both endpoints and monotonic, guarantees the
// obvious `a + t * (b - a)` does not provide. The device kernel transcribes that
// algorithm and is compiled -ffp-contract=off to keep it.

#include "quasar/core/device_observations.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::core::DevicePointCloud;
using ::quasar::core::LineProbe;
using ::quasar::core::ObservationGrid;
using ::quasar::core::PlaneSlice;
using ::quasar::core::PointSoA;

namespace {

void expect_matches_host(const PointSoA& device, const PointSoA& host) {
  ASSERT_EQ(device.n_points(), host.n_points());
  for (std::size_t i = 0; i < host.n_points(); ++i) {
    EXPECT_EQ(device.px[i], host.px[i]) << "x at " << i;
    EXPECT_EQ(device.py[i], host.py[i]) << "y at " << i;
    EXPECT_EQ(device.pz[i], host.pz[i]) << "z at " << i;
  }
}

}  // namespace

TEST(DeviceObservations, GridExpansionMatchesHostBitExactly) {
  ObservationGrid g;
  // A translated, anisotropically spaced grid: the offsets are not exact
  // binary fractions, so the fma in the expansion has something to round.
  g.origin = Vec3{-0.3125, 7.1, -1234.567};
  g.spacing = Vec3{0.017, -0.29, 3.0e-4};
  g.dims = {7, 5, 3};

  expect_matches_host(g.to_device_point_cloud().to_host(), g.to_point_soa());
}

TEST(DeviceObservations, PlaneExpansionMatchesHostBitExactly) {
  PlaneSlice s;
  s.origin = Vec3{1.0e6, -2.5, 0.75};
  s.u_step = Vec3{0.031, 0.0, -0.007};
  s.v_step = Vec3{0.0, 0.019, 0.011};
  s.nu = 9;
  s.nv = 6;

  expect_matches_host(s.to_device_point_cloud().to_host(), s.to_point_soa());
}

TEST(DeviceObservations, LineExpansionMatchesHostBitExactlyAndPinsEndpoints) {
  LineProbe l;
  // Opposite signs across the origin, which selects std::lerp's first branch,
  // and an odd point count so the midpoint is a sampled site.
  l.start = Vec3{-1.0e-3, 4.0, -7.25};
  l.end = Vec3{2.0e-3, -4.0, 9.5};
  l.n_points = 33;

  const PointSoA device = l.to_device_point_cloud().to_host();
  expect_matches_host(device, l.to_point_soa());

  // std::lerp is exact at both ends. If the kernel had used a + t*(b - a) this
  // is what would drift.
  const std::size_t last = device.n_points() - 1;
  EXPECT_EQ(device.px[0], l.start.x);
  EXPECT_EQ(device.py[0], l.start.y);
  EXPECT_EQ(device.pz[0], l.start.z);
  EXPECT_EQ(device.px[last], l.end.x);
  EXPECT_EQ(device.py[last], l.end.y);
  EXPECT_EQ(device.pz[last], l.end.z);
}

TEST(DeviceObservations, SameSignLineAlsoMatchesHostBitExactly) {
  // Both endpoints on the same side of zero takes std::lerp's second branch,
  // including its monotonicity clamp. The first branch alone would not exercise
  // that path.
  LineProbe l;
  l.start = Vec3{1.0, 2.0, 3.0};
  l.end = Vec3{1.0 + 1.0e-12, 2.5, 3.25};
  l.n_points = 17;

  expect_matches_host(l.to_device_point_cloud().to_host(), l.to_point_soa());
}

TEST(DeviceObservations, OverflowingExpansionIsRejected) {
  // A spacing that walks off the top of the exponent range. The host
  // checked_coordinate() threw std::invalid_argument here; the device sets a
  // status bit and the host raises the same type.
  ObservationGrid g;
  g.origin = Vec3{0, 0, 0};
  g.spacing = Vec3{1.0e308, 1.0, 1.0};
  g.dims = {4, 1, 1};

  EXPECT_THROW((void)g.to_device_point_cloud(), std::invalid_argument);
}

TEST(DeviceObservations, SingleSiteGridExpandsToItsOrigin) {
  ObservationGrid g;
  g.origin = Vec3{0.5, -0.25, 8.0};
  g.spacing = Vec3{1.0, 1.0, 1.0};
  g.dims = {1, 1, 1};

  const PointSoA device = g.to_device_point_cloud().to_host();
  ASSERT_EQ(device.n_points(), 1u);
  EXPECT_EQ(device.px[0], g.origin.x);
  EXPECT_EQ(device.py[0], g.origin.y);
  EXPECT_EQ(device.pz[0], g.origin.z);
}
