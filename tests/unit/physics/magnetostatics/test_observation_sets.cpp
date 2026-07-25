#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::LineProbe;
using ::quasar::magnetostatics::ObservationGrid;
using ::quasar::magnetostatics::PlaneSlice;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::PointSoA;

namespace {

Real dist(Vec3 a, Vec3 b) noexcept { return length(a - b); }

}  // namespace

TEST(PointCloud, AddManySafelyAcceptsItsOwnStorage) {
  PointCloud points;
  points.add(Vec3{1, 2, 3});
  points.add(Vec3{4, 5, 6});
  const auto& original = points.points();
  points.add(std::span<const Vec3>{original.data(), original.size()});

  ASSERT_EQ(points.size(), 4u);
  EXPECT_EQ(points.points()[0].x, Real{1});
  EXPECT_EQ(points.points()[1].x, Real{4});
  EXPECT_EQ(points.points()[2].x, Real{1});
  EXPECT_EQ(points.points()[3].x, Real{4});
}

// ----- ObservationGrid -----------------------------------------------------

TEST(ObservationGrid, PointAtMatchesAffineFormula) {
  ObservationGrid g;
  g.origin  = Vec3{Real{1}, Real{2}, Real{3}};
  g.spacing = Vec3{Real{0.5}, Real{0.25}, Real{2.0}};
  g.dims    = {4, 3, 2};

  EXPECT_EQ(g.size(), 24u);

  for (int k = 0; k < g.dims[2]; ++k) {
    for (int j = 0; j < g.dims[1]; ++j) {
      for (int i = 0; i < g.dims[0]; ++i) {
        const Vec3 want{g.origin.x + i * g.spacing.x,
                        g.origin.y + j * g.spacing.y,
                        g.origin.z + k * g.spacing.z};
        EXPECT_LT(dist(g.point_at(i, j, k), want), Real{1e-14});
      }
    }
  }
}

TEST(ObservationGrid, PointAtRejectsIndicesOutsideDeclaredGrid) {
  ObservationGrid g;
  g.dims = {2, 3, 4};
  EXPECT_THROW((void)g.point_at(-1, 0, 0), std::invalid_argument);
  EXPECT_THROW((void)g.point_at(2, 0, 0), std::invalid_argument);
  EXPECT_THROW((void)g.point_at(0, 3, 0), std::invalid_argument);
  EXPECT_THROW((void)g.point_at(0, 0, 4), std::invalid_argument);
}

TEST(ObservationGrid, ToPointSoaIsXFastestLayout) {
  ObservationGrid g;
  g.origin  = Vec3{Real{0}, Real{0}, Real{0}};
  g.spacing = Vec3{Real{1}, Real{10}, Real{100}};
  g.dims    = {3, 2, 2};

  const PointSoA s = g.to_point_soa();
  ASSERT_EQ(s.n_points(), 12u);

  // i + dims[0] * (j + dims[1] * k); spot-check a few entries.
  EXPECT_EQ(s.px[0],  Real{0});   EXPECT_EQ(s.py[0],  Real{0});   EXPECT_EQ(s.pz[0],   Real{0});
  EXPECT_EQ(s.px[1],  Real{1});   EXPECT_EQ(s.py[1],  Real{0});   EXPECT_EQ(s.pz[1],   Real{0});
  EXPECT_EQ(s.px[3],  Real{0});   EXPECT_EQ(s.py[3],  Real{10});  EXPECT_EQ(s.pz[3],   Real{0});
  EXPECT_EQ(s.px[6],  Real{0});   EXPECT_EQ(s.py[6],  Real{0});   EXPECT_EQ(s.pz[6],   Real{100});
  EXPECT_EQ(s.px[11], Real{2});   EXPECT_EQ(s.py[11], Real{10});  EXPECT_EQ(s.pz[11],  Real{100});
}

TEST(ObservationGrid, ToPointCloudHasSameCountAsSoa) {
  ObservationGrid g;
  g.origin  = Vec3{Real{-1}, Real{0}, Real{0}};
  g.spacing = Vec3{Real{0.1}, Real{0.1}, Real{0.1}};
  g.dims    = {5, 5, 4};

  const PointCloud pc = g.to_point_cloud();
  const PointSoA   s  = g.to_point_soa();
  ASSERT_EQ(pc.size(), s.n_points());
  ASSERT_EQ(pc.size(), 100u);

  for (std::size_t idx = 0; idx < pc.size(); ++idx) {
    EXPECT_LT(dist(pc.points()[idx],
                   Vec3{s.px[idx], s.py[idx], s.pz[idx]}),
              Real{1e-14});
  }
}

TEST(ObservationGrid, RejectsZeroOrNegativeDims) {
  ObservationGrid g;
  g.dims = {0, 1, 1};
  EXPECT_THROW(ObservationGrid::validate(g), std::invalid_argument);
  EXPECT_THROW(g.to_point_cloud(),           std::invalid_argument);
  g.dims = {2, -1, 2};
  EXPECT_THROW(g.to_point_soa(),             std::invalid_argument);
}

TEST(ObservationGrid, RejectsPointCountOverflowBeforeMaterialization) {
  ObservationGrid g;
  g.dims = {std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(), 2};
  EXPECT_THROW(ObservationGrid::validate(g), std::invalid_argument);
  EXPECT_THROW((void)g.size(), std::invalid_argument);
}

TEST(ObservationGrid, RejectsCollapsedAdjacentCoordinates) {
  ObservationGrid g;
  g.origin = Vec3{1.0e308, 0, 0};
  g.spacing = Vec3{1, 1, 1};
  g.dims = {2, 1, 1};
  EXPECT_THROW(ObservationGrid::validate(g), std::invalid_argument);
  EXPECT_THROW((void)g.point_at(1, 0, 0), std::invalid_argument);
}

TEST(ObservationGrid, AffineCancellationAvoidsIntermediateOverflow) {
  const Real largest = std::numeric_limits<Real>::max();
  ObservationGrid g;
  g.origin = Vec3{-largest, 0, 0};
  g.spacing = Vec3{largest, 1, 1};
  g.dims = {3, 1, 1};
  const PointSoA soa = g.to_point_soa();
  ASSERT_EQ(soa.n_points(), 3u);
  EXPECT_EQ(soa.px[0], -largest);
  EXPECT_EQ(soa.px[1], Real{0});
  EXPECT_EQ(soa.px[2], largest);
}

// ----- PlaneSlice ----------------------------------------------------------

TEST(PlaneSlice, PointAtMatchesAffineFormula) {
  PlaneSlice s;
  s.origin = Vec3{Real{0}, Real{0}, Real{0.5}};
  s.u_step = Vec3{Real{0.2}, Real{0}, Real{0}};
  s.v_step = Vec3{Real{0}, Real{0.1}, Real{0}};
  s.nu = 5;
  s.nv = 4;

  EXPECT_EQ(s.size(), 20u);

  for (int j = 0; j < s.nv; ++j) {
    for (int i = 0; i < s.nu; ++i) {
      const Vec3 want{Real{0} + i * Real{0.2},
                      Real{0} + j * Real{0.1},
                      Real{0.5}};
      EXPECT_LT(dist(s.point_at(i, j), want), Real{1e-14});
    }
  }
}

TEST(PlaneSlice, PointAtRejectsIndicesOutsideDeclaredSlice) {
  PlaneSlice s;
  s.nu = 2;
  s.nv = 3;
  EXPECT_THROW((void)s.point_at(-1, 0), std::invalid_argument);
  EXPECT_THROW((void)s.point_at(2, 0), std::invalid_argument);
  EXPECT_THROW((void)s.point_at(0, 3), std::invalid_argument);
}

TEST(PlaneSlice, ToPointSoaUsesUFastestLayout) {
  PlaneSlice s;
  s.origin = Vec3{Real{0}, Real{0}, Real{0}};
  s.u_step = Vec3{Real{1}, Real{0}, Real{0}};
  s.v_step = Vec3{Real{0}, Real{10}, Real{0}};
  s.nu = 3;
  s.nv = 2;

  const PointSoA soa = s.to_point_soa();
  ASSERT_EQ(soa.n_points(), 6u);

  EXPECT_EQ(soa.px[0], Real{0}); EXPECT_EQ(soa.py[0], Real{0});
  EXPECT_EQ(soa.px[1], Real{1}); EXPECT_EQ(soa.py[1], Real{0});
  EXPECT_EQ(soa.px[2], Real{2}); EXPECT_EQ(soa.py[2], Real{0});
  EXPECT_EQ(soa.px[3], Real{0}); EXPECT_EQ(soa.py[3], Real{10});
  EXPECT_EQ(soa.px[5], Real{2}); EXPECT_EQ(soa.py[5], Real{10});
}

TEST(PlaneSlice, RejectsZeroSize) {
  PlaneSlice s;
  s.nu = 0;
  EXPECT_THROW(PlaneSlice::validate(s), std::invalid_argument);
}

TEST(PlaneSlice, RejectsPointCountBeyondEvaluatorIndexRange) {
  PlaneSlice s;
  s.nu = std::numeric_limits<int>::max();
  s.nv = 2;
  EXPECT_THROW(PlaneSlice::validate(s), std::invalid_argument);
}

TEST(PlaneSlice, RejectsCollapsedSampleStep) {
  PlaneSlice s;
  s.origin = Vec3{1.0e308, 0, 0};
  s.u_step = Vec3{1, 0, 0};
  s.nu = 2;
  s.nv = 1;
  EXPECT_THROW(PlaneSlice::validate(s), std::invalid_argument);
  EXPECT_THROW((void)s.point_at(1, 0), std::invalid_argument);
}

TEST(PlaneSlice, RejectsCollinearSamplingAxes) {
  PlaneSlice s;
  s.u_step = Vec3{1, 2, 3};
  s.v_step = Vec3{2, 4, 6};
  s.nu = 2;
  s.nv = 2;
  EXPECT_THROW(PlaneSlice::validate(s), std::invalid_argument);
}

TEST(PlaneSlice, RejectsCombinedCornerOverflow) {
  const Real step = Real{0.75} * std::numeric_limits<Real>::max();
  PlaneSlice s;
  s.u_step = Vec3{step, 0, 0};
  s.v_step = Vec3{step, 1, 0};
  s.nu = 2;
  s.nv = 2;
  EXPECT_THROW(PlaneSlice::validate(s), std::invalid_argument);
}

TEST(PlaneSlice, RejectsBoundaryCollapseAfterOtherAxisTranslation) {
  PlaneSlice s;
  s.u_step = Vec3{1, 0, 0};
  s.v_step = Vec3{1.0e308, 1, 0};
  s.nu = 2;
  s.nv = 2;
  EXPECT_THROW(PlaneSlice::validate(s), std::invalid_argument);
}

TEST(PlaneSlice, AffineCancellationAvoidsIntermediateOverflow) {
  const Real largest = std::numeric_limits<Real>::max();
  PlaneSlice s;
  s.origin = Vec3{largest, 0, 0};
  s.u_step = Vec3{-largest, 0, 0};
  s.v_step = Vec3{};
  s.nu = 3;
  s.nv = 1;
  const PointSoA soa = s.to_point_soa();
  ASSERT_EQ(soa.n_points(), 3u);
  EXPECT_EQ(soa.px[0], largest);
  EXPECT_EQ(soa.px[1], Real{0});
  EXPECT_EQ(soa.px[2], -largest);
}

// ----- LineProbe -----------------------------------------------------------

TEST(LineProbe, EndpointsAreIncluded) {
  LineProbe l;
  l.start    = Vec3{Real{1}, Real{2}, Real{3}};
  l.end      = Vec3{Real{4}, Real{2}, Real{3}};
  l.n_points = 7;

  EXPECT_LT(dist(l.point_at(0), l.start),           Real{1e-14});
  EXPECT_LT(dist(l.point_at(l.n_points - 1), l.end), Real{1e-14});
}

TEST(LineProbe, PointAtRejectsIndicesOutsideDeclaredProbe) {
  LineProbe l;
  l.n_points = 3;
  EXPECT_THROW((void)l.point_at(-1), std::invalid_argument);
  EXPECT_THROW((void)l.point_at(3), std::invalid_argument);
}

TEST(LineProbe, SamplesAreEquallySpaced) {
  LineProbe l;
  l.start    = Vec3{Real{0}, Real{0}, Real{0}};
  l.end      = Vec3{Real{1}, Real{1}, Real{1}};
  l.n_points = 11;

  const Real expected_step = length(l.end - l.start) / Real{10};
  for (int i = 0; i + 1 < l.n_points; ++i) {
    EXPECT_NEAR(dist(l.point_at(i), l.point_at(i + 1)), expected_step,
                Real{1e-12});
  }
}

TEST(LineProbe, RejectsCollapseAtFinalEndpoint) {
  LineProbe l;
  l.start = Vec3{std::nextafter(Real{1},
                                std::numeric_limits<Real>::infinity()), 0, 0};
  l.end = Vec3{1, 0, 0};
  l.n_points = 3;
  EXPECT_THROW(LineProbe::validate(l), std::invalid_argument);
  EXPECT_THROW((void)l.point_at(1), std::invalid_argument);
}

TEST(LineProbe, ToPointCloudAndSoaAgree) {
  LineProbe l;
  l.start    = Vec3{Real{-1}, Real{0}, Real{0}};
  l.end      = Vec3{Real{1}, Real{0}, Real{0}};
  l.n_points = 5;

  const PointCloud pc = l.to_point_cloud();
  const PointSoA   s  = l.to_point_soa();
  ASSERT_EQ(pc.size(), s.n_points());
  ASSERT_EQ(pc.size(), 5u);

  for (std::size_t i = 0; i < pc.size(); ++i) {
    EXPECT_LT(dist(pc.points()[i],
                   Vec3{s.px[i], s.py[i], s.pz[i]}),
              Real{1e-14});
  }
}

TEST(LineProbe, RejectsFewerThanTwoPoints) {
  LineProbe l;
  l.n_points = 1;
  EXPECT_THROW(LineProbe::validate(l), std::invalid_argument);
}
