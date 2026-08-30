#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "filament_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::PointSoA;
using ::quasar::magnetostatics::DeviceSegmentSoA;
using ::quasar::magnetostatics::SegmentSoA;

static_assert(std::is_copy_constructible_v<ConductorSystem>);
static_assert(std::is_copy_assignable_v<ConductorSystem>);
static_assert(std::is_move_constructible_v<ConductorSystem>);
static_assert(std::is_move_assignable_v<ConductorSystem>);
static_assert(std::is_nothrow_move_constructible_v<ConductorSystem>);
static_assert(std::is_nothrow_move_assignable_v<ConductorSystem>);

namespace {

// The vertices live on the device now, so a test fixture uploads them. This
// deliberately bypasses generic_polyline's validation: several tests below feed
// degenerate or non-finite geometry precisely to check that the flatten
// rejects it.
Filament make_filament(std::string name, Real current_A,
                       const std::vector<Vec3>& points) {
  return quasar::test::filament(std::move(name), current_A,
                  points);
}

}  // namespace

TEST(ConductorSoa, FlattensThreeFilamentsInOrder) {
  ConductorSystem cs;
  cs.add(make_filament("loop", Real{1.0}, {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}}));
  cs.add(make_filament("line", Real{2.5}, {Vec3{-1, 0, 0}, Vec3{1, 0, 0}}));
  cs.add(make_filament("zig", Real{3.0}, {Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{0, 0, 2}, Vec3{0, 0, 3}}));

  const SegmentSoA s = cs.to_segments_soa();
  ASSERT_EQ(s.n_segments(), 2u + 1u + 3u);
  ASSERT_EQ(s.ax.size(), s.n_segments());
  ASSERT_EQ(s.I .size(), s.n_segments());

  // Filament 0, segment 0: (0,0,0) -> (1,0,0), I=1
  EXPECT_EQ(s.ax[0], Real{0});
  EXPECT_EQ(s.ay[0], Real{0});
  EXPECT_EQ(s.az[0], Real{0});
  EXPECT_EQ(s.bx[0], Real{1});
  EXPECT_EQ(s.by[0], Real{0});
  EXPECT_EQ(s.bz[0], Real{0});
  EXPECT_EQ(s.I [0], Real{1});

  // Filament 1, segment 0: (-1,0,0) -> (1,0,0), I=2.5 -> index 2.
  EXPECT_EQ(s.ax[2], Real{-1});
  EXPECT_EQ(s.bx[2], Real{1});
  EXPECT_EQ(s.I [2], Real{2.5});

  // Filament 2, segment 1: (0,0,1) -> (0,0,2), I=3 -> index 4.
  EXPECT_EQ(s.az[4], Real{1});
  EXPECT_EQ(s.bz[4], Real{2});
  EXPECT_EQ(s.I [4], Real{3.0});
}

TEST(ConductorSoa, ThrowsWhenFilamentHasFewerThanTwoPoints) {
  ConductorSystem cs;
  cs.add(make_filament("bad", Real{1.0}, {Vec3{0, 0, 0}}));
  EXPECT_THROW(cs.to_segments_soa(), std::invalid_argument);
}

TEST(ConductorSoa, ThrowsOnZeroLengthSegment) {
  ConductorSystem cs;
  cs.add(make_filament("bad", Real{1.0}, {Vec3{0, 0, 0}, Vec3{0, 0, 0}}));
  EXPECT_THROW(cs.to_segments_soa(), std::invalid_argument);
}

TEST(ConductorSoa, PreservesDistinctSubFemtometreSegment) {
  ConductorSystem cs;
  cs.add(make_filament("tiny", Real{1.0}, {Vec3{0, 0, 0}, Vec3{1e-20, 0, 0}}));
  const SegmentSoA s = cs.to_segments_soa();
  ASSERT_EQ(s.n_segments(), 1u);
  EXPECT_EQ(s.bx[0] - s.ax[0], Real{1e-20});
}

TEST(ConductorSoa, ConcurrentConstLookupsShareOneStableCache) {
  ConductorSystem cs;
  std::vector<Vec3> points;
  constexpr std::size_t segment_count = 8192;
  points.reserve(segment_count + 1);
  for (std::size_t i = 0; i <= segment_count; ++i) {
    points.push_back(Vec3{static_cast<Real>(i), Real{0}, Real{0}});
  }
  cs.add(make_filament("many", Real{3.5}, points));

  constexpr std::size_t thread_count = 16;
  std::barrier start{static_cast<std::ptrdiff_t>(thread_count)};
  std::array<const DeviceSegmentSoA*, thread_count> results{};
  std::array<std::thread, thread_count> workers;
  for (std::size_t i = 0; i < thread_count; ++i) {
    workers[i] = std::thread{[&, i] {
      start.arrive_and_wait();
      results[i] = &cs.device_segments();
    }};
  }
  for (auto& worker : workers) worker.join();

  ASSERT_NE(results[0], nullptr);
  ASSERT_EQ(results[0]->n_segments(), segment_count);
  // The cache holds device planes; reading two of them back is an output
  // boundary, which is exactly what to_host() is for.
  const SegmentSoA staged = results[0]->to_host();
  EXPECT_EQ(staged.I.front(), Real{3.5});
  EXPECT_EQ(staged.I.back(), Real{3.5});
  for (const DeviceSegmentSoA* result : results) {
    EXPECT_EQ(result, results[0]);
  }
}

TEST(ConductorSoa, CopiesOwnIndependentInvalidatableCaches) {
  ConductorSystem original;
  original.add(make_filament("first", Real{1}, {Vec3{0, 0, 0}, Vec3{1, 0, 0}}));
  ASSERT_EQ(original.device_segments().n_segments(), 1u);  // Prime its cache.

  ConductorSystem copied = original;
  const DeviceSegmentSoA* copied_cache = &copied.device_segments();
  EXPECT_NE(copied_cache, &original.device_segments());
  ASSERT_EQ(copied_cache->n_segments(), 1u);

  original.add(make_filament("second", Real{2}, {Vec3{0, 0, 0}, Vec3{0, 1, 0}}));
  EXPECT_EQ(original.device_segments().n_segments(), 2u);
  EXPECT_EQ(copied.device_segments().n_segments(), 1u);

  ConductorSystem assigned;
  assigned = copied;
  EXPECT_NE(&assigned.device_segments(), &copied.device_segments());
  EXPECT_EQ(assigned.device_segments().n_segments(), 1u);
}

TEST(ConductorSoa, MovesPreserveGeometryAndResetMovedFromCache) {
  ConductorSystem source;
  source.add(make_filament("wire", Real{4}, {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{2, 0, 0}}));
  ASSERT_EQ(source.device_segments().n_segments(), 2u);  // Prime its cache.

  ConductorSystem moved = std::move(source);
  EXPECT_EQ(moved.device_segments().n_segments(), 2u);
  EXPECT_TRUE(source.empty());
  EXPECT_TRUE(source.device_segments().empty());

  ConductorSystem assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.device_segments().n_segments(), 2u);
  EXPECT_TRUE(moved.empty());
  EXPECT_TRUE(moved.device_segments().empty());
}

TEST(ConductorSoa, ThrowsOnNonFiniteCoordinate) {
  const Real nan = std::nan("");
  ConductorSystem cs;
  cs.add(make_filament("bad", Real{1.0}, {Vec3{0, 0, 0}, Vec3{nan, 0, 0}}));
  EXPECT_THROW(cs.to_segments_soa(), std::invalid_argument);
}

TEST(PointCloudSoa, ReshapesAddOneAndAddRange) {
  PointCloud pc;
  pc.add(Vec3{1, 2, 3});
  const Vec3 batch[] = {Vec3{4, 5, 6}, Vec3{7, 8, 9}};
  pc.add(std::span<const Vec3>{batch, 2});

  const PointSoA s = pc.to_point_soa();
  ASSERT_EQ(s.n_points(), 3u);

  EXPECT_EQ(s.px[0], Real{1}); EXPECT_EQ(s.py[0], Real{2}); EXPECT_EQ(s.pz[0], Real{3});
  EXPECT_EQ(s.px[1], Real{4}); EXPECT_EQ(s.py[1], Real{5}); EXPECT_EQ(s.pz[1], Real{6});
  EXPECT_EQ(s.px[2], Real{7}); EXPECT_EQ(s.py[2], Real{8}); EXPECT_EQ(s.pz[2], Real{9});
}

TEST(PointCloudSoa, EmptyReshapesToEmpty) {
  PointCloud pc;
  const PointSoA s = pc.to_point_soa();
  EXPECT_EQ(s.n_points(), 0u);
}

TEST(PublicSoaValidation, RejectsMismatchedComponentLengths) {
  PointSoA points;
  points.px = {1.0, 2.0};
  points.py = {3.0};
  points.pz = {4.0, 5.0};
  EXPECT_THROW(points.validate(), std::invalid_argument);

  SegmentSoA segments;
  segments.ax = {0.0};
  segments.ay = {0.0};
  segments.az = {0.0};
  segments.bx = {1.0};
  segments.by = {0.0};
  segments.bz = {0.0};
  // Missing current plane: sizing every upload from ax.size() would read I
  // out of bounds without the cross-plane validation.
  EXPECT_THROW(segments.validate(), std::invalid_argument);
}

TEST(PublicSoaValidation, RejectsNonFinitePayloads) {
  const Real inf = std::numeric_limits<Real>::infinity();
  PointSoA points{{0.0}, {inf}, {0.0}};
  EXPECT_THROW(points.validate(), std::invalid_argument);

  SegmentSoA segments;
  segments.ax = {0.0};
  segments.ay = {0.0};
  segments.az = {0.0};
  segments.bx = {1.0};
  segments.by = {0.0};
  segments.bz = {0.0};
  segments.I = {inf};
  EXPECT_THROW(segments.validate(), std::invalid_argument);
}

TEST(PointCloudSoa, RejectsNonFinitePointsAtomically) {
  PointCloud pc;
  pc.add(Vec3{1, 2, 3});
  EXPECT_THROW(pc.add(Vec3{std::numeric_limits<Real>::quiet_NaN(), 0, 0}),
               std::invalid_argument);
  EXPECT_EQ(pc.size(), 1u);

  const Vec3 batch[] = {Vec3{4, 5, 6},
                        Vec3{0, std::numeric_limits<Real>::infinity(), 0}};
  EXPECT_THROW(pc.add(std::span<const Vec3>{batch, 2}), std::invalid_argument);
  EXPECT_EQ(pc.size(), 1u);
}
