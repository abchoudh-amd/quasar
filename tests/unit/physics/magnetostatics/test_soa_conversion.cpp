#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

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
using ::quasar::magnetostatics::SegmentSoA;

static_assert(std::is_copy_constructible_v<ConductorSystem>);
static_assert(std::is_copy_assignable_v<ConductorSystem>);
static_assert(std::is_move_constructible_v<ConductorSystem>);
static_assert(std::is_move_assignable_v<ConductorSystem>);
static_assert(std::is_nothrow_move_constructible_v<ConductorSystem>);
static_assert(std::is_nothrow_move_assignable_v<ConductorSystem>);

TEST(ConductorSoa, FlattensThreeFilamentsInOrder) {
  ConductorSystem cs;
  cs.add({/*name=*/"loop", /*current_A=*/Real{1.0},
          /*points=*/{Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}}});
  cs.add({"line", Real{2.5},
          {Vec3{-1, 0, 0}, Vec3{1, 0, 0}}});
  cs.add({"zig", Real{3.0},
          {Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{0, 0, 2}, Vec3{0, 0, 3}}});

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
  cs.add({"bad", Real{1.0}, {Vec3{0, 0, 0}}});
  EXPECT_THROW(cs.to_segments_soa(), std::invalid_argument);
}

TEST(ConductorSoa, ThrowsOnZeroLengthSegment) {
  ConductorSystem cs;
  cs.add({"bad", Real{1.0}, {Vec3{0, 0, 0}, Vec3{0, 0, 0}}});
  EXPECT_THROW(cs.to_segments_soa(), std::invalid_argument);
}

TEST(ConductorSoa, PreservesDistinctSubFemtometreSegment) {
  ConductorSystem cs;
  cs.add({"tiny", Real{1.0}, {Vec3{0, 0, 0}, Vec3{1e-20, 0, 0}}});
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
  cs.add({"many", Real{3.5}, std::move(points)});

  constexpr std::size_t thread_count = 16;
  std::barrier start{static_cast<std::ptrdiff_t>(thread_count)};
  std::array<const SegmentSoA*, thread_count> results{};
  std::array<std::thread, thread_count> workers;
  for (std::size_t i = 0; i < thread_count; ++i) {
    workers[i] = std::thread{[&, i] {
      start.arrive_and_wait();
      results[i] = &cs.segments_soa();
    }};
  }
  for (auto& worker : workers) worker.join();

  ASSERT_NE(results[0], nullptr);
  ASSERT_EQ(results[0]->n_segments(), segment_count);
  EXPECT_EQ(results[0]->I.front(), Real{3.5});
  EXPECT_EQ(results[0]->I.back(), Real{3.5});
  for (const SegmentSoA* result : results) {
    EXPECT_EQ(result, results[0]);
  }
}

TEST(ConductorSoa, CopiesOwnIndependentInvalidatableCaches) {
  ConductorSystem original;
  original.add({"first", Real{1}, {Vec3{0, 0, 0}, Vec3{1, 0, 0}}});
  ASSERT_EQ(original.segments_soa().n_segments(), 1u);  // Prime its cache.

  ConductorSystem copied = original;
  const SegmentSoA* copied_cache = &copied.segments_soa();
  EXPECT_NE(copied_cache, &original.segments_soa());
  ASSERT_EQ(copied_cache->n_segments(), 1u);

  original.add({"second", Real{2}, {Vec3{0, 0, 0}, Vec3{0, 1, 0}}});
  EXPECT_EQ(original.segments_soa().n_segments(), 2u);
  EXPECT_EQ(copied.segments_soa().n_segments(), 1u);

  ConductorSystem assigned;
  assigned = copied;
  EXPECT_NE(&assigned.segments_soa(), &copied.segments_soa());
  EXPECT_EQ(assigned.segments_soa().n_segments(), 1u);
}

TEST(ConductorSoa, MovesPreserveGeometryAndResetMovedFromCache) {
  ConductorSystem source;
  source.add({"wire", Real{4},
              {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{2, 0, 0}}});
  ASSERT_EQ(source.segments_soa().n_segments(), 2u);  // Prime its cache.

  ConductorSystem moved = std::move(source);
  EXPECT_EQ(moved.segments_soa().n_segments(), 2u);
  EXPECT_TRUE(source.empty());
  EXPECT_TRUE(source.segments_soa().ax.empty());

  ConductorSystem assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.segments_soa().n_segments(), 2u);
  EXPECT_TRUE(moved.empty());
  EXPECT_TRUE(moved.segments_soa().ax.empty());
}

TEST(ConductorSoa, ThrowsOnNonFiniteCoordinate) {
  const Real nan = std::nan("");
  ConductorSystem cs;
  cs.add({"bad", Real{1.0}, {Vec3{0, 0, 0}, Vec3{nan, 0, 0}}});
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
