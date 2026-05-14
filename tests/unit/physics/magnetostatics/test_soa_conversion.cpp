#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::PointSoA;
using ::quasar::magnetostatics::SegmentSoA;

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
