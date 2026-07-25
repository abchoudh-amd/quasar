#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

using ::quasar::pi;
using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::generic_polyline;
using ::quasar::magnetostatics::helix;
using ::quasar::magnetostatics::polygon;
using ::quasar::magnetostatics::racetrack;
using ::quasar::magnetostatics::solenoid;

namespace {

Real dist(Vec3 a, Vec3 b) noexcept { return length(a - b); }

}  // namespace

TEST(CircularLoop, ProducesClosedRingOnCorrectCircle) {
  const Vec3 center{Real{1}, Real{2}, Real{3}};
  const Vec3 axis{Real{0}, Real{0}, Real{1}};
  const Real R = Real{0.5};
  const int  N = 32;

  const Filament fil = circular_loop(center, axis, R, N, Real{2.0}, "loop_test");
  ASSERT_EQ(fil.points.size(), static_cast<std::size_t>(N) + 1u);

  for (const auto& p : fil.points) {
    const Vec3 r = p - center;
    EXPECT_NEAR(r.z, Real{0}, Real{1e-14});
    EXPECT_NEAR(std::sqrt(r.x * r.x + r.y * r.y), R, Real{1e-14});
  }
  EXPECT_EQ(fil.points.front().x, fil.points.back().x);
  EXPECT_EQ(fil.points.front().y, fil.points.back().y);
  EXPECT_EQ(fil.points.front().z, fil.points.back().z);

  EXPECT_EQ(fil.current_A, Real{2.0});
  EXPECT_EQ(fil.name, "loop_test");
}

TEST(CircularLoop, ThrowsOnBadArguments) {
  EXPECT_THROW(circular_loop({0,0,0}, {0,0,1}, /*r=*/Real{-1}, 4, Real{1}),
               std::invalid_argument);
  EXPECT_THROW(circular_loop({0,0,0}, {0,0,1}, Real{1}, /*n=*/0, Real{1}),
               std::invalid_argument);
  EXPECT_THROW(circular_loop({0,0,0}, {0,0,1}, Real{1}, /*n=*/1, Real{1}),
               std::invalid_argument);
  EXPECT_THROW(circular_loop({0,0,0}, {0,0,1}, Real{1}, /*n=*/2, Real{1}),
               std::invalid_argument);
  EXPECT_THROW(circular_loop({0,0,0}, {0,0,1}, Real{1},
                             std::numeric_limits<int>::max(), Real{1}),
               std::invalid_argument);
  EXPECT_THROW(circular_loop({0,0,0}, {0,0,0}, Real{1}, 4, Real{1}),
               std::invalid_argument);
}

TEST(CircularLoop, NormalizesExtremeFiniteAxisWithoutOverflow) {
  const Real largest = std::numeric_limits<Real>::max();
  const auto fil = circular_loop({0,0,0}, {largest, largest, largest},
                                 Real{1}, 8, Real{1});
  ASSERT_EQ(fil.points.size(), 9u);
  for (const Vec3 p : fil.points) {
    EXPECT_TRUE(std::isfinite(p.x));
    EXPECT_TRUE(std::isfinite(p.y));
    EXPECT_TRUE(std::isfinite(p.z));
    EXPECT_NEAR(length(p), Real{1}, Real{2e-15});
  }
}

TEST(CircularLoop, RejectsRadiusThatDisappearsAtTranslatedCenter) {
  const Real largest = std::numeric_limits<Real>::max();
  EXPECT_THROW(circular_loop({largest, 0, 0}, {0, 0, 1}, Real{1}, 4, Real{1}),
               std::invalid_argument);
}

TEST(Helix, RejectsDiscretizationThatCannotRepresentATurn) {
  EXPECT_THROW(helix({0,0,0}, {0,0,1}, Real{1}, Real{0}, 1, 1, Real{1}),
               std::invalid_argument);
  EXPECT_THROW(helix({0,0,0}, {0,0,1}, Real{1}, Real{0.1}, 1, 2, Real{1}),
               std::invalid_argument);
  EXPECT_THROW(helix({0,0,0}, {0,0,1}, Real{1}, Real{0.1},
                     std::numeric_limits<int>::max(), 3, Real{1}),
               std::invalid_argument);
}

TEST(Helix, AdvancesByPitchPerTurnAndKeepsRadiusConstant) {
  const Vec3 axis{Real{0}, Real{0}, Real{1}};
  const Real R     = Real{0.1};
  const Real pitch = Real{0.05};
  const int  turns = 3;
  const int  n_per = 16;
  const Filament fil = helix({0,0,0}, axis, R, pitch, turns, n_per, Real{1}, "h");

  const int N = turns * n_per;
  ASSERT_EQ(fil.points.size(), static_cast<std::size_t>(N) + 1u);

  const Real total = static_cast<Real>(turns) * pitch;
  EXPECT_NEAR(fil.points.front().z, -total / Real{2}, Real{1e-12});
  EXPECT_NEAR(fil.points.back ().z, +total / Real{2}, Real{1e-12});

  for (const auto& p : fil.points) {
    EXPECT_NEAR(std::sqrt(p.x * p.x + p.y * p.y), R, Real{1e-12});
  }

  EXPECT_NEAR(fil.points[n_per].z - fil.points[0].z, pitch, Real{1e-12});
  EXPECT_NEAR(fil.points[2 * n_per].z - fil.points[0].z, Real{2} * pitch,
              Real{1e-12});
}

TEST(Helix, RejectsTranslatedOrSubsegmentGeometryThatCannotBeRepresented) {
  const Real largest = std::numeric_limits<Real>::max();
  EXPECT_THROW(helix({largest, 0, 0}, {0, 0, 1}, Real{1}, Real{1},
                     1, 4, Real{1}),
               std::invalid_argument);

  const Real smallest = std::numeric_limits<Real>::denorm_min();
  EXPECT_THROW(helix({0, 0, 0}, {0, 0, 1}, Real{1}, smallest,
                     1, 3, Real{1}),
               std::invalid_argument);
}

TEST(Solenoid, EquivalentToHelixWithPitchEqLengthOverTurns) {
  const Vec3 axis{Real{0}, Real{1}, Real{0}};
  const Real R     = Real{0.05};
  const Real L     = Real{0.6};
  const int  turns = 6;
  const int  n_per = 12;

  const Filament a = solenoid({0,0,0}, axis, R, L,            turns, n_per, Real{1}, "s");
  const Filament b = helix   ({0,0,0}, axis, R, L / turns,    turns, n_per, Real{1}, "h");

  ASSERT_EQ(a.points.size(), b.points.size());
  for (std::size_t i = 0; i < a.points.size(); ++i) {
    EXPECT_LT(dist(a.points[i], b.points[i]), Real{1e-14});
  }
}

TEST(Solenoid, RejectsPositiveLengthWhosePitchUnderflows) {
  const Real smallest = std::numeric_limits<Real>::denorm_min();
  EXPECT_THROW(solenoid({0, 0, 0}, {0, 0, 1}, Real{1}, smallest,
                        2, 4, Real{1}),
               std::invalid_argument);
}

TEST(Racetrack, HasExpectedVertexCountAndPerimeter) {
  const int  n_arc = 8;
  const Real L = Real{1.0};
  const Real R = Real{0.2};

  const Filament fil = racetrack({0,0,0}, {0,0,1}, L, R, n_arc, Real{1}, "rt");
  ASSERT_EQ(fil.points.size(), static_cast<std::size_t>(2 * n_arc + 3));

  EXPECT_LT(dist(fil.points.front(), fil.points.back()), Real{1e-14});

  Real perim = Real{0};
  for (std::size_t i = 0; i + 1 < fil.points.size(); ++i) {
    perim += dist(fil.points[i], fil.points[i + 1]);
  }
  const Real exact = Real{2} * L + Real{2} * pi * R;
  // n_arc=8 polygon-approximation underestimate ~0.25%; allow 1% slack.
  EXPECT_NEAR(perim, exact, Real{0.01} * exact);
}

TEST(Racetrack, RejectsZeroStraightLengthInsteadOfCreatingDuplicateEdges) {
  EXPECT_THROW(racetrack({0,0,0}, {0,0,1}, Real{0}, Real{0.2}, 8, Real{1}),
               std::invalid_argument);
}

TEST(Racetrack, RejectsLocalDimensionsLostAtTranslatedCenter) {
  const Real largest = std::numeric_limits<Real>::max();
  EXPECT_THROW(racetrack({largest, 0, 0}, {0, 0, 1}, Real{2}, Real{1},
                         4, Real{1}),
               std::invalid_argument);
}

TEST(Polygon, HasEqualSideLengthsAndClosesBack) {
  const int  N = 6;
  const Real R = Real{1.0};
  const Filament fil = polygon({0,0,0}, {0,0,1}, R, N, Real{1}, "hex");
  ASSERT_EQ(fil.points.size(), static_cast<std::size_t>(N) + 1u);

  const Real side_expected = Real{2} * R * std::sin(pi / static_cast<Real>(N));
  for (std::size_t i = 0; i + 1 < fil.points.size(); ++i) {
    EXPECT_NEAR(dist(fil.points[i], fil.points[i + 1]), side_expected,
                Real{1e-12});
  }
}

TEST(Polygon, RejectsTooFewSides) {
  EXPECT_THROW(polygon({0,0,0}, {0,0,1}, Real{1}, /*n_sides=*/2, Real{1}),
               std::invalid_argument);
}

TEST(Polygon, RejectsRadiusThatDisappearsAtTranslatedCenter) {
  const Real largest = std::numeric_limits<Real>::max();
  EXPECT_THROW(polygon({largest, 0, 0}, {0, 0, 1}, Real{1}, 4, Real{1}),
               std::invalid_argument);
}

TEST(GenericPolyline, PassesThroughGivenVertices) {
  const std::vector<Vec3> pts = {
    Vec3{0,0,0}, Vec3{1,0,0}, Vec3{1,1,0}, Vec3{1,1,1}
  };
  const Filament fil = generic_polyline(pts, Real{2.5}, "gp");
  ASSERT_EQ(fil.points.size(), pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i) {
    EXPECT_LT(dist(fil.points[i], pts[i]), Real{1e-14});
  }
  EXPECT_EQ(fil.current_A, Real{2.5});
  EXPECT_EQ(fil.name, "gp");
}

TEST(GenericPolyline, ThrowsOnFewerThanTwoPoints) {
  EXPECT_THROW(generic_polyline({Vec3{0,0,0}}, Real{1}, "bad"),
               std::invalid_argument);
}

TEST(GenericPolyline, RejectsNonFiniteAndDuplicateVertices) {
  EXPECT_THROW(generic_polyline({Vec3{0,0,0}, Vec3{0,0,0}}, Real{1}, "dup"),
               std::invalid_argument);
  EXPECT_THROW(generic_polyline(
                   {Vec3{0,0,0}, Vec3{std::numeric_limits<Real>::infinity(),0,0}},
                   Real{1}, "inf"),
               std::invalid_argument);
}

TEST(GeometryGenerators, OutputsAreValidForConductorSystemFlattening) {
  // End-to-end sanity: each generator's Filament should pass through
  // ConductorSystem::to_segments_soa() without throwing.
  using ::quasar::magnetostatics::ConductorSystem;

  ConductorSystem cs;
  cs.add(circular_loop({0,0,0}, {0,0,1}, Real{0.1}, 36, Real{1.0}));
  cs.add(helix       ({0,0,0}, {0,0,1}, Real{0.05}, Real{0.02}, 4, 24,
                       Real{1.0}));
  cs.add(solenoid    ({0.2, 0, 0}, {0,0,1}, Real{0.04}, Real{0.1}, 8, 24,
                       Real{1.0}));
  cs.add(racetrack   ({0,0.5,0}, {0,0,1}, Real{0.2}, Real{0.05}, 16,
                       Real{1.0}));
  cs.add(polygon     ({0,-0.5,0}, {0,0,1}, Real{0.05}, 5, Real{1.0}));

  EXPECT_NO_THROW(cs.to_segments_soa());

  const auto soa = cs.to_segments_soa();
  EXPECT_GT(soa.n_segments(), 0u);
}
