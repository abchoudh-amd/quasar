#include "quasar/physics/magnetostatics/geometry.hpp"

#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quasar::magnetostatics {

namespace {

struct Basis {
  Vec3 axis_hat;
  Vec3 u;
  Vec3 v;
};

Basis make_basis(Vec3 axis_input, const std::string& gen) {
  if (!std::isfinite(axis_input.x) || !std::isfinite(axis_input.y)
      || !std::isfinite(axis_input.z)) {
    throw std::invalid_argument{gen + ": axis vector must be non-zero and finite"};
  }
  const Real scale = std::max({std::abs(axis_input.x), std::abs(axis_input.y),
                               std::abs(axis_input.z)});
  if (!(scale > 0)) {
    throw std::invalid_argument{gen + ": axis vector must be non-zero and finite"};
  }
  const Vec3 scaled = axis_input / scale;
  const Real scaled_length = std::hypot(scaled.x, scaled.y, scaled.z);
  const Vec3 ah = scaled / scaled_length;
  // Pick a reference vector that is not (nearly) parallel to ah.
  const Vec3 ref = (std::abs(ah.x) < Real{0.9}) ? Vec3{Real{1}, Real{0}, Real{0}}
                                                 : Vec3{Real{0}, Real{1}, Real{0}};
  const Vec3 u   = normalized(cross(ref, ah));
  const Vec3 v   = cross(ah, u);  // unit, by construction
  return Basis{ah, u, v};
}

void check_radius(const std::string& gen, Real r) {
  if (!(r > 0) || !std::isfinite(r)) {
    throw std::invalid_argument{gen + ": radius must be > 0 and finite"};
  }
}

void check_segment_count(const std::string& gen, int n, int minimum = 1) {
  if (n < minimum) {
    throw std::invalid_argument{gen + ": segment count must be >= "
                                + std::to_string(minimum)};
  }
}

// Upper bound on generated filament vertices, guarding the reserve()/loop below
// against signed-int overflow from a hostile or typo'd deck (n_turns *
// n_segments_per_turn is computed in size_t here, not int).
constexpr std::size_t kMaxFilamentPoints = std::size_t{1} << 26;  // ~67M vertices

std::size_t checked_total_points(const std::string& gen, int n_turns,
                                 int n_segments_per_turn) {
  const std::size_t turns = static_cast<std::size_t>(n_turns);
  const std::size_t per_turn = static_cast<std::size_t>(n_segments_per_turn);
  if (turns > kMaxFilamentPoints / per_turn) {
    throw std::invalid_argument{
        gen + ": n_turns * n_segments_per_turn exceeds the vertex limit"};
  }
  const std::size_t total = turns * per_turn;
  if (total > kMaxFilamentPoints) {
    throw std::invalid_argument{
        gen + ": n_turns * n_segments_per_turn exceeds the vertex limit"};
  }
  return total;
}

void check_point_count(const std::string& gen, std::size_t n) {
  if (n > kMaxFilamentPoints) {
    throw std::invalid_argument{gen + ": generated vertex count exceeds the limit"};
  }
}

void check_finite(const std::string& gen, const std::string& field, Real x) {
  if (!std::isfinite(x)) {
    throw std::invalid_argument{gen + ": " + field + " must be finite"};
  }
}

constexpr Real kMaterialComponent =
    Real{64} * std::numeric_limits<Real>::epsilon();

[[noreturn]] void throw_unresolved(const std::string& gen,
                                   const std::string& dimension) {
  throw std::invalid_argument{
      gen + ": " + dimension
      + " is not representable at the requested center"};
}

// Form a requested local displacement without silently losing a material
// component to underflow.  Components smaller than a few ulp of the full
// direction are immaterial (and include the expected sin(pi) residue), while
// every material component must retain at least one useful binary digit.
Vec3 scaled_direction(Real distance, Vec3 direction, const std::string& gen,
                      const std::string& dimension) {
  const Vec3 offset = distance * direction;
  if (distance == Real{0}) return offset;

  const Real magnitude = std::abs(distance);
  const Real direction_components[] = {direction.x, direction.y, direction.z};
  const Real offset_components[] = {offset.x, offset.y, offset.z};
  for (int component = 0; component < 3; ++component) {
    const Real unit_component = std::abs(direction_components[component]);
    if (unit_component <= kMaterialComponent) continue;
    const Real actual = offset_components[component];
    if (actual == Real{0} || !std::isfinite(actual)
        || std::signbit(actual)
            != std::signbit(distance * direction_components[component])) {
      throw_unresolved(gen, dimension);
    }
    // Dividing back by the requested distance avoids forming an exact
    // distance*direction reference that may itself be subnormal.  A ratio
    // outside [1/2, 3/2] means even the first significant bit of this local
    // component was lost.
    const Real ratio = std::abs((actual / magnitude) / unit_component);
    if (!std::isfinite(ratio) || ratio < Real{0.5} || ratio > Real{1.5}) {
      throw_unresolved(gen, dimension);
    }
  }
  return offset;
}

// Translate a locally meaningful displacement into global coordinates, then
// round-trip it back.  Merely checking that the final point is finite is not
// enough: at a large center, one entire radial/axial dimension can round away
// while the remaining coordinates still form non-zero segments.
Vec3 translated_point(Vec3 origin, Vec3 offset, Real local_scale,
                      const std::string& gen,
                      const std::string& dimension) {
  const Vec3 point = origin + offset;
  if (!std::isfinite(point.x) || !std::isfinite(point.y)
      || !std::isfinite(point.z)) {
    throw std::invalid_argument{gen + ": generated coordinate overflowed"};
  }

  if (local_scale > Real{0}) {
    const Vec3 recovered = point - origin;
    const Real intended_components[] = {offset.x, offset.y, offset.z};
    const Real recovered_components[] = {recovered.x, recovered.y, recovered.z};
    for (int component = 0; component < 3; ++component) {
      const Real intended = intended_components[component];
      if (std::abs(intended) <= kMaterialComponent * local_scale) continue;
      const Real actual = recovered_components[component];
      if (actual == Real{0} || !std::isfinite(actual)
          || std::signbit(actual) != std::signbit(intended)) {
        throw_unresolved(gen, dimension);
      }
      const Real ratio = std::abs(actual / intended);
      if (!std::isfinite(ratio) || ratio < Real{0.5} || ratio > Real{1.5}) {
        throw_unresolved(gen, dimension);
      }
    }
  }
  return point;
}

void append_point(Filament& f, Vec3 p, const std::string& gen) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
    throw std::invalid_argument{gen + ": generated coordinate overflowed"};
  }
  f.points.push_back(p);
}

void validate_segments(const Filament& f, const std::string& gen) {
  for (std::size_t i = 1; i < f.points.size(); ++i) {
    const Vec3& a = f.points[i - 1];
    const Vec3& b = f.points[i];
    if (a.x == b.x && a.y == b.y && a.z == b.z) {
      throw std::invalid_argument{
          gen + ": generated a zero-length segment in host precision"};
    }
  }
}

}  // namespace

Filament circular_loop(Vec3 center, Vec3 axis, Real radius_m,
                       int n_segments, Real current_A,
                       std::string name) {
  check_segment_count("circular_loop", n_segments, 3);
  check_finite("circular_loop", "center.x", center.x);
  check_finite("circular_loop", "center.y", center.y);
  check_finite("circular_loop", "center.z", center.z);
  check_radius("circular_loop", radius_m);
  check_finite("circular_loop", "current_A", current_A);
  check_point_count("circular_loop", static_cast<std::size_t>(n_segments) + 1u);
  const Basis b = make_basis(axis, "circular_loop");

  Filament f{std::move(name), current_A, {}};
  f.points.reserve(static_cast<std::size_t>(n_segments) + 1u);
  for (int k = 0; k < n_segments; ++k) {
    const Real theta = Real{2} * pi * static_cast<Real>(k)
                                    / static_cast<Real>(n_segments);
    const Vec3 radial_direction =
        std::cos(theta) * b.u + std::sin(theta) * b.v;
    const Vec3 radial = scaled_direction(
        radius_m, radial_direction, "circular_loop", "radius");
    append_point(f, translated_point(center, radial, radius_m,
                                     "circular_loop", "radius"),
                 "circular_loop");
  }
  // Copy, rather than recompute at 2*pi: closure is bitwise exact and cannot
  // create a microscopic extra segment from sin(2*pi) round-off.
  f.points.push_back(f.points.front());
  validate_segments(f, "circular_loop");
  return f;
}

Filament helix(Vec3 center, Vec3 axis, Real radius_m, Real pitch_m,
               int n_turns, int n_segments_per_turn, Real current_A,
               std::string name) {
  if (n_turns < 1) {
    throw std::invalid_argument{"helix: n_turns must be >= 1"};
  }
  check_segment_count("helix", n_segments_per_turn, 3);
  check_finite("helix", "center.x", center.x);
  check_finite("helix", "center.y", center.y);
  check_finite("helix", "center.z", center.z);
  check_radius("helix", radius_m);
  check_finite("helix", "pitch_m",   pitch_m);
  check_finite("helix", "current_A", current_A);
  const Basis b = make_basis(axis, "helix");

  const std::size_t n_total = checked_total_points("helix", n_turns,
                                                   n_segments_per_turn);
  check_point_count("helix", n_total + 1u);
  const Real total_length = static_cast<Real>(n_turns) * pitch_m;
  if (!std::isfinite(total_length)) {
    throw std::invalid_argument{"helix: n_turns * pitch_m must be finite"};
  }
  if (pitch_m != Real{0}
      && pitch_m / static_cast<Real>(n_segments_per_turn) == Real{0}) {
    throw_unresolved("helix", "axial advance per segment");
  }
  const Real half_length = total_length / Real{2};
  if (total_length != Real{0} && half_length == Real{0}) {
    throw_unresolved("helix", "axial half-length");
  }

  Filament f{std::move(name), current_A, {}};
  f.points.reserve(n_total + 1u);
  for (std::size_t k = 0; k <= n_total; ++k) {
    const Real fraction = static_cast<Real>(k) / static_cast<Real>(n_total);
    const Real axial_distance = std::lerp(-half_length, half_length, fraction);
    const Vec3 axial_offset = scaled_direction(
        axial_distance, b.axis_hat, "helix", "axial position");
    const Vec3 turn_center = translated_point(
        center, axial_offset, std::abs(axial_distance),
        "helix", "axial position");
    // Radial phase is periodic. Reducing the integer index before converting to
    // an angle prevents argument-reduction drift after many turns.
    const std::size_t phase = k % static_cast<std::size_t>(n_segments_per_turn);
    const Real theta = Real{2} * pi * static_cast<Real>(phase)
                     / static_cast<Real>(n_segments_per_turn);
    const Vec3 radial_direction =
        std::cos(theta) * b.u + std::sin(theta) * b.v;
    const Vec3 radial = scaled_direction(
        radius_m, radial_direction, "helix", "radius");
    append_point(f, translated_point(turn_center, radial, radius_m,
                                     "helix", "radius"),
                 "helix");
  }
  validate_segments(f, "helix");
  return f;
}

Filament solenoid(Vec3 center, Vec3 axis, Real radius_m, Real length_m,
                  int n_turns, int n_segments_per_turn, Real current_A,
                  std::string name) {
  if (n_turns < 1) {
    throw std::invalid_argument{"solenoid: n_turns must be >= 1"};
  }
  if (!(length_m > 0) || !std::isfinite(length_m)) {
    throw std::invalid_argument{"solenoid: length_m must be > 0 and finite"};
  }
  const Real pitch = length_m / static_cast<Real>(n_turns);
  if (pitch == Real{0}) {
    throw_unresolved("solenoid", "pitch");
  }
  return helix(center, axis, radius_m, pitch, n_turns, n_segments_per_turn,
               current_A, std::move(name));
}

Filament racetrack(Vec3 center, Vec3 axis,
                   Real straight_length_m, Real arc_radius_m,
                   int n_arc_segments, Real current_A,
                   std::string name) {
  check_segment_count("racetrack", n_arc_segments);
  check_finite("racetrack", "center.x", center.x);
  check_finite("racetrack", "center.y", center.y);
  check_finite("racetrack", "center.z", center.z);
  check_radius("racetrack", arc_radius_m);
  if (!(straight_length_m > 0) || !std::isfinite(straight_length_m)) {
    throw std::invalid_argument{"racetrack: straight_length_m must be > 0 and finite"};
  }
  check_finite("racetrack", "current_A", current_A);
  const Basis b = make_basis(axis, "racetrack");

  const std::size_t n_points = std::size_t{2}
                             * static_cast<std::size_t>(n_arc_segments) + 3u;
  check_point_count("racetrack", n_points);

  const Real L = straight_length_m;
  const Real R = arc_radius_m;
  const Real half_length = L / Real{2};
  if (half_length == Real{0}) {
    throw_unresolved("racetrack", "straight half-length");
  }
  const Vec3 right_offset = scaled_direction(
      half_length, b.u, "racetrack", "straight length");
  const Vec3 left_offset = scaled_direction(
      -half_length, b.u, "racetrack", "straight length");
  const Vec3 right_center = translated_point(
      center, right_offset, half_length, "racetrack", "straight length");
  const Vec3 left_center = translated_point(
      center, left_offset, half_length, "racetrack", "straight length");

  Filament f{std::move(name), current_A, {}};
  f.points.reserve(n_points);

  // Start at bottom-right corner (transition: bottom straight -> right arc).
  const Vec3 bottom_radial = scaled_direction(
      -R, b.v, "racetrack", "arc radius");
  const Vec3 bottom_right = translated_point(
      right_center, bottom_radial, R, "racetrack", "arc radius");
  append_point(f, bottom_right, "racetrack");

  // Right semicircular arc: theta in (-pi/2, +pi/2], CCW looking down `axis`.
  for (int k = 1; k <= n_arc_segments; ++k) {
    const Real theta = -pi / Real{2}
                       + pi * static_cast<Real>(k)
                              / static_cast<Real>(n_arc_segments);
    const Vec3 radial_direction =
        std::cos(theta) * b.u + std::sin(theta) * b.v;
    const Vec3 radial = scaled_direction(
        R, radial_direction, "racetrack", "arc radius");
    append_point(f, translated_point(right_center, radial, R,
                                     "racetrack", "arc radius"),
                 "racetrack");
  }

  // Top straight: top-right corner -> top-left corner.
  const Vec3 top_radial = scaled_direction(
      R, b.v, "racetrack", "arc radius");
  append_point(f, translated_point(left_center, top_radial, R,
                                   "racetrack", "arc radius"),
               "racetrack");

  // Left semicircular arc: theta in (pi/2, 3*pi/2].
  for (int k = 1; k <= n_arc_segments; ++k) {
    const Real theta = pi / Real{2}
                       + pi * static_cast<Real>(k)
                              / static_cast<Real>(n_arc_segments);
    const Vec3 radial_direction =
        std::cos(theta) * b.u + std::sin(theta) * b.v;
    const Vec3 radial = scaled_direction(
        R, radial_direction, "racetrack", "arc radius");
    append_point(f, translated_point(left_center, radial, R,
                                     "racetrack", "arc radius"),
                 "racetrack");
  }

  // Bottom straight closes back to start.
  append_point(f, bottom_right, "racetrack");

  validate_segments(f, "racetrack");
  return f;
}

Filament polygon(Vec3 center, Vec3 axis, Real circumradius_m, int n_sides,
                 Real current_A, std::string name) {
  if (n_sides < 3) {
    throw std::invalid_argument{"polygon: n_sides must be >= 3"};
  }
  return circular_loop(center, axis, circumradius_m, n_sides, current_A,
                       std::move(name));
}

Filament generic_polyline(std::vector<Vec3> points, Real current_A,
                          std::string name) {
  check_finite("generic_polyline", "current_A", current_A);
  if (points.size() < 2) {
    throw std::invalid_argument{"generic_polyline: needs at least 2 points"};
  }
  check_point_count("generic_polyline", points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!std::isfinite(points[i].x) || !std::isfinite(points[i].y)
        || !std::isfinite(points[i].z)) {
      throw std::invalid_argument{
          "generic_polyline: point " + std::to_string(i) + " must be finite"};
    }
    if (i > 0 && points[i].x == points[i - 1].x
        && points[i].y == points[i - 1].y
        && points[i].z == points[i - 1].z) {
      throw std::invalid_argument{
          "generic_polyline: consecutive points must be distinct"};
    }
  }
  return Filament{std::move(name), current_A, std::move(points)};
}

}  // namespace quasar::magnetostatics
