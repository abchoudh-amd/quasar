#include "quasar/physics/magnetostatics/geometry.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/kernels.hpp"

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

// One orthonormal frame per generator call, built from one axis vector. Scalar
// configuration math: it does not scale with the vertex count, so it stays on
// the host.
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

// Upper bound on generated filament vertices, guarding the allocation below
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

// Host counterparts of the device resolvability checks, retained for the few
// scalar corner points a racetrack computes outside its arc kernels. The device
// versions in src/backend/hip/magnetostatics/geometry_hip.hip are the ones that
// run per vertex; these are the same algorithm at a handful of call sites.
//
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

int checked_vertex_count(const std::string& gen, std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument{gen + ": generated vertex count exceeds the limit"};
  }
  return static_cast<int>(n);
}

void write_vec3(double (&destination)[3], Vec3 v) {
  const Real components[3] = {v.x, v.y, v.z};
  for (int c = 0; c < 3; ++c) destination[c] = components[c];
}

// Every generator kernel reports through the same status word, and each raises
// the same three exceptions from it. `radial` and `axial` name the dimension a
// bit refers to, matching the messages the host loops used. What is lost
// relative to those loops is the index of the offending vertex: an OR-reduced
// predicate does not carry one.
void check_generator_status(int flags, const std::string& gen,
                            const std::string& radial,
                            const std::string& axial = {}) {
  if ((flags & 1) != 0) throw_unresolved(gen, radial);
  if ((flags & 2) != 0) throw_unresolved(gen, axial);
  if ((flags & 4) != 0) {
    throw std::invalid_argument{gen + ": generated coordinate overflowed"};
  }
}

// Allocates the device planes, runs one generator launch into them, and applies
// the status contract above.
template <class Launch>
FilamentPoints generate(std::size_t n, const std::string& gen,
                        const std::string& radial, const std::string& axial,
                        Launch&& launch) {
  FilamentPoints points(n);
  if (n == 0) return points;
  backend::DeviceBuffer<int> status(1);
  launch(points, status);
  int flags = 0;
  status.copy_to_host(&flags, 1);
  check_generator_status(flags, gen, radial, axial);
  return points;
}

void validate_segments(const FilamentPoints& points, const std::string& gen) {
  if (points.size() < 2) return;
  backend::DeviceBuffer<int> status(1);
  ::launch_ms_validate_segments(
      points.x(), points.y(), points.z(),
      checked_vertex_count(gen, points.size()), status.device_ptr(), nullptr);
  int flags = 0;
  status.copy_to_host(&flags, 1);
  if ((flags & 1) != 0) {
    throw std::invalid_argument{
        gen + ": generated a zero-length segment in host precision"};
  }
}

// Writes one scalar corner point into the device planes. Three coordinates, not
// a sweep, so the arithmetic that produced them is ordinary host configuration
// math and only the store crosses to the device.
void write_vertex(FilamentPoints& points, std::size_t index, Vec3 p,
                  const std::string& gen) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
    throw std::invalid_argument{gen + ": generated coordinate overflowed"};
  }
  const Real components[3] = {p.x, p.y, p.z};
  Real* const planes[3] = {points.x(), points.y(), points.z()};
  for (int c = 0; c < 3; ++c) {
    backend::device_memcpy_h2d(planes[c] + index, &components[c], sizeof(Real));
  }
}

// Shared arc-kernel parameter block. theta_k = bias + scale * index_k / divisor,
// with index_k = k + offset, reduced modulo `modulus` when that is nonzero.
QuasarMsArcParams arc_params(Vec3 centre, const Basis& b, Real radius,
                             Real theta_bias, Real theta_scale,
                             int index_offset, int index_modulus,
                             int theta_divisor) {
  QuasarMsArcParams params{};
  write_vec3(params.centre, centre);
  write_vec3(params.u, b.u);
  write_vec3(params.v, b.v);
  params.radius = radius;
  params.theta_bias = theta_bias;
  params.theta_scale = theta_scale;
  params.index_offset = index_offset;
  params.index_modulus = index_modulus;
  params.theta_divisor = theta_divisor;
  params.material_component = kMaterialComponent;
  return params;
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
  const std::size_t n_points = static_cast<std::size_t>(n_segments) + 1u;
  check_point_count("circular_loop", n_points);
  const Basis b = make_basis(axis, "circular_loop");

  const QuasarMsArcParams params = arc_params(
      center, b, radius_m, Real{0}, Real{2} * pi, 0, 0, n_segments);
  FilamentPoints points = generate(
      n_points, "circular_loop", "radius", {},
      [&](FilamentPoints& out, backend::DeviceBuffer<int>& status) {
        ::launch_ms_arc_points(params, n_segments, out.x(), out.y(), out.z(),
                               status.device_ptr(), nullptr);
      });
  // Copy, rather than recompute at 2*pi: closure is bitwise exact and cannot
  // create a microscopic extra segment from sin(2*pi) round-off.
  points.copy_vertex(0, n_points - 1);
  validate_segments(points, "circular_loop");
  return Filament{std::move(name), current_A, std::move(points)};
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

  QuasarMsHelixParams params{};
  write_vec3(params.centre, center);
  write_vec3(params.axis, b.axis_hat);
  write_vec3(params.u, b.u);
  write_vec3(params.v, b.v);
  params.radius = radius_m;
  params.half_length = half_length;
  params.n_total = static_cast<long long>(n_total);
  params.segments_per_turn = n_segments_per_turn;
  params.material_component = kMaterialComponent;

  const std::size_t n_points = n_total + 1u;
  const int count = checked_vertex_count("helix", n_points);
  FilamentPoints points = generate(
      n_points, "helix", "radius", "axial position",
      [&](FilamentPoints& out, backend::DeviceBuffer<int>& status) {
        ::launch_ms_helix_points(params, count, out.x(), out.y(), out.z(),
                                 status.device_ptr(), nullptr);
      });
  validate_segments(points, "helix");
  return Filament{std::move(name), current_A, std::move(points)};
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
  // Four scalar frame points and two corners. None of these scale with
  // n_arc_segments, so they are configuration math and stay on the host; only
  // the two arcs are swept.
  const Vec3 right_offset = scaled_direction(
      half_length, b.u, "racetrack", "straight length");
  const Vec3 left_offset = scaled_direction(
      -half_length, b.u, "racetrack", "straight length");
  const Vec3 right_center = translated_point(
      center, right_offset, half_length, "racetrack", "straight length");
  const Vec3 left_center = translated_point(
      center, left_offset, half_length, "racetrack", "straight length");

  // Start at bottom-right corner (transition: bottom straight -> right arc).
  const Vec3 bottom_radial = scaled_direction(
      -R, b.v, "racetrack", "arc radius");
  const Vec3 bottom_right = translated_point(
      right_center, bottom_radial, R, "racetrack", "arc radius");
  // Top straight: top-right corner -> top-left corner.
  const Vec3 top_radial = scaled_direction(
      R, b.v, "racetrack", "arc radius");
  const Vec3 top_left = translated_point(
      left_center, top_radial, R, "racetrack", "arc radius");

  // Right semicircular arc: theta in (-pi/2, +pi/2], CCW looking down `axis`.
  const QuasarMsArcParams right_arc = arc_params(
      right_center, b, R, -pi / Real{2}, pi, 1, 0, n_arc_segments);
  // Left semicircular arc: theta in (pi/2, 3*pi/2].
  const QuasarMsArcParams left_arc = arc_params(
      left_center, b, R, pi / Real{2}, pi, 1, 0, n_arc_segments);

  const std::size_t left_arc_base =
      static_cast<std::size_t>(n_arc_segments) + 2u;
  FilamentPoints points = generate(
      n_points, "racetrack", "arc radius", {},
      [&](FilamentPoints& out, backend::DeviceBuffer<int>& status) {
        ::launch_ms_arc_points(right_arc, n_arc_segments, out.x() + 1,
                               out.y() + 1, out.z() + 1, status.device_ptr(),
                               nullptr);
        ::launch_ms_arc_points(
            left_arc, n_arc_segments, out.x() + left_arc_base,
            out.y() + left_arc_base, out.z() + left_arc_base,
            status.device_ptr(), nullptr);
      });

  write_vertex(points, 0, bottom_right, "racetrack");
  write_vertex(points, static_cast<std::size_t>(n_arc_segments) + 1u, top_left,
               "racetrack");
  // Bottom straight closes back to start.
  write_vertex(points, n_points - 1, bottom_right, "racetrack");

  validate_segments(points, "racetrack");
  return Filament{std::move(name), current_A, std::move(points)};
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
  // These coordinates came from a deck, not from a calculation, so they are
  // validated here and uploaded rather than generated. The loop is over data
  // the host already holds.
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
  return Filament{std::move(name), current_A, FilamentPoints::upload(points)};
}

}  // namespace quasar::magnetostatics
