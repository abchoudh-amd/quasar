#include "quasar/physics/magnetostatics/geometry.hpp"

#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"

#include <cmath>
#include <cstddef>
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
  const Real al = length(axis_input);
  if (!(al > 0) || !std::isfinite(al)) {
    throw std::invalid_argument{gen + ": axis vector must be non-zero and finite"};
  }
  const Vec3 ah  = axis_input / al;
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

void check_segment_count(const std::string& gen, int n) {
  if (n < 1) {
    throw std::invalid_argument{gen + ": segment count must be >= 1"};
  }
}

void check_finite(const std::string& gen, const std::string& field, Real x) {
  if (!std::isfinite(x)) {
    throw std::invalid_argument{gen + ": " + field + " must be finite"};
  }
}

}  // namespace

Filament circular_loop(Vec3 center, Vec3 axis, Real radius_m,
                       int n_segments, Real current_A,
                       std::string name) {
  check_segment_count("circular_loop", n_segments);
  check_radius("circular_loop", radius_m);
  check_finite("circular_loop", "current_A", current_A);
  const Basis b = make_basis(axis, "circular_loop");

  Filament f{std::move(name), current_A, {}};
  f.points.reserve(static_cast<std::size_t>(n_segments) + 1u);
  for (int k = 0; k <= n_segments; ++k) {
    const Real theta = Real{2} * pi * static_cast<Real>(k)
                                    / static_cast<Real>(n_segments);
    f.points.push_back(center + radius_m * (std::cos(theta) * b.u
                                            + std::sin(theta) * b.v));
  }
  return f;
}

Filament helix(Vec3 center, Vec3 axis, Real radius_m, Real pitch_m,
               int n_turns, int n_segments_per_turn, Real current_A,
               std::string name) {
  if (n_turns < 1) {
    throw std::invalid_argument{"helix: n_turns must be >= 1"};
  }
  check_segment_count("helix", n_segments_per_turn);
  check_radius("helix", radius_m);
  check_finite("helix", "pitch_m",   pitch_m);
  check_finite("helix", "current_A", current_A);
  const Basis b = make_basis(axis, "helix");

  const int  n_total      = n_turns * n_segments_per_turn;
  const Real total_length = static_cast<Real>(n_turns) * pitch_m;
  const Vec3 start        = center - (total_length / Real{2}) * b.axis_hat;

  Filament f{std::move(name), current_A, {}};
  f.points.reserve(static_cast<std::size_t>(n_total) + 1u);
  for (int k = 0; k <= n_total; ++k) {
    const Real t        = static_cast<Real>(k) / static_cast<Real>(n_segments_per_turn);
    const Real theta    = Real{2} * pi * t;
    const Real z_offset = t * pitch_m;
    f.points.push_back(start + z_offset * b.axis_hat
                       + radius_m * (std::cos(theta) * b.u
                                     + std::sin(theta) * b.v));
  }
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
  return helix(center, axis, radius_m, pitch, n_turns, n_segments_per_turn,
               current_A, std::move(name));
}

Filament racetrack(Vec3 center, Vec3 axis,
                   Real straight_length_m, Real arc_radius_m,
                   int n_arc_segments, Real current_A,
                   std::string name) {
  check_segment_count("racetrack", n_arc_segments);
  check_radius("racetrack", arc_radius_m);
  if (!(straight_length_m >= 0) || !std::isfinite(straight_length_m)) {
    throw std::invalid_argument{"racetrack: straight_length_m must be >= 0 and finite"};
  }
  check_finite("racetrack", "current_A", current_A);
  const Basis b = make_basis(axis, "racetrack");

  const Real L = straight_length_m;
  const Real R = arc_radius_m;
  const Vec3 right_center = center + (L / Real{2}) * b.u;
  const Vec3 left_center  = center - (L / Real{2}) * b.u;

  Filament f{std::move(name), current_A, {}};
  f.points.reserve(static_cast<std::size_t>(2 * n_arc_segments + 3));

  // Start at bottom-right corner (transition: bottom straight -> right arc).
  const Vec3 bottom_right = right_center - R * b.v;
  f.points.push_back(bottom_right);

  // Right semicircular arc: theta in (-pi/2, +pi/2], CCW looking down `axis`.
  for (int k = 1; k <= n_arc_segments; ++k) {
    const Real theta = -pi / Real{2}
                       + pi * static_cast<Real>(k)
                              / static_cast<Real>(n_arc_segments);
    f.points.push_back(right_center + R * std::cos(theta) * b.u
                                    + R * std::sin(theta) * b.v);
  }

  // Top straight: top-right corner -> top-left corner.
  f.points.push_back(left_center + R * b.v);

  // Left semicircular arc: theta in (pi/2, 3*pi/2].
  for (int k = 1; k <= n_arc_segments; ++k) {
    const Real theta = pi / Real{2}
                       + pi * static_cast<Real>(k)
                              / static_cast<Real>(n_arc_segments);
    f.points.push_back(left_center + R * std::cos(theta) * b.u
                                   + R * std::sin(theta) * b.v);
  }

  // Bottom straight closes back to start.
  f.points.push_back(bottom_right);

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
  return Filament{std::move(name), current_A, std::move(points)};
}

}  // namespace quasar::magnetostatics
