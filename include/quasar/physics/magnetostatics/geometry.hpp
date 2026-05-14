#pragma once

#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"

#include <string>
#include <vector>

namespace quasar::magnetostatics {

// Closed circular current loop of radius `radius_m` in the plane perpendicular
// to `axis`, centered at `center`. The polyline has `n_segments` straight
// edges and (n_segments + 1) vertices; the last vertex coincides with the
// first to close the loop.
Filament circular_loop(Vec3 center, Vec3 axis, Real radius_m,
                       int  n_segments, Real current_A,
                       std::string name = "loop");

// Cylindrical helix of `n_turns` wound around `axis`. Axial advance per turn
// is `pitch_m`; the helix is centered on `center`. Discretization is
// `n_segments_per_turn`, so the polyline has `n_turns * n_segments_per_turn`
// segments and one more vertex.
Filament helix(Vec3 center, Vec3 axis, Real radius_m, Real pitch_m,
               int  n_turns, int n_segments_per_turn, Real current_A,
               std::string name = "helix");

// Solenoid: uniform-pitch helix sized by total axial length instead of pitch.
// pitch_m = length_m / n_turns; otherwise identical to helix(...).
Filament solenoid(Vec3 center, Vec3 axis, Real radius_m, Real length_m,
                  int  n_turns, int n_segments_per_turn, Real current_A,
                  std::string name = "solenoid");

// Racetrack coil in the plane perpendicular to `axis`: two straight sections
// of length `straight_length_m` joined by semicircular ends of radius
// `arc_radius_m`. `n_arc_segments` controls discretization per semicircle.
// Closed polyline; total segments = 2*n_arc_segments + 2.
Filament racetrack(Vec3 center, Vec3 axis,
                   Real straight_length_m, Real arc_radius_m,
                   int  n_arc_segments, Real current_A,
                   std::string name = "racetrack");

// Regular `n_sides`-gon inscribed in a circle of circumradius `circumradius_m`
// in the plane perpendicular to `axis`, centered at `center`. Closed polyline.
Filament polygon(Vec3 center, Vec3 axis, Real circumradius_m,
                 int  n_sides, Real current_A,
                 std::string name = "polygon");

// Open polyline through `points` carrying `current_A`. Caller-provided
// vertices are not modified.
Filament generic_polyline(std::vector<Vec3> points, Real current_A,
                          std::string name = "polyline");

}  // namespace quasar::magnetostatics
