#pragma once

// Builds a Filament from a literal vertex list, for tests and benchmarks.
//
// Filament vertices live on the device, so the aggregate initialization tests
// used to write -- `Filament{"wire", 1.0, {Vec3{0,0,0}, Vec3{1,0,0}}}` -- no
// longer compiles. This restores that shape with an upload.
//
// It deliberately bypasses `generic_polyline`, which validates finiteness and
// rejects coincident consecutive vertices: several tests feed exactly that kind
// of degenerate geometry in order to check that the flatten or the evaluator
// rejects it downstream.

#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace quasar::test {

inline magnetostatics::Filament filament(std::string name, Real current_A,
                                         const std::vector<Vec3>& vertices) {
  return magnetostatics::Filament{
      std::move(name), current_A,
      magnetostatics::FilamentPoints::upload(vertices)};
}

}  // namespace quasar::test
