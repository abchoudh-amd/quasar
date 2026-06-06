#pragma once

#include "quasar/backend/memory.hpp"  // backend::DeviceBuffer
#include "quasar/core/grid.hpp"       // Grid2D
#include "quasar/core/types.hpp"      // Real

#include <cstddef>
#include <string>

namespace quasar::mhd {

// Solver-owned static background magnetic field B0. The components use the same
// staggering as the evolved state's B in MhdField2D: the in-plane B0x/B0y live on
// cell faces (CT primary storage) and the out-of-plane toroidal B0z is
// cell-centered. Each buffer is sized to the grid's full padded storage
// (grid.storage_size()), allocated identically to MhdField2D's bx_face/by_face/
// bz_cell. When `active` is false the background is identically zero and solvers
// take a fast path that skips reading these buffers.
template <class T>
struct MhdBackgroundField {
  Grid2D grid{};
  backend::DeviceBuffer<T> b0x_face{};  // same staggering as MhdField2D::bx_face
  backend::DeviceBuffer<T> b0y_face{};  // same staggering as MhdField2D::by_face
  backend::DeviceBuffer<T> b0z_cell{};  // same staggering as MhdField2D::bz_cell
  bool active{false};                   // false => B0 identically zero (fast path)

  MhdBackgroundField() = default;
  explicit MhdBackgroundField(Grid2D g)
    : grid{g},
      b0x_face{g.storage_size()}, b0y_face{g.storage_size()},
      b0z_cell{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }
};

// Deck-facing background specification. `profile` is a registry name selecting
// the background construction scheme (default: a constant uniform vector). The
// bx0/by0/bz0 fields are the uniform-vector parameters consumed when
// profile == "uniform".
struct MhdBackgroundSpec {
  bool enabled{false};
  std::string profile{"uniform"};  // registry name (default uniform vector)
  Real bx0{}, by0{}, bz0{};        // uniform-vector params (profile="uniform")
};

}  // namespace quasar::mhd
