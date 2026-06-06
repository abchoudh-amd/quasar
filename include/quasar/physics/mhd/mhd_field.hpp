#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"

#include <cstddef>

namespace quasar::mhd {

// Cell-centered conserved MHD state plus constrained-transport (CT) magnetic
// field. The in-plane field components live on cell faces (CT primary storage)
// so the discrete divergence stays at round-off; the out-of-plane (toroidal)
// component is cell-centered. Every buffer is sized to the grid's full padded
// storage (grid.storage_size()), mirroring YeeField2D in core/yee_field.hpp.
template <class T>
struct MhdField2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> rho{};      // cell-centered mass density
  backend::DeviceBuffer<T> mx{};       // cell-centered momentum density x
  backend::DeviceBuffer<T> my{};       // cell-centered momentum density y
  backend::DeviceBuffer<T> mz{};       // cell-centered momentum density z
  backend::DeviceBuffer<T> energy{};   // cell-centered total energy density
  backend::DeviceBuffer<T> bx_face{};  // face-staggered Bx (CT primary storage)
  backend::DeviceBuffer<T> by_face{};  // face-staggered By (CT primary storage)
  backend::DeviceBuffer<T> bz_cell{};  // out-of-plane toroidal Bz (cell-centered)

  MhdField2D() = default;
  explicit MhdField2D(Grid2D g)
    : grid{g},
      rho{g.storage_size()}, mx{g.storage_size()}, my{g.storage_size()},
      mz{g.storage_size()}, energy{g.storage_size()},
      bx_face{g.storage_size()}, by_face{g.storage_size()},
      bz_cell{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }
};

// Edge-staggered electromotive force (EMF) used by the CT update. The toroidal
// Ez lives on cell edges (the z-edge / cell corner in 2D), while the in-plane
// Ex/Ey live on their respective edges. Each buffer is sized to the grid's full
// padded storage, mirroring the MhdField2D constructor style above.
template <class T>
struct EmfField2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> ez_edge{};  // out-of-plane Ez on cell edges (corners)
  backend::DeviceBuffer<T> ex_edge{};  // in-plane Ex on cell edges
  backend::DeviceBuffer<T> ey_edge{};  // in-plane Ey on cell edges

  EmfField2D() = default;
  explicit EmfField2D(Grid2D g)
    : grid{g},
      ez_edge{g.storage_size()}, ex_edge{g.storage_size()},
      ey_edge{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }
};

}  // namespace quasar::mhd
