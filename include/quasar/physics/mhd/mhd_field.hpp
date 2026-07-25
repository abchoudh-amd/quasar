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
  // Cell-centred evolved energy. Without a prescribed background this is the
  // ordinary ideal-MHD total energy. With B=B0+b splitting it is
  // E'=rho*e+|m|^2/(2*rho)+|b|^2/2 and contains no B0 magnetic baseline.
  backend::DeviceBuffer<T> energy{};
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

// Auxiliary momentum-flux channels used by the active-background field split.
// The primary MhdField2D flux buffers carry the material stress H, while these
// buffers carry
//
//   * wave_{x,y,z}: the HLL/HLLD Rankine--Hugoniot correction W, and
//   * cross_b{xyz}: the effective perturbation field entering the background-
//     linear Maxwell stress C(B0,b).
//
// C is deliberately kept factored as B0 times cross_b.  Forming C at each face
// and then differencing two rounded O(B0*b) values would lose a representable
// gradient when a common guide-field stress dominates it.  The update kernel
// instead expands these factors directly into the same exponent accumulator as
// H, W, and the static-background stress.
template <class T>
struct MhdMomentumFluxParts2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> wave_x{};
  backend::DeviceBuffer<T> wave_y{};
  backend::DeviceBuffer<T> wave_z{};
  backend::DeviceBuffer<T> cross_bx{};
  backend::DeviceBuffer<T> cross_by{};
  backend::DeviceBuffer<T> cross_bz{};

  MhdMomentumFluxParts2D() = default;
  explicit MhdMomentumFluxParts2D(Grid2D g)
    : grid{g},
      wave_x{g.storage_size()}, wave_y{g.storage_size()},
      wave_z{g.storage_size()}, cross_bx{g.storage_size()},
      cross_by{g.storage_size()}, cross_bz{g.storage_size()} {}

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
