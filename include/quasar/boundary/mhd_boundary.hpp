#pragma once

#include <array>
#include <string>

#include "quasar/core/grid.hpp"  // Side (enum class Side) + Grid2D
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::boundary {

// Per-side boundary selection for the MHD vertical slice (order: x_lo, x_hi,
// y_lo, y_hi). Each name maps directly to a QUASAR_REGISTER_MHD_*_BOUNDARY key,
// so adding a boundary is a single registration with no enum / switch / binding
// to update. Names: "periodic" | "outflow" | "reflecting". The fluid axis
// (rho, m, energy) and the magnetic-field axis (bx_face, by_face, bz_cell) have
// independent selections because a reflecting wall imposes different symmetries
// on momentum than on the magnetic field.
struct MhdBoundarySpec {
  std::array<std::string, 4> fluid{"periodic", "periodic", "periodic", "periodic"};
  std::array<std::string, 4> field{"periodic", "periodic", "periodic", "periodic"};
};

// Boundary applied to the conserved fluid state (rho, mx, my, mz, energy). The
// concrete implementations self-register under their string name via
// QUASAR_REGISTER_MHD_FLUID_BOUNDARY.
class IMhdFluidBoundary {
 public:
  virtual ~IMhdFluidBoundary() = default;

  // Fill the ghost layers of the fluid components on the given side from the
  // interior, applying the BC's symmetry (wrap / zero-gradient / wall mirror).
  virtual void fill_ghosts(mhd::MhdField2D<Real>& field, Side side) const = 0;

  // Optional post-update correction hook (image-current-style closures). The
  // periodic / outflow / reflecting BCs do all their work in fill_ghosts, so
  // this defaults to a no-op.
  virtual void correct(mhd::MhdField2D<Real>& /*field*/, Side /*side*/) const {}
};

// Boundary applied to the magnetic field (bx_face, by_face face-staggered;
// bz_cell cell-centered). Self-registers via QUASAR_REGISTER_MHD_FIELD_BOUNDARY.
class IMhdFieldBoundary {
 public:
  virtual ~IMhdFieldBoundary() = default;

  // Fill the ghost layers of the magnetic-field components on the given side
  // from the interior, applying the BC's symmetry for face- and cell-staggered
  // storage.
  virtual void fill_ghosts(mhd::MhdField2D<Real>& field, Side side) const = 0;
};

}  // namespace quasar::boundary
