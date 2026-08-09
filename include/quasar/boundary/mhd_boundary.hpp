#pragma once

#include <array>
#include <string>

#include "quasar/core/grid.hpp"  // Side (enum class Side) + Grid2D
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::boundary {

// Per-side boundary selection for the MHD vertical slice (order: x_lo, x_hi,
// y_lo, y_hi). Each name maps directly to a QUASAR_REGISTER_MHD_*_BOUNDARY key,
// so adding a boundary is a single registration with no enum / switch / binding
// to update. Names: "periodic" | "outflow" | "wall" | "axis".  "axis" is
// the cylindrical r=0 parity closure and is valid only on x_lo. The fluid axis
// (rho, m, energy) and the magnetic-field axis (bx_face, by_face, bz_cell) have
// independent selections because a wall imposes different symmetries
// on momentum than on the magnetic field.  "internal" is reserved for the
// distributed tile builder and is not valid physics-deck vocabulary.
struct MhdBoundarySpec {
  std::array<std::string, 4> fluid{"periodic", "periodic", "periodic", "periodic"};
  std::array<std::string, 4> field{"periodic", "periodic", "periodic", "periodic"};
};

// Classify a boundary name as periodic (two-sided wrap) vs non-periodic
// (one-sided closure). Returns true iff `name == "periodic"`; every other
// registered name ("outflow", "wall", ...) is non-periodic. The solver
// uses this to thread per-side one-sided flags into the reconstruction / CT
// kernels (a non-periodic side drops the ghost-GRADIENT dependence; the ghost
// VALUES are still filled and read by the existing fill_ghosts closures).
bool mhd_boundary_is_periodic(const std::string& name);

// Boundary applied to the conserved fluid state (rho, mx, my, mz, energy). The
// concrete implementations self-register under their string name via
// QUASAR_REGISTER_MHD_FLUID_BOUNDARY.
class IMhdFluidBoundary {
 public:
  virtual ~IMhdFluidBoundary() = default;
  virtual int ghost_continuation_mode() const noexcept = 0;
  virtual bool is_periodic() const noexcept {
    return ghost_continuation_mode() == 0;
  }
  virtual bool is_internal_cut() const noexcept {
    return ghost_continuation_mode() == 4;
  }

  // Fill the ghost layers of the fluid components on the given side from the
  // interior, applying the BC's symmetry (wrap / zero-gradient / wall mirror).
  virtual void fill_ghosts(mhd::MhdField2D<Real>& field, Side side) const = 0;

  // Optional post-update correction hook (image-current-style closures). The
  // periodic / outflow / wall BCs do all their work in fill_ghosts, so
  // this defaults to a no-op.
  virtual void correct(mhd::MhdField2D<Real>& /*field*/, Side /*side*/) const {}
};

// Boundary applied to the magnetic field (bx_face, by_face face-staggered;
// bz_cell cell-centered). Self-registers via QUASAR_REGISTER_MHD_FIELD_BOUNDARY.
class IMhdFieldBoundary {
 public:
  virtual ~IMhdFieldBoundary() = default;
  virtual int ghost_continuation_mode() const noexcept = 0;
  virtual bool is_periodic() const noexcept {
    return ghost_continuation_mode() == 0;
  }
  virtual bool is_internal_cut() const noexcept {
    return ghost_continuation_mode() == 4;
  }

  // Fill the ghost layers of the magnetic-field components on the given side
  // from the interior, applying the BC's symmetry for face- and cell-staggered
  // storage.
  virtual void fill_ghosts(mhd::MhdField2D<Real>& field, Side side) const = 0;
};

}  // namespace quasar::boundary
