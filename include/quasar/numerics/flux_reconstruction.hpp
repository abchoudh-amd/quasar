#pragma once

// High-order finite-volume reconstruction for ideal MHD. Given cell-average
// conserved states, each scheme reconstructs LEFT/RIGHT states on every face
// normal to `dir`. A sibling Riemann solver (HLLD) consumes them to form the
// numerical face flux whose control-volume difference advances those averages;
// CT/EMF handling is owned elsewhere.
//
// Three concrete schemes self-register by name (defined in
// src/numerics/flux_reconstruction.cpp):
//   "muscl_minmod" -- 2nd-order minmod-limited reconstruction in PRIMITIVE
//                      variables (nghost=2). Robust baseline; primitive, so
//                      is_characteristic()==false.
//   "mp5"          -- Suresh-Huynh (1997) 5th-order monotonicity-preserving
//                      reconstruction in CHARACTERISTIC variables (nghost=3).
//   "mp7"          -- 7th-order extension with the same MP limiting machinery,
//                      in CHARACTERISTIC variables (nghost=4).
//
// Reconstruction is performed dimension-by-direction (each call reconstructs
// only along the requested normal `dir`); the genuinely-multidimensional corner
// coupling is delegated to the CT/EMF stage that combines the per-direction
// interface states. This is a documented v1 limitation (see the .cpp).

#include "quasar/core/types.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::numerics {

// Interface for a flux-reconstruction scheme. Concrete schemes are stateless and
// selected by string name through the plugin registry.
class IFluxReconstruction {
 public:
  virtual ~IFluxReconstruction() = default;

  // Ghost-cell halo the scheme reads past each boundary (muscl_minmod=2,
  // mp5=3, mp7=4). The solver ctor guards that the grid carries at least this
  // many ghosts before reconstruct_faces runs.
  virtual int required_nghost() const = 0;

  // True when the scheme reconstructs in characteristic variables (mp5, mp7);
  // false for the primitive-variable muscl_minmod baseline.
  virtual bool is_characteristic() const = 0;

  // Reconstruct the LEFT and RIGHT conserved interface states on every interface
  // normal to `dir` (0=x, 1=y), writing them into `out` (out.dir must equal dir).
  // For dir=0, face (i,j) is between cells (i-1,j) and (i,j): the L state is the
  // right-biased extrapolation from the cell-(i-1) side and the R state is the
  // left-biased extrapolation from the cell-(i) side. `gamma` is the adiabatic
  // index used to build the EOS / characteristic eigensystem.
  virtual void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
                                 MhdInterfaceStates<Real>& out, Real gamma) const = 0;
};

// Spatial order of the built-in device reconstruction kernel selected by a
// scheme's required_nghost(): 2 -> 2 (muscl_minmod), 3 -> 5 (mp5), 4 -> 7 (mp7).
//
// The halo width, not the registry name, is the selector: MhdSolver2D sizes its
// working grid from required_nghost() and the device kernel branches on the
// order this returns. Keeping the mapping here -- beside the required_nghost()
// declaration that feeds it -- makes halo and order one fact with one owner, so
// a new scheme cannot pick up a halo and an order that disagree. The Python
// deck layer reads both through the _core.mhd.reconstruction_halo /
// reconstruction_order bindings rather than mirroring this table.
constexpr int reconstruction_order_from_nghost(int nghost) noexcept {
  switch (nghost) {
    case 3: return 5;
    case 4: return 7;
    default: return 2;
  }
}

}  // namespace quasar::numerics
