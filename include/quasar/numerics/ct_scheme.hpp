#pragma once

// Constrained-transport (CT) scheme interface for the ideal-MHD numerics axis.
//
// A CT scheme keeps the discrete magnetic divergence at round-off by storing the
// in-plane field on cell faces (MhdField2D::bx_face / by_face) and evolving them
// from a corner-staggered electromotive force (EMF). The two-step contract is:
//
//   1. compute_emf : build the corner Ez (EmfField2D::ez_edge) from the SAME
//      reconstructed interface states the Riemann flux uses, so the induction
//      update is consistent with the fluid update.
//   2. update_face_b : advance bx_face / by_face by the discrete curl of that
//      corner Ez. The stencil is chosen so the change in the cell-centered
//      discrete divergence telescopes to *identically* zero for any Ez field --
//      this is the machine-epsilon div(B) guarantee (Evans & Hawley 1988;
//      Balsara & Spicer 1999).
//
// divergence_b_linf is a host-side diagnostic returning max |div B| over the
// interior cells using the same discrete operator the update annihilates.
//
// Staggering convention (must match mhd_field.hpp and the sibling solver):
//   bx_face(i,j) = Bx on the x_lo (left)   face of cell (i,j)
//   by_face(i,j) = By on the y_lo (bottom) face of cell (i,j)
//   ez_edge(i,j) = Ez at the lower-left corner of cell (i,j), shared by cells
//                  (i-1,j-1), (i,j-1), (i-1,j), (i,j).
//   divB(i,j) = (bx_face(i+1,j) - bx_face(i,j))/dx
//             + (by_face(i,j+1) - by_face(i,j))/dy

#include "quasar/core/types.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::numerics {

class ICtScheme {
 public:
  virtual ~ICtScheme() = default;

  // Build the corner-staggered EMF (emf.ez_edge) from the conserved field `u`
  // and the dir=0 / dir=1 reconstructed interface states. `gamma` is the
  // adiabatic index (passed for parity with the flux path; the Ez construction
  // here is purely kinematic, E = v x B).
  virtual void compute_emf(const quasar::mhd::MhdField2D<Real>& u,
                           const MhdInterfaceStates<Real>& ifx,   // dir=0 faces
                           const MhdInterfaceStates<Real>& ify,   // dir=1 faces
                           quasar::mhd::EmfField2D<Real>& emf,
                           Real gamma) const = 0;

  // Advance the face-staggered in-plane B from the corner Ez by the discrete
  // curl of E (dB/dt = -curl E). Conserves div(B) to round-off by construction.
  virtual void update_face_b(quasar::mhd::MhdField2D<Real>& u,
                             const quasar::mhd::EmfField2D<Real>& emf,
                             Real dt) const = 0;

  // Host diagnostic: L-infinity norm of the discrete cell-centered div(B) over
  // the interior cells.
  virtual Real divergence_b_linf(const quasar::mhd::MhdField2D<Real>& u) const = 0;
};

}  // namespace quasar::numerics
