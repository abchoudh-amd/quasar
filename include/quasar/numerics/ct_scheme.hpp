#pragma once

// Constrained-transport (CT) scheme interface for the ideal-MHD numerics axis.
//
// A CT scheme keeps the discrete magnetic divergence at round-off by storing the
// in-plane field on cell faces (MhdField2D::bx_face / by_face) and evolving them
// from a corner-staggered electromotive force (EMF). The two steps are:
//
//   1. Corner EMF (launch_mhd_ct_emf_prepare / _finish): rerun the directional
//      HLLD magnetic fluxes from the SAME reconstructed interface states as the
//      conservative update, interpret them as upwind face electric fields, then
//      interpolate those face values to the shared corner.
//   2. Face-B rate (launch_mhd_emf_curl_rate): write the discrete curl of that
//      corner Ez into the residual's face-B slots as a RATE, which rk_stage then
//      advances. The stencil is chosen so the change in the cell-centered
//      discrete divergence telescopes to *identically* zero for any Ez field --
//      this is the machine-epsilon div(B) guarantee (Evans & Hawley 1988;
//      Balsara & Spicer 1999). A convex combination of divergence-free fields is
//      divergence-free, so the guarantee survives every SSP-RK stage.
//
// Both steps are direct device launches from MhdSolver2D, not virtual calls.
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
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::numerics {

// Only the div(B) diagnostic is virtual. The EMF construction and the face-B
// advance are deliberately NOT on this interface: MhdSolver2D builds the corner
// EMF with launch_mhd_ct_emf_prepare/_finish (passing its own background,
// boundary flags, and MP5/MP7 order) and then advances face B as an ordinary
// residual component via launch_mhd_emf_curl_rate + rk_stage, so face B rides
// the same SSP-RK convex combination as the other seven components. A virtual
// compute_emf/update_face_b pair could only ever describe a reduced
// periodic/no-background second-order path that nothing calls -- and applying
// the curl directly to the field on top of the flux divergence was the
// double-count bug that shape invites. Keep the seam at the diagnostic, which
// MhdSolver2D::divergence_b_max() genuinely dispatches through.
class ICtScheme {
 public:
  virtual ~ICtScheme() = default;

  // Host diagnostic: L-infinity norm of the discrete cell-centered div(B) over
  // the interior cells.
  virtual Real divergence_b_linf(const quasar::mhd::MhdField2D<Real>& u) const = 0;
};

}  // namespace quasar::numerics
