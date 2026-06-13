#pragma once

// Axisymmetric cylindrical (r,z) geometric source term for the ideal-MHD slice.
//
// In cylindrical mode the in-plane axes are r (x-axis, column index i) and z
// (y-axis, row index j); the third coordinate phi is ignored (m=0 axisymmetry).
// The conserved-component slot map matched throughout the MHD CT path is:
//
//   mx      -> m_r    (radial momentum)
//   my      -> m_z    (axial  momentum)
//   mz      -> m_phi  (toroidal / azimuthal momentum)
//   bx_face -> B_r    (radial field)
//   by_face -> B_z    (axial  field)
//   bz_cell -> B_phi  (toroidal field)
//
// -- Radial-flux convention and which geometric terms this source supplies -----
// The solver's residual (src/physics/mhd/mhd_solver.cpp) accumulates the
// divergence via launch_mhd_flux_difference, which is a *Cartesian-style*
// flux difference  -(F_{i+1/2} - F_{i-1/2})/dr  in BOTH directions (it does not
// apply the cylindrical conservative operator (1/r) d(r F_r)/dr). With that
// planar radial difference, EVERY conserved component whose radial flux is
// nonzero needs a leftover geometric source equal to (true cylindrical operator
// minus the applied Cartesian dF/dr). This source supplies that FULL 1/r
// correction for all fluid components so axisymmetric mass/momentum/energy are
// conserved and a balanced (r,z) equilibrium is stationary.
//
// -- Exact formulas implemented (authoritative device kernel,
//    src/backend/hip/mhd/mhd_update.hip :: geometric_source_kernel) -----------
// With v = m/rho per slot, p* = p_gas + B^2/2, v.B = v_r B_r + v_z B_z +
// v_phi B_phi, and the leftover = (cylindrical operator) - (Cartesian dF/dr):
//
//   S_{rho}   = -(rho v_r) / r                              -> dudt.rho
//   S_{m_r}   = (rho v_phi^2 - B_phi^2 - rho v_r^2 + B_r^2)/r -> dudt.mx
//   S_{m_z}   = -(rho v_r v_z - B_r B_z) / r                -> dudt.my  (axial)
//   S_{m_phi} = -2 (rho v_r v_phi - B_r B_phi) / r          -> dudt.mz
//   S_{energy}= -((E + p*) v_r - B_r (v.B)) / r             -> dudt.energy
//   S_{B_phi} = S_{B_r} = S_{B_z} = 0
//
// S_{m_r} groups the centrifugal (rho v_phi^2/r) and hoop-tension (-B_phi^2/r)
// terms with the radial self-flux curvature -(rho v_r^2 - B_r^2)/r (p* cancels
// between the cylindrical and Cartesian operators). S_{m_phi} is the angular-
// momentum 2 G/r curvature (the phi flux carries an r^2 metric weight). The
// toroidal field B_phi has NO geometric source: (curl E)_phi = dE_r/dz -
// dE_z/dr is metric-free, so the Cartesian flux difference is already exact;
// the poloidal faces B_r/B_z are advanced by the CT EMF curl, not by this source.
//
// -- On-axis (r -> 0) handling ------------------------------------------------
// The radius is grid.r_at_cell_center(i) = (i+0.5)*dr for a domain starting on
// the axis (origin_x = 0), so EVERY cell center has r > 0 -- the innermost cell
// i=0 sits at r = 0.5*dr and its 1/r factor (= 2/dr) is finite. The geometric
// source is therefore applied at the i=0 column like any other; for a state with
// nonzero radial flow at the axis the source is genuinely nonzero there, and
// skipping it (as an over-eager r <= 0.5*dr guard would) drops an O(1) term while
// the Cartesian flux difference still runs, breaking conservation at the axis
// row. Only a non-positive / non-finite cell-center radius is skipped, which a
// valid grid never produces.
//
// add() is a thin host wrapper that delegates to launch_mhd_geometric_source
// (kernels.hpp); the device kernel is the single authoritative implementation
// of the formulas above. The solver only invokes add() in cylindrical mode (it
// is a pure accumulate, so a Cartesian run never calls it).

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::mhd {

struct MhdGeometricSource {
  // Accumulate the axisymmetric (r,z) geometric source S(u) into dudt (does NOT
  // overwrite). dir convention: r = x (index i), z = y (index j). Delegates to
  // the device kernel launch_mhd_geometric_source on the default stream.
  static void add(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                  const Grid2D& grid, Real gamma);
};

}  // namespace quasar::mhd
