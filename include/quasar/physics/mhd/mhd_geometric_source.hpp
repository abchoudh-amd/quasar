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
// planar radial difference, the leftover geometric source that the cylindrical
// divergence would otherwise produce is the standard axisymmetric ideal-MHD
// curvature source. This source supplies that FULL 1/r correction so a balanced
// (r,z) equilibrium is stationary.
//
// -- Exact formulas implemented (authoritative device kernel,
//    src/backend/hip/mhd/mhd_update.hip :: geometric_source_kernel) -----------
// With v_r = m_r/rho, v_phi = m_phi/rho, B_r, B_phi read from the slots above:
//
//   S_{m_r}   = (rho * v_phi^2 - B_phi^2) / r      -> dudt.mx      += S_{m_r}
//   S_{B_phi} = -(v_r * B_phi - v_phi * B_r) / r   -> dudt.bz_cell += S_{B_phi}
//
//   S_{m_z}, S_{m_phi}, S_{rho}, S_{energy}, S_{B_r}, S_{B_z} = 0
//
// S_{m_r} is the centrifugal term (rho v_phi^2 / r) plus the toroidal magnetic
// hoop-tension term (-B_phi^2 / r). S_{B_phi} is the toroidal-induction
// curvature term arising from the cylindrical curl of the kinematic EMF. The
// remaining components carry no 1/r geometric source under this splitting.
//
// -- On-axis (r -> 0) guard ---------------------------------------------------
// The radius is grid.r_at_cell_center(i). Cells with r <= 0.5*dr (i.e. the
// on-axis column when the domain starts at r=0) are skipped: the source is
// regular there and the 1/r factor would otherwise overflow. This matches the
// PIC cylindrical on-axis convention (src/physics/pic), which zeroes the
// inside-axis radial flux at the r=0 face.
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
