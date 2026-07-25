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
// For continuity, axial momentum, and energy, launch_mhd_flux_difference uses
// the exact annular face/volume weights
//
//   -[r_{i+1} F_{i+1/2} - r_i F_{i-1/2}]
//     / [0.5 (r_{i+1}^2-r_i^2)].
//
// Thus those components need no pointwise F/r repair.
// Azimuthal momentum instead uses its exact piecewise-constant angular moment,
//
//   -[r_hi^2 F_{m_phi,hi} - r_lo^2 F_{m_phi,lo}]
//     / int_cell(r^2 dr),
//
// directly in the flux kernel. This telescopes under the natural discrete
// angular-momentum weight and removes the non-conservative cell-centred -F/r
// source. The production solver does not combine an annular aggregate radial-
// momentum flux with this source: it overwrites that component with the fused
// pressure-free tensor form
//
//   -d_r T_rr - d_z T_zr + (T_phiphi-T_rr)/r,
//
// so the curvature cancellation happens before rounding. This class remains a
// standalone source-term API for callers that assemble the conventional
// annular radial-momentum divergence themselves. B_phi uses its metric-free
// induction equation, while B_r/B_z are advanced by the matching annular
// constrained-transport curl.
//
// -- Exact formulas implemented (authoritative device kernel,
//    src/backend/hip/mhd/mhd_update.hip :: geometric_source_kernel) -----------
// With v = m/rho per slot, total B = B0+b, and
// p* = p_gas + |B|^2/2:
//
//   S_{m_r} = [rho v_phi^2 + p* - B_phi^2] / r -> dudt.mx
//   S_rho = S_{m_z} = S_{m_phi} = S_energy = S_B = 0
//
// The p*/r term cancels the pressure part introduced when F_rr is placed under
// the annular divergence; the remaining pieces are centrifugal acceleration and
// toroidal magnetic hoop stress. Angular-momentum conservation is already exact
// in the radial flux operator. All magnetic quantities in the radial curvature
// term are TOTAL fields, even though the evolved state stores only b and
// perturbation magnetic energy.
//
// -- On-axis (r -> 0) handling ------------------------------------------------
// The radius is grid.r_at_cell_center(i) = (i+0.5)*dr for a domain starting on
// the axis (origin_x = 0), so EVERY cell center has r > 0 -- the innermost cell
// i=0 sits at r = 0.5*dr and its 1/r factor (= 2/dr) is finite. The geometric
// source is therefore applied at the i=0 column like any other. Annular domains
// with origin_x > 0 use the same expression. Only a non-positive cell-center
// radius is skipped defensively; valid cylindrical grids never contain one.
//
// add() is a thin host wrapper that delegates to launch_mhd_geometric_source
// (kernels.hpp); the device kernel is the single authoritative implementation
// of the conventional source above. The production solver uses its fused
// cylindrical residual instead.

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::mhd {

struct MhdGeometricSource {
  // Accumulate the conventional annular-form axisymmetric source S(u) into
  // dudt (does NOT overwrite). This is a source only; it does not supply the
  // radial tensor derivatives used by the solver's fused residual. Direction
  // convention: r = x (index i), z = y (index j). Delegates to the device
  // kernel launch_mhd_geometric_source on the default stream.
  static void add(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                  const MhdBackgroundField<Real>& b0,
                  const Grid2D& grid, Real gamma,
                  int collocation_order = 0);
};

}  // namespace quasar::mhd
