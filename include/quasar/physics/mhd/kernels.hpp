#pragma once

// Backend-neutral declaration of the ideal-MHD kernel-launch ABI.
//
// This is the MHD analogue of include/quasar/physics/pic/kernels.hpp: a
// per-physics seam that speaks MHD types (MhdField2D / MhdInterfaceStates /
// EmfField2D), so it lives under physics/mhd/ rather than backend/ -- the
// backend axis stays physics-neutral (device.hpp / memory.hpp only). Every
// launch_mhd_* entry point defined under src/backend/hip/mhd/ is declared here
// exactly once and included both by its .hip definition (so a signature drift
// is a compile error) and by the sibling MHD solver (the ONLY caller). Callers
// reach the backend only through this header; do not hand-redeclare these.
//
// This header is pure declarations and is safe to include from non-HIP host
// translation units (no __global__, no <hip/hip_runtime.h>). The .hip
// definitions translate the backend-neutral stream handle to hipStream_t.
//
// Ownership contract: the kernels allocate NOTHING. Every buffer (the input
// field, the interface/flux scratch, the EMF, the dudt accumulator, the output
// field) is owned by the solver and passed in by reference; the launch wrappers
// extract the device pointers (DeviceBuffer::device_ptr()) on the host before
// dispatch, mirroring the PIC backend convention.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::mhd {

// The kernel-launch ABI speaks the backend-neutral stream handle so the solver
// never includes a HIP header; the .hip definitions cast it back internally.
using stream_t = ::quasar::backend::stream_t;

// -- Flux reconstruction -----------------------------------------------------
// Reconstruct the LEFT/RIGHT conserved interface states on every interface
// normal to `dir` (0=x, 1=y) into `out` (out.dir must equal dir). `scheme_order`
// selects the spatial order: 2 = MUSCL-minmod.
//
// NOTE (device-path simplification, flagged for review): the high-order host
// schemes "mp5"/"mp7" (Suresh-Huynh monotonicity-preserving in characteristic
// variables) are NOT yet ported to device -- flux_reconstruction.hpp exposes no
// QUASAR_HOST_DEVICE limiter helpers to reuse. The device kernel therefore
// implements a robust 2nd-order MUSCL-minmod reconstruction (in primitive
// variables) for ALL scheme_order values; scheme_order 5/7 currently resolve to
// the same 2nd-order device path. The host registry retains the full MP5/MP7
// schemes. This is a documented build-safety-first simplification.
void launch_mhd_reconstruct(const MhdField2D<Real>& u, int dir,
                            quasar::numerics::MhdInterfaceStates<Real>& out,
                            int scheme_order, Real gamma, stream_t stream);

// -- Riemann flux ------------------------------------------------------------
// Pointwise numerical flux at every interface normal to `dir` from the L/R
// interface states, written into `flux_out` (the 8 conserved-component buffers
// of an MhdField2D reused as flux storage: rho,mx,my,mz,energy,bx_face,by_face,
// bz_cell map to the flux components rho,mx,my,mz,energy,bx,by,bz).
//
// Device flux: full HLLD (Miyoshi & Kusano 2005) with degeneracy guards; if a
// sub-fan (e.g. near-zero normal B or a non-physical intermediate density) is
// hit, the kernel falls back to the single-intermediate HLL flux for that
// interface so the build stays divergence-/positivity-safe.
void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface, int dir,
                          MhdField2D<Real>& flux_out, Real gamma, stream_t stream);

// -- Conservative flux difference --------------------------------------------
// Accumulate the conservative finite-difference flux divergence into `dudt`:
//   dudt += -(F_{i+1/2} - F_{i-1/2}) / d(dir)
// where `flux` holds the interface flux at face (i,j) (the x_lo / y_lo face of
// cell (i,j), same staggering as the interface states). One call per direction;
// the caller zeroes dudt before the first direction. `flux` carries the same
// 8-component MhdField2D layout the Riemann kernel wrote.
void launch_mhd_flux_difference(const MhdField2D<Real>& flux, int dir,
                                MhdField2D<Real>& dudt, stream_t stream);

// -- Constrained-transport EMF -----------------------------------------------
// Build the corner-staggered EMF (emf.ez_edge at the lower-left corner of cell
// (i,j)) from the conserved field `u` and the dir=0 / dir=1 interface states,
// via the kinematic E = -(v x B) averaged from the four surrounding interface
// upwind values. ex_edge / ey_edge are also populated for parity.
void launch_mhd_ct_emf(const MhdField2D<Real>& u,
                       const quasar::numerics::MhdInterfaceStates<Real>& ifx,
                       const quasar::numerics::MhdInterfaceStates<Real>& ify,
                       EmfField2D<Real>& emf, Real gamma, stream_t stream);

// -- Face-B update -----------------------------------------------------------
// Advance the face-staggered in-plane B from the corner Ez by the discrete curl
// of E so the cell-centered div(B) telescopes to identically zero:
//   bx_face(i,j) -= dt * (ez(i,j+1) - ez(i,j)) / dy
//   by_face(i,j) += dt * (ez(i+1,j) - ez(i,j)) / dx
void launch_mhd_face_b_update(MhdField2D<Real>& u, const EmfField2D<Real>& emf,
                              Real dt, stream_t stream);

// -- CT EMF curl rate into the residual --------------------------------------
// Write the CT EMF curl RATE (dB/dt, no dt factor) into ONLY dudt.bx_face /
// dudt.by_face, OVERWRITING (assign, not accumulate) whatever the Godunov flux
// difference left in those two slots. Every other dudt component (rho, mx, my,
// mz, energy, bz_cell) is left exactly as flux_difference wrote it. This routes
// the face-B advance solely through the CT curl + the single rk_stage, so face B
// is no longer double-updated by both the flux difference and a separate
// face_b_update.
//   dudt.bx_face(i,j) = -(ez_edge(i,j+1) - ez_edge(i,j)) / dy
//   dudt.by_face(i,j) = +(ez_edge(i+1,j) - ez_edge(i,j)) / dx
// Stencil-, dx/dy-, and interior-bounds-identical to launch_mhd_face_b_update
// (with dt factored out), so the telescoping div(B) cancellation is bit-for-bit
// the same. `grid` is passed by value (like launch_mhd_geometric_source).
//
// NOTE: launch_mhd_face_b_update is retained but is now unused on the solver
// path -- the solver advances face B via this rate + rk_stage instead of
// applying the curl directly to the field.
void launch_mhd_emf_curl_rate(const EmfField2D<Real>& emf, MhdField2D<Real>& dudt,
                              Grid2D grid, stream_t stream);

// -- SSP-RK stage combine ----------------------------------------------------
// Pointwise Shu-Osher stage combine over all 8 conserved components:
//   out = a * un + b * ustage + c_dt * dudt
// (c_dt already folds the dt factor of the stage). All four fields share the
// grid; out may alias ustage.
void launch_mhd_rk_stage(MhdField2D<Real>& out, const MhdField2D<Real>& un,
                         const MhdField2D<Real>& ustage, const MhdField2D<Real>& dudt,
                         Real a, Real b, Real c_dt, stream_t stream);

// -- Positivity floors -------------------------------------------------------
// Clamp density to `rho_floor` and gas pressure to `p_floor` (re-deriving total
// energy from the floored pressure, holding momentum and B fixed) at every cell.
void launch_mhd_apply_floors(MhdField2D<Real>& u, Real rho_floor, Real p_floor,
                             Real gamma, stream_t stream);

// -- Cylindrical geometric source --------------------------------------------
// Accumulate the axisymmetric (r,z) geometric source into `dudt`:
//   dudt += S(u, r)
// with the radius read from grid.r_at_cell_center(i) and an on-axis (r -> 0)
// guard. No-op contribution where r is at/below the guard floor. The solver
// only calls this in cylindrical mode (it is a pure add, so a Cartesian solver
// simply never invokes it).
void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 Grid2D grid, Real gamma, stream_t stream);

}  // namespace quasar::mhd
