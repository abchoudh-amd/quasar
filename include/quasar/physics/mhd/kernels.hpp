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
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::mhd {

// The kernel-launch ABI speaks the backend-neutral stream handle so the solver
// never includes a HIP header; the .hip definitions cast it back internally.
using stream_t = ::quasar::backend::stream_t;

// Per-side non-periodic boundary flags threaded into the reconstruction and CT
// EMF kernels. The four entries follow the canonical side order
// [x_lo, x_hi, y_lo, y_hi]; 1 = non-periodic (the device path drops the ghost
// GRADIENT dependence and uses a one-sided stencil at that boundary), 0 =
// periodic (the two-sided wrap stencil). The solver computes these from the
// per-side field-boundary names; an all-zero flags set is the periodic fast
// path and is bit-identical to the no-flags behavior.
struct BoundaryFlags4 {
  int side[4];  // [x_lo, x_hi, y_lo, y_hi]; 1 = non-periodic, 0 = periodic
};

// -- Flux reconstruction -----------------------------------------------------
// Reconstruct the LEFT/RIGHT conserved interface states on every interface
// normal to `dir` (0=x, 1=y) into `out` (out.dir must equal dir). `scheme_order`
// selects the spatial order and the device kernel HONORS it:
//   2 = MUSCL-minmod (2nd-order TVD, primitive-variable slope limiting),
//   5 = MP5  (5th-order monotonicity-preserving, Suresh-Huynh),
//   7 = MP7  (7th-order monotonicity-preserving, Suresh-Huynh).
// The MP5/MP7 paths run the SAME characteristic-variable reconstruction as the
// host registry: the per-interface 7-wave eigensystem
// (quasar::numerics::MhdEigensystem) projects the conserved delta into
// characteristic space (quasar::numerics::CharacteristicProjector / CharVec7),
// the scalar MP limiter (quasar/numerics/mp_limiter.hpp) is applied wave-by-wave,
// and the result is mapped back to conserved variables -- all device-callable, so
// scheme_order 5/7 run the full high-order characteristic MP reconstruction on
// device (they achieve their design 5th/7th order; see mp_limiter.hpp for the
// point-value interpolation-coefficient correction that makes this so -- results
// are not bit-identical to the pre-port host MP5/MP7 output).
//
// Field-split + one-sided boundary extensions:
//   `b0` is the static background magnetic field B = B0 + b. When b0.active is
//   false the kernel takes the zero-background fast path (the stored b IS the
//   total field) and the result is bit-identical to the original body. When
//   active, the wave speeds / interface reconstruction see the total field.
//   `flags` marks per-side non-periodic boundaries (see BoundaryFlags4); at a
//   non-periodic boundary the reconstruction uses a one-sided stencil that
//   drops the ghost-GRADIENT dependence (ghost VALUES are still read). With
//   `flags` all-zero and `b0` inactive this is bit-identical to the periodic
//   no-background path.
void launch_mhd_reconstruct(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                            int dir, quasar::numerics::MhdInterfaceStates<Real>& out,
                            int scheme_order, BoundaryFlags4 flags, Real gamma, stream_t stream);

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
//
// `b0` is the static background field (B = B0 + b); inactive => zero-background
// fast path, bit-identical to the original body.
void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface,
                          const MhdBackgroundField<Real>& b0, int dir,
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
//
// `b0` is the static background field (B = B0 + b); inactive => zero-background
// fast path. `flags` marks per-side non-periodic boundaries (see
// BoundaryFlags4): at a non-periodic boundary the corner-EMF averaging uses a
// one-sided stencil that drops the ghost-GRADIENT dependence. With `flags`
// all-zero and `b0` inactive this is bit-identical to the original body.
void launch_mhd_ct_emf(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                       const quasar::numerics::MhdInterfaceStates<Real>& ifx,
                       const quasar::numerics::MhdInterfaceStates<Real>& ify,
                       BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma, stream_t stream);

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
//
// `b0` is the static background field (B = B0 + b); inactive => zero-background
// fast path, bit-identical to the original body. With a nonzero B0 the stored
// magnetic energy is the perturbation-only 0.5|b|^2, so the floor re-derivation
// is consistent with the field-split EOS.
void launch_mhd_apply_floors(MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                             Real rho_floor, Real p_floor, Real gamma, stream_t stream);

// -- Cylindrical geometric source --------------------------------------------
// Accumulate the axisymmetric (r,z) geometric source into `dudt`:
//   dudt += S(u, r)
// with the radius read from grid.r_at_cell_center(i) = (i+0.5)*dr, which is > 0
// at every cell center (the innermost cell i=0 is at r = 0.5*dr), so the source
// is applied at every column including the axis column; only a non-positive /
// non-finite cell-center radius (which a valid grid never produces) is skipped.
// The solver only calls this in cylindrical mode (it is a pure add, so a
// Cartesian solver simply never invokes it).
void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 Grid2D grid, Real gamma, stream_t stream);

// -- CFL maximum signal rate -------------------------------------------------
// Reduce the maximum ADDITIVE directional signal rate
//   (|v_x| + c_fast_x)/dx + (|v_y| + c_fast_y)/dy
// over all INTERIOR cells and write the single scalar into *host_max_rate; the
// caller's stable step is then dt = cfl / max_rate. This is the multidimensional
// unsplit Courant condition (the residual sums both directional flux differences
// into one dudt per RK stage), which is stricter than a per-direction max signal
// speed over min(dx,dy). The fast speed sees the TOTAL magnetic field B = b + B0
// (b0-aware): when b0.active is false this is the zero-background fast path (the
// stored b IS the total field). Cells with non-positive or non-finite density are
// SKIPPED (they contribute no rate) so a transiently floored cell cannot poison
// the reduction. The result is read back to the host pointer before this returns.
void launch_mhd_cfl_max_rate(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                             Real gamma, Real* host_max_rate, stream_t stream);

// -- Constrained-transport div(B) L-infinity ---------------------------------
// Reduce the maximum interior |div B| -- the cell-centered divergence of the
// face-staggered in-plane field,
//   div B(i,j) = (bx_face(i+1,j) - bx_face(i,j)) / dx
//              + (by_face(i,j+1) - by_face(i,j)) / dy
// -- over all interior cells and write the single scalar into *host_linf. This
// is the CT consistency diagnostic; with the CT face-B update it should remain at
// round-off. The result is read back to the host pointer before this returns.
void launch_mhd_ct_divb_linf(const MhdField2D<Real>& u, Real* host_linf, stream_t stream);

// -- Ghost-layer fill (fluid components) -------------------------------------
// Fill the ghost layers of ONE boundary `side` (canonical order 0=x_lo, 1=x_hi,
// 2=y_lo, 3=y_hi) for the FLUID conserved components (rho, mx, my, mz, energy,
// bz_cell) according to `mode`:
//   mode 0 = periodic   : wrap from the opposite interior layer,
//   mode 1 = outflow    : zero-gradient copy of the adjacent interior cell,
//   mode 2 = reflecting : mirror the interior cells; the NORMAL momentum
//                         component (mx at an x-side, my at a y-side) is sign-
//                         flipped, every tangential momentum component and all
//                         scalars (rho, energy, bz_cell) are copied even.
void launch_mhd_fill_ghosts_fluid(MhdField2D<Real>& u, int side, int mode, stream_t stream);

// -- Ghost-layer fill (face-staggered field components) -----------------------
// Fill the ghost layers of ONE boundary `side` (same order as above) for the
// face-staggered in-plane magnetic components (bx_face, by_face) according to
// `mode`:
//   mode 0 = periodic   : wrap from the opposite interior layer,
//   mode 1 = outflow    : zero-gradient copy of the adjacent interior face,
//   mode 2 = reflecting : mirror the interior faces; the NORMAL face-B component
//                         (bx_face at an x-side, by_face at a y-side) is sign-
//                         flipped, the tangential face-B component is copied
//                         even. Run after the fluid fill so the CT face field
//                         and the cell field carry a consistent boundary.
void launch_mhd_fill_ghosts_field(MhdField2D<Real>& u, int side, int mode, stream_t stream);

}  // namespace quasar::mhd
