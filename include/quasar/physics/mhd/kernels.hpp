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

// Per-side boundary modes threaded into reconstruction and CT.  The entries are
// [x_lo,x_hi,y_lo,y_hi]: 0 periodic, 1 outflow, 2 conducting wall, 3 r=0 axis.
// Reconstruction treats every nonzero value as one-sided; CT additionally uses
// the distinction to pin the tangential EMF on conducting/axis edges.
struct BoundaryFlags4 {
  int side[4];
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
// device. The evolved state is a finite-volume cell average, and the MP helpers
// use the matching cell-average-to-face coefficients; the conservative flux
// residual therefore achieves the requested design order on smooth data.
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
//
// `rate_only` is reserved for the CFL preflight.  It permits a finite
// conservative face state whose primitive velocity or physical wave speed is
// outside binary64, because the CFL kernel evaluates the corresponding
// speed/spacing rate without materializing either quantity.  The ordinary
// Riemann path keeps the stricter finite-primitive contract.
void launch_mhd_reconstruct(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                            int dir, quasar::numerics::MhdInterfaceStates<Real>& out,
                            int scheme_order, BoundaryFlags4 flags, Real gamma,
                            stream_t stream, bool rate_only = false);

// -- Riemann flux ------------------------------------------------------------
// Numerical face flux at every interface normal to `dir` from the L/R
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
// fast path, bit-identical to the original body. `flags` makes the high physical
// face read the low-face Riemann data on a fully periodic axis, so the two
// representations of the same periodic face remain bit-identical.
void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface,
                          const MhdBackgroundField<Real>& b0, int dir,
                          MhdField2D<Real>& flux_out, BoundaryFlags4 flags,
                          Real gamma, stream_t stream,
                          bool hll_only = false,
                          MhdMomentumFluxParts2D<Real>* momentum_parts = nullptr);

// -- Conservative flux difference --------------------------------------------
// Accumulate the conservative finite-volume face-flux divergence into `dudt`:
//   dudt += -(F_{i+1/2} - F_{i-1/2}) / d(dir)
// where `flux` holds the interface flux at face (i,j) (the x_lo / y_lo face of
// cell (i,j), same staggering as the interface states). One call per direction;
// the caller zeroes dudt before the first direction. `flux` carries the same
// 8-component MhdField2D layout the Riemann kernel wrote.
void launch_mhd_flux_difference(const MhdField2D<Real>& flux, int dir,
                                MhdField2D<Real>& dudt, stream_t stream,
                                bool cylindrical = false);

// Add the complete static Maxwell-stress force once per residual. Both
// directions (plus cylindrical curvature/angular weights) share one
// common-exponent reduction per momentum component. Inactive B0 is a no-op.
void launch_mhd_background_stress_correction(
    const MhdBackgroundField<Real>& b0, MhdField2D<Real>& dudt,
    BoundaryFlags4 flags, stream_t stream, bool cylindrical = false,
    int collocation_order = 0);

// Active-background momentum residual.  The two directional face fluxes carry
// material stress in their momentum slots, while `parts_*` carry a factored
// background-linear stress and the Riemann wave correction.  This kernel
// flattens both directions plus the static T0 correction into one exponent
// reduction per component and overwrites dudt.{mx,my,mz}.  In cylindrical
// geometry it writes the axial/azimuthal components; the radial tensor
// residual (which also needs cell-centred curvature) is completed below.
void launch_mhd_split_momentum_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_x,
    const MhdMomentumFluxParts2D<Real>& parts_x,
    const MhdField2D<Real>& flux_y,
    const MhdMomentumFluxParts2D<Real>& parts_y,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical = false, int collocation_order = 0);

// Overwrite the cylindrical radial-momentum rate with the pressure-free tensor
// form in one common-exponent reduction:
//   -d_r(F_rr) - d_z(F_zr) + (T_phiphi - T_rr)/r.
// The split/static tensor difference is expanded before rounding, so gas
// pressure and axial-field terms cancel symbolically instead of being
// reintroduced after they may already have rounded out of an aggregate face
// flux. The maximum active path has 20 terms. This launcher is solver-only;
// launch_mhd_geometric_source remains the standalone source-term API.
void launch_mhd_cylindrical_radial_momentum_residual(
    const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_r, const MhdField2D<Real>& flux_z,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    int collocation_order = 0,
    const MhdMomentumFluxParts2D<Real>* parts_r = nullptr,
    const MhdMomentumFluxParts2D<Real>* parts_z = nullptr);

// -- Constrained-transport EMF -----------------------------------------------
// Build the corner-staggered EMF (emf.ez_edge at the lower-left corner of cell
// (i,j)) from the dir=0 / dir=1 reconstructed interface states. The kernel
// recomputes the directional HLLD magnetic fluxes, maps them to the two upwind
// face electric fields, and interpolates those face values to the corner using
// the requested reconstruction order. ex_edge / ey_edge are also populated for
// parity.
//
// `b0` is the static background field (B = B0 + b); inactive => zero-background
// fast path. `flags` marks per-side non-periodic boundaries (see
// BoundaryFlags4): at a non-periodic boundary the corner-EMF averaging uses a
// one-sided stencil that drops the ghost-GRADIENT dependence. With `flags`
// all-zero and `b0` inactive this is bit-identical to the original body.
void launch_mhd_ct_emf(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                       const quasar::numerics::MhdInterfaceStates<Real>& ifx,
                       const quasar::numerics::MhdInterfaceStates<Real>& ify,
                       BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma,
                       stream_t stream, int scheme_order = 2,
                       bool cylindrical = false, bool hll_only = false);

// -- Face-B update -----------------------------------------------------------
// Advance the face-staggered in-plane B from the corner Ez by the discrete curl
// of E so the cell-centered div(B) telescopes to identically zero:
//   bx_face(i,j) -= dt * (ez(i,j+1) - ez(i,j)) / dy
//   by_face(i,j) += dt * (ez(i+1,j) - ez(i,j)) / dx
void launch_mhd_face_b_update(MhdField2D<Real>& u, const EmfField2D<Real>& emf,
                              Real dt, stream_t stream,
                              bool cylindrical = false);

// -- CT EMF curl rate into the residual --------------------------------------
// Write the CT EMF curl RATE (dB/dt, no dt factor) into ONLY dudt.bx_face /
// dudt.by_face, OVERWRITING (assign, not accumulate) whatever the Godunov flux
// difference left in those two slots. Every other dudt component (rho, mx, my,
// mz, energy, bz_cell) is left unchanged by this kernel (active-B0 energy is
// overwritten later by launch_mhd_split_energy_residual). This routes
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
                              Grid2D grid, stream_t stream,
                              bool cylindrical = false);

// Overwrite `actual_rate.energy` with the active-background split equation
//   -D_E(F_E') + v . [(curl B0) x (B0+b)].
// The Riemann flux already carries F_E'=F_E-B0.F_B. curl(B0) is formed before
// multiplication, so a curl-free dominant background produces no B0^2
// intermediate; the flux divergence and expanded Lorentz-power terms share one
// exponent reduction. Cylindrical D_E is annular in r.
void launch_mhd_split_energy_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& state,
    const MhdField2D<Real>& flux_x, const MhdField2D<Real>& flux_y,
    MhdField2D<Real>& actual_rate, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical = false, int collocation_order = 0);

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

// Compute the largest global convex fraction theta in [0,1] for which every
// cell on the segment base + theta*(candidate-base) has rho > rho_floor and
// p > p_floor. Each cell computes its own density/internal-energy bound; the
// launcher returns their minimum. The conservative solver passes zero bounds,
// since strict mathematical positivity (unlike an arbitrary positive floor) is
// the invariant-domain contract. Pressure admissibility uses the concavity of
// E-|m|^2/(2 rho)-|b|^2/2 and a bracketed bisection, so no cell state is changed.
// The caller owns/reuses `scratch` for the block-min reduction.
void launch_mhd_admissible_fraction(
    const MhdField2D<Real>& base, const MhdField2D<Real>& candidate,
    Real rho_floor, Real p_floor, Real gamma,
    backend::DeviceBuffer<Real>& scratch, Real* host_theta, stream_t stream,
    int collocation_order = 0);

// -- Cylindrical geometric source --------------------------------------------
// Accumulate the axisymmetric (r,z) geometric source into `dudt`:
//   dudt += S(u, r)
// with the radius read from grid.r_at_cell_center(i) = (i+0.5)*dr, which is > 0
// at every cell center (the innermost cell i=0 is at r = 0.5*dr), so the source
// is applied at every column including the axis column; only a non-positive /
// non-finite cell-center radius (which a valid grid never produces) is skipped.
// The only nonzero component is radial-momentum curvature. Azimuthal momentum
// is already conservative under its exact int(r^2 dr) flux operator and has no
// cell-centred geometric source.
// This is the conventional standalone source-only API. The production solver
// uses launch_mhd_cylindrical_radial_momentum_residual instead so its tensor
// derivatives and pressure-free curvature share one reduction.
void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 const MhdBackgroundField<Real>& b0,
                                 Grid2D grid, Real gamma, stream_t stream,
                                 int collocation_order = 0);

// -- CFL maximum signal rate -------------------------------------------------
// Reduce the maximum finite-volume Courant coefficient from the four incident
// faces of every interior cell. At each face alpha=max_side(|v_n|+c_fast,n) is
// evaluated from the exact reconstructed L/R interface states consumed by the
// Riemann solver, including its shared CT normal B. Thus both a staggered face
// that cancels in the cell average and an MP-reconstructed face overshoot enter
// the bound. Cartesian uses
//   0.5*(alpha_xlo+alpha_xhi)/dx +
//   0.5*(alpha_ylo+alpha_yhi)/dy.
// Cylindrical takes the maximum of all three radial operators: the annular
// fluid rate
//   0.5*(r_lo*alpha_xlo+r_hi*alpha_xhi)/(r_c*dr),
// the exact piecewise-constant angular-momentum rate
//   0.5*(r_lo^2*alpha_xlo+r_hi^2*alpha_xhi)
//       / [dr*(r_c^2+dr^2/12)],
// and the metric-free B_phi rate 0.5*(alpha_xlo+alpha_xhi)/dr. The ordinary
// axial half-sum is then added. At the axis the angular-momentum coefficient is
// 1.5*alpha_xhi/dr. Cartesian uniform states reduce to the familiar additive
// (|v_x|+c_fast,x)/dx+(|v_y|+c_fast,y)/dy bound. Rates are formed before any
// unscaled velocity or fast speed, so a finite rate remains representable even
// when the corresponding physical speed does not. The fast rate sees total
// B=b+B0. Invalid/non-finite face states contribute infinity.
//
// `scratch` is a caller-owned block-partials buffer reused across calls (the
// auto-dt loop calls this every step): the launcher (re)sizes it only when it is
// smaller than the per-launch block count, so the caller avoids a per-step
// hipMalloc/hipFree. Ownership stays with the caller, honoring the
// scratch-ownership contract; the kernel writes every block slot
// before any read, so a larger reused buffer is safe.
void launch_mhd_cfl_max_rate(
    const quasar::numerics::MhdInterfaceStates<Real>& ifx,
    const quasar::numerics::MhdInterfaceStates<Real>& ify,
    const MhdBackgroundField<Real>& b0, Real gamma,
    backend::DeviceBuffer<Real>& scratch, Real* host_max_rate,
    stream_t stream, bool low_order = false, bool cylindrical = false,
    BoundaryFlags4 flags = BoundaryFlags4{});

// -- Constrained-transport div(B) L-infinity ---------------------------------
// Reduce the maximum interior |div B| -- the cell-centered divergence of the
// face-staggered in-plane field,
//   div B(i,j) = (bx_face(i+1,j) - bx_face(i,j)) / dx
//              + (by_face(i,j+1) - by_face(i,j)) / dy
// -- over all interior cells and write the single scalar into *host_linf. This
// is the CT consistency diagnostic; with the CT face-B update it should remain at
// round-off. The result is read back to the host pointer before this returns.
//
// `scratch` is a caller-owned block-partials buffer reused across calls (same
// contract as launch_mhd_cfl_max_rate): the launcher (re)sizes it only when it is
// too small, so the caller (the CT scheme) avoids a per-call hipMalloc/hipFree.
void launch_mhd_ct_divb_linf(const MhdField2D<Real>& u,
                             backend::DeviceBuffer<Real>& scratch,
                             Real* host_linf, stream_t stream,
                             bool cylindrical = false);

// Scale-free discrete-solenoidality diagnostic used by the live-state
// preflight.  Each cell reports
//
//   |sum_k t_k| / sum_k |t_k|,
//
// where t_k are the signed face/spacing terms in the exact Cartesian or
// annular CT divergence stencil.  Forming the ratio at a common binary
// exponent makes it meaningful for subnormal and near-overflow field scales
// without introducing a unit-dependent absolute floor.  A zero field reports
// zero; non-finite face data reports infinity.
void launch_mhd_ct_divb_relative_linf(
    const MhdField2D<Real>& u, backend::DeviceBuffer<Real>& scratch,
    Real* host_linf, stream_t stream, bool cylindrical = false);

// -- Ghost-layer fill (fluid components) -------------------------------------
// Fill the ghost layers of ONE boundary `side` (canonical order 0=x_lo, 1=x_hi,
// 2=y_lo, 3=y_hi) for the FLUID conserved components (rho, mx, my, mz, energy,
// bz_cell) according to `mode`:
//   mode 0 = periodic   : wrap from the opposite interior layer,
//   mode 1 = outflow    : zero-gradient copy of the adjacent interior cell,
//   mode 2 = wall       : mirror the interior cells; the NORMAL momentum
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
//   mode 2 = wall       : mirror the interior faces; the NORMAL face-B component
//                         (bx_face at an x-side, by_face at a y-side) is sign-
//                         flipped, the tangential face-B component is copied
//                         even. Run after the fluid fill so the CT face field
//                         and the cell field carry a consistent boundary.
void launch_mhd_fill_ghosts_field(MhdField2D<Real>& u, int side, int mode, stream_t stream);

}  // namespace quasar::mhd
