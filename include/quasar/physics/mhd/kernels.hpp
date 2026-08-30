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
// dispatch, mirroring the PIC backend convention. A trailing inactive
// RadialTablesView selects Cartesian coefficients. An active view must match
// the launch grid and scheme/collocation order, and its owning RadialTables must
// remain alive until the asynchronous kernel has completed.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::mhd {

// The kernel-launch ABI speaks the backend-neutral stream handle so the solver
// never includes a HIP header; the .hip definitions cast it back internally.
using stream_t = ::quasar::backend::stream_t;

// Per-side boundary modes threaded into reconstruction and CT.  The entries are
// [x_lo,x_hi,y_lo,y_hi]: 0 periodic, 1 outflow, 2 conducting wall, 3 r=0 axis,
// 4 exchanged internal tile side.  Reconstruction treats physical modes 1--3
// as one-sided; mode 4 consumes the already-exchanged guard cells without
// wrapping or applying a physical closure. CT additionally uses the distinction
// to pin the tangential EMF on conducting/axis edges.
struct BoundaryFlags4 {
  int side[4];
};

// Per-side ownership of the two boundary faces in each directional HLLD
// launch, ordered [x_lo,x_hi,y_lo,y_hi]. Serial launches keep the all-owned
// default. A distributed tile clears only a shared side for which another tile
// is the canonical owner; the skipped output is populated by the subsequent
// owner-record exchange before any flux divergence consumes it.
struct FaceOwnershipFlags4 {
  int side[4]{1, 1, 1, 1};
};

// -- Distributed halo packing ----------------------------------------------
// Device-side packing keeps the distributed runtime's fixed eight-lane wire
// representation independent of the concrete solver scratch object being
// exchanged. A lane can expose ordinary Real storage, an integer table, or one
// member of a ScaledValue table. Missing lanes are written as exact zeroes and
// ignored on receive, preserving the symmetric payload used by every MHD halo
// family without allocating dummy full-field buffers.
inline constexpr std::size_t mhd_device_halo_component_count = 8;

enum class MhdDeviceHaloDirection : int {
  x_low = 0,
  x_high = 1,
  y_low = 2,
  y_high = 3,
};

enum class MhdDeviceHaloValueKind : int {
  absent = 0,
  real = 1,
  int32 = 2,
  scaled_mantissa = 3,
  scaled_exponent = 4,
};

enum class MhdDeviceHaloLayout : int {
  cell = 0,
  cell_extended_y = 1,
  x_face = 2,
  x_face_extended_y = 3,
  y_face = 4,
  node = 5,
};

struct MhdDeviceHaloConstComponent {
  const void* values{nullptr};
  MhdDeviceHaloValueKind kind{MhdDeviceHaloValueKind::absent};
};

struct MhdDeviceHaloComponent {
  void* values{nullptr};
  MhdDeviceHaloValueKind kind{MhdDeviceHaloValueKind::absent};
  MhdDeviceHaloLayout layout{MhdDeviceHaloLayout::cell};
};

struct MhdDeviceHaloConstComponents {
  MhdDeviceHaloConstComponent component[mhd_device_halo_component_count]{};
};

struct MhdDeviceHaloComponents {
  MhdDeviceHaloComponent component[mhd_device_halo_component_count]{};
};

// The payload layout is byte-for-byte identical to
// distributed::pack_mhd_register_halo: x transfers contain nghost+1 columns
// by ny+1 rows per lane; y transfers contain nghost+1 rows over the complete
// padded pitch. `receive_shared_face` applies the canonical face-owner rule.
void launch_mhd_device_halo_pack(
    Grid2D grid, MhdDeviceHaloDirection direction,
    const MhdDeviceHaloConstComponents& components,
    backend::DeviceBuffer<Real>& payload, stream_t stream);
void launch_mhd_device_halo_unpack(
    Grid2D grid, MhdDeviceHaloDirection direction,
    const backend::DeviceBuffer<Real>& payload,
    const MhdDeviceHaloComponents& components, bool receive_shared_face,
    stream_t stream);

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
                            stream_t stream, bool rate_only = false,
                            quasar::numerics::RadialTablesView radial_tables = {});

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
// representations of the same periodic face remain bit-identical.  MP5/MP7
// require an order-matched transverse Gauss rule: their reconstructed states are
// face averages, and the nonlinear Riemann map must be evaluated at recovered
// point states before being integrated back to a face-average flux.
void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface,
                          const MhdBackgroundField<Real>& b0, int dir,
                          MhdField2D<Real>& flux_out, BoundaryFlags4 flags,
                          Real gamma, stream_t stream,
                          bool hll_only = false,
                          MhdMomentumFluxParts2D<Real>* momentum_parts = nullptr,
                          int scheme_order = 2,
                          FaceOwnershipFlags4 ownership = {},
                          quasar::numerics::RadialTablesView radial_tables = {});

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
    int collocation_order = 0, int scheme_order = 2,
    quasar::numerics::RadialTablesView radial_tables = {});

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
    bool cylindrical = false, int collocation_order = 0,
    int scheme_order = 2,
    quasar::numerics::RadialTablesView radial_tables = {});

// Overwrite the cylindrical radial-momentum rate with the pressure-free tensor
// form in one common-exponent reduction:
//   -d_r(F_rr) - d_z(F_zr) + (T_phiphi - T_rr)/r.
// For MP5/MP7 the ring-volume average additionally includes the exact metric
// defect [<T_rr>_uniform-(F_rr,hi+F_rr,lo)/2]/r. Both uniform-cell tensors are
// evaluated at recovered tensor Gauss points; inadmissible point states retain
// the established cell-centred fallback for that cell.
// The split/static tensor difference is expanded before rounding, so gas
// pressure and axial-field terms cancel symbolically instead of being
// reintroduced after they may already have rounded out of an aggregate face
// flux. The quadrature-expanded active path has a proven 308-term maximum and
// a 320-term bound in its exact radix accumulator. This launcher is solver-only;
// launch_mhd_geometric_source remains the standalone source-term API.
void launch_mhd_cylindrical_radial_momentum_residual(
    const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_r, const MhdField2D<Real>& flux_z,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    Real gamma, int collocation_order = 0, int scheme_order = 2,
    const MhdMomentumFluxParts2D<Real>* parts_r = nullptr,
    const MhdMomentumFluxParts2D<Real>* parts_z = nullptr,
    quasar::numerics::RadialTablesView radial_tables = {});

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
//
// The prepare/finish seam lets the distributed runtime exchange the derived
// cell/face tables and exact one-dimensional masks without enlarging the
// numerical state halo.  The ordinary launcher remains the serial API and
// invokes both phases back-to-back on one stream.
void launch_mhd_ct_emf_prepare(
    const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
    const quasar::numerics::MhdInterfaceStates<Real>& ifx,
    const quasar::numerics::MhdInterfaceStates<Real>& ify,
    BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma,
    stream_t stream, int scheme_order = 2, bool hll_only = false,
    quasar::numerics::RadialTablesView radial_tables = {});
void launch_mhd_ct_emf_finish(BoundaryFlags4 flags, EmfField2D<Real>& emf,
                              stream_t stream, int scheme_order = 2,
                              bool cylindrical = false,
                              quasar::numerics::RadialTablesView radial_tables = {});
void launch_mhd_ct_emf(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                       const quasar::numerics::MhdInterfaceStates<Real>& ifx,
                       const quasar::numerics::MhdInterfaceStates<Real>& ify,
                       BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma,
                       stream_t stream, int scheme_order = 2,
                       bool cylindrical = false, bool hll_only = false,
                       quasar::numerics::RadialTablesView radial_tables = {});

// -- CT EMF curl rate into the residual --------------------------------------
// Write the CT EMF curl RATE (dB/dt, no dt factor) into ONLY dudt.bx_face /
// dudt.by_face, OVERWRITING (assign, not accumulate) whatever the Godunov flux
// difference left in those two slots. Every other dudt component (rho, mx, my,
// mz, energy, bz_cell) is left unchanged by this kernel (active-B0 energy is
// overwritten later by launch_mhd_split_energy_residual). This routes the
// face-B advance solely through the CT curl + the single rk_stage, so face B is
// never double-updated by both the flux difference and a separate direct curl.
//   dudt.bx_face(i,j) = -(ez_edge(i,j+1) - ez_edge(i,j)) / dy
//   dudt.by_face(i,j) = +(ez_edge(i+1,j) - ez_edge(i,j)) / dx
// The stencil telescopes the cell-centered div(B) change to identically zero for
// any corner Ez, and a convex combination of divergence-free fields stays
// divergence-free, so the guarantee survives every SSP-RK stage. `grid` is
// passed by value (like launch_mhd_geometric_source).
void launch_mhd_emf_curl_rate(const EmfField2D<Real>& emf, MhdField2D<Real>& dudt,
                              Grid2D grid, stream_t stream,
                              bool cylindrical = false);

// Overwrite `actual_rate.energy` with the CT-consistent active-background
// change of variables
//   -D_E(F_E' + B0.F_B) - <B0 . db/dt>.
// `parts_x/y` retain the small transverse covariance of B0 and F_B from the
// Riemann face quadrature; the kernel conditions all remaining products around
// the cell-average B0 before the final reduction.
// `actual_rate` already contains the final CT in-plane rate and conservative
// out-of-plane rate.  MP5/MP7 evaluate the cell inner product at tensor Gauss
// points. Cylindrical D_E is annular in r.
void launch_mhd_split_energy_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_x,
    const MhdMomentumFluxParts2D<Real>& parts_x,
    const MhdField2D<Real>& flux_y,
    const MhdMomentumFluxParts2D<Real>& parts_y,
    MhdField2D<Real>& actual_rate, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical = false, int collocation_order = 0,
    int scheme_order = 2,
    quasar::numerics::RadialTablesView radial_tables = {});

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
                             Real rho_floor, Real p_floor, Real gamma,
                             stream_t stream, int collocation_order = 0,
                             quasar::numerics::RadialTablesView radial_tables = {});

// Compute the largest global convex fraction theta in [0,1] for which every
// cell on the segment base + theta*(candidate-base) has rho > rho_floor and
// p > p_floor. Each cell computes its own density/internal-energy bound; the
// launcher returns their minimum. The conservative solver passes zero bounds,
// since strict mathematical positivity (unlike an arbitrary positive floor) is
// its acceptance contract. Pressure admissibility uses the concavity of
// E-|m|^2/(2 rho)-|b|^2/2 and a bracketed bisection, so no cell state is changed.
// The caller owns/reuses `scratch` for the block-min reduction.
void launch_mhd_admissible_fraction(
    const MhdField2D<Real>& base, const MhdField2D<Real>& candidate,
    Real rho_floor, Real p_floor, Real gamma,
    backend::DeviceBuffer<Real>& scratch, Real* host_theta, stream_t stream,
    int collocation_order = 0,
    quasar::numerics::RadialTablesView radial_tables = {});

// -- Cylindrical geometric source --------------------------------------------
// Accumulate the axisymmetric (r,z) geometric source into `dudt`:
//   dudt += S(u, r)
// with the radius read from grid.r_at_cell_center(i). On an axis-starting grid
// this is (i+0.5)*dr, so the innermost cell i=0 is at r = 0.5*dr and the source
// is applied there like every other column. Only a non-positive /
// non-finite cell-center radius (which a valid grid never produces) is skipped.
// The only nonzero component is radial-momentum curvature. Azimuthal momentum
// is already conservative under its exact int(r^2 dr) flux operator and has no
// cell-centred geometric source.
// This standalone launcher selects equation-native MP5/MP7 tensor recovery
// when an active compatible RadialTablesView is supplied, with the range-safe
// cell-centred rule as its low-order and inadmissible-point fallback. The
// production solver uses launch_mhd_cylindrical_radial_momentum_residual instead
// so its tensor derivatives and pressure-free curvature share one reduction.
void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 const MhdBackgroundField<Real>& b0,
                                 Grid2D grid, Real gamma, stream_t stream,
                                 int collocation_order = 0,
                                 quasar::numerics::RadialTablesView radial_tables = {});

// -- CFL maximum signal rate -------------------------------------------------
// Reduce the maximum finite-volume Courant coefficient from the four incident
// faces of every interior cell. At each configured-HLLD face,
// alpha=max_side(|v_n|)+max_side(c_fast,n), matching HLLD's common outer fast
// speed. The piecewise-constant LF retry instead uses its actual
// alpha=max_side(|v_n|+c_fast,n). Both are evaluated from the exact
// reconstructed L/R interface states consumed by the Riemann solver, including
// its shared CT normal B. Thus both a staggered face that cancels in the cell
// average and an MP-reconstructed point overshoot enter the bound. Cartesian
// uses
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
// 1.5*alpha_xhi/dr. When each face has coincident L/R states, Cartesian mode
// reduces to the familiar additive
// (|v_x|+c_fast,x)/dx+(|v_y|+c_fast,y)/dy bound. Each physical face alpha must
// itself be finite because the Riemann solver consumes that wave fan before any
// mesh scaling. The alpha/spacing rates and their multidimensional sum are
// evaluated with a homogeneous power-of-two scale when necessary, so a finite
// positive timestep is retained even when the unscaled aggregate rate exceeds
// binary64. The fast speed sees total B=b+B0. Invalid/non-finite face states or
// unrepresentable physical face fans contribute infinity.
//
// ``ScaledCflRate`` retains the scale used by that homogeneous reduction:
//
//   true max rate = scaled_max_rate / rate_scale,
//   dt            = cfl * rate_scale / scaled_max_rate.
//
// The ordinary path always reports rate_scale=1. If its aggregate overflows,
// the launcher retries once with rate_scale=2^-64. A retry that still overflows
// cannot have a positive binary64 reciprocal (and physical-fan failures remain
// infinity under either scale).
//
// `scratch` is a caller-owned block-partials buffer reused across calls (the
// auto-dt loop calls this every step): the launcher (re)sizes it only when it is
// smaller than the per-launch block count, so the caller avoids a per-step
// hipMalloc/hipFree. Ownership stays with the caller, honoring the
// scratch-ownership contract; the kernel writes every block slot
// before any read, so a larger reused buffer is safe.
struct ScaledCflRate {
  Real scaled_max_rate{};
  Real rate_scale{Real{1}};
};

void launch_mhd_cfl_max_rate(
    const quasar::numerics::MhdInterfaceStates<Real>& ifx,
    const quasar::numerics::MhdInterfaceStates<Real>& ify,
    const MhdBackgroundField<Real>& b0, Real gamma,
    backend::DeviceBuffer<Real>& scratch, ScaledCflRate* host_rate,
    stream_t stream, int scheme_order = 2, bool cylindrical = false,
    BoundaryFlags4 flags = BoundaryFlags4{},
    quasar::numerics::RadialTablesView radial_tables = {});

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
// preflight. Each cell reports
//
//   ||d_1+d_2||_inf / || |d_1|+|d_2| ||_inf,
//
// where each d is a complete directional divergence contribution. Cartesian
// face offsets cancel before normalization; the cylindrical radial d retains
// the physical B_r/r term. The global directional norm also supplies a
// meaningful scale at local derivative nulls. Forming both norms from scaled
// directional values handles subnormal and near-overflow fields without a
// unit-dependent absolute floor. By default the 1024-face-ulp forward-error
// envelope is available only at a genuine opposite-sign cross-direction
// cancellation. ``solver_owned`` widens that envelope to an internally
// accepted CT/RK state whose exact-arithmetic divergence was already proved;
// callers must never infer this provenance from field values. A zero stencil
// reports zero; non-finite face data reports infinity.
void launch_mhd_ct_divb_relative_linf(
    const MhdField2D<Real>& u, backend::DeviceBuffer<Real>& scratch,
    Real* host_linf, stream_t stream, bool cylindrical = false,
    bool solver_owned = false);

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

// Apply the identical staggered magnetic closure directly to a static
// background field. This avoids staging all three B0 arrays through host
// memory after an internal halo exchange.
// -- Prescribed-background validation ----------------------------------------
//
// A supplied B0 must be discretely divergence-free, compatible with the
// configured homogeneous boundary closure, and -- when the deck or the profile
// asserts it -- discretely curl-free. These sweeps touch every padded cell, so
// they run on device; the host keeps only the final comparison against the
// tolerance and the throw, which is what turns a number into a diagnosable
// error message.
//
// Each result deliberately reports a MEASURED defect plus flags rather than a
// bare pass/fail. The caller needs the distinction: a non-finite sample, a
// resolved divergence, and a divergence explained entirely by representation
// roundoff are three different situations with three different messages, and
// collapsing them on device would throw away the information the message needs.
struct MhdBackgroundSolenoidalResult {
  // Largest residual and the largest directional scale it is judged against,
  // both as scaled values so a thin annulus at a large radius survives. The
  // ratio is formed on the host by numerics::normalized_scaled_ratio.
  quasar::numerics::ScaledValue residual_linf{};
  quasar::numerics::ScaledValue directional_scale_linf{};
  // Any b0x/b0y/b0z sample in padded storage was not finite.
  int nonfinite_sample{0};
};

struct MhdBackgroundCurlFreeResult {
  // Poloidal curl defect at interior corners, already normalized.
  Real relative_linf{Real{0}};
  // The curl components that must vanish exactly rather than to a tolerance:
  // stored-value differences of the out-of-plane component, and (on an
  // axis-touching cylindrical grid) the out-of-plane component itself. A
  // regular axisymmetric curl-free field on a domain containing r=0 has no
  // toroidal component at all; the formal annular solution Bphi=C/r is
  // singular there and represents a distributional axial current, so this is
  // not a tolerance question.
  int single_derivative_is_zero{1};
  // A cylindrical interior face was at negative radius.
  int negative_interior_radius{0};
};

// One bit per closure rule, ordered so the host reports the same rule the
// sequential sweep would have reported first.
enum MhdBackgroundBoundaryRule : unsigned {
  mhd_background_rule_periodic_x = 1u << 0,
  mhd_background_rule_axis_parity = 1u << 1,
  mhd_background_rule_axis_constraint = 1u << 2,
  mhd_background_rule_x_wall_parity = 1u << 3,
  mhd_background_rule_x_wall_normal = 1u << 4,
  mhd_background_rule_periodic_y = 1u << 5,
  mhd_background_rule_y_wall_parity = 1u << 6,
  mhd_background_rule_y_wall_normal = 1u << 7,
};

struct MhdBackgroundBoundaryResult {
  unsigned violated_rules{0};
};

// `field_modes` holds the four boundary mode codes in side order
// (x_lo, x_hi, y_lo, y_hi): 0 = ignored, 1 = periodic, 2 = wall, 3 = axis.
MhdBackgroundSolenoidalResult launch_mhd_validate_background_solenoidal(
    const MhdBackgroundField<Real>& b0, Grid2D grid, int cylindrical,
    stream_t stream);

MhdBackgroundBoundaryResult launch_mhd_validate_background_boundaries(
    const MhdBackgroundField<Real>& b0, Grid2D grid,
    const int field_modes[4], stream_t stream);

MhdBackgroundCurlFreeResult launch_mhd_validate_background_curl_free(
    const MhdBackgroundField<Real>& b0, Grid2D grid, int cylindrical,
    stream_t stream);

void launch_mhd_fill_ghosts_background(MhdBackgroundField<Real>& b0, int side,
                                       int mode, stream_t stream);

}  // namespace quasar::mhd
