// Backend-neutral declaration of the PIC kernel-launch ABI.
//
// This is a per-physics seam (it speaks pic types like ParticleSpecies / JField2D),
// so it lives under physics/pic/ rather than backend/ — the backend axis stays
// physics-neutral (device.hpp / memory.hpp only). Every launch_pic_* entry point
// defined under src/backend/hip/pic/ is declared here exactly once and included
// both by its .hip definition (so a signature drift is a compile error) and by
// every caller in the physics/boundary/numerics layers. Callers reach the backend
// only through this header; do not hand-redeclare these extern "C" prototypes.
#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/species.hpp"

#include <cstddef>

// The kernel-launch ABI speaks the backend-neutral stream handle so callers in
// the physics/boundary/numerics layers never include a HIP header. The .hip
// definitions cast it back to quasar_stream_t internally.
using quasar_stream_t = ::quasar::backend::stream_t;

// The field-data ABI is phrased in quasar::Real (and YeeField2D<Real> /
// JField2D<Real>), not literal double, so the kernel boundary tracks the same
// precision typedef as the rest of the PIC stack: if Real ever changes, the
// solver's Real* device pointers cannot silently mismatch a double* ABI slot.
extern "C" {

// -- FDTD field updates ------------------------------------------------------
void launch_pic_fdtd_b_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_b_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);

// -- Cylindrical (r,z) m=0 FDTD field updates --------------------------------
// Axisymmetric counterparts of launch_pic_fdtd_{b,e}: the curls use the
// 1/r and (1/r) d(r .)/dr radial operators with the on-axis (i=0) regularized
// closure, reading the radius from Grid2D's r_at_* helpers.  The order-four
// radial divergence is algebraically D(A)/dr+M(A)/r: it is equivalent to
// applying the staggered derivative to r*A and dividing by the cell-centre
// radius, but never materializes the potentially overflowing product r*A.
void launch_pic_fdtd_b_cyl_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_b_cyl_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_cyl_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_cyl_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);

// -- Cylindrical (r,z) particle gather + push --------------------------------
// Boris push in (r,z) with the vz ABI slot carrying v_phi and the coordinate
// (centrifugal/azimuthal) terms applied across r. `periodic_x`/`periodic_y`
// mirror the Cartesian gather flags: the axial (y) axis can still be periodic
// while the radial axis at i=0 is on-axis (never periodic).
void launch_pic_gather_push_cyl_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::BField2D<quasar::Real>&, int periodic_x,
                                       int periodic_y, quasar::Real force_dt,
                                       quasar::Real position_dt,
                                       quasar::Real previous_b_weight,
                                       quasar::Real current_b_weight, quasar_stream_t);
void launch_pic_gather_push_cyl_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::BField2D<quasar::Real>&, int periodic_x,
                                       int periodic_y, quasar::Real force_dt,
                                       quasar::Real position_dt,
                                       quasar::Real previous_b_weight,
                                       quasar::Real current_b_weight, quasar_stream_t);

// -- Cylindrical (r,z) current deposition ------------------------------------
// Charge-conserving Esirkepov deposit with ring/volume weights proportional to
// the radius (cell_volume(i)), so the discrete cylindrical continuity residual
// stays within tolerance. `periodic_x`/`periodic_y` mirror the Cartesian flags.
void launch_pic_deposit_cyl_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                   quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                                   int periodic_y, quasar_stream_t);
void launch_pic_deposit_cyl_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                   quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                                   int periodic_y, quasar_stream_t);
void launch_pic_charge_cyl_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                  quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                                  int periodic_y, quasar_stream_t);
void launch_pic_charge_cyl_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                  quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                                  int periodic_y, quasar_stream_t);

// -- Cylindrical on-axis (r=0) boundary --------------------------------------
// Field closure: enforces the r=0 parity of the field components in the i=0
// ghost/edge so the 1/r curls stay regular on the axis. Run after each curl
// (and as a ghost fill) on the x_lo side only.
void launch_pic_boundary_axis_fields(const quasar::Grid2D&,
                                     quasar::YeeField2D<quasar::Real>&, quasar_stream_t);
// Particle closure: folds particles that cross r=0 back into the domain
// (r -> -r, vr -> -vr), keeping the on-axis approach reflectionless. Run on the
// x_lo side only.
void launch_pic_boundary_axis_particles(const quasar::Grid2D&,
                                        quasar::pic::ParticleSpecies&, quasar_stream_t);

// -- Particle gather + push --------------------------------------------------
// `periodic_x`/`periodic_y` (0/1) select per-axis field-gather indexing: a
// periodic axis wraps; a non-periodic (wall) axis reads the padded interpolation
// stencil (the evolved field's boundary closure and the prescribed field sampled
// at those same ghost coordinates), clamping only at the allocation limit for
// low-level caller safety instead of wrapping to the far edge. Mirrors the
// deposit.
void launch_pic_gather_push_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::BField2D<quasar::Real>&, int periodic_x,
                                   int periodic_y, quasar::Real force_dt,
                                   quasar::Real position_dt,
                                   quasar::Real previous_b_weight,
                                   quasar::Real current_b_weight, quasar_stream_t);
void launch_pic_gather_push_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::BField2D<quasar::Real>&, int periodic_x,
                                   int periodic_y, quasar::Real force_dt,
                                   quasar::Real position_dt,
                                   quasar::Real previous_b_weight,
                                   quasar::Real current_b_weight, quasar_stream_t);
// Snapshot B^{n-1/2} immediately before Faraday advances it to B^{n+1/2}.
void launch_pic_copy_b(const quasar::Grid2D&, const quasar::YeeField2D<quasar::Real>&,
                       quasar::BField2D<quasar::Real>&, quasar_stream_t);

// -- Current deposition ------------------------------------------------------
// `periodic_x`/`periodic_y` (0/1) select per-axis node indexing: a periodic axis
// wraps (historical behaviour), a non-periodic (wall) axis deposits into ghost
// cells without wrapping so launch_pic_boundary_specular_foldback can reflect the
// boundary-crossing current back into the interior.
void launch_pic_deposit_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                               int periodic_y, quasar_stream_t);
void launch_pic_deposit_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                               int periodic_y, quasar_stream_t);
void launch_pic_charge_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                              quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                              int periodic_y, quasar_stream_t);
void launch_pic_charge_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                              quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                              int periodic_y, quasar_stream_t);
void launch_pic_add_uniform_charge(const quasar::Grid2D&,
                                   quasar::ScalarGrid2D<quasar::Real>&,
                                   quasar::Real density, quasar_stream_t);
// The fourth-order staggered derivative factors as D4+ = D2+ S with
// S=(13/12)I-(1/24)(T+ + T-). This solves Sx*Jx=Jx_raw and Sy*Jy=Jy_raw after
// the ordinary Esirkepov prefix deposit, making its forward D2 continuity
// identity exactly compatible with the order-four Ampere/Gauss operator.
void launch_pic_current_correct_order4(const quasar::Grid2D&,
                                       quasar::JField2D<quasar::Real>&,
                                       quasar::Real* rhs_x, quasar::Real* rhs_y,
                                       quasar::Real* iter_x, quasar::Real* iter_y,
                                       int x_lo_mode, int x_hi_mode,
                                       int y_lo_mode, int y_hi_mode,
                                       quasar_stream_t);
// Boundary modes used by both compact correction launchers are 0=periodic,
// 1=even normal-E continuation (PEC), 2=linear continuation (outflow), and
// 3=the cylindrical r=0 axis. The correction must use the same continuation as
// the field ghost closure or D4(J_corrected)=D2(J_raw) fails at boundary cells.
// Cylindrical variant solves the radial compact system without materialising
// r*Jr, and the axial system for Jz.
void launch_pic_current_correct_cyl_order4(const quasar::Grid2D&,
                                           quasar::JField2D<quasar::Real>&,
                                           quasar::Real* rhs_r, quasar::Real* rhs_z,
                                           quasar::Real* iter_r, quasar::Real* iter_z,
                                           int r_lo_mode, int r_hi_mode,
                                           int z_lo_mode, int z_hi_mode,
                                           quasar_stream_t);
// Restore the duplicate physical high normal-current faces on periodic axes.
// Must run after every filter/order correction and immediately before Ampere.
void launch_pic_current_periodic_high_faces(
    const quasar::Grid2D&, quasar::JField2D<quasar::Real>&,
    int periodic_x, int periodic_y, quasar_stream_t);
// Reads + clears the species' persistent deposit-overflow flag and throws a
// std::runtime_error if any deposit since the last check spilled outside the
// deposition window. The solver drains it immediately after initial charge and
// each step's current/next-charge deposits; finalize() is a defensive last drain.
void launch_pic_deposit_overflow_check(const quasar::pic::ParticleSpecies&,
                                       quasar_stream_t);
// Drains the gather/push sticky state-error flag and throws synchronously. The
// public pusher calls this after every launch so invalid state never reaches a
// boundary/deposit kernel, including for direct C++ callers outside EmPic2D3V.
void launch_pic_particle_error_check(const quasar::pic::ParticleSpecies&,
                                     quasar_stream_t);

// -- Current filtering -------------------------------------------------------
// `scratch` is caller-owned ping-pong storage of at least 3*grid.storage_size()
// Real values (one strip per current component; the three are filtered in one
// fused launch), hoisted out of the per-step path. `periodic_x`/`periodic_y`
// select whether the smoothing stencil wraps on that axis; non-periodic axes
// clamp at the edge so a filter cannot couple current across a wall.
void launch_pic_filter_binomial(const quasar::Grid2D&, quasar::JField2D<quasar::Real>&,
                                quasar::Real* scratch, int, int periodic_x, int periodic_y,
                                int cylindrical, quasar_stream_t);
void launch_pic_filter_compensated(const quasar::Grid2D&, quasar::JField2D<quasar::Real>&,
                                   quasar::Real* scratch, int, int periodic_x, int periodic_y,
                                   int cylindrical, quasar_stream_t);

// -- Particle boundary conditions --------------------------------------------
void launch_pic_boundary_absorb_particles(const quasar::Grid2D&,
                                          quasar::pic::ParticleSpecies&, int, quasar_stream_t);
void launch_pic_boundary_prepare_absorb(const quasar::Grid2D&,
                                        quasar::pic::ParticleSpecies&, int side,
                                        int shape_order, quasar_stream_t);
void launch_pic_boundary_specular_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int, quasar_stream_t);
// Reflects current deposited into one reflecting side's ghost cells back into the
// interior (image-charge fold) and zeroes those ghosts. Run after the deposit and
// before the current filter / E-update on every specular side.
void launch_pic_boundary_specular_foldback(const quasar::Grid2D&,
                                           quasar::JField2D<quasar::Real>&, int side,
                                           int cylindrical, quasar_stream_t);
void launch_pic_boundary_specular_foldback_charge(
    const quasar::Grid2D&, quasar::ScalarGrid2D<quasar::Real>&, int side,
    int cylindrical, quasar_stream_t);
// `side` is the Side enum (0=x_lo,1=x_hi,2=y_lo,3=y_hi). A per-side periodic BC
// wraps only particles that cross that side, so a one-sided periodic wall does
// not hide exits through the opposite non-periodic wall.
void launch_pic_boundary_periodic_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int side,
                                            quasar_stream_t);

// -- Field boundary conditions -----------------------------------------------
void launch_pic_boundary_periodic_fields(const quasar::Grid2D&,
                                         quasar::YeeField2D<quasar::Real>&,
                                         int, quasar_stream_t);

// Stagger-aware non-periodic ghost closures. PEC applies the exact even/odd
// continuation at the physical wall; outflow supplies linear halo continuation
// before its characteristic tangential-E update. `cylindrical` selects the
// radial staggering of the axisymmetric scheme.
void launch_pic_boundary_pec_fields(const quasar::Grid2D&,
                                    quasar::YeeField2D<quasar::Real>&,
                                    int side, int cylindrical,
                                    quasar_stream_t);
void launch_pic_boundary_outflow_fill_fields(const quasar::Grid2D&,
                                             quasar::YeeField2D<quasar::Real>&,
                                             int side, int cylindrical,
                                             quasar_stream_t);
// Backward-compatible internal cylindrical spellings.
void launch_pic_boundary_cyl_pec_fields(const quasar::Grid2D&,
                                        quasar::YeeField2D<quasar::Real>&,
                                        int side, quasar_stream_t);
void launch_pic_boundary_cyl_outflow_fields(const quasar::Grid2D&,
                                            quasar::YeeField2D<quasar::Real>&,
                                            int side, quasar_stream_t);

// First-order Mur characteristic update for the two tangential-E component
// lattices. `stride` is max tangential line length (normal-axis-independent),
// and `cylindrical` enables sqrt(r)-scaled radial characteristics.
void launch_pic_boundary_outflow_correct_e(
    const quasar::Grid2D&, quasar::YeeField2D<quasar::Real>&, int side,
    quasar::Real dt, quasar::Real* mur_strips, int stride, int init,
    int skip_lo, int skip_hi, int cylindrical, quasar_stream_t);
// Diagonal Mur closure for a corner shared by two outflow sides. `corner_mask`
// bits enumerate (xlo,ylo), (xhi,ylo), (xlo,yhi), (xhi,yhi).
void launch_pic_boundary_outflow_corners(const quasar::Grid2D&,
                                         quasar::YeeField2D<quasar::Real>&,
                                         quasar::Real dt, unsigned int corner_mask,
                                         quasar::Real* history, int init,
                                         int cylindrical,
                                         quasar_stream_t);

// -- Particle array compaction -----------------------------------------------
// Compacts alive particles to the front of every species array (order is not
// preserved — irrelevant for PIC) and shrinks the active count. Returns the new
// alive count via set_count() on the species.
void launch_pic_particle_compact(quasar::pic::ParticleSpecies&, quasar_stream_t);

// -- Alive-particle count ----------------------------------------------------
// Single-pass device reduction of the alive flags; returns the count to the
// host. Cheaper than a full HostSnapshot when only the scalar is needed.
std::size_t launch_pic_alive_count(const quasar::pic::ParticleSpecies&, quasar_stream_t);

}  // extern "C"
