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

extern "C" {

// -- FDTD field updates ------------------------------------------------------
void launch_pic_fdtd_b_order2(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*, double,
                              quasar_stream_t);
void launch_pic_fdtd_b_order4(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*, double,
                              quasar_stream_t);
void launch_pic_fdtd_e_order2(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*,
                              const double*, const double*, const double*, double,
                              quasar_stream_t);
void launch_pic_fdtd_e_order4(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*,
                              const double*, const double*, const double*, double,
                              quasar_stream_t);

// -- Particle gather + push --------------------------------------------------
// `periodic_x`/`periodic_y` (0/1) select per-axis field-gather indexing: a
// periodic axis wraps; a non-periodic (wall) axis clamps the interpolation
// stencil into the ghost layer (reading the field-boundary closure / replicated
// external field) instead of wrapping to the far edge. Mirrors the deposit.
void launch_pic_gather_push_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<double>&,
                                   const quasar::YeeField2D<double>&, int periodic_x,
                                   int periodic_y, double, quasar_stream_t);
void launch_pic_gather_push_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<double>&,
                                   const quasar::YeeField2D<double>&, int periodic_x,
                                   int periodic_y, double, quasar_stream_t);

// -- Current deposition ------------------------------------------------------
// `periodic_x`/`periodic_y` (0/1) select per-axis node indexing: a periodic axis
// wraps (historical behaviour), a non-periodic (wall) axis deposits into ghost
// cells without wrapping so launch_pic_boundary_specular_foldback can reflect the
// boundary-crossing current back into the interior.
void launch_pic_deposit_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<double>&, double, int periodic_x,
                               int periodic_y, quasar_stream_t);
void launch_pic_deposit_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<double>&, double, int periodic_x,
                               int periodic_y, quasar_stream_t);

// -- Current filtering -------------------------------------------------------
// `scratch` is caller-owned ping-pong storage of at least grid.storage_size()
// doubles (hoisted out of the per-step path).
void launch_pic_filter_binomial(const quasar::Grid2D&, quasar::JField2D<double>&,
                                double* scratch, int, quasar_stream_t);
void launch_pic_filter_compensated(const quasar::Grid2D&, quasar::JField2D<double>&,
                                   double* scratch, int, quasar_stream_t);

// -- Particle boundary conditions --------------------------------------------
void launch_pic_boundary_absorb_particles(const quasar::Grid2D&,
                                          quasar::pic::ParticleSpecies&, int, quasar_stream_t);
void launch_pic_boundary_specular_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int, quasar_stream_t);
// Reflects current deposited into one reflecting side's ghost cells back into the
// interior (image-charge fold) and zeroes those ghosts. Run after the deposit and
// before the current filter / E-update on every specular side.
void launch_pic_boundary_specular_foldback(const quasar::Grid2D&,
                                           quasar::JField2D<double>&, int side,
                                           quasar_stream_t);
// `axis` selects the wrapped axis: 0 = x, 1 = y. A per-side periodic BC wraps
// only its own axis so it does not interfere with non-periodic walls.
void launch_pic_boundary_periodic_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int axis,
                                            quasar_stream_t);

// -- Field boundary conditions -----------------------------------------------
void launch_pic_boundary_periodic_fields(const quasar::Grid2D&, quasar::YeeField2D<double>&,
                                         int, quasar_stream_t);

// Characteristic-wall (PEC) one-sided closures, run after the interior curl.
// correct_b rewrites the boundary-node tangential B with a one-sided normal-axis
// derivative (low walls only; high walls are interior-correct at 2nd order) and
// pins normal B to zero; correct_e pins tangential E to zero (all sides). `side`
// is the Side enum (0=x_lo,1=x_hi,2=y_lo,3=y_hi). The order-4 variant
// (outer-two-layer closure) is added with the 4th-order PEC commit.
void launch_pic_boundary_wall_correct_b_order2(const quasar::Grid2D&,
                                               quasar::YeeField2D<double>&, int side,
                                               double dt, quasar_stream_t);
void launch_pic_boundary_wall_correct_e(const quasar::Grid2D&,
                                        quasar::YeeField2D<double>&, int side,
                                        quasar_stream_t);
// 4th-order variants: the outer two boundary layers are reduced to the 2nd-order
// staggered scheme (interior-only) on the normal-axis derivative, then the wall
// closure / pin is applied. correct_e_order4 additionally re-closes the second
// interior layer the forward E-curl overran on the high wall.
void launch_pic_boundary_wall_correct_b_order4(const quasar::Grid2D&,
                                               quasar::YeeField2D<double>&, int side,
                                               double dt, quasar_stream_t);
void launch_pic_boundary_wall_correct_e_order4(const quasar::Grid2D&,
                                               quasar::YeeField2D<double>&, int side,
                                               double dt, quasar_stream_t);

// First-order Mur outflow (open boundary). correct_b applies the same one-sided
// B closure as the wall but pins nothing; correct_e advances the tangential-E
// components with the Mur one-way-wave update using `mur_strips` (a per-face
// history buffer of 4*len doubles: [a_u0, a_u1, b_u0, b_u1]). `init` non-zero
// seeds the strips from the current field (first step) without advancing.
void launch_pic_boundary_outflow_correct_b_order2(const quasar::Grid2D&,
                                                  quasar::YeeField2D<double>&, int side,
                                                  double dt, quasar_stream_t);
void launch_pic_boundary_outflow_correct_e_order2(const quasar::Grid2D&,
                                                  quasar::YeeField2D<double>&, int side,
                                                  double dt, double* mur_strips, int init,
                                                  int skip_lo, int skip_hi, quasar_stream_t);
// 4th-order variants: B closure reduces the outer two boundary layers to the
// interior-only stencil (no pin); E re-closes the second interior layer the
// forward curl overran (high wall) and then Mur-advances the wall node (Mur
// itself stays first order regardless of FDTD order).
void launch_pic_boundary_outflow_correct_b_order4(const quasar::Grid2D&,
                                                  quasar::YeeField2D<double>&, int side,
                                                  double dt, quasar_stream_t);
void launch_pic_boundary_outflow_correct_e_order4(const quasar::Grid2D&,
                                                  quasar::YeeField2D<double>&, int side,
                                                  double dt, double* mur_strips, int init,
                                                  int skip_lo, int skip_hi, quasar_stream_t);

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
