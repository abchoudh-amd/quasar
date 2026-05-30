// Single source of truth for the PIC backend kernel-launch ABI.
//
// Every launch_pic_* entry point defined under src/backend/hip/pic/ is declared
// here exactly once and included both by its .hip definition (so a signature
// drift is a compile error) and by every caller in the physics/boundary/numerics
// layers. Do not hand-redeclare these extern "C" prototypes anywhere else.
#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

extern "C" {

// -- FDTD field updates ------------------------------------------------------
void launch_pic_fdtd_b_order2(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*, double,
                              hipStream_t);
void launch_pic_fdtd_b_order4(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*, double,
                              hipStream_t);
void launch_pic_fdtd_e_order2(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*,
                              const double*, const double*, const double*, double,
                              hipStream_t);
void launch_pic_fdtd_e_order4(const quasar::Grid2D&, double*, double*, double*,
                              const double*, const double*, const double*,
                              const double*, const double*, const double*, double,
                              hipStream_t);

// -- Particle gather + push --------------------------------------------------
void launch_pic_gather_push_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<double>&,
                                   const quasar::YeeField2D<double>&, double, hipStream_t);
void launch_pic_gather_push_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<double>&,
                                   const quasar::YeeField2D<double>&, double, hipStream_t);

// -- Current deposition ------------------------------------------------------
void launch_pic_deposit_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<double>&, double, hipStream_t);
void launch_pic_deposit_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<double>&, double, hipStream_t);

// -- Current filtering -------------------------------------------------------
// `scratch` is caller-owned ping-pong storage of at least grid.storage_size()
// doubles (hoisted out of the per-step path).
void launch_pic_filter_binomial(const quasar::Grid2D&, quasar::JField2D<double>&,
                                double* scratch, int, hipStream_t);
void launch_pic_filter_compensated(const quasar::Grid2D&, quasar::JField2D<double>&,
                                   double* scratch, int, hipStream_t);

// -- Particle boundary conditions --------------------------------------------
void launch_pic_boundary_absorb_particles(const quasar::Grid2D&,
                                          quasar::pic::ParticleSpecies&, int, hipStream_t);
void launch_pic_boundary_specular_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int, hipStream_t);
void launch_pic_boundary_periodic_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, hipStream_t);

// -- Field boundary conditions -----------------------------------------------
void launch_pic_boundary_pec_fields(const quasar::Grid2D&, quasar::YeeField2D<double>&,
                                    int, hipStream_t);
void launch_pic_boundary_periodic_fields(const quasar::Grid2D&, quasar::YeeField2D<double>&,
                                         int, hipStream_t);

// -- Particle array compaction -----------------------------------------------
// Compacts alive particles to the front of every species array (order is not
// preserved — irrelevant for PIC) and shrinks the active count. Returns the new
// alive count via set_count() on the species.
void launch_pic_particle_compact(quasar::pic::ParticleSpecies&, hipStream_t);

// -- Alive-particle count ----------------------------------------------------
// Single-pass device reduction of the alive flags; returns the count to the
// host. Cheaper than a full HostSnapshot when only the scalar is needed.
std::size_t launch_pic_alive_count(const quasar::pic::ParticleSpecies&, hipStream_t);

}  // extern "C"
