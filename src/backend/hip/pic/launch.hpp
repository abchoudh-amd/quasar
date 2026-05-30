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
void launch_pic_filter_binomial(const quasar::Grid2D&, quasar::JField2D<double>&,
                                int, hipStream_t);
void launch_pic_filter_compensated(const quasar::Grid2D&, quasar::JField2D<double>&,
                                   int, hipStream_t);

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
void launch_pic_particle_compact(quasar::pic::ParticleSpecies&, hipStream_t);

}  // extern "C"
