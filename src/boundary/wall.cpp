#include "quasar/boundary/wall.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

extern "C" void launch_pic_boundary_pec_fields(const quasar::Grid2D&,
                                               quasar::YeeField2D<double>&,
                                               int, hipStream_t);
extern "C" void launch_pic_boundary_specular_particles(const quasar::Grid2D&,
                                                       quasar::pic::ParticleSpecies&,
                                                       int, hipStream_t);
extern "C" void launch_pic_boundary_absorb_particles(const quasar::Grid2D&,
                                                     quasar::pic::ParticleSpecies&,
                                                     int, hipStream_t);

namespace quasar::boundary {

void PecFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_pec_fields(field.grid, field, static_cast<int>(side), nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void SpecularParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_specular_particles(species.grid(), species, static_cast<int>(side), nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void AbsorbingParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_absorb_particles(species.grid(), species, static_cast<int>(side), nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

QUASAR_REGISTER_FIELD_BOUNDARY("pec", PecFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("specular", SpecularParticleBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("absorbing", AbsorbingParticleBC)

}  // namespace quasar::boundary
