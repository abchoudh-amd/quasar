#include "quasar/boundary/wall.hpp"

#include "quasar/physics/pic/species.hpp"

#include "quasar/backend/pic_kernels.hpp"

namespace quasar::boundary {

void PecFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_pec_fields(field.grid, field, static_cast<int>(side), nullptr);
}

void SpecularParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_specular_particles(species.grid(), species, static_cast<int>(side), nullptr);
}

void AbsorbingParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_absorb_particles(species.grid(), species, static_cast<int>(side), nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("pec", PecFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("specular", SpecularParticleBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("absorbing", AbsorbingParticleBC)

}  // namespace quasar::boundary
