#include "quasar/boundary/periodic.hpp"

#include "quasar/physics/pic/species.hpp"

#include "quasar/physics/pic/kernels.hpp"

namespace quasar::boundary {

void PeriodicFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_periodic_fields(field.grid, field, static_cast<int>(side), nullptr);
}

void PeriodicParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_periodic_particles(species.grid(), species,
                                           static_cast<int>(side), nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("periodic", PeriodicFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("periodic", PeriodicParticleBC)

}  // namespace quasar::boundary
