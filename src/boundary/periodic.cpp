#include "quasar/boundary/periodic.hpp"

#include "quasar/physics/pic/species.hpp"

#include "quasar/backend/pic_kernels.hpp"

namespace quasar::boundary {

void PeriodicFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_periodic_fields(field.grid, field, static_cast<int>(side), nullptr);
}

void PeriodicParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  // x_lo/x_hi wrap the x axis; y_lo/y_hi wrap the y axis. Wrapping only the
  // BC's own axis lets periodic and non-periodic walls coexist per side.
  const int axis = (side == Side::x_lo || side == Side::x_hi) ? 0 : 1;
  ::launch_pic_boundary_periodic_particles(species.grid(), species, axis, nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("periodic", PeriodicFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("periodic", PeriodicParticleBC)

}  // namespace quasar::boundary
