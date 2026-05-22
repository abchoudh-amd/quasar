#include "quasar/boundary/periodic.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

extern "C" void launch_pic_boundary_periodic_particles(const quasar::Grid2D&,
                                                       quasar::pic::ParticleSpecies&,
                                                       hipStream_t);
extern "C" void launch_pic_boundary_periodic_fields(const quasar::Grid2D&,
                                                    quasar::YeeField2D<double>&,
                                                    int, hipStream_t);

namespace quasar::boundary {

void PeriodicFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_periodic_fields(field.grid, field, static_cast<int>(side), nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void PeriodicParticleBC::apply(pic::ParticleSpecies& species, Side) const {
  ::launch_pic_boundary_periodic_particles(species.grid(), species, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

QUASAR_REGISTER_FIELD_BOUNDARY("periodic", PeriodicFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("periodic", PeriodicParticleBC)

}  // namespace quasar::boundary
