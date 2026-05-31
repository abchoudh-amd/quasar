#include "quasar/boundary/wall.hpp"

#include "quasar/physics/pic/species.hpp"

#include "quasar/backend/pic_kernels.hpp"

namespace quasar::boundary {

void PecFieldBC::fill_ghosts(YeeField2D<Real>& /*field*/, Side /*side*/) const {
  // No-op: the one-sided closure corrects boundary nodes after each curl, so no
  // ghost fill is needed at either order.
}

void PecFieldBC::correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {
  if (order_ == 4) {
    ::launch_pic_boundary_wall_correct_b_order4(field.grid, field,
                                                static_cast<int>(side), dt, nullptr);
  } else {
    ::launch_pic_boundary_wall_correct_b_order2(field.grid, field,
                                                static_cast<int>(side), dt, nullptr);
  }
}

void PecFieldBC::correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const {
  if (order_ == 4) {
    ::launch_pic_boundary_wall_correct_e_order4(field.grid, field,
                                                static_cast<int>(side), dt, nullptr);
  } else {
    ::launch_pic_boundary_wall_correct_e(field.grid, field, static_cast<int>(side), nullptr);
  }
}

void SpecularParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_specular_particles(species.grid(), species, static_cast<int>(side), nullptr);
}

void SpecularParticleBC::fold_current(JField2D<Real>& current, Side side) const {
  ::launch_pic_boundary_specular_foldback(current.grid, current, static_cast<int>(side), nullptr);
}

void AbsorbingParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_absorb_particles(species.grid(), species, static_cast<int>(side), nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("pec", PecFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("specular", SpecularParticleBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("absorbing", AbsorbingParticleBC)

}  // namespace quasar::boundary
