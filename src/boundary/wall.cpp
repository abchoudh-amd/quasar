#include "quasar/boundary/wall.hpp"

#include "quasar/physics/pic/species.hpp"

#include "quasar/physics/pic/kernels.hpp"

namespace quasar::boundary {

void PecFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_pec_fields(field.grid, field, static_cast<int>(side),
                                   cylindrical_ ? 1 : 0, nullptr);
}

void PecFieldBC::correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {
  (void)dt;
  fill_ghosts(field, side);
}

void PecFieldBC::correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const {
  (void)dt;
  fill_ghosts(field, side);
}

void SpecularParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_specular_particles(species.grid(), species, static_cast<int>(side), nullptr);
}

void SpecularParticleBC::fold_current(JField2D<Real>& current, Side side) const {
  ::launch_pic_boundary_specular_foldback(
      current.grid, current, static_cast<int>(side), cylindrical_ ? 1 : 0,
      nullptr);
}

void SpecularParticleBC::fold_charge(ScalarGrid2D<Real>& charge, Side side) const {
  ::launch_pic_boundary_specular_foldback_charge(
      charge.grid, charge, static_cast<int>(side), cylindrical_ ? 1 : 0,
      nullptr);
}

void AbsorbingParticleBC::prepare_deposit(pic::ParticleSpecies& species,
                                          Side side,
                                          int shape_order) const {
  ::launch_pic_boundary_prepare_absorb(species.grid(), species,
                                       static_cast<int>(side), shape_order,
                                       nullptr);
}

void AbsorbingParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  ::launch_pic_boundary_absorb_particles(species.grid(), species, static_cast<int>(side), nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("pec", PecFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("specular", SpecularParticleBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("absorbing", AbsorbingParticleBC)

}  // namespace quasar::boundary
