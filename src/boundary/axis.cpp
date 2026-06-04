#include "quasar/boundary/axis.hpp"

#include "quasar/physics/pic/species.hpp"

#include "quasar/physics/pic/kernels.hpp"

namespace quasar::boundary {

// The cylindrical axis lives at r=0, the x_lo (i=0) face. The solver runs every
// field BC over all four sides, so gate the axis closure to x_lo and leave the
// other faces untouched (their deck-driven BCs own them).
void AxisFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  if (side != Side::x_lo) return;
  ::launch_pic_boundary_axis_fields(field.grid, field, nullptr);
}

void AxisFieldBC::correct_after_b(YeeField2D<Real>& field, Side side, Real /*dt*/) const {
  if (side != Side::x_lo) return;
  ::launch_pic_boundary_axis_fields(field.grid, field, nullptr);
}

void AxisFieldBC::correct_after_e(YeeField2D<Real>& field, Side side, Real /*dt*/) const {
  if (side != Side::x_lo) return;
  ::launch_pic_boundary_axis_fields(field.grid, field, nullptr);
}

void AxisParticleBC::apply(pic::ParticleSpecies& species, Side side) const {
  if (side != Side::x_lo) return;
  ::launch_pic_boundary_axis_particles(species.grid(), species, nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("axis", AxisFieldBC)
QUASAR_REGISTER_PARTICLE_BOUNDARY("axis", AxisParticleBC)

}  // namespace quasar::boundary
