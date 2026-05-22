#include "quasar/physics/pic/diagnostics.hpp"

namespace quasar::pic {

Real total_kinetic_energy(const ParticleSpecies& species) {
  (void)species;
  return Real{0};
}

Real total_em_energy(const YeeField2D<Real>&, const Grid2D&) {
  return Real{0};
}

Real gauss_residual(const YeeField2D<Real>&, const JField2D<Real>&) {
  return Real{0};
}

std::size_t alive_count(const ParticleSpecies& species) {
  return species.size();
}

}  // namespace quasar::pic
