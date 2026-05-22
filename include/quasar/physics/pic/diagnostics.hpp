#pragma once

#include "quasar/physics/pic/pic_solver.hpp"

namespace quasar::pic {

Real total_kinetic_energy(const ParticleSpecies& species);
Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid);
Real gauss_residual(const YeeField2D<Real>& fields, const JField2D<Real>& current);
std::size_t alive_count(const ParticleSpecies& species);

}  // namespace quasar::pic
