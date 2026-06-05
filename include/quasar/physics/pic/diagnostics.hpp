#pragma once

#include "quasar/physics/pic/pic_solver.hpp"

namespace quasar::pic {

Real total_kinetic_energy(const ParticleSpecies& species);
// EM field energy u = 0.5*(E^2 + B^2) integrated over the interior. When
// `cylindrical` is true the per-cell weight is the axisymmetric ring volume
// 2*pi*r*dr*dz (Grid2D::cell_volume) instead of the flat Cartesian area dx*dy.
Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     bool cylindrical = false);
// L2 norm of the discrete divergence of E. When `cylindrical` is true the
// axisymmetric divergence (1/r) d(r E_r)/dr + d E_z/dz is used with a
// non-wrapping radial axis; otherwise the Cartesian backward differences.
Real gauss_residual(const YeeField2D<Real>& fields, const JField2D<Real>& current,
                    bool cylindrical = false);
std::size_t alive_count(const ParticleSpecies& species);

}  // namespace quasar::pic
