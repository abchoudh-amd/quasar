#pragma once

#include "quasar/physics/pic/pic_solver.hpp"

namespace quasar::pic {

Real total_kinetic_energy(const ParticleSpecies& species);
// Yee discrete EM energy: 0.5*sum(component^2 * dual_control_volume), evaluated
// on each component's own logical lattice (never average-then-square). The
// overload without a BoundarySpec retains the historical all-periodic topology;
// use the boundary-aware overload for wall/open domains. Cylindrical weights are
// exact radial dual annuli times the axial dual length.
Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     bool cylindrical = false);
Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     const boundary::BoundarySpec& boundary,
                     bool cylindrical = false);
Real total_em_energy(const EmPic2D3V& solver);
// Volume-weighted RMS of the solver-compatible Gauss residual div(E)-rho. The
// derivative order and boundary topology must match the field solver exactly.
Real gauss_residual(const YeeField2D<Real>& fields,
                    const ScalarGrid2D<Real>& charge, int fdtd_order,
                    const boundary::BoundarySpec& boundary,
                    bool cylindrical = false);
Real gauss_residual(EmPic2D3V& solver);
// Vacuum-only divergence norm retained under an honest name for callers that do
// not have a particle charge field.
Real electric_divergence_norm(const YeeField2D<Real>& fields, int fdtd_order,
                              const boundary::BoundarySpec& boundary,
                              bool cylindrical = false);
std::size_t alive_count(const ParticleSpecies& species);

}  // namespace quasar::pic
