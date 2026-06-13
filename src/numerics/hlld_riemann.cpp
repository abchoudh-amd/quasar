// HLLD approximate Riemann solver for ideal MHD (registry adapter).
//
// Reference: T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann
// solver for ideal magnetohydrodynamics", J. Comput. Phys. 208 (2005) 315-344.
//
// The seven-wave HLLD algebra itself lives in the shared, host/device-callable
// core include/quasar/numerics/hlld_core.hpp -- the SINGLE source of truth that
// the device hot path (src/backend/hip/mhd/mhd_riemann.hip) also calls, so the
// two cannot drift in their degeneracy guards or intermediate-state formulas.
// This file is only the registry adapter: it pins gamma and wires the
// IRiemannSolver virtual interface to hlld::hlld_flux_x via the shared
// rotate_in/rotate_out.

#include "quasar/numerics/riemann_solver.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/numerics/hlld_core.hpp"

#include <cmath>

namespace quasar::numerics {

// File-local concrete solver, registered by name "hlld". Carries the adiabatic
// index gamma (default 5/3); the solver driver may override via set_gamma()
// after the registry default-constructs the object.
class HlldRiemann : public IRiemannSolver {
 public:
  void set_gamma(Real gamma) { gamma_ = gamma; }
  Real gamma() const { return gamma_; }

  void flux(const MhdState& Lin, const MhdState& Rin, int dir, MhdFlux& out) const override {
    // Rotate into the canonical +x-normal frame, solve via the shared core,
    // rotate the flux back.
    const MhdState L = hlld::rotate_in(Lin, dir);
    const MhdState R = hlld::rotate_in(Rin, dir);
    const MhdFlux  f = hlld::hlld_flux_x(L, R, gamma_);
    out = hlld::rotate_out(f, dir);
  }

  Real max_wavespeed(const MhdState& state, int dir, Real gamma) const override {
    const Real inv_rho = Real{1} / state.rho;
    const Real vn = (dir == 0) ? state.mx * inv_rho : state.my * inv_rho;
    return std::abs(vn) + fast_magnetosonic_speed(state, dir, gamma);
  }

 private:
  Real gamma_{Real{5} / Real{3}};
};

}  // namespace quasar::numerics

QUASAR_REGISTER_RIEMANN_SOLVER("hlld", ::quasar::numerics::HlldRiemann)
