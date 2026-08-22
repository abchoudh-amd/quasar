#pragma once

// Riemann-solver interface for the ideal-MHD numerics axis. A Riemann solver
// consumes the left/right reconstructed conserved states at a cell interface
// and returns the numerical (Godunov) flux along the interface normal `dir`
// (0 = x, 1 = y). Concrete solvers self-register by name via
// QUASAR_REGISTER_RIEMANN_SOLVER so the input deck selects them by string.
//
// Scope: this interface is a HOST test seam, not the evolution path. MhdSolver2D
// pins riemann='hlld' by name and calls launch_mhd_hlld_flux directly; nothing in
// production dispatches through IRiemannSolver. Its value is that the registered
// adapter reaches the same host/device-shared hlld_core.hpp the GPU runs, so unit
// tests can exercise the real seven-wave algebra from hand-built states. Adding a
// second Riemann solver means teaching the device path about it -- registering a
// class here alone would not change what the solver computes (the constructor
// gate rejects any other name rather than silently ignoring it).
//
// The interface is intentionally gamma-free in flux(): the adiabatic index is a
// solver-construction-time property. Because the registry default-constructs
// solvers (argument-free factory), the concrete HLLD solver defaults gamma to
// 5/3 and exposes a non-virtual set_gamma() that the solver driver may call
// after create() to pin a different value. max_wavespeed() does take gamma
// explicitly so the CFL pass need not assume a stored value.

#include "quasar/numerics/mhd_state.hpp"

namespace quasar::numerics {

class IRiemannSolver {
 public:
  virtual ~IRiemannSolver() = default;

  // Numerical flux across the interface whose outward normal is `dir`
  // (0 = x, 1 = y), given the reconstructed left/right conserved states.
  virtual void flux(const MhdState& L, const MhdState& R, int dir, MhdFlux& out) const = 0;

  // Upper bound on the signal speed used for the CFL condition:
  // |v_dir| + fast_magnetosonic_speed(state, dir, gamma).
  virtual Real max_wavespeed(const MhdState& state, int dir, Real gamma) const = 0;
};

}  // namespace quasar::numerics
