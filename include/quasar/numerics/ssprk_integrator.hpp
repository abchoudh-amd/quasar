#pragma once

// SSP-RK (strong-stability-preserving Runge-Kutta) time-integrator interface for
// the ideal-MHD solver. The integrator is intentionally STATE-FREE: it owns no
// buffers and applies no Shu-Osher coefficients itself. It only sequences calls
// on the solver's residual/stage seam (see mhd_solver.hpp):
//
//   for stage s in [0, n_stages):
//     solver.compute_residual(<stage-input register>, solver.residual_register());
//     solver.combine_stage(s, dt);
//
// All register routing, the Shu-Osher combine coefficients, the CT face-B update,
// and the positivity floors live inside the solver's combine_stage(). Keeping the
// integrator a thin host loop means a new time scheme is selectable by registry
// name without touching the solver's buffer ownership.
//
// MhdSolver2D is forward-declared (not included) on purpose: mhd_solver.hpp
// includes THIS header, so including mhd_solver.hpp here would form a cycle. The
// .cpp that defines the concrete integrators includes mhd_solver.hpp to call its
// methods.

#include "quasar/core/types.hpp"

namespace quasar::mhd {
class MhdSolver2D;  // defined in quasar/physics/mhd/mhd_solver.hpp
}  // namespace quasar::mhd

namespace quasar::numerics {

// Pluggable SSP-RK integrator. Concrete schemes self-register through
// QUASAR_REGISTER_INTEGRATOR so the deck selects them by string name.
class ISsprkIntegrator {
 public:
  virtual ~ISsprkIntegrator() = default;

  // Number of Runge-Kutta stages this scheme drives per step (3 for ssprk3).
  virtual int n_stages() const = 0;

  // Advance the solver's live state by one full step of size `dt`. A thin host
  // loop over the stages calling solver.compute_residual / solver.combine_stage;
  // it allocates nothing and applies no coefficients itself.
  virtual void advance(quasar::mhd::MhdSolver2D& solver, Real dt) const = 0;
};

}  // namespace quasar::numerics
