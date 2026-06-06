#pragma once

// Positivity-preserving limiter interface for the ideal-MHD solver. apply()
// enforces a density floor and a gas-pressure floor on a conserved MHD field so a
// stage that produced a non-physical (negative-density or negative-pressure) cell
// is corrected back to a positive, physically admissible state.
//
// Two consumers share the SAME floor semantics:
//   - the solver's per-stage path runs the DEVICE floors
//     (quasar::mhd::launch_mhd_apply_floors, see physics/mhd/kernels.hpp);
//   - a registry-created IPositivityLimiter object exposes the equivalent floor
//     on the host via apply(), so a unit test can construct the limiter by name
//     and floor an MhdField2D directly.
// Both clamp rho to rho_floor and re-derive total energy from the floored
// pressure while holding momentum and B fixed, so the two paths agree.

#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::numerics {

// Pluggable positivity limiter. Concrete schemes self-register through
// QUASAR_REGISTER_POSITIVITY_LIMITER so the deck selects them by string name.
class IPositivityLimiter {
 public:
  virtual ~IPositivityLimiter() = default;

  // Floor `u` in place: clamp density below rho_floor up to rho_floor, and raise
  // total energy in any cell whose gas pressure is below p_floor so its pressure
  // becomes exactly p_floor (momentum and B unchanged). Already-positive cells
  // are left untouched to round-off. `gamma` is the adiabatic index.
  virtual void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor,
                     Real p_floor, Real gamma) const = 0;
};

}  // namespace quasar::numerics
