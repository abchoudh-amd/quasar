#pragma once

// Positivity-preserving limiter interface for the ideal-MHD solver. Evolution
// uses admissible_fraction() to constrain a conservative update before accepting
// it. apply() remains an explicit repair utility for callers that hand the
// registry object an already-invalid standalone field; it is not used by the
// conservative solver path.
//
// The two operations deliberately have different mutation contracts:
//   - admissible_fraction() is read-only and drives conservative retry/subcycling;
//   - apply() is an explicit in-place repair for an already-invalid standalone
//     field, retained for registry/API compatibility and diagnostic workflows.
// Conservative evolution passes zero bounds to admissible_fraction(), enforcing
// strict rho>0 and internal energy>0. Positive configured floors are reserved
// for explicit repair because they are not invariant sets of the conservative
// equations; automatic initial-state construction does not clamp to them.

#include "quasar/core/types.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

namespace quasar::numerics {

// Pluggable positivity limiter. Concrete schemes self-register through
// QUASAR_REGISTER_POSITIVITY_LIMITER so the deck selects them by string name.
class IPositivityLimiter {
 public:
  virtual ~IPositivityLimiter() = default;

  // Explicitly repair `u` in place: clamp density below rho_floor and raise
  // total energy in any cell whose gas pressure is below p_floor so its pressure
  // becomes exactly p_floor (momentum and B unchanged). Already-positive cells
  // are left untouched to round-off. The evolution solver does not call this
  // nonconservative utility. `gamma` is the adiabatic index.
  // `collocation_order` selects the staggered-field face-to-cell rule;
  // cylindrical callers pass the matching solver-owned `radial_tables` view.
  // The inactive default retains Cartesian collocation.
  virtual void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor,
                     Real p_floor, Real gamma, int collocation_order = 0,
                     RadialTablesView radial_tables = {}) const = 0;

  // Return min_cell theta_cell in [0,1] such that the convex segment from an
  // admissible `base` state to `candidate` remains strictly above both bounds.
  // The conservative solver passes zero bounds (mathematical positivity); an
  // explicit caller may request positive bounds. A return of 1 accepts the
  // candidate unchanged. The solver uses theta<1 to retry the conservative
  // SSP-RK update with a smaller CFL-coupled substep; it never applies a
  // cell-local mass/energy repair to an evolved state. Magnetic pressure uses
  // the collocation selected by `collocation_order` and `radial_tables`, with
  // the inactive default retaining the Cartesian rule.
  virtual Real admissible_fraction(
      const quasar::mhd::MhdField2D<Real>& base,
      const quasar::mhd::MhdField2D<Real>& candidate,
      Real rho_floor, Real p_floor, Real gamma,
      int collocation_order = 0,
      RadialTablesView radial_tables = {}) const = 0;
};

}  // namespace quasar::numerics
