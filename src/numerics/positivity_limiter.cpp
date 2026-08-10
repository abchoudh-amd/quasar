// Troubled-cell positivity limiter for ideal MHD, self-registered "troubled_cell".
//
// Evolution calls admissible_fraction() with zero bounds: each cell bounds the
// convex segment from a known-positive base state to a stage candidate, and a
// device block-min reduction returns the most restrictive theta. MhdSolver2D
// retries the entire conservative SSP-RK substep when theta<1, using
// piecewise-constant HLL as its low-order retry anchor. The full coupled
// cylindrical/CT/background operator is checked for admissibility at runtime;
// no unconditional invariant-domain progress claim is made for it. A request
// that cannot make a positive representable advance is restored and reported
// as an error. No evolved cell is locally repaired. Positive configured floors
// are used only by the explicit repair surface; they cannot be invariant under
// an arbitrary conservative flux and automatic initialization does not clamp
// to them.
//
// apply() remains a thin launcher over apply_floors_kernel for explicit callers
// that already hold an invalid standalone field. Keeping that repair in one
// device implementation avoids host/device formula drift.
//
// Explicit-repair semantics (realized in apply_floors_kernel): for every cell,
//   1. density floor: if rho < rho_floor, set rho = rho_floor;
//   2. pressure floor: with momentum m and magnetic field B held FIXED, if the
//      gas pressure p = (gamma-1)*(E - 0.5*|m|^2/rho - 0.5*|B|^2) < p_floor, raise
//      total energy to E = p_floor/(gamma-1) + 0.5*|m|^2/rho + 0.5*|B|^2 so the
//      cell's pressure becomes exactly p_floor.
// A cell already above both floors is untouched to round-off. This is the
// "troubled cell" correction: only flagged (non-physical) cells are modified.
//
// The background field B0 is inert under the field-split EOS: the floor
// re-derivation does not depend on it, and launch_mhd_apply_floors reads no B0
// buffers when inactive. apply() therefore passes a default-inactive background.

#include "quasar/numerics/positivity_limiter.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"

namespace quasar::numerics {

// Conservative-update controller plus explicit standalone repair. Registered
// "troubled_cell" for deck compatibility.
class TroubledCellLimiter : public IPositivityLimiter {
 public:
  void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor, Real p_floor,
             Real gamma, int collocation_order = 0,
             RadialTablesView radial_tables = {}) const override {
    // Default-inactive background: B0 identically zero, so the device floor takes
    // the zero-background fast path (bit-identical to the original host floor) and
    // reads no B0 buffers.
    quasar::mhd::MhdBackgroundField<Real> b0{};
    quasar::mhd::launch_mhd_apply_floors(u, b0, rho_floor, p_floor, gamma,
                                         /*stream=*/nullptr, collocation_order,
                                         radial_tables);
    // The kernel is launched on the default stream; synchronize so apply() is a
    // completed in-place floor on return (host callers may read u immediately).
    backend::device_synchronize(nullptr);
  }

  Real admissible_fraction(
      const quasar::mhd::MhdField2D<Real>& base,
      const quasar::mhd::MhdField2D<Real>& candidate,
      Real rho_floor, Real p_floor, Real gamma,
      int collocation_order = 0,
      RadialTablesView radial_tables = {}) const override {
    Real theta = Real{0};
    quasar::mhd::launch_mhd_admissible_fraction(
        base, candidate, rho_floor, p_floor, gamma, fraction_scratch_,
        &theta, /*stream=*/nullptr, collocation_order, radial_tables);
    return theta;
  }

 private:
  mutable quasar::backend::DeviceBuffer<Real> fraction_scratch_{};
};

}  // namespace quasar::numerics

QUASAR_REGISTER_POSITIVITY_LIMITER("troubled_cell", ::quasar::numerics::TroubledCellLimiter)
