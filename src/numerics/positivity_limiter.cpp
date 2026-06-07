// Troubled-cell positivity limiter for ideal MHD, self-registered "troubled_cell".
//
// This translation unit is a THIN LAUNCHER over the single authoritative floor
// implementation, the device kernel
// (src/backend/hip/mhd/mhd_update.hip :: apply_floors_kernel) declared via
// launch_mhd_apply_floors in physics/mhd/kernels.hpp. apply() forwards to it on
// the default device stream (nullptr) and then synchronizes, mirroring how the
// sibling MHD host wrapper (physics/mhd/mhd_geometric_source.cpp) forwards to
// launch_mhd_geometric_source. Keeping one device implementation avoids a
// host/device floor-formula drift.
//
// Floor semantics (UNCHANGED, realized in apply_floors_kernel): for every cell,
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

// Floor-based troubled-cell limiter. Registered "troubled_cell".
class TroubledCellLimiter : public IPositivityLimiter {
 public:
  void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor, Real p_floor,
             Real gamma) const override {
    // Default-inactive background: B0 identically zero, so the device floor takes
    // the zero-background fast path (bit-identical to the original host floor) and
    // reads no B0 buffers.
    quasar::mhd::MhdBackgroundField<Real> b0{};
    quasar::mhd::launch_mhd_apply_floors(u, b0, rho_floor, p_floor, gamma,
                                         /*stream=*/nullptr);
    // The kernel is launched on the default stream; synchronize so apply() is a
    // completed in-place floor on return (host callers may read u immediately).
    backend::device_synchronize(nullptr);
  }
};

}  // namespace quasar::numerics

QUASAR_REGISTER_POSITIVITY_LIMITER("troubled_cell", ::quasar::numerics::TroubledCellLimiter)
