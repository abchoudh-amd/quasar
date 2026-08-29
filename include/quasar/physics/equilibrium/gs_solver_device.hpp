#pragma once

// Device-backed free-boundary Grad-Shafranov solver.
//
// Same contract as GsSolver -- same GsConfig in, same GsResult out, same
// GsStatus failure semantics -- with every arithmetic step running on the GPU.
// It exists alongside the host solver rather than replacing it outright so that
// the two can be compared on a complete solve before the host implementation is
// deleted. That comparison is the port's acceptance gate; see
// tests/unit/physics/equilibrium/test_gs_solver_device.cpp.
//
// -- What "device" does and does not mean here ---------------------------------
// All arithmetic is on the GPU. The Picard loop itself -- the iteration counter,
// the status transitions, the stall window -- stays on the host, because it is
// control flow rather than computation and each step launches a sequence of
// kernels. Two scalars are read back per iteration (the raw plasma current, for
// the I_p normalization, and the residual norm, for the convergence and stall
// tests). Those readbacks are load-bearing: both feed branches that decide what
// the next iteration does, and eliminating them would change the iteration
// count and therefore the answer.
//
// -- Profile restriction -------------------------------------------------------
// The profile must be a PolynomialProfile. IEquilibriumProfile is virtual and a
// vtable cannot cross to the device, so the profile is lowered to a
// ProfileCoefficients POD at construction. A non-polynomial profile is a
// construction-time error rather than a silent fallback; adding one means
// giving it a device evaluator (see the note on ProfileCoefficients).

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"

#include <memory>

namespace quasar::equilibrium {

class GsSolverDevice {
 public:
  GsSolverDevice(GsConfig cfg, std::shared_ptr<IEquilibriumProfile> profile);

  GsResult solve();

 private:
  GsConfig cfg_;
  ProfileCoefficients profile_{};
};

}  // namespace quasar::equilibrium
