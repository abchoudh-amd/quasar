// SSP-RK3 (three-stage, third-order strong-stability-preserving Runge-Kutta)
// integrator for the ideal-MHD solver, self-registered as "ssprk3".
//
// The integrator is a THIN, state-free host loop. It owns no buffers and applies
// no Shu-Osher coefficients: the solver's combine_stage() owns the Shu-Osher
// combine, the CT face-B rate, and the positivity admissibility check. The
// integrator only sequences the seam.
//
// Register routing (mhd_solver.hpp: "rk_[0] is the live state U; rk_[1], rk_[2]
// are the SSP-RK3 stage registers."). The solver's combine_stage writes:
//   stage 0: out = rk_[1] = U1   (reads Un  = rk_[0])
//   stage 1: out = rk_[2] = U2   (reads U1  = rk_[1])
//   stage 2: out = rk_[0] = Un+1 (reads U2  = rk_[2])
// So the residual L(u) at each stage is evaluated on the register that holds the
// stage INPUT: rk_register(0) at stage 0, rk_register(1) at stage 1,
// rk_register(2) at stage 2 -- i.e. the stage-input index equals the stage index.
// One full SSP-RK3 step is therefore:
//
//   for s in {0,1,2}:
//     compute_residual(rk_register(s), residual_register());  // dudt = L(u_s)
//     combine_stage(s, dt);                                    // solver routes
//
// after which the live state rk_[0] holds U^{n+1}.

#include "quasar/numerics/ssprk_integrator.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace quasar::numerics {

// Three-stage third-order SSP-RK (Shu & Osher 1988). Registered "ssprk3".
class Ssprk3 : public ISsprkIntegrator {
 public:
  int n_stages() const override { return 3; }

  void advance(quasar::mhd::MhdSolver2D& solver, Real dt) const override {
    if (!(dt > Real{0}) || !std::isfinite(dt)) {
      throw std::invalid_argument{"Ssprk3::advance: dt must be finite and positive"};
    }
    const int stages = n_stages();
    for (int s = 0; s < stages; ++s) {
      // The stage-input register index equals the stage index (see header note
      // and the solver's combine_stage routing): stage 0 reads rk_register(0)=Un,
      // stage 1 reads rk_register(1)=U1, stage 2 reads rk_register(2)=U2.
      solver.compute_residual(solver.rk_register(s), solver.residual_register());
      solver.combine_stage(s, dt);
    }
  }
};

}  // namespace quasar::numerics

QUASAR_REGISTER_INTEGRATOR("ssprk3", ::quasar::numerics::Ssprk3)
