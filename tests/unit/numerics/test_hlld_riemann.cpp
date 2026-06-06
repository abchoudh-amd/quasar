// RED-phase tests for the HLLD ideal-MHD Riemann solver, obtained via the
// registry under the string name "hlld".
//
// Targets the blind contract in include/quasar/numerics/riemann_solver.hpp:
//   class IRiemannSolver {
//    public: virtual ~IRiemannSolver()=default;
//     virtual void flux(const MhdState& L, const MhdState& R, int dir,
//                       MhdFlux& out) const = 0;
//     virtual Real max_wavespeed(const MhdState&, int dir, Real gamma) const = 0;
//   };
// and the registry API in include/quasar/core/registry.hpp:
//   quasar::Registry<IRiemannSolver>::instance().create("hlld").
//
// The MHD numerics module is built WHOLE_ARCHIVE so the "hlld" registration
// survives the link once implemented; this CPU-only test exercises it directly.
//
// NOTE: IRiemannSolver::flux as declared does not take gamma. These tests use
// gamma values consistent with the constructed states; the implementation is
// expected to carry gamma via construction/default for the registry-created
// solver, while max_wavespeed takes gamma explicitly per the contract.

#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

namespace {

using quasar::Real;
using quasar::numerics::IRiemannSolver;
using quasar::numerics::MhdFlux;
using quasar::numerics::MhdState;

constexpr Real kFluxTol = 1e-10;

// The gamma the registry-created "hlld" solver is expected to use by default.
// Brio-Wu and the consistency states below all use gamma=2 to match.
constexpr Real kGamma = 2.0;

std::unique_ptr<IRiemannSolver> make_hlld() {
  return ::quasar::Registry<IRiemannSolver>::instance().create("hlld");
}

// Build a conserved MHD state from primitive quantities (total-energy form).
MhdState conserved(Real rho, Real vx, Real vy, Real vz, Real p, Real bx, Real by,
                   Real bz, Real gamma) {
  const Real e_int = p / ((gamma - 1.0) * rho);
  MhdState s{};
  s.rho = rho;
  s.mx = rho * vx;
  s.my = rho * vy;
  s.mz = rho * vz;
  s.energy = rho * e_int + 0.5 * rho * (vx * vx + vy * vy + vz * vz) +
             0.5 * (bx * bx + by * by + bz * bz);
  s.bx = bx;
  s.by = by;
  s.bz = bz;
  return s;
}

// Analytic conserved-variable ideal-MHD flux in the x-direction (dir=0).
// Local helper for the consistency check; does not call library internals.
MhdFlux analytic_flux_x(const MhdState& s, Real gamma) {
  const Real rho = s.rho;
  const Real vx = s.mx / rho;
  const Real vy = s.my / rho;
  const Real vz = s.mz / rho;
  const Real bx = s.bx, by = s.by, bz = s.bz;
  const Real b2 = bx * bx + by * by + bz * bz;
  const Real v2 = vx * vx + vy * vy + vz * vz;
  const Real vdotb = vx * bx + vy * by + vz * bz;
  const Real p_gas = (gamma - 1.0) * (s.energy - 0.5 * rho * v2 - 0.5 * b2);
  const Real p_tot = p_gas + 0.5 * b2;

  MhdFlux f{};
  f.rho = rho * vx;
  f.mx = rho * vx * vx + p_tot - bx * bx;
  f.my = rho * vx * vy - bx * by;
  f.mz = rho * vx * vz - bx * bz;
  f.energy = (s.energy + p_tot) * vx - bx * vdotb;
  f.bx = 0.0;  // normal field has zero flux in its own direction
  f.by = by * vx - bx * vy;
  f.bz = bz * vx - bx * vz;
  return f;
}

bool finite_flux(const MhdFlux& f) {
  return std::isfinite(f.rho) && std::isfinite(f.mx) && std::isfinite(f.my) &&
         std::isfinite(f.mz) && std::isfinite(f.energy) && std::isfinite(f.bx) &&
         std::isfinite(f.by) && std::isfinite(f.bz);
}

// Hydrodynamic + magnetic fast magnetosonic speed in dir=0 for a uniform state,
// computed locally for the max_wavespeed lower-bound check.
Real fast_speed_x(const MhdState& s, Real gamma) {
  const Real rho = s.rho;
  const Real p = (gamma - 1.0) *
                 (s.energy - 0.5 * (s.mx * s.mx + s.my * s.my + s.mz * s.mz) / rho -
                  0.5 * (s.bx * s.bx + s.by * s.by + s.bz * s.bz));
  const Real a2 = gamma * p / rho;
  const Real b2 = (s.bx * s.bx + s.by * s.by + s.bz * s.bz) / rho;
  const Real bn2 = (s.bx * s.bx) / rho;  // normal field for dir=0
  const Real term = a2 + b2;
  const Real disc = std::sqrt(std::max(0.0, term * term - 4.0 * a2 * bn2));
  return std::sqrt(0.5 * (term + disc));
}

}  // namespace

// The registry must yield a non-null HLLD solver instance.
TEST(HlldRiemann, RegistryProducesSolver) {
  std::unique_ptr<IRiemannSolver> solver = make_hlld();
  ASSERT_NE(solver, nullptr);
}

// Consistency: flux(L, L) equals the analytic physical flux of the single state.
TEST(HlldRiemann, ConsistencyEqualsAnalyticFlux) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const MhdState s = conserved(1.3, 0.4, -0.2, 0.3, 1.1, 0.6, 0.5, -0.4, kGamma);
  const MhdFlux ref = analytic_flux_x(s, kGamma);

  MhdFlux out{};
  solver->flux(s, s, /*dir=*/0, out);

  EXPECT_NEAR(out.rho, ref.rho, kFluxTol);
  EXPECT_NEAR(out.mx, ref.mx, kFluxTol);
  EXPECT_NEAR(out.my, ref.my, kFluxTol);
  EXPECT_NEAR(out.mz, ref.mz, kFluxTol);
  EXPECT_NEAR(out.energy, ref.energy, kFluxTol);
  EXPECT_NEAR(out.by, ref.by, kFluxTol);
  EXPECT_NEAR(out.bz, ref.bz, kFluxTol);
}

// Uniform state: with identical L and R the interface flux equals the analytic
// flux and the normal magnetic flux is zero (Bx is continuous, F_Bx == 0).
TEST(HlldRiemann, UniformStateNormalFieldFluxIsZero) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const MhdState s = conserved(0.9, 0.1, 0.2, -0.1, 0.8, 0.75, 0.3, 0.2, kGamma);
  MhdFlux out{};
  solver->flux(s, s, /*dir=*/0, out);
  EXPECT_NEAR(out.bx, 0.0, kFluxTol);
}

// Brio-Wu-like interface: standard left/right states, gamma=2, zero velocity.
// L: rho=1, p=1, Bx=0.75, By=1 ; R: rho=0.125, p=0.1, Bx=0.75, By=-1.
TEST(HlldRiemann, BrioWuInterfaceIsFiniteWithContinuousNormalField) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const Real bx = 0.75;
  const MhdState L = conserved(1.0, 0.0, 0.0, 0.0, 1.0, bx, 1.0, 0.0, kGamma);
  const MhdState R = conserved(0.125, 0.0, 0.0, 0.0, 0.1, bx, -1.0, 0.0, kGamma);

  MhdFlux out{};
  solver->flux(L, R, /*dir=*/0, out);

  EXPECT_TRUE(finite_flux(out));
  // Bx is continuous across the interface => its normal flux vanishes.
  EXPECT_NEAR(out.bx, 0.0, kFluxTol);
}

// max_wavespeed is positive and at least the fast-magnetosonic speed of both
// the left and right states for the Brio-Wu interface.
TEST(HlldRiemann, MaxWavespeedBoundsFastSpeeds) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const Real bx = 0.75;
  const MhdState L = conserved(1.0, 0.0, 0.0, 0.0, 1.0, bx, 1.0, 0.0, kGamma);
  const MhdState R = conserved(0.125, 0.0, 0.0, 0.0, 0.1, bx, -1.0, 0.0, kGamma);

  const Real sL = solver->max_wavespeed(L, /*dir=*/0, kGamma);
  const Real sR = solver->max_wavespeed(R, /*dir=*/0, kGamma);

  EXPECT_GT(sL, 0.0);
  EXPECT_GT(sR, 0.0);
  EXPECT_GE(sL, fast_speed_x(L, kGamma) - kFluxTol);
  EXPECT_GE(sR, fast_speed_x(R, kGamma) - kFluxTol);
}

// Degenerate B -> 0 interface (pure hydro): the flux stays finite (no NaN).
TEST(HlldRiemann, FiniteAtVanishingFieldInterface) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const MhdState L = conserved(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, kGamma);
  const MhdState R = conserved(0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, kGamma);

  MhdFlux out{};
  solver->flux(L, R, /*dir=*/0, out);
  EXPECT_TRUE(finite_flux(out));

  EXPECT_GT(solver->max_wavespeed(L, /*dir=*/0, kGamma), 0.0);
  EXPECT_GT(solver->max_wavespeed(R, /*dir=*/0, kGamma), 0.0);
}
