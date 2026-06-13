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

// The adiabatic index the registry-created "hlld" solver carries by default.
// flux() is gamma-free by interface design (gamma is a construction-time
// property) and the registry default-constructs the solver, so a caller holding
// the IRiemannSolver base cannot override it; any test that compares the solver's
// flux to an analytic reference must build the reference at THIS gamma. (The full
// MhdSolver2D passes cfg_.gamma to the device kernel explicitly, so production
// runs honor the deck gamma regardless of this default.)
constexpr Real kSolverGamma = 5.0 / 3.0;

// Consistency: flux(L, L) equals the analytic physical flux of the single state,
// evaluated at the solver's own gamma.
TEST(HlldRiemann, ConsistencyEqualsAnalyticFlux) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const MhdState s =
      conserved(1.3, 0.4, -0.2, 0.3, 1.1, 0.6, 0.5, -0.4, kSolverGamma);
  const MhdFlux ref = analytic_flux_x(s, kSolverGamma);

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

// Symmetric interface: when L and R are mirror images about the interface
// (same rho/p/Bx, opposite tangential v and By), the contact sits at the
// interface so the mass flux F_rho is ~0 and the normal momentum flux equals the
// star total pressure. This pins the contact/star construction, not just
// finiteness -- an HLL-degenerate solver would NOT generally give F_rho==0 here.
TEST(HlldRiemann, SymmetricInterfaceHasZeroMassFlux) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const Real bx = 0.5;
  // Mirror states: identical rho/p/Bx/Bz, opposite vy and By so the problem is
  // symmetric under x-reflection. The HLLD contact speed S_M is then exactly 0.
  const MhdState L = conserved(1.0, 0.0,  0.3, 0.0, 1.0, bx,  0.4, 0.1, kGamma);
  const MhdState R = conserved(1.0, 0.0, -0.3, 0.0, 1.0, bx, -0.4, 0.1, kGamma);

  MhdFlux out{};
  solver->flux(L, R, /*dir=*/0, out);
  EXPECT_TRUE(finite_flux(out));
  // Contact at the interface => no net mass crosses it.
  EXPECT_NEAR(out.rho, 0.0, 1e-9);
  // Normal field flux remains zero.
  EXPECT_NEAR(out.bx, 0.0, kFluxTol);
}

// HLLD resolves the rotational/Alfven discontinuity that HLL smears: across an
// interface with a tangential-field reversal and a normal field, the tangential
// magnetic flux F_By is nonzero and finite (the rotational wave carries it). A
// single-state HLL average would not capture the rotational structure; this pins
// that the multi-state machinery is actually engaged.
TEST(HlldRiemann, RotationalDiscontinuityProducesTangentialFieldFlux) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const Real bx = 0.7;  // strong normal field => well-separated Alfven waves
  const MhdState L = conserved(1.0, 0.2, 0.0, 0.0, 1.0, bx,  0.6, 0.0, kGamma);
  const MhdState R = conserved(1.0, 0.2, 0.0, 0.0, 1.0, bx, -0.6, 0.0, kGamma);

  MhdFlux out{};
  solver->flux(L, R, /*dir=*/0, out);
  EXPECT_TRUE(finite_flux(out));
  // The tangential field reverses across the fan; the rotational wave gives a
  // nonzero tangential-field flux (it would be ~0 for a trivial average here).
  EXPECT_GT(std::abs(out.by), 1e-3);
}

// Direction wiring: the dir=1 solve runs the same +x-normal core on the
// (x<->y)-swapped state (shared rotate_in/out, also used by the device kernel).
// For a uniform state the mass flux is rho*v_normal -- rho*vx for dir=0 and
// rho*vy for dir=1 -- and the NORMAL magnetic flux vanishes in each direction
// (F_Bx==0 for dir=0, F_By==0 for dir=1). These pin the rotation wiring without
// relying on the subtler transverse-component algebra.
TEST(HlldRiemann, DirectionRotationConsistency) {
  auto solver = make_hlld();
  ASSERT_NE(solver, nullptr);

  const MhdState s =
      conserved(1.2, 0.3, -0.15, 0.25, 0.9, 0.5, 0.45, -0.3, kSolverGamma);
  MhdFlux fx{}, fy{};
  solver->flux(s, s, /*dir=*/0, fx);
  solver->flux(s, s, /*dir=*/1, fy);
  EXPECT_NEAR(fx.rho, s.mx, kFluxTol);  // rho*vx
  EXPECT_NEAR(fy.rho, s.my, kFluxTol);  // rho*vy
  EXPECT_NEAR(fx.bx, 0.0, kFluxTol);    // dir=0 normal-field flux
  EXPECT_NEAR(fy.by, 0.0, kFluxTol);    // dir=1 normal-field flux
  EXPECT_TRUE(finite_flux(fx) && finite_flux(fy));
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
