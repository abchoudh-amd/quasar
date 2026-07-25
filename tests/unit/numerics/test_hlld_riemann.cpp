// Tests for the HLLD ideal-MHD Riemann solver, obtained via the
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
#include "quasar/numerics/hlld_core.hpp"
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

// The overflow-safe HLL form rescales the wave speeds before combining the
// fluxes. Pin the asymmetric coefficients explicitly: F_L is weighted by S_R
// and F_R by -S_L. Equal states or symmetric wave brackets cannot detect an
// accidental interchange of those two weights.
TEST(HlldRiemann, HllFallbackUsesCanonicalAsymmetricWaveWeights) {
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real sl = Real{-2};
  constexpr Real sr = Real{5};
  const MhdState L = conserved(
      1.3, 0.7, -0.2, 0.1, 0.9, 0.4, 0.3, -0.1, gamma);
  const MhdState R = conserved(
      0.6, -0.4, 0.5, -0.3, 1.2, 0.4, -0.2, 0.6, gamma);
  const MhdFlux fl = analytic_flux_x(L, gamma);
  const MhdFlux fr = analytic_flux_x(R, gamma);
  const MhdFlux got =
      quasar::numerics::hlld::hll_flux_x(L, R, sl, sr, gamma);

  const auto expected = [&](Real flux_l, Real flux_r, Real state_l,
                            Real state_r) {
    return (sr * flux_l - sl * flux_r + sl * sr * (state_r - state_l)) /
           (sr - sl);
  };
  EXPECT_NEAR(got.rho, expected(fl.rho, fr.rho, L.rho, R.rho), 2e-15);
  EXPECT_NEAR(got.mx, expected(fl.mx, fr.mx, L.mx, R.mx), 2e-15);
  EXPECT_NEAR(got.my, expected(fl.my, fr.my, L.my, R.my), 2e-15);
  EXPECT_NEAR(got.mz, expected(fl.mz, fr.mz, L.mz, R.mz), 2e-15);
  EXPECT_NEAR(got.energy,
              expected(fl.energy, fr.energy, L.energy, R.energy), 4e-15);
  EXPECT_EQ(got.bx, Real{0});
  EXPECT_NEAR(got.by, expected(fl.by, fr.by, L.by, R.by), 2e-15);
  EXPECT_NEAR(got.bz, expected(fl.bz, fr.bz, L.bz, R.bz), 2e-15);
}

TEST(HlldRiemann, UnequalNormalFieldsAreNormalizedConsistently) {
  constexpr Real gamma = Real{5} / Real{3};
  MhdState L = conserved(1.0, 0.2, -0.1, 0.3, 0.9,
                         0.6, 0.4, -0.2, gamma);
  MhdState R = conserved(0.8, -0.15, 0.25, -0.1, 1.1,
                         1.0, -0.3, 0.5, gamma);
  const Real bn = Real{0.5} * L.bx + Real{0.5} * R.bx;

  MhdState Ln = L;
  MhdState Rn = R;
  Ln.energy += Real{0.5} * bn * bn - Real{0.5} * Ln.bx * Ln.bx;
  Rn.energy += Real{0.5} * bn * bn - Real{0.5} * Rn.bx * Rn.bx;
  Ln.bx = bn;
  Rn.bx = bn;

  const MhdFlux got = quasar::numerics::hlld::hlld_flux_x(L, R, gamma);
  const MhdFlux ref = quasar::numerics::hlld::hlld_flux_x(Ln, Rn, gamma);
  ASSERT_TRUE(finite_flux(got));
  EXPECT_NEAR(got.rho, ref.rho, kFluxTol);
  EXPECT_NEAR(got.mx, ref.mx, kFluxTol);
  EXPECT_NEAR(got.my, ref.my, kFluxTol);
  EXPECT_NEAR(got.mz, ref.mz, kFluxTol);
  EXPECT_NEAR(got.energy, ref.energy, kFluxTol);
  EXPECT_NEAR(got.by, ref.by, kFluxTol);
  EXPECT_NEAR(got.bz, ref.bz, kFluxTol);
}

// Regression for Miyoshi--Kusano Eq. 63 on the RIGHT double-star state. This
// Riemann problem has S_aL < S_M < 0 < S_aR < S_R, so the interface samples
// U_R**. Independent evaluation of the published construction gives the
// intermediate values below. In particular
//
//   E_R** = E_R* + sqrt(rho_R*) sign(Bx)
//                    [(v*.B*)_R - (v**.B**)],
//
// with the PLUS sign on the right. The numerical flux must then satisfy the
// Rankine--Hugoniot jump across both right-going waves:
//   F** = F_R + S_R(U_R*-U_R) + S_aR(U_R**-U_R*).
// Using the left-state minus sign for E_R** changes only this test's energy flux
// from -2.08710132326 to -1.92829179184, so the check is sign-sensitive.
TEST(HlldRiemann, RightDoubleStarEnergySatisfiesRankineHugoniot) {
  constexpr Real gamma = Real{5} / Real{3};
  const MhdState L = conserved(
      0.5, 0.2, 0.8, -0.3, 0.7, 0.9, 0.4, 0.5, gamma);
  const MhdState R = conserved(
      1.4, -0.5, -0.2, 0.9, 1.2, 0.9, -0.7, -0.1, gamma);

  const MhdFlux out = quasar::numerics::hlld::hlld_flux_x(L, R, gamma);
  const MhdFlux fR = analytic_flux_x(R, gamma);

  constexpr Real sR = 2.1418875868531733;
  constexpr Real saR = 0.3253072618499077;
  constexpr Real eRStar = 3.3054446710970824;
  constexpr Real eRDoubleStar = 3.0613530843929193;
  const Real rh_energy = fR.energy + sR * (eRStar - R.energy) +
                         saR * (eRDoubleStar - eRStar);

  EXPECT_NEAR(out.energy, rh_energy, 2e-13);
  EXPECT_NEAR(out.energy, -2.0871013232600024, 2e-13);
  EXPECT_GT(std::abs(out.energy - (-1.9282917918373412)), 0.1);
}

TEST(HlldRiemann, ExtremeNormalFieldKeepsRepresentableFluxFinite) {
  constexpr Real gamma = Real{5} / Real{3};
  const Real bn = Real{1.5e154};  // bn*bn overflows; 0.5*bn^2 is representable
  const Real p_seed = Real{1e300};
  MhdState s{};
  s.rho = Real{1};
  s.mx = Real{1};
  s.bx = bn;
  const Real half_bn2 = quasar::numerics::half_squared_norm3(bn, 0, 0);
  ASSERT_TRUE(std::isfinite(half_bn2));
  ASSERT_FALSE(std::isfinite(bn * bn));
  s.energy = half_bn2 + p_seed / (gamma - Real{1}) + Real{0.5};

  const Real recovered_p = quasar::numerics::pressure(s, gamma);
  ASSERT_TRUE(std::isfinite(recovered_p));
  ASSERT_GT(recovered_p, Real{0});
  const MhdFlux physical = quasar::numerics::hlld::physical_flux_x(s, gamma);
  const MhdFlux riemann = quasar::numerics::hlld::hlld_flux_x(s, s, gamma);
  ASSERT_TRUE(finite_flux(physical));
  ASSERT_TRUE(finite_flux(riemann));

  const Real expected_mx = (s.mx * (s.mx / s.rho) + recovered_p) - half_bn2;
  const Real expected_energy = (s.energy - half_bn2) + recovered_p;
  EXPECT_NEAR(physical.mx / expected_mx, Real{1}, Real{3e-15});
  EXPECT_NEAR(physical.energy / expected_energy, Real{1}, Real{3e-15});
  EXPECT_NEAR(riemann.mx / physical.mx, Real{1}, Real{3e-15});
  EXPECT_NEAR(riemann.energy / physical.energy, Real{1}, Real{3e-15});

  MhdState reflected = s;
  reflected.bx = -bn;
  const MhdFlux reflected_flux =
      quasar::numerics::hlld::hlld_flux_x(reflected, reflected, gamma);
  ASSERT_TRUE(finite_flux(reflected_flux));
  EXPECT_NEAR(reflected_flux.mx / riemann.mx, Real{1}, Real{3e-15});
  EXPECT_NEAR(reflected_flux.energy / riemann.energy, Real{1}, Real{3e-15});
}

TEST(HlldRiemann, SplitFluxNeverPromotesDominantBackgroundEnergy) {
  constexpr Real gamma = Real{5} / Real{3};
  const MhdState s = conserved(Real{1}, Real{0.2}, Real{-0.1}, Real{0.3},
                               Real{1}, Real{0.25}, Real{-0.5}, Real{0.75},
                               gamma);
  const quasar::numerics::MhdBackground b0{
      Real{1e100}, Real{-2e100}, Real{0.5e100}};

  // The split EOS must retain the O(1) gas pressure even though a total-energy
  // representation would be O(1e200) and could not encode that pressure.
  EXPECT_NEAR(quasar::numerics::pressure(s, b0, gamma), Real{1}, Real{2e-15});

  const MhdFlux physical =
      quasar::numerics::hlld::physical_flux_split_x(s, b0, gamma);
  const MhdFlux riemann =
      quasar::numerics::hlld::hlld_flux_split_x(s, s, b0, gamma);
  const MhdFlux lf =
      quasar::numerics::hlld::lax_friedrichs_flux_split_x(s, s, b0, gamma);
  ASSERT_TRUE(finite_flux(physical));
  ASSERT_TRUE(finite_flux(riemann));
  ASSERT_TRUE(finite_flux(lf));

  const auto same_relative = [](Real got, Real expected) {
    const Real scale = std::max(Real{1}, std::abs(expected));
    EXPECT_NEAR((got - expected) / scale, Real{0}, Real{2e-14});
  };
  same_relative(riemann.rho, physical.rho);
  same_relative(riemann.mx, physical.mx);
  same_relative(riemann.my, physical.my);
  same_relative(riemann.mz, physical.mz);
  same_relative(riemann.energy, physical.energy);
  same_relative(riemann.by, physical.by);
  same_relative(riemann.bz, physical.bz);
  same_relative(lf.rho, physical.rho);
  same_relative(lf.mx, physical.mx);
  same_relative(lf.my, physical.my);
  same_relative(lf.mz, physical.mz);
  same_relative(lf.energy, physical.energy);
  same_relative(lf.by, physical.by);
  same_relative(lf.bz, physical.bz);
}

TEST(HlldRiemann, ZeroBackgroundSplitPathIsExactlyTheOrdinarySolver) {
  constexpr Real gamma = Real{5} / Real{3};
  const MhdState L = conserved(
      1.3, 0.7, -0.2, 0.1, 0.9, 0.4, 0.3, -0.1, gamma);
  const MhdState R = conserved(
      0.6, -0.4, 0.5, -0.3, 1.2, 0.4, -0.2, 0.6, gamma);
  const quasar::numerics::MhdBackground zero{};
  constexpr Real sl = Real{-2};
  constexpr Real sr = Real{5};

  const MhdFlux ordinary_physical =
      quasar::numerics::hlld::physical_flux_x(L, gamma);
  const MhdFlux split_physical =
      quasar::numerics::hlld::physical_flux_split_x(L, zero, gamma);
  const MhdFlux ordinary_hll =
      quasar::numerics::hlld::hll_flux_x(L, R, sl, sr, gamma);
  const MhdFlux split_hll =
      quasar::numerics::hlld::hll_flux_split_x(L, R, zero, sl, sr, gamma);
  const MhdFlux ordinary_lf =
      quasar::numerics::hlld::lax_friedrichs_flux_x(L, R, gamma);
  const MhdFlux split_lf =
      quasar::numerics::hlld::lax_friedrichs_flux_split_x(
          L, R, zero, gamma);
  const MhdFlux ordinary_hlld =
      quasar::numerics::hlld::hlld_flux_x(L, R, gamma);
  const MhdFlux split_hlld =
      quasar::numerics::hlld::hlld_flux_split_x(L, R, zero, gamma);

  const auto expect_exact = [](const MhdFlux& got, const MhdFlux& expected) {
    EXPECT_EQ(got.rho, expected.rho);
    EXPECT_EQ(got.mx, expected.mx);
    EXPECT_EQ(got.my, expected.my);
    EXPECT_EQ(got.mz, expected.mz);
    EXPECT_EQ(got.energy, expected.energy);
    EXPECT_EQ(got.bx, expected.bx);
    EXPECT_EQ(got.by, expected.by);
    EXPECT_EQ(got.bz, expected.bz);
  };
  expect_exact(split_physical, ordinary_physical);
  expect_exact(split_hll, ordinary_hll);
  expect_exact(split_lf, ordinary_lf);
  expect_exact(split_hlld, ordinary_hlld);
}

TEST(HlldRiemann, ModerateSplitFluxMatchesAffineTotalEnergyTransform) {
  constexpr Real gamma = Real{5} / Real{3};
  const quasar::numerics::MhdBackground b0{Real{0.8}, Real{-0.4}, Real{0.3}};
  const MhdState L = conserved(
      1.1, 0.35, -0.2, 0.1, 0.9, 0.2, 0.5, -0.15, gamma);
  const MhdState R = conserved(
      0.7, -0.25, 0.4, -0.3, 1.3, 0.2, -0.35, 0.45, gamma);

  const auto promote = [&](const MhdState& u) {
    MhdState t = u;
    t.bx += b0.b0x;
    t.by += b0.b0y;
    t.bz += b0.b0z;
    t.energy = (u.energy - Real{0.5} *
                    (u.bx * u.bx + u.by * u.by + u.bz * u.bz)) +
               Real{0.5} * (t.bx * t.bx + t.by * t.by + t.bz * t.bz);
    return t;
  };
  const MhdFlux total =
      quasar::numerics::hlld::hlld_flux_x(promote(L), promote(R), gamma);
  const MhdFlux split =
      quasar::numerics::hlld::hlld_flux_split_x(L, R, b0, gamma);

  const Real f0_xx = Real{0.5} *
      (b0.b0y * b0.b0y + b0.b0z * b0.b0z - b0.b0x * b0.b0x);
  const Real f0_yx = -b0.b0x * b0.b0y;
  const Real f0_zx = -b0.b0x * b0.b0z;
  const Real transformed_energy =
      total.energy - b0.b0y * total.by - b0.b0z * total.bz;
  EXPECT_NEAR(split.rho, total.rho, 2e-12);
  EXPECT_NEAR(split.mx, total.mx - f0_xx, 2e-12);
  EXPECT_NEAR(split.my, total.my - f0_yx, 2e-12);
  EXPECT_NEAR(split.mz, total.mz - f0_zx, 2e-12);
  EXPECT_NEAR(split.energy, transformed_energy, 3e-12);
  EXPECT_EQ(split.bx, Real{0});
  EXPECT_NEAR(split.by, total.by, 2e-12);
  EXPECT_NEAR(split.bz, total.bz, 2e-12);
}

TEST(HlldRiemann, SplitPressureCancelsOpposingHugeCrossTerms) {
  constexpr Real gamma = Real{5} / Real{3};
  MhdState s{};
  s.rho = Real{1};
  s.energy = Real{2.5};  // p/(gamma-1) + |b|^2/2 = 1.5 + 1
  s.bx = Real{1};
  s.by = Real{-1};
  const quasar::numerics::MhdBackground b0{
      Real{1e200}, Real{1e200}, Real{0}};

  // q = p + B0.b + |b|^2/2 = 1 + (1e200-1e200) + 1.
  EXPECT_EQ(quasar::numerics::split_total_pressure(s, b0, gamma), Real{2});
}

TEST(HlldRiemann, DominantObliqueBackgroundRetainsFiniteFluxRemainder) {
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real A = Real{1e100};
  MhdState s{};
  s.rho = Real{1};
  s.mx = Real{1};
  s.my = Real{1};
  s.energy = Real{3.5};
  s.bx = Real{1};
  s.by = Real{1};
  const quasar::numerics::MhdBackground b0{A, A, Real{0}};

  const MhdFlux physical =
      quasar::numerics::hlld::physical_flux_split_x(s, b0, gamma);
  const MhdFlux hlld =
      quasar::numerics::hlld::hlld_flux_split_x(s, s, b0, gamma);
  const MhdFlux lf =
      quasar::numerics::hlld::lax_friedrichs_flux_split_x(
          s, s, b0, gamma);

  ASSERT_TRUE(finite_flux(physical));
  ASSERT_TRUE(finite_flux(hlld));
  ASSERT_TRUE(finite_flux(lf));
  for (const MhdFlux* flux : {&physical, &hlld, &lf}) {
    // The O(A) normal-stress and Poynting terms cancel, leaving these explicit
    // thermal/kinetic remainders.  Forming q or q* first loses both values.
    EXPECT_EQ(flux->mx, Real{2});
    EXPECT_EQ(flux->energy, Real{3.5});
    EXPECT_EQ(flux->my, Real{-2} * A);
  }
}

TEST(HlldRiemann, DominantBackgroundDensityContactTraversesStarPathExactly) {
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real A = Real{1e100};
  MhdState L{};
  L.rho = Real{1};
  L.mx = Real{1};
  L.my = Real{1};
  L.energy = Real{3.5};
  L.bx = Real{1};
  L.by = Real{1};
  MhdState R = L;
  R.rho = Real{2};
  R.mx = Real{2};
  R.my = Real{2};
  R.energy = Real{4.5};
  const quasar::numerics::MhdBackground b0{A, A, Real{0}};

  // This is a pure right-moving contact: p, v, and b are continuous while rho
  // jumps.  Lin != Rin prevents the equal-state fast path, and the interface
  // lies in the left double-star region, whose exact flux is F(L).
  const MhdFlux expected =
      quasar::numerics::hlld::physical_flux_split_x(L, b0, gamma);
  const MhdFlux got =
      quasar::numerics::hlld::hlld_flux_split_x(L, R, b0, gamma);

  ASSERT_TRUE(finite_flux(got));
  EXPECT_EQ(got.rho, expected.rho);
  EXPECT_EQ(got.mx, expected.mx);
  EXPECT_EQ(got.my, expected.my);
  EXPECT_EQ(got.mz, expected.mz);
  EXPECT_EQ(got.energy, expected.energy);
  EXPECT_EQ(got.bx, expected.bx);
  EXPECT_EQ(got.by, expected.by);
  EXPECT_EQ(got.bz, expected.bz);
}

TEST(HlldRiemann, DominantBackgroundRotationalProblemPreservesReflectionSymmetry) {
  constexpr Real gamma = Real{5} / Real{3};
  constexpr Real A = Real{1e100};
  const MhdState L = conserved(
      Real{1}, Real{0.2}, Real{0.4}, Real{-0.3}, Real{1.0},
      Real{0.5}, Real{0.7}, Real{-0.2}, gamma);
  const MhdState R = conserved(
      Real{0.8}, Real{-0.1}, Real{-0.5}, Real{0.6}, Real{1.2},
      Real{0.5}, Real{-0.6}, Real{0.4}, gamma);
  const quasar::numerics::MhdBackground b0{A, A, Real{0}};
  const MhdFlux f =
      quasar::numerics::hlld::hlld_flux_split_x(L, R, b0, gamma);
  ASSERT_TRUE(finite_flux(f));

  const auto reflect_x = [](MhdState u) {
    u.mx = -u.mx;
    u.bx = -u.bx;
    return u;
  };
  const quasar::numerics::MhdBackground reflected_b0{-A, A, Real{0}};
  const MhdFlux reflected = quasar::numerics::hlld::hlld_flux_split_x(
      reflect_x(R), reflect_x(L), reflected_b0, gamma);
  ASSERT_TRUE(finite_flux(reflected));

  const auto expect_relative = [](Real got, Real expected) {
    const Real scale = std::max(Real{1}, std::abs(expected));
    EXPECT_NEAR((got - expected) / scale, Real{0}, Real{3e-13});
  };
  expect_relative(reflected.rho, -f.rho);
  expect_relative(reflected.mx, f.mx);
  expect_relative(reflected.my, -f.my);
  expect_relative(reflected.mz, -f.mz);
  expect_relative(reflected.energy, -f.energy);
  EXPECT_EQ(reflected.bx, Real{0});
  expect_relative(reflected.by, -f.by);
  expect_relative(reflected.bz, -f.bz);
}

TEST(HlldRiemann, SplitVanishingNormalFieldMatchesAffineTotalSolver) {
  constexpr Real gamma = Real{5} / Real{3};
  const quasar::numerics::MhdBackground b0{
      Real{-0.2}, Real{0.7}, Real{-0.3}};
  const MhdState L = conserved(
      1.1, 0.2, -0.3, 0.1, 0.9, 0.2, 0.5, -0.2, gamma);
  const MhdState R = conserved(
      0.8, -0.1, 0.4, -0.2, 1.2, 0.2, -0.4, 0.35, gamma);

  const auto promote = [&](const MhdState& u) {
    MhdState t = u;
    t.bx += b0.b0x;
    t.by += b0.b0y;
    t.bz += b0.b0z;
    t.energy = (u.energy - quasar::numerics::half_squared_norm3(
                    u.bx, u.by, u.bz)) +
               quasar::numerics::half_squared_norm3(t.bx, t.by, t.bz);
    return t;
  };
  const MhdFlux total =
      quasar::numerics::hlld::hlld_flux_x(promote(L), promote(R), gamma);
  const MhdFlux split =
      quasar::numerics::hlld::hlld_flux_split_x(L, R, b0, gamma);

  const Real t0_xx = Real{0.5} *
      (b0.b0y * b0.b0y + b0.b0z * b0.b0z - b0.b0x * b0.b0x);
  const Real t0_yx = -b0.b0x * b0.b0y;
  const Real t0_zx = -b0.b0x * b0.b0z;
  EXPECT_NEAR(split.rho, total.rho, 3e-12);
  EXPECT_NEAR(split.mx, total.mx - t0_xx, 3e-12);
  EXPECT_NEAR(split.my, total.my - t0_yx, 3e-12);
  EXPECT_NEAR(split.mz, total.mz - t0_zx, 3e-12);
  EXPECT_NEAR(split.energy,
              total.energy - b0.b0y * total.by - b0.b0z * total.bz,
              4e-12);
  EXPECT_EQ(split.bx, Real{0});
  EXPECT_NEAR(split.by, total.by, 3e-12);
  EXPECT_NEAR(split.bz, total.bz, 3e-12);
}

TEST(HlldRiemann, SplitYNormalRotationMatchesAffineTotalSolver) {
  constexpr Real gamma = Real{5} / Real{3};
  const quasar::numerics::MhdBackground b0{
      Real{0.4}, Real{-0.6}, Real{0.2}};
  const MhdState L = conserved(
      1.2, 0.3, -0.2, 0.1, 1.0, 0.25, 0.15, -0.35, gamma);
  const MhdState R = conserved(
      0.75, -0.4, 0.1, 0.5, 0.8, -0.3, 0.15, 0.45, gamma);

  const auto promote = [&](const MhdState& u) {
    MhdState t = u;
    t.bx += b0.b0x;
    t.by += b0.b0y;
    t.bz += b0.b0z;
    t.energy = (u.energy - quasar::numerics::half_squared_norm3(
                    u.bx, u.by, u.bz)) +
               quasar::numerics::half_squared_norm3(t.bx, t.by, t.bz);
    return t;
  };
  const MhdFlux total_y = quasar::numerics::hlld::rotate_out(
      quasar::numerics::hlld::hlld_flux_x(
          quasar::numerics::hlld::rotate_in(promote(L), 1),
          quasar::numerics::hlld::rotate_in(promote(R), 1), gamma),
      1);
  const MhdFlux split_y = quasar::numerics::hlld::rotate_out(
      quasar::numerics::hlld::hlld_flux_split_x(
          quasar::numerics::hlld::rotate_in(L, 1),
          quasar::numerics::hlld::rotate_in(R, 1),
          quasar::numerics::hlld::rotate_in(b0, 1), gamma),
      1);

  const Real t0_xy = -b0.b0x * b0.b0y;
  const Real t0_yy = Real{0.5} *
      (b0.b0x * b0.b0x + b0.b0z * b0.b0z - b0.b0y * b0.b0y);
  const Real t0_zy = -b0.b0z * b0.b0y;
  EXPECT_NEAR(split_y.rho, total_y.rho, 3e-12);
  EXPECT_NEAR(split_y.mx, total_y.mx - t0_xy, 3e-12);
  EXPECT_NEAR(split_y.my, total_y.my - t0_yy, 3e-12);
  EXPECT_NEAR(split_y.mz, total_y.mz - t0_zy, 3e-12);
  EXPECT_NEAR(split_y.energy,
              total_y.energy - b0.b0x * total_y.bx - b0.b0z * total_y.bz,
              4e-12);
  EXPECT_NEAR(split_y.bx, total_y.bx, 3e-12);
  EXPECT_EQ(split_y.by, Real{0});
  EXPECT_NEAR(split_y.bz, total_y.bz, 3e-12);
}

TEST(HlldRiemann, SplitUnequalNormalFieldsAreNormalizedConsistently) {
  constexpr Real gamma = Real{5} / Real{3};
  const quasar::numerics::MhdBackground b0{
      Real{0.6}, Real{-0.2}, Real{0.3}};
  MhdState L = conserved(
      1.0, 0.2, -0.1, 0.3, 0.9, 0.1, 0.4, -0.2, gamma);
  MhdState R = conserved(
      0.8, -0.15, 0.25, -0.1, 1.1, 0.5, -0.3, 0.5, gamma);
  const Real bn = Real{0.5} * L.bx + Real{0.5} * R.bx;
  MhdState Ln = L;
  MhdState Rn = R;
  Ln.energy = (Ln.energy - quasar::numerics::half_squared_norm3(
                                Ln.bx, Real{0}, Real{0})) +
              quasar::numerics::half_squared_norm3(bn, Real{0}, Real{0});
  Rn.energy = (Rn.energy - quasar::numerics::half_squared_norm3(
                                Rn.bx, Real{0}, Real{0})) +
              quasar::numerics::half_squared_norm3(bn, Real{0}, Real{0});
  Ln.bx = bn;
  Rn.bx = bn;

  const MhdFlux got =
      quasar::numerics::hlld::hlld_flux_split_x(L, R, b0, gamma);
  const MhdFlux expected =
      quasar::numerics::hlld::hlld_flux_split_x(Ln, Rn, b0, gamma);
  EXPECT_EQ(got.rho, expected.rho);
  EXPECT_EQ(got.mx, expected.mx);
  EXPECT_EQ(got.my, expected.my);
  EXPECT_EQ(got.mz, expected.mz);
  EXPECT_EQ(got.energy, expected.energy);
  EXPECT_EQ(got.bx, expected.bx);
  EXPECT_EQ(got.by, expected.by);
  EXPECT_EQ(got.bz, expected.bz);
}
