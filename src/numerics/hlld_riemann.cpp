// HLLD approximate Riemann solver for ideal MHD.
//
// Reference: T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann
// solver for ideal magnetohydrodynamics", J. Comput. Phys. 208 (2005) 315-344.
// We resolve all seven waves of the ideal-MHD system: the fast magnetosonic
// fronts S_L / S_R (eqs. that bound the Riemann fan), the rotational/Alfven
// discontinuities S_L* / S_R* (eq. 51), and the entropy/contact wave S_M
// (eq. 38), giving the four intermediate states U*_{L,R} and U**_{L,R}.
//
// Conventions of this file:
//   - States are conserved 8-tuples (rho, mx, my, mz, energy, bx, by, bz).
//   - The interface normal is `dir` (0 = x, 1 = y). We handle dir == 1 by an
//     in-plane rotation: swap the (x <-> y) momentum and (Bx <-> By) magnetic
//     components on input, run the canonical x-normal HLLD core, then swap the
//     same pair back on the output flux. mz / bz (out-of-plane) are untouched,
//     so the single core routine serves both directions.
//   - The normal field B_n (Bx in the rotated frame) is continuous across the
//     interface for a well-posed constrained-transport input. If the supplied
//     L.bx and R.bx disagree we take their arithmetic mean as the single B_n,
//     which is the standard HLLD choice and keeps the normal-B flux component
//     (which is identically zero in the induction equation for the normal
//     component) consistent. The returned out.bx is set to 0 in the rotated
//     frame because dBx/dt has no x-flux; rotation maps that to the correct
//     normal-component-zero flux for either dir.
//
// Robustness / HLL fallback: when the HLLD intermediate construction is
// ill-conditioned -- B_n^2 negligible relative to the total pressure, or the
// rotational denominators (rho* fast-speed factors) underflow -- we fall back
// to the single-intermediate-state HLL flux (Harten-Lax-van Leer), exactly as
// Miyoshi & Kusano recommend for the degenerate Bn -> 0 limit. The HLL flux is
// itself consistent (it reduces to F(L) when L == R and the fan is one-sided),
// so the L == R consistency property is preserved on both code paths.

#include "quasar/numerics/riemann_solver.hpp"

#include "quasar/core/registry.hpp"

#include <algorithm>
#include <cmath>

namespace quasar::numerics {

namespace {

// ---- small helpers -------------------------------------------------------

// Physical ideal-MHD flux F(U) along the +x normal, in conserved-variable
// order. Derived from the primitive form with total pressure p* = p + B^2/2.
inline MhdFlux physical_flux_x(const MhdState& u, Real gamma) {
  const Real inv_rho = Real{1} / u.rho;
  const Real vx = u.mx * inv_rho;
  const Real vy = u.my * inv_rho;
  const Real vz = u.mz * inv_rho;
  const Real p  = pressure(u, gamma);
  const Real b2 = u.bx * u.bx + u.by * u.by + u.bz * u.bz;
  const Real ptot = p + Real{0.5} * b2;
  const Real vdotb = vx * u.bx + vy * u.by + vz * u.bz;

  MhdFlux f;
  f.rho    = u.rho * vx;
  f.mx     = u.rho * vx * vx + ptot - u.bx * u.bx;
  f.my     = u.rho * vx * vy - u.bx * u.by;
  f.mz     = u.rho * vx * vz - u.bx * u.bz;
  f.energy = (u.energy + ptot) * vx - u.bx * vdotb;
  f.bx     = Real{0};                       // dBx/dt has no x-flux (normal comp.)
  f.by     = u.by * vx - u.bx * vy;
  f.bz     = u.bz * vx - u.bx * vz;
  return f;
}

// Rotate in-plane components for dir == 1: swap (mx,my) and (bx,by).
inline MhdState rotate_in(const MhdState& u, int dir) {
  if (dir == 0) return u;
  MhdState r = u;
  std::swap(r.mx, r.my);
  std::swap(r.bx, r.by);
  return r;
}

// Inverse rotation applied to a flux for dir == 1: swap (mx,my) and (bx,by).
inline MhdFlux rotate_out(const MhdFlux& f, int dir) {
  if (dir == 0) return f;
  MhdFlux r = f;
  std::swap(r.mx, r.my);
  std::swap(r.bx, r.by);
  return r;
}

inline bool finite_safe(Real x) {
  return std::isfinite(x);
}

// Single-intermediate-state HLL flux along +x. Used both as the degenerate
// fallback for HLLD and to share the S_L/S_R bookkeeping. Returns F(L) when the
// whole fan is to the right (S_L >= 0) and F(R) when to the left (S_R <= 0), so
// L == R consistency holds.
inline MhdFlux hll_flux_x(const MhdState& L, const MhdState& R, Real gamma,
                          Real sl, Real sr) {
  if (sl >= Real{0}) return physical_flux_x(L, gamma);
  if (sr <= Real{0}) return physical_flux_x(R, gamma);

  const MhdFlux fl = physical_flux_x(L, gamma);
  const MhdFlux fr = physical_flux_x(R, gamma);
  const Real inv = Real{1} / (sr - sl);

  MhdFlux f;
  auto blend = [&](Real fL, Real fR, Real uL, Real uR) {
    return (sr * fL - sl * fR + sl * sr * (uR - uL)) * inv;
  };
  f.rho    = blend(fl.rho, fr.rho, L.rho, R.rho);
  f.mx     = blend(fl.mx, fr.mx, L.mx, R.mx);
  f.my     = blend(fl.my, fr.my, L.my, R.my);
  f.mz     = blend(fl.mz, fr.mz, L.mz, R.mz);
  f.energy = blend(fl.energy, fr.energy, L.energy, R.energy);
  f.bx     = Real{0};
  f.by     = blend(fl.by, fr.by, L.by, R.by);
  f.bz     = blend(fl.bz, fr.bz, L.bz, R.bz);
  return f;
}

}  // namespace

// File-local concrete solver, registered by name "hlld". Carries the adiabatic
// index gamma (default 5/3); the solver driver may override via set_gamma()
// after the registry default-constructs the object.
class HlldRiemann : public IRiemannSolver {
 public:
  void set_gamma(Real gamma) { gamma_ = gamma; }
  Real gamma() const { return gamma_; }

  void flux(const MhdState& Lin, const MhdState& Rin, int dir, MhdFlux& out) const override {
    // Rotate into the canonical +x-normal frame, solve, rotate the flux back.
    const MhdState L = rotate_in(Lin, dir);
    const MhdState R = rotate_in(Rin, dir);
    const MhdFlux  f = solve_x(L, R);
    out = rotate_out(f, dir);
  }

  Real max_wavespeed(const MhdState& state, int dir, Real gamma) const override {
    const Real inv_rho = Real{1} / state.rho;
    const Real vn = (dir == 0) ? state.mx * inv_rho : state.my * inv_rho;
    return std::abs(vn) + fast_magnetosonic_speed(state, dir, gamma);
  }

 private:
  // Canonical x-normal HLLD core (Miyoshi & Kusano 2005).
  MhdFlux solve_x(const MhdState& L, const MhdState& R) const {
    const Real gamma = gamma_;

    // Continuous normal field B_n: average if the CT input disagrees.
    const Real bn = Real{0.5} * (L.bx + R.bx);

    // Primitive decode and total pressures.
    const Real inv_rhoL = Real{1} / L.rho;
    const Real inv_rhoR = Real{1} / R.rho;
    const Real vxL = L.mx * inv_rhoL, vyL = L.my * inv_rhoL, vzL = L.mz * inv_rhoL;
    const Real vxR = R.mx * inv_rhoR, vyR = R.my * inv_rhoR, vzR = R.mz * inv_rhoR;
    const Real pL = pressure(L, gamma);
    const Real pR = pressure(R, gamma);
    const Real ptL = pL + Real{0.5} * (L.bx * L.bx + L.by * L.by + L.bz * L.bz);
    const Real ptR = pR + Real{0.5} * (R.bx * R.bx + R.by * R.by + R.bz * R.bz);

    // Fast-magnetosonic outer wave speeds (eq. 12 / 67): bracket the fan with
    // the min/max of left and right fast speeds about the respective normal
    // velocities.
    const Real cfL = fast_magnetosonic_speed(L, 0, gamma);
    const Real cfR = fast_magnetosonic_speed(R, 0, gamma);
    const Real sl = std::min(vxL, vxR) - std::max(cfL, cfR);
    const Real sr = std::max(vxL, vxR) + std::max(cfL, cfR);

    // One-sided fans: return the upwind physical flux (also gives L==R -> F(L)).
    if (sl >= Real{0}) return physical_flux_x(L, gamma);
    if (sr <= Real{0}) return physical_flux_x(R, gamma);

    // Contact / entropy speed S_M (eq. 38). Denominator is the HLL mass
    // weighting; guard against underflow.
    const Real denomM = (sr - vxR) * R.rho - (sl - vxL) * L.rho;
    if (!(std::abs(denomM) > kEps)) {
      return hll_flux_x(L, R, gamma, sl, sr);
    }
    const Real sm = ((sr - vxR) * R.rho * vxR - (sl - vxL) * L.rho * vxL - ptR + ptL) / denomM;

    // Total pressure in the star region is constant across the contact (eq. 41).
    const Real pt_star =
        ((sr - vxR) * R.rho * ptL - (sl - vxL) * L.rho * ptR
         + L.rho * R.rho * (sr - vxR) * (sl - vxL) * (vxR - vxL)) / denomM;

    // ---- single-star states U*_{L,R} (eqs. 43-48) ------------------------
    auto star_state = [&](const MhdState& U, Real s, Real vx, Real vy, Real vz,
                          Real pt) -> MhdState {
      const Real rho = U.rho;
      const Real denom = rho * (s - vx) * (s - sm) - bn * bn;
      MhdState us;
      us.rho = rho * (s - vx) / (s - sm);
      us.bx  = bn;
      if (!(std::abs(denom) > kEps)) {
        // Degenerate: transverse components revert to the advected base state.
        us.by = U.by;
        us.bz = U.bz;
        const Real vys = vy;
        const Real vzs = vz;
        us.mx = us.rho * sm;
        us.my = us.rho * vys;
        us.mz = us.rho * vzs;
        const Real vdotb_s = sm * us.bx + vys * us.by + vzs * us.bz;
        const Real vdotb   = vx * U.bx + vy * U.by + vz * U.bz;
        us.energy = ((s - vx) * U.energy - pt * vx + pt_star * sm
                     + bn * (vdotb - vdotb_s)) / (s - sm);
        return us;
      }
      const Real inv_denom = Real{1} / denom;
      const Real fac = bn * (sm - vx) * inv_denom;
      const Real vys = vy - U.by * fac;
      const Real vzs = vz - U.bz * fac;
      const Real bcoef = (rho * (s - vx) * (s - vx) - bn * bn) * inv_denom;
      const Real bys = U.by * bcoef;
      const Real bzs = U.bz * bcoef;

      us.mx = us.rho * sm;
      us.my = us.rho * vys;
      us.mz = us.rho * vzs;
      us.by = bys;
      us.bz = bzs;
      const Real vdotb_s = sm * us.bx + vys * bys + vzs * bzs;
      const Real vdotb   = vx * U.bx + vy * U.by + vz * U.bz;
      us.energy = ((s - vx) * U.energy - pt * vx + pt_star * sm
                   + bn * (vdotb - vdotb_s)) / (s - sm);
      return us;
    };

    const MhdState usL = star_state(L, sl, vxL, vyL, vzL, ptL);
    const MhdState usR = star_state(R, sr, vxR, vyR, vzR, ptR);

    // Alfven / rotational speeds bounding the double-star region (eq. 51).
    const Real sqrt_rhoLs = std::sqrt(usL.rho);
    const Real sqrt_rhoRs = std::sqrt(usR.rho);
    const Real saL = sm - std::abs(bn) / sqrt_rhoLs;
    const Real saR = sm + std::abs(bn) / sqrt_rhoRs;

    const MhdFlux fl = physical_flux_x(L, gamma);
    const MhdFlux fr = physical_flux_x(R, gamma);

    auto flux_from_star = [&](const MhdState& U, const MhdState& us, Real s) {
      MhdFlux f;
      const MhdFlux fU = physical_flux_x(U, gamma);
      f.rho    = fU.rho    + s * (us.rho - U.rho);
      f.mx     = fU.mx     + s * (us.mx - U.mx);
      f.my     = fU.my     + s * (us.my - U.my);
      f.mz     = fU.mz     + s * (us.mz - U.mz);
      f.energy = fU.energy + s * (us.energy - U.energy);
      f.bx     = Real{0};
      f.by     = fU.by     + s * (us.by - U.by);
      f.bz     = fU.bz     + s * (us.bz - U.bz);
      return f;
    };

    // If Bn ~ 0 the two Alfven waves collapse onto S_M and there is no
    // double-star region (Miyoshi & Kusano sec. 3.3): use the single-star flux,
    // which is well-conditioned there.
    const bool bn_negligible =
        (bn * bn) <= kEps * (std::abs(pt_star) + std::abs(ptL) + std::abs(ptR) + kEps);

    if (bn_negligible) {
      MhdFlux f;
      if (sl <= Real{0} && Real{0} <= sm) {
        f = flux_from_star(L, usL, sl);
      } else {
        f = flux_from_star(R, usR, sr);
      }
      if (!finite_safe(f.energy) || !finite_safe(f.rho)) {
        return hll_flux_x(L, R, gamma, sl, sr);
      }
      return f;
    }

    // ---- double-star states U**_{L,R} (eqs. 59-63) -----------------------
    const Real inv_sqrt_sum = Real{1} / (sqrt_rhoLs + sqrt_rhoRs);
    const Real sgn_bn = (bn >= Real{0}) ? Real{1} : Real{-1};

    const Real vyLs = usL.my / usL.rho;
    const Real vzLs = usL.mz / usL.rho;
    const Real vyRs = usR.my / usR.rho;
    const Real vzRs = usR.mz / usR.rho;

    const Real vyss = (sqrt_rhoLs * vyLs + sqrt_rhoRs * vyRs
                       + (usR.by - usL.by) * sgn_bn) * inv_sqrt_sum;
    const Real vzss = (sqrt_rhoLs * vzLs + sqrt_rhoRs * vzRs
                       + (usR.bz - usL.bz) * sgn_bn) * inv_sqrt_sum;
    const Real byss = (sqrt_rhoLs * usR.by + sqrt_rhoRs * usL.by
                       + sqrt_rhoLs * sqrt_rhoRs * (vyRs - vyLs) * sgn_bn) * inv_sqrt_sum;
    const Real bzss = (sqrt_rhoLs * usR.bz + sqrt_rhoRs * usL.bz
                       + sqrt_rhoLs * sqrt_rhoRs * (vzRs - vzLs) * sgn_bn) * inv_sqrt_sum;

    auto double_star = [&](const MhdState& us, Real sqrt_rhos) -> MhdState {
      MhdState uss;
      uss.rho = us.rho;
      uss.bx  = bn;
      uss.mx  = uss.rho * sm;
      uss.my  = uss.rho * vyss;
      uss.mz  = uss.rho * vzss;
      uss.by  = byss;
      uss.bz  = bzss;
      const Real vdotb_ss = sm * bn + vyss * byss + vzss * bzss;
      const Real vdotb_s  = (us.mx * us.bx + us.my * us.by + us.mz * us.bz) / us.rho;
      uss.energy = us.energy - sqrt_rhos * sgn_bn * (vdotb_s - vdotb_ss);
      return uss;
    };

    const MhdState ussL = double_star(usL, sqrt_rhoLs);
    const MhdState ussR = double_star(usR, sqrt_rhoRs);

    // ---- region selection (eq. 66) --------------------------------------
    MhdFlux f;
    if (Real{0} <= sl) {
      f = fl;
    } else if (Real{0} <= saL) {
      // U*_L : F_L + S_L (U*_L - U_L)
      f = flux_from_star(L, usL, sl);
    } else if (Real{0} <= sm) {
      // U**_L : F_L + S_L*(U**_L - U*_L) + S_L(U*_L - U_L)
      f.rho    = fl.rho    + saL * (ussL.rho - usL.rho)    + sl * (usL.rho - L.rho);
      f.mx     = fl.mx     + saL * (ussL.mx - usL.mx)      + sl * (usL.mx - L.mx);
      f.my     = fl.my     + saL * (ussL.my - usL.my)      + sl * (usL.my - L.my);
      f.mz     = fl.mz     + saL * (ussL.mz - usL.mz)      + sl * (usL.mz - L.mz);
      f.energy = fl.energy + saL * (ussL.energy - usL.energy) + sl * (usL.energy - L.energy);
      f.bx     = Real{0};
      f.by     = fl.by     + saL * (ussL.by - usL.by)      + sl * (usL.by - L.by);
      f.bz     = fl.bz     + saL * (ussL.bz - usL.bz)      + sl * (usL.bz - L.bz);
    } else if (Real{0} <= saR) {
      // U**_R : F_R + S_R*(U**_R - U*_R) + S_R(U*_R - U_R)
      f.rho    = fr.rho    + saR * (ussR.rho - usR.rho)    + sr * (usR.rho - R.rho);
      f.mx     = fr.mx     + saR * (ussR.mx - usR.mx)      + sr * (usR.mx - R.mx);
      f.my     = fr.my     + saR * (ussR.my - usR.my)      + sr * (usR.my - R.my);
      f.mz     = fr.mz     + saR * (ussR.mz - usR.mz)      + sr * (usR.mz - R.mz);
      f.energy = fr.energy + saR * (ussR.energy - usR.energy) + sr * (usR.energy - R.energy);
      f.bx     = Real{0};
      f.by     = fr.by     + saR * (ussR.by - usR.by)      + sr * (usR.by - R.by);
      f.bz     = fr.bz     + saR * (ussR.bz - usR.bz)      + sr * (usR.bz - R.bz);
    } else if (Real{0} <= sr) {
      // U*_R : F_R + S_R (U*_R - U_R)
      f = flux_from_star(R, usR, sr);
    } else {
      f = fr;
    }

    // Final NaN/inf guard: if anything went non-finite (pathological inputs),
    // fall back to the robust HLL flux.
    if (!finite_safe(f.rho) || !finite_safe(f.mx) || !finite_safe(f.my)
        || !finite_safe(f.mz) || !finite_safe(f.energy) || !finite_safe(f.by)
        || !finite_safe(f.bz)) {
      return hll_flux_x(L, R, gamma, sl, sr);
    }
    return f;
  }

  Real gamma_{Real{5} / Real{3}};
};

}  // namespace quasar::numerics

QUASAR_REGISTER_RIEMANN_SOLVER("hlld", ::quasar::numerics::HlldRiemann)
