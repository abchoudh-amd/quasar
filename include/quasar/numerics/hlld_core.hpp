#pragma once

// Shared, host/device-callable HLLD core for ideal MHD (Miyoshi & Kusano 2005).
//
// This is the SINGLE source of truth for the seven-wave HLLD algebra. Both the
// host registry solver (src/numerics/hlld_riemann.cpp) and the device hot path
// (src/backend/hip/mhd/mhd_riemann.hip) call hlld_flux_x() so the two can never
// drift in their degeneracy guards, fallbacks, or intermediate-state formulas.
// Everything is `QUASAR_HOST_DEVICE inline` over the shared MhdState/MhdFlux PODs
// and the header-inline EOS in mhd_state.hpp, so it compiles in a .cpp and a
// .hip identically.
//
// Frame convention: hlld_flux_x() assumes the interface normal is +x. Callers
// handle dir==1 (y-interfaces) by rotating the in-plane (mx,my)/(bx,by) pairs in
// before the call and rotating the flux back out after; rotate_in/rotate_out
// here provide that rotation so both callers share it too.
//
// Robustness: when the HLLD intermediate construction is ill-conditioned (B_n^2
// negligible vs the total pressure, or a rotational/contact denominator
// underflows, or a star density is non-positive, or any output goes non-finite)
// the routine falls back to the single-intermediate HLL flux, exactly as
// Miyoshi & Kusano recommend for the degenerate B_n -> 0 limit. HLL is itself
// consistent, so the L==R -> F(L) consistency property holds on both paths.

#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <cmath>

namespace quasar::numerics::hlld {

// Rotate a conserved state into the +x-normal frame: dir==0 identity, dir==1
// swaps the (x,y) momentum and (x,y) magnetic pairs.
QUASAR_HOST_DEVICE inline MhdState rotate_in(const MhdState& s, int dir) {
  if (dir == 0) return s;
  MhdState r = s;
  r.mx = s.my; r.my = s.mx;
  r.bx = s.by; r.by = s.bx;
  return r;
}

// Inverse rotation of a flux back to the lab frame for dir==1.
QUASAR_HOST_DEVICE inline MhdFlux rotate_out(const MhdFlux& f, int dir) {
  if (dir == 0) return f;
  MhdFlux r = f;
  r.mx = f.my; r.my = f.mx;
  r.bx = f.by; r.by = f.bx;
  return r;
}

// Physical ideal-MHD flux along +x with total pressure p* = p + B^2/2.
QUASAR_HOST_DEVICE inline MhdFlux physical_flux_x(const MhdState& u, Real gamma) {
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
  f.bx     = Real{0};  // dBx/dt has no x-flux (normal component)
  f.by     = u.by * vx - u.bx * vy;
  f.bz     = u.bz * vx - u.bx * vz;
  return f;
}

// Single-intermediate HLL flux along +x -- robust fallback. Returns F(L) when
// the fan is fully right-going and F(R) when fully left-going.
QUASAR_HOST_DEVICE inline MhdFlux hll_flux_x(const MhdState& L, const MhdState& R,
                                             Real sl, Real sr, Real gamma) {
  const MhdFlux fl = physical_flux_x(L, gamma);
  const MhdFlux fr = physical_flux_x(R, gamma);
  if (sl >= Real{0}) return fl;
  if (sr <= Real{0}) return fr;
  const Real inv = Real{1} / (sr - sl);
  MhdFlux f;
  f.rho    = (sr * fl.rho    - sl * fr.rho    + sl * sr * (R.rho    - L.rho))    * inv;
  f.mx     = (sr * fl.mx     - sl * fr.mx     + sl * sr * (R.mx     - L.mx))     * inv;
  f.my     = (sr * fl.my     - sl * fr.my     + sl * sr * (R.my     - L.my))     * inv;
  f.mz     = (sr * fl.mz     - sl * fr.mz     + sl * sr * (R.mz     - L.mz))     * inv;
  f.energy = (sr * fl.energy - sl * fr.energy + sl * sr * (R.energy - L.energy)) * inv;
  f.bx     = Real{0};
  f.by     = (sr * fl.by     - sl * fr.by     + sl * sr * (R.by     - L.by))     * inv;
  f.bz     = (sr * fl.bz     - sl * fr.bz     + sl * sr * (R.bz     - L.bz))     * inv;
  return f;
}

// Canonical +x-normal HLLD flux. Falls back to HLL on any degeneracy.
QUASAR_HOST_DEVICE inline MhdFlux hlld_flux_x(const MhdState& L, const MhdState& R,
                                              Real gamma) {
  using std::fabs;
  using std::sqrt;
  using std::isfinite;

  // Continuous normal field B_n (CT keeps it so; average if the input disagrees).
  const Real bn = Real{0.5} * (L.bx + R.bx);
  const Real bn2 = bn * bn;

  const Real inv_rhoL = Real{1} / L.rho;
  const Real inv_rhoR = Real{1} / R.rho;
  const Real vxL = L.mx * inv_rhoL, vyL = L.my * inv_rhoL, vzL = L.mz * inv_rhoL;
  const Real vxR = R.mx * inv_rhoR, vyR = R.my * inv_rhoR, vzR = R.mz * inv_rhoR;
  const Real pL = pressure(L, gamma);
  const Real pR = pressure(R, gamma);
  const Real ptL = pL + Real{0.5} * (L.bx * L.bx + L.by * L.by + L.bz * L.bz);
  const Real ptR = pR + Real{0.5} * (R.bx * R.bx + R.by * R.by + R.bz * R.bz);

  // Fast-magnetosonic outer wave speeds (Davis bracket).
  const Real cfL = fast_magnetosonic_speed(L, 0, gamma);
  const Real cfR = fast_magnetosonic_speed(R, 0, gamma);
  const Real cfmax = (cfL > cfR) ? cfL : cfR;
  const Real sl = ((vxL < vxR) ? vxL : vxR) - cfmax;
  const Real sr = ((vxL > vxR) ? vxL : vxR) + cfmax;

  // One-sided fans: upwind physical flux (also gives L==R -> F(L)).
  if (sl >= Real{0}) return physical_flux_x(L, gamma);
  if (sr <= Real{0}) return physical_flux_x(R, gamma);

  // Contact / entropy speed S_M. Guard the HLL-mass denominator relatively.
  const Real denomM = (sr - vxR) * R.rho - (sl - vxL) * L.rho;
  if (!(fabs(denomM) > kEps)) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }
  const Real sm = ((sr - vxR) * R.rho * vxR - (sl - vxL) * L.rho * vxL - ptR + ptL) / denomM;

  // Total pressure is constant across the contact (eq. 41).
  const Real pt_star =
      ((sr - vxR) * R.rho * ptL - (sl - vxL) * L.rho * ptR
       + L.rho * R.rho * (sr - vxR) * (sl - vxL) * (vxR - vxL)) / denomM;

  // Outer star densities; non-positive => degenerate, fall back to HLL.
  const Real rhoLs = L.rho * (sl - vxL) / (sl - sm);
  const Real rhoRs = R.rho * (sr - vxR) / (sr - sm);
  if (!(rhoLs > Real{0}) || !(rhoRs > Real{0})) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }

  // Single-star state U*_{L,R} (eqs. 43-48). Writes the transverse v/B too.
  struct Star { MhdState u; Real vy; Real vz; Real by; Real bz; bool ok; };
  auto star_state = [&](const MhdState& U, Real s, Real vx, Real vy, Real vz,
                        Real pt, Real rhos) -> Star {
    Star out;
    out.ok = true;
    const Real sMinusV = s - vx;
    const Real denom = U.rho * sMinusV * (s - sm) - bn2;
    MhdState us;
    us.rho = rhos;
    us.bx  = bn;
    Real vys, vzs, bys, bzs;
    if (!(fabs(denom) > kEps)) {
      // Degenerate: transverse field/velocity revert to the advected base state.
      bys = U.by;
      bzs = U.bz;
      vys = vy;
      vzs = vz;
    } else {
      const Real inv_denom = Real{1} / denom;
      const Real fac = bn * (sm - vx) * inv_denom;
      vys = vy - U.by * fac;
      vzs = vz - U.bz * fac;
      const Real bcoef = (U.rho * sMinusV * sMinusV - bn2) * inv_denom;
      bys = U.by * bcoef;
      bzs = U.bz * bcoef;
    }
    us.mx = us.rho * sm;
    us.my = us.rho * vys;
    us.mz = us.rho * vzs;
    us.by = bys;
    us.bz = bzs;
    const Real vdotb   = vx * U.bx + vy * U.by + vz * U.bz;
    const Real vdotb_s = sm * bn + vys * bys + vzs * bzs;
    us.energy = (sMinusV * U.energy - pt * vx + pt_star * sm
                 + bn * (vdotb - vdotb_s)) / (s - sm);
    out.u = us; out.vy = vys; out.vz = vzs; out.by = bys; out.bz = bzs;
    return out;
  };

  const Star sL = star_state(L, sl, vxL, vyL, vzL, ptL, rhoLs);
  const Star sR = star_state(R, sr, vxR, vyR, vzR, ptR, rhoRs);
  const MhdState usL = sL.u;
  const MhdState usR = sR.u;

  const MhdFlux fl = physical_flux_x(L, gamma);
  const MhdFlux fr = physical_flux_x(R, gamma);

  auto flux_from_star = [&](const MhdFlux& fU, const MhdState& U,
                            const MhdState& us, Real s) {
    MhdFlux f;
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

  const Real sqrt_rhoLs = sqrt(rhoLs);
  const Real sqrt_rhoRs = sqrt(rhoRs);
  const Real saL = sm - fabs(bn) / sqrt_rhoLs;
  const Real saR = sm + fabs(bn) / sqrt_rhoRs;

  // If B_n ~ 0 the Alfven waves collapse onto S_M: no double-star region.
  const bool bn_negligible =
      (bn2) <= kEps * (fabs(pt_star) + fabs(ptL) + fabs(ptR) + kEps);

  MhdFlux f;
  if (bn_negligible) {
    if (sl <= Real{0} && Real{0} <= sm) {
      f = flux_from_star(fl, L, usL, sl);
    } else {
      f = flux_from_star(fr, R, usR, sr);
    }
    if (!isfinite(f.rho) || !isfinite(f.energy)) {
      return hll_flux_x(L, R, sl, sr, gamma);
    }
    return f;
  }

  // Double-star states U**_{L,R} (eqs. 59-63).
  const Real inv_sqrt_sum = Real{1} / (sqrt_rhoLs + sqrt_rhoRs);
  const Real sgn_bn = (bn >= Real{0}) ? Real{1} : Real{-1};

  const Real vyss = (sqrt_rhoLs * sL.vy + sqrt_rhoRs * sR.vy
                     + (sR.by - sL.by) * sgn_bn) * inv_sqrt_sum;
  const Real vzss = (sqrt_rhoLs * sL.vz + sqrt_rhoRs * sR.vz
                     + (sR.bz - sL.bz) * sgn_bn) * inv_sqrt_sum;
  const Real byss = (sqrt_rhoLs * sR.by + sqrt_rhoRs * sL.by
                     + sqrt_rhoLs * sqrt_rhoRs * (sR.vy - sL.vy) * sgn_bn) * inv_sqrt_sum;
  const Real bzss = (sqrt_rhoLs * sR.bz + sqrt_rhoRs * sL.bz
                     + sqrt_rhoLs * sqrt_rhoRs * (sR.vz - sL.vz) * sgn_bn) * inv_sqrt_sum;

  auto double_star = [&](const Star& s, Real sqrt_rhos) -> MhdState {
    MhdState uss;
    uss.rho = s.u.rho;
    uss.bx  = bn;
    uss.mx  = uss.rho * sm;
    uss.my  = uss.rho * vyss;
    uss.mz  = uss.rho * vzss;
    uss.by  = byss;
    uss.bz  = bzss;
    const Real vdotb_ss = sm * bn + vyss * byss + vzss * bzss;
    const Real vdotb_s  = sm * bn + s.vy * s.by + s.vz * s.bz;
    uss.energy = s.u.energy - sqrt_rhos * sgn_bn * (vdotb_s - vdotb_ss);
    return uss;
  };

  const MhdState ussL = double_star(sL, sqrt_rhoLs);
  const MhdState ussR = double_star(sR, sqrt_rhoRs);

  // Region selection (eq. 66).
  if (Real{0} <= saL) {
    f = flux_from_star(fl, L, usL, sl);
  } else if (Real{0} <= sm) {
    f.rho    = fl.rho    + saL * (ussL.rho - usL.rho)       + sl * (usL.rho - L.rho);
    f.mx     = fl.mx     + saL * (ussL.mx - usL.mx)         + sl * (usL.mx - L.mx);
    f.my     = fl.my     + saL * (ussL.my - usL.my)         + sl * (usL.my - L.my);
    f.mz     = fl.mz     + saL * (ussL.mz - usL.mz)         + sl * (usL.mz - L.mz);
    f.energy = fl.energy + saL * (ussL.energy - usL.energy) + sl * (usL.energy - L.energy);
    f.bx     = Real{0};
    f.by     = fl.by     + saL * (ussL.by - usL.by)         + sl * (usL.by - L.by);
    f.bz     = fl.bz     + saL * (ussL.bz - usL.bz)         + sl * (usL.bz - L.bz);
  } else if (Real{0} <= saR) {
    f.rho    = fr.rho    + saR * (ussR.rho - usR.rho)       + sr * (usR.rho - R.rho);
    f.mx     = fr.mx     + saR * (ussR.mx - usR.mx)         + sr * (usR.mx - R.mx);
    f.my     = fr.my     + saR * (ussR.my - usR.my)         + sr * (usR.my - R.my);
    f.mz     = fr.mz     + saR * (ussR.mz - usR.mz)         + sr * (usR.mz - R.mz);
    f.energy = fr.energy + saR * (ussR.energy - usR.energy) + sr * (usR.energy - R.energy);
    f.bx     = Real{0};
    f.by     = fr.by     + saR * (ussR.by - usR.by)         + sr * (usR.by - R.by);
    f.bz     = fr.bz     + saR * (ussR.bz - usR.bz)         + sr * (usR.bz - R.bz);
  } else {
    f = flux_from_star(fr, R, usR, sr);
  }

  // Final NaN/inf guard: pathological inputs fall back to robust HLL.
  if (!isfinite(f.rho) || !isfinite(f.mx) || !isfinite(f.my) || !isfinite(f.mz)
      || !isfinite(f.energy) || !isfinite(f.by) || !isfinite(f.bz)) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }
  return f;
}

}  // namespace quasar::numerics::hlld
