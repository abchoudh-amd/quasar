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
#include <limits>

namespace quasar::numerics::hlld {

// HLLD is intentionally a device call boundary.  Inlining the complete split
// solver into every Riemann/CT call site duplicates its cancellation-safe local
// workspaces and can make LLVM promote thread-local arrays to the full LDS
// budget.  Host builds retain the usual header-inline definition; HIP device
// builds emit one callable implementation per translation unit.
#if defined(__HIP_DEVICE_COMPILE__)
#define QUASAR_HLLD_DEVICE_NOINLINE __attribute__((noinline))
#else
#define QUASAR_HLLD_DEVICE_NOINLINE
#endif

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

// Rotate a prescribed background field into the same normal frame as a state.
QUASAR_HOST_DEVICE inline MhdBackground rotate_in(const MhdBackground& b, int dir) {
  if (dir == 0) return b;
  return MhdBackground{b.b0y, b.b0x, b.b0z};
}

// Cancellation-preserving momentum-flux representation for the field split.
// `material` contains the complete non-momentum fluxes and only the material
// momentum stress
//
//   H_nj = rho*v_n*v_j + delta_nj*(p + |b|^2/2) - b_n*b_j.
//
// `cross_b` keeps the effective perturbation field that enters the linear
// background stress
//
//   C_nj = delta_nj*(B0.b) - B0_n*b_j - b_n*B0_j
//
// in factored form, and `wave` contains only the HLL/HLLD momentum correction.
// A face-to-cell update must reduce H, the factors of C, and W together; adding
// the already-rounded aggregate H+C+W from adjacent faces loses gas-pressure
// gradients beside a dominant guide-field stress.
struct MhdMomentumFluxParts {
  MhdFlux material{};
  MhdFlux wave{};
  MhdBackground cross_b{};
};

QUASAR_HOST_DEVICE inline MhdMomentumFluxParts rotate_out(
    const MhdMomentumFluxParts& f, int dir) {
  if (dir == 0) return f;
  MhdMomentumFluxParts r = f;
  r.material = rotate_out(f.material, dir);
  r.wave = rotate_out(f.wave, dir);
  r.cross_b = rotate_in(f.cross_b, dir);  // an x/y swap is its own inverse
  return r;
}

// Collapse the split channels for legacy/standalone callers.  Production
// finite-volume evolution does not call this for momentum: it preserves the
// channels through the fused divergence kernel below the Riemann layer.
QUASAR_HOST_DEVICE inline MhdFlux compose_momentum_flux_parts(
    const MhdMomentumFluxParts& parts, const MhdBackground& b0) {
  MhdFlux f = parts.material;
  {
    const Real a[5] = {parts.material.mx, parts.wave.mx, b0.b0x,
                       b0.b0y, b0.b0z};
    const Real b[5] = {Real{1}, Real{1}, -parts.cross_b.b0x,
                       parts.cross_b.b0y, parts.cross_b.b0z};
    f.mx = scaled_product_sum(a, b, 5);
  }
  f.my = product_sum4(parts.material.my, Real{1}, parts.wave.my, Real{1},
                      b0.b0x, -parts.cross_b.b0y,
                      b0.b0y, -parts.cross_b.b0x);
  f.mz = product_sum4(parts.material.mz, Real{1}, parts.wave.mz, Real{1},
                      b0.b0x, -parts.cross_b.b0z,
                      b0.b0z, -parts.cross_b.b0x);
  return f;
}

// Physical ideal-MHD flux along +x with total pressure p* = p + B^2/2.
QUASAR_HOST_DEVICE inline MhdFlux physical_flux_x(
    const MhdState& u, Real gamma) {
  const Real vx = u.mx / u.rho;
  const Real vy = u.my / u.rho;
  const Real vz = u.mz / u.rho;
  const Real p  = pressure(u, gamma);
  const Real half_bn2 = half_squared_norm3(u.bx, Real{0}, Real{0});
  const Real half_bt2 = half_squared_norm3(Real{0}, u.by, u.bz);
  MhdFlux f;
  f.rho    = u.mx;
  {
    const Real a[5] = {u.mx, p, u.by, u.bz, u.bx};
    const Real b[5] = {vx, Real{1}, Real{0.5} * u.by,
                       Real{0.5} * u.bz, Real{-0.5} * u.bx};
    f.mx = scaled_product_sum(a, b, 5);
  }
  f.my = product_sum2(u.mx, vy, u.bx, -u.by);
  f.mz = product_sum2(u.mx, vz, u.bx, -u.bz);
  // Expand the Poynting cancellation analytically:
  //   (E+p+B^2/2)vx - Bx(v.B)
  // = vx(E+p+Bt^2/2-Bx^2/2) - Bx(vt.Bt).
  // E-Bx^2/2 is formed first because both are representable conserved-energy
  // terms even when Bx*Bx is not.
  const Real normal_enthalpy = ((u.energy - half_bn2) + p) + half_bt2;
  const Real tangential_vdotb = product_sum2(vy, u.by, vz, u.bz);
  f.energy = product_sum2(vx, normal_enthalpy, u.bx, -tangential_vdotb);
  f.bx     = Real{0};  // dBx/dt has no x-flux (normal component)
  f.by     = product_sum2(u.by, vx, u.bx, -vy);
  f.bz     = product_sum2(u.bz, vx, u.bx, -vz);
  return f;
}

// Single-intermediate HLL flux along +x -- robust fallback. Returns F(L) when
// the fan is fully right-going and F(R) when fully left-going.
QUASAR_HOST_DEVICE inline MhdFlux hll_flux_x(
    const MhdState& L, const MhdState& R, Real sl, Real sr, Real gamma) {
  const MhdFlux fl = physical_flux_x(L, gamma);
  const MhdFlux fr = physical_flux_x(R, gamma);
  if (sl >= Real{0}) return fl;
  if (sr <= Real{0}) return fr;
  const Real speed_scale = std::fmax(std::fabs(sl), std::fabs(sr));
  if (!(speed_scale > Real{0}) || !std::isfinite(speed_scale)) return fl;
  const Real sln = sl / speed_scale;
  const Real srn = sr / speed_scale;
  const Real denom = srn - sln;
  if (!(denom > Real{0}) || !std::isfinite(denom)) return fl;
  // Canonical HLL weights:
  //   F_HLL = [S_R F_L - S_L F_R + S_L S_R (U_R-U_L)]
  //           / (S_R-S_L).
  const Real wl = srn / denom;
  const Real wr = -sln / denom;
  const Real dissipation = speed_scale * (sln * srn / denom);
  MhdFlux f;
  f.rho    = product_sum3(wl, fl.rho, wr, fr.rho,
                           dissipation, R.rho - L.rho);
  f.mx     = product_sum3(wl, fl.mx, wr, fr.mx,
                           dissipation, R.mx - L.mx);
  f.my     = product_sum3(wl, fl.my, wr, fr.my,
                           dissipation, R.my - L.my);
  f.mz     = product_sum3(wl, fl.mz, wr, fr.mz,
                           dissipation, R.mz - L.mz);
  f.energy = product_sum3(wl, fl.energy, wr, fr.energy,
                           dissipation, R.energy - L.energy);
  f.bx     = Real{0};
  f.by     = product_sum3(wl, fl.by, wr, fr.by,
                           dissipation, R.by - L.by);
  f.bz     = product_sum3(wl, fl.bz, wr, fr.bz,
                           dissipation, R.bz - L.bz);
  return f;
}

// Positivity-oriented local Lax-Friedrichs (Rusanov) anchor. A generic HLL flux
// with asymmetric Davis speeds is robust as a Riemann fallback, but it is not
// the Lax-Friedrichs splitting used by the multidimensional ideal-MHD
// invariant-domain result. The symmetric bracket +/- max(|v_n|+c_f) makes the
// first-order retry the required convex LF update under the additive CFL bound.
QUASAR_HOST_DEVICE inline MhdFlux lax_friedrichs_flux_x(
    const MhdState& L, const MhdState& R, Real gamma) {
  const Real vxL = L.mx / L.rho;
  const Real vxR = R.mx / R.rho;
  const Real cfL = fast_magnetosonic_speed(L, 0, gamma);
  const Real cfR = fast_magnetosonic_speed(R, 0, gamma);
  const Real alphaL = std::fabs(vxL) + cfL;
  const Real alphaR = std::fabs(vxR) + cfR;
  const Real alpha = (alphaL > alphaR) ? alphaL : alphaR;
  return hll_flux_x(L, R, -alpha, alpha, gamma);
}

// Canonical +x-normal HLLD flux. Falls back to HLL on any degeneracy.
QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdFlux hlld_flux_x(
    const MhdState& Lin, const MhdState& Rin, Real gamma) {
  using std::fabs;
  using std::sqrt;
  using std::isfinite;

  // Relative guards are scaled by the terms forming each denominator.  A fixed
  // 1e-30 cutoff is neither dimensionless nor useful for ordinary O(1) states.
  constexpr Real rel_tol = Real{64} * std::numeric_limits<Real>::epsilon();
  auto resolved = [&](Real value, Real scale) {
    return isfinite(value) && fabs(value) > rel_tol * (scale + std::numeric_limits<Real>::min());
  };

  // Continuous normal field B_n (CT normally makes the inputs identical). If a
  // caller supplies a mismatch, normalize BOTH states to the same half-scaled
  // mean and adjust their total energies to preserve gas pressure. All pressure,
  // wave-speed, physical-flux, and star-state algebra below then sees one
  // internally consistent normal field.
  const Real bn = Real{0.5} * Lin.bx + Real{0.5} * Rin.bx;
  MhdState L = Lin;
  MhdState R = Rin;
  const Real half_bn2 = half_squared_norm3(bn, Real{0}, Real{0});
  L.energy = (L.energy - half_squared_norm3(L.bx, Real{0}, Real{0})) + half_bn2;
  R.energy = (R.energy - half_squared_norm3(R.bx, Real{0}, Real{0})) + half_bn2;
  L.bx = bn;
  R.bx = bn;
  const bool bn2_representable =
      half_bn2 <= Real{0.5} * std::numeric_limits<Real>::max();
  const Real bn2 = bn2_representable ? Real{2} * half_bn2 : Real{0};

  const Real inv_rhoL = Real{1} / L.rho;
  const Real inv_rhoR = Real{1} / R.rho;
  const Real vxL = L.mx * inv_rhoL, vyL = L.my * inv_rhoL, vzL = L.mz * inv_rhoL;
  const Real vxR = R.mx * inv_rhoR, vyR = R.my * inv_rhoR, vzR = R.mz * inv_rhoR;
  const Real pL = pressure(L, gamma);
  const Real pR = pressure(R, gamma);
  const Real ptL = pL + half_squared_norm3(L.bx, L.by, L.bz);
  const Real ptR = pR + half_squared_norm3(R.bx, R.by, R.bz);

  // Fast-magnetosonic outer wave speeds (Davis bracket).
  const Real cfL = fast_magnetosonic_speed(L, 0, gamma);
  const Real cfR = fast_magnetosonic_speed(R, 0, gamma);
  const Real cfmax = (cfL > cfR) ? cfL : cfR;
  const Real sl = ((vxL < vxR) ? vxL : vxR) - cfmax;
  const Real sr = ((vxL > vxR) ? vxL : vxR) + cfmax;

  // One-sided fans: upwind physical flux (also gives L==R -> F(L)).
  if (sl >= Real{0}) return physical_flux_x(L, gamma);
  if (sr <= Real{0}) return physical_flux_x(R, gamma);
  // The seven-wave star formulas contain Bn^2 explicitly. If that square is
  // outside binary64 even though Bn^2/2 (and the physical/HLL flux) is still
  // representable, skip those formulas before they can form inf-inf.
  if (!bn2_representable || !isfinite(ptL) || !isfinite(ptR)) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }

  // Contact / entropy speed S_M. Guard the HLL-mass denominator relatively.
  const Real denomM = (sr - vxR) * R.rho - (sl - vxL) * L.rho;
  const Real denomM_scale = fabs((sr - vxR) * R.rho) +
                            fabs((sl - vxL) * L.rho);
  if (!resolved(denomM, denomM_scale)) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }
  const Real sm = ((sr - vxR) * R.rho * vxR - (sl - vxL) * L.rho * vxL - ptR + ptL) / denomM;
  if (!isfinite(sm)) return hll_flux_x(L, R, sl, sr, gamma);

  // Total pressure is constant across the contact (eq. 41).
  const Real pt_star =
      ((sr - vxR) * R.rho * ptL - (sl - vxL) * L.rho * ptR
       + L.rho * R.rho * (sr - vxR) * (sl - vxL) * (vxR - vxL)) / denomM;
  if (!isfinite(pt_star)) return hll_flux_x(L, R, sl, sr, gamma);

  // Outer star densities; non-positive => degenerate, fall back to HLL.
  const Real dLs = sl - sm;
  const Real dRs = sr - sm;
  const Real wave_scale = fabs(sl) + fabs(sr) + fabs(sm);
  if (!resolved(dLs, wave_scale) || !resolved(dRs, wave_scale)) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }
  const Real rhoLs = L.rho * (sl - vxL) / dLs;
  const Real rhoRs = R.rho * (sr - vxR) / dRs;
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
    const Real denom_scale = fabs(U.rho * sMinusV * (s - sm)) + bn2;
    MhdState us;
    us.rho = rhos;
    us.bx  = bn;
    Real vys, vzs, bys, bzs;
    if (!resolved(denom, denom_scale)) {
      out.ok = false;
      return out;
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
    const Real vdotb = product_sum3(vx, U.bx, vy, U.by, vz, U.bz);
    const Real vdotb_s = product_sum3(sm, bn, vys, bys, vzs, bzs);
    us.energy = (sMinusV * U.energy - pt * vx + pt_star * sm
                 + bn * (vdotb - vdotb_s)) / (s - sm);
    out.u = us; out.vy = vys; out.vz = vzs; out.by = bys; out.bz = bzs;
    return out;
  };

  const Star sL = star_state(L, sl, vxL, vyL, vzL, ptL, rhoLs);
  const Star sR = star_state(R, sr, vxR, vyR, vzR, ptR, rhoRs);
  if (!sL.ok || !sR.ok) return hll_flux_x(L, R, sl, sr, gamma);
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
      bn2 <= rel_tol * (fabs(pt_star) + fabs(ptL) + fabs(ptR) +
                        std::numeric_limits<Real>::min());

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

  auto double_star = [&](const Star& s, Real sqrt_rhos, Real side_sign) -> MhdState {
    MhdState uss;
    uss.rho = s.u.rho;
    uss.bx  = bn;
    uss.mx  = uss.rho * sm;
    uss.my  = uss.rho * vyss;
    uss.mz  = uss.rho * vzss;
    uss.by  = byss;
    uss.bz  = bzss;
    const Real vdotb_ss = product_sum3(sm, bn, vyss, byss, vzss, bzss);
    const Real vdotb_s = product_sum3(sm, bn, s.vy, s.by, s.vz, s.bz);
    // Miyoshi--Kusano Eq. 63 uses the upper (minus) sign on the left and
    // lower (plus) sign on the right rotational state.
    uss.energy = s.u.energy + side_sign * sqrt_rhos * sgn_bn *
                                      (vdotb_s - vdotb_ss);
    return uss;
  };

  const MhdState ussL = double_star(sL, sqrt_rhoLs, Real{-1});
  const MhdState ussR = double_star(sR, sqrt_rhoRs, Real{+1});

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

// ---------------------------------------------------------------------------
// Cancellation-free static-background field split
// ---------------------------------------------------------------------------
//
// The evolved state below is U'=(rho,m,E',b), where
//
//   B = B0 + b,                  E' = e + |m|^2/(2 rho) + |b|^2/2.
//
// Constructing the ordinary total-energy state would add |B0|^2/2 to E' and
// subsequently subtract it to recover the gas pressure.  That destroys all
// thermodynamics once |B0|^2 is more than roughly 1/epsilon times the thermal
// energy.  The split solver instead removes the prescribed, face-constant
// Maxwell stress
//
//   T0 = |B0|^2 I/2 - B0 B0
//
// from the numerical momentum flux.  The finite-volume caller adds -div(T0)
// separately for a non-uniform/current-carrying background.  This core therefore
// never stores an O(|B0|^2) flux beside the physically important O(B0*b) terms.

// Physical +x flux for U', with T0 removed from the momentum flux.  The energy
// flux is F_E' = F_E - B0.F_B and the induction flux is the physical total-field
// induction flux (dB/dt=db/dt because B0 is static).
QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdMomentumFluxParts
physical_flux_split_parts_x(
    const MhdState& u, const MhdBackground& b0, Real gamma) {
  // Preserve the established no-background arithmetic exactly, including for an
  // explicitly enabled but all-zero background profile.
  if (background_is_zero(b0)) {
    MhdMomentumFluxParts parts;
    parts.material = physical_flux_x(u, gamma);
    return parts;
  }

  const Real vx = u.mx / u.rho;
  const Real vy = u.my / u.rho;
  const Real vz = u.mz / u.rho;
  const Real p = pressure(u, gamma);

  MhdMomentumFluxParts parts;
  MhdFlux& f = parts.material;
  f.rho = u.mx;

  // Keep the material stress separate from the background-linear stress.  In
  // particular, p must remain representable when C_xx=B0t.bt is enormous but
  // common to adjacent faces.
  {
    const Real a[5] = {u.mx, p, u.bx, u.by, u.bz};
    const Real b[5] = {vx, Real{1}, Real{-0.5} * u.bx,
                       Real{0.5} * u.by, Real{0.5} * u.bz};
    f.mx = scaled_product_sum(a, b, 5);
  }
  f.my = product_sum2(u.mx, vy, u.bx, -u.by);
  f.mz = product_sum2(u.mx, vz, u.bx, -u.bz);
  parts.cross_b = MhdBackground{u.bx, u.by, u.bz};

  // Expand the normal Poynting cancellation analytically.  In particular the
  // B0x*bx*vx term cancels before either side is formed:
  //
  // F_E' = vx [E'+p+B0.b+|b|^2/2] - (B0x+bx)(v.b)
  //      = vx H - B0x(vt.bt) - bx(vt.bt),
  // H = E'+p+B0t.bt+(|bt|^2-|bx|^2)/2.
  {
    // Do not first collapse H or vt.bt to one Real: in an oblique strong field,
    // vx*B0t.bt and -B0x*vt.bt may be O(1e200) yet cancel to an O(1)
    // thermal/kinetic flux.  Accumulate the fully expanded eleven terms at a
    // common exponent and round only once.
    ScaledQuaternaryAccumulator energy_sum;
    append_scaled_quaternary_product(
        energy_sum, vx, u.energy, Real{1}, Real{1});
    append_scaled_quaternary_product(
        energy_sum, vx, p, Real{1}, Real{1});
    append_scaled_quaternary_product(energy_sum, vx, b0.b0y, u.by, Real{1});
    append_scaled_quaternary_product(energy_sum, vx, b0.b0z, u.bz, Real{1});
    append_scaled_quaternary_product(
        energy_sum, vx, u.bx, Real{-0.5} * u.bx, Real{1});
    append_scaled_quaternary_product(
        energy_sum, vx, u.by, Real{0.5} * u.by, Real{1});
    append_scaled_quaternary_product(
        energy_sum, vx, u.bz, Real{0.5} * u.bz, Real{1});
    append_scaled_quaternary_product(energy_sum, b0.b0x, -vy, u.by, Real{1});
    append_scaled_quaternary_product(energy_sum, b0.b0x, -vz, u.bz, Real{1});
    append_scaled_quaternary_product(energy_sum, u.bx, -vy, u.by, Real{1});
    append_scaled_quaternary_product(energy_sum, u.bx, -vz, u.bz, Real{1});
    f.energy = finish_scaled_quaternary_sum(energy_sum);
  }

  f.bx = Real{0};
  f.by = product_sum4(u.by, vx, u.bx, -vy,
                      b0.b0y, vx, b0.b0x, -vy);
  f.bz = product_sum4(u.bz, vx, u.bx, -vz,
                      b0.b0z, vx, b0.b0x, -vz);
  return parts;
}

QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdFlux
physical_flux_split_x(
    const MhdState& u, const MhdBackground& b0, Real gamma) {
  return compose_momentum_flux_parts(
      physical_flux_split_parts_x(u, b0, gamma), b0);
}

// HLL flux for the reduced state and the background-stress-shifted momentum
// flux.  For one fixed interface B0 this is exactly the affine transform of the
// ordinary total-field HLL flux, but it never constructs total energy or T0.
QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdMomentumFluxParts
hll_flux_split_parts_x(
    const MhdState& L, const MhdState& R, const MhdBackground& b0,
    Real sl, Real sr, Real gamma) {
  if (background_is_zero(b0)) {
    MhdMomentumFluxParts parts;
    parts.material = hll_flux_x(L, R, sl, sr, gamma);
    return parts;
  }
  const MhdMomentumFluxParts fl = physical_flux_split_parts_x(L, b0, gamma);
  const MhdMomentumFluxParts fr = physical_flux_split_parts_x(R, b0, gamma);
  if (sl >= Real{0}) return fl;
  if (sr <= Real{0}) return fr;
  const Real speed_scale = std::fmax(std::fabs(sl), std::fabs(sr));
  if (!(speed_scale > Real{0}) || !std::isfinite(speed_scale)) return fl;
  const Real sln = sl / speed_scale;
  const Real srn = sr / speed_scale;
  const Real denom = srn - sln;
  if (!(denom > Real{0}) || !std::isfinite(denom)) return fl;
  const Real wl = srn / denom;
  const Real wr = -sln / denom;
  const Real dissipation = speed_scale * (sln * srn / denom);
  MhdMomentumFluxParts f;
  f.material.rho = product_sum3(wl, fl.material.rho, wr, fr.material.rho,
                       dissipation, R.rho - L.rho);
  f.material.mx = product_sum2(
      wl, fl.material.mx, wr, fr.material.mx);
  f.material.my = product_sum2(
      wl, fl.material.my, wr, fr.material.my);
  f.material.mz = product_sum2(
      wl, fl.material.mz, wr, fr.material.mz);
  f.wave.mx = dissipation * (R.mx - L.mx);
  f.wave.my = dissipation * (R.my - L.my);
  f.wave.mz = dissipation * (R.mz - L.mz);
  f.cross_b.b0x = product_sum2(wl, L.bx, wr, R.bx);
  f.cross_b.b0y = product_sum2(wl, L.by, wr, R.by);
  f.cross_b.b0z = product_sum2(wl, L.bz, wr, R.bz);
  f.material.energy = product_sum3(
      wl, fl.material.energy, wr, fr.material.energy,
                          dissipation, R.energy - L.energy);
  f.material.bx = Real{0};
  f.material.by = product_sum3(wl, fl.material.by, wr, fr.material.by,
                      dissipation, R.by - L.by);
  f.material.bz = product_sum3(wl, fl.material.bz, wr, fr.material.bz,
                      dissipation, R.bz - L.bz);
  return f;
}

QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdFlux hll_flux_split_x(
    const MhdState& L, const MhdState& R, const MhdBackground& b0,
    Real sl, Real sr, Real gamma) {
  return compose_momentum_flux_parts(
      hll_flux_split_parts_x(L, R, b0, sl, sr, gamma), b0);
}

QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdMomentumFluxParts
lax_friedrichs_flux_split_parts_x(
    const MhdState& L, const MhdState& R, const MhdBackground& b0,
    Real gamma) {
  if (background_is_zero(b0)) {
    MhdMomentumFluxParts parts;
    parts.material = lax_friedrichs_flux_x(L, R, gamma);
    return parts;
  }
  const Real vxL = L.mx / L.rho;
  const Real vxR = R.mx / R.rho;
  const Real cfL = fast_magnetosonic_speed(L, b0, 0, gamma);
  const Real cfR = fast_magnetosonic_speed(R, b0, 0, gamma);
  const Real alphaL = std::fabs(vxL) + cfL;
  const Real alphaR = std::fabs(vxR) + cfR;
  const Real alpha = (alphaL > alphaR) ? alphaL : alphaR;
  return hll_flux_split_parts_x(L, R, b0, -alpha, alpha, gamma);
}

QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdFlux
lax_friedrichs_flux_split_x(
    const MhdState& L, const MhdState& R, const MhdBackground& b0,
    Real gamma) {
  return compose_momentum_flux_parts(
      lax_friedrichs_flux_split_parts_x(L, R, b0, gamma), b0);
}

// Append outer0*outer1*q(u,B0) as seven four-factor products.  Keeping q in
// this unevaluated form lets HLLD combine q_L-q_R and the q/q* work terms before
// rounding; a common O(B0*b) pressure offset therefore cannot erase an O(1) gas
// pressure jump.
QUASAR_HOST_DEVICE inline void append_split_pressure_terms(
    const MhdState& u, const MhdBackground& b0, Real gamma,
    Real outer0, Real outer1, ScaledQuaternaryAccumulator& sum) {
  const Real p = pressure(u, gamma);
  append_scaled_quaternary_product(
      sum, outer0, outer1, p, Real{1});
  append_scaled_quaternary_product(
      sum, outer0, outer1, b0.b0x, u.bx);
  append_scaled_quaternary_product(
      sum, outer0, outer1, b0.b0y, u.by);
  append_scaled_quaternary_product(
      sum, outer0, outer1, b0.b0z, u.bz);
  append_scaled_quaternary_product(
      sum, outer0, outer1, u.bx, Real{0.5} * u.bx);
  append_scaled_quaternary_product(
      sum, outer0, outer1, u.by, Real{0.5} * u.by);
  append_scaled_quaternary_product(
      sum, outer0, outer1, u.bz, Real{0.5} * u.bz);
}

QUASAR_HOST_DEVICE inline Real split_pressure_linear_combination(
    const MhdState& L, Real l0, Real l1,
    const MhdState& R, Real r0, Real r1,
    const MhdBackground& b0, Real gamma) {
  ScaledQuaternaryAccumulator sum;
  append_split_pressure_terms(L, b0, gamma, l0, l1, sum);
  append_split_pressure_terms(R, b0, gamma, r0, r1, sum);
  return finish_scaled_quaternary_sum(sum);
}

// Seven-wave HLLD in reduced variables.  This is the exact face-local affine
// transform of Miyoshi--Kusano HLLD for a prescribed constant B0.  Every star
// magnetic state is formed directly as b*=B*-B0, so an O(1) perturbation is not
// obtained by subtracting two O(B0) rounded numbers.
QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdMomentumFluxParts
hlld_flux_split_parts_x(
    const MhdState& Lin, const MhdState& Rin, const MhdBackground& b0,
    Real gamma) {
  using std::fabs;
  using std::isfinite;
  using std::sqrt;

  if (background_is_zero(b0)) {
    MhdMomentumFluxParts parts;
    parts.material = hlld_flux_x(Lin, Rin, gamma);
    return parts;
  }

  // Exact consistency is particularly important for a dominant background:
  // reconstructing an algebraically identical star energy can round by one ulp
  // and the subsequent O(|B0|) Rankine--Hugoniot multiplier would amplify that
  // harmless state-space rounding into an O(|B0|*ulp(E')) flux error.
  if (Lin.rho == Rin.rho && Lin.mx == Rin.mx && Lin.my == Rin.my &&
      Lin.mz == Rin.mz && Lin.energy == Rin.energy && Lin.bx == Rin.bx &&
      Lin.by == Rin.by && Lin.bz == Rin.bz) {
    return physical_flux_split_parts_x(Lin, b0, gamma);
  }

  constexpr Real rel_tol = Real{64} * std::numeric_limits<Real>::epsilon();
  auto resolved = [&](Real value, Real scale) {
    return isfinite(value) &&
           fabs(value) > rel_tol * (scale + std::numeric_limits<Real>::min());
  };

  // CT supplies one perturbation normal field.  Normalize a mismatched stand-
  // alone caller while preserving the split gas pressure on each side.
  const Real bn_pert = Real{0.5} * Lin.bx + Real{0.5} * Rin.bx;
  MhdState L = Lin;
  MhdState R = Rin;
  const Real half_bn_pert2 = half_squared_norm3(bn_pert, Real{0}, Real{0});
  L.energy = (L.energy - half_squared_norm3(L.bx, Real{0}, Real{0})) +
             half_bn_pert2;
  R.energy = (R.energy - half_squared_norm3(R.bx, Real{0}, Real{0})) +
             half_bn_pert2;
  L.bx = bn_pert;
  R.bx = bn_pert;

  const Real bn = b0.b0x + bn_pert;
  const Real half_bn2 = half_squared_norm3(bn, Real{0}, Real{0});
  const bool bn2_representable =
      half_bn2 <= Real{0.5} * std::numeric_limits<Real>::max();
  const Real bn2 = bn2_representable ? Real{2} * half_bn2 : Real{0};

  const Real vxL = L.mx / L.rho;
  const Real vyL = L.my / L.rho;
  const Real vzL = L.mz / L.rho;
  const Real vxR = R.mx / R.rho;
  const Real vyR = R.my / R.rho;
  const Real vzR = R.mz / R.rho;
  const Real pL = pressure(L, gamma);
  const Real pR = pressure(R, gamma);

  const Real cfL = fast_magnetosonic_speed(L, b0, 0, gamma);
  const Real cfR = fast_magnetosonic_speed(R, b0, 0, gamma);
  const Real cfmax = (cfL > cfR) ? cfL : cfR;
  const Real sl = ((vxL < vxR) ? vxL : vxR) - cfmax;
  const Real sr = ((vxL > vxR) ? vxL : vxR) + cfmax;

  if (sl >= Real{0}) return physical_flux_split_parts_x(L, b0, gamma);
  if (sr <= Real{0}) return physical_flux_split_parts_x(R, b0, gamma);
  if (!bn2_representable) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }

  const Real ar = (sr - vxR) * R.rho;
  const Real al = (sl - vxL) * L.rho;
  const Real denomM = product_sum2(sr - vxR, R.rho,
                                   sl - vxL, -L.rho);
  const Real denomM_scale = fabs(ar) + fabs(al);
  if (!resolved(denomM, denomM_scale)) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }

  // The |B0|^2/2 baseline cancels identically from S_M.  Keep q_L-q_R expanded
  // until the opposing background/perturbation products have cancelled, so a
  // finite gas-pressure jump is not rounded out by a common O(B0*b) offset.
  const Real q_difference = split_pressure_linear_combination(
      L, Real{1}, Real{1}, R, Real{-1}, Real{1}, b0, gamma);
  const Real sm_numerator = product_sum3(
      sr - vxR, R.mx, sl - vxL, -L.mx, q_difference, Real{1});
  const Real sm = sm_numerator / denomM;
  if (!isfinite(sm)) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }

  // q* = wL*qL + wR*qR + wL*AL*(vxR-vxL).  Do not materialize it: each star
  // energy below combines -vx*q(side) with SM*q* in one scaled reduction.
  const Real wL = ar / denomM;
  const Real wR = -al / denomM;

  const Real dLs = sl - sm;
  const Real dRs = sr - sm;
  const Real wave_scale = fabs(sl) + fabs(sr) + fabs(sm);
  if (!resolved(dLs, wave_scale) || !resolved(dRs, wave_scale)) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }
  const Real rhoLs = L.rho * (sl - vxL) / dLs;
  const Real rhoRs = R.rho * (sr - vxR) / dRs;
  if (!(rhoLs > Real{0}) || !(rhoRs > Real{0})) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }

  struct SplitStar {
    MhdState u;
    MhdState jump;
    Real vy;
    Real vz;
    Real by;
    Real bz;
    bool ok;
  };

  auto star_state = [&](const MhdState& U, Real s, Real vx, Real vy,
                        Real vz, Real rhos, bool left_side) -> SplitStar {
    SplitStar out{};
    out.ok = true;
    const Real sMinusV = s - vx;
    const Real mass_wave = U.rho * sMinusV;
    const Real denom = product_sum2(mass_wave, s - sm, bn, -bn);
    const Real denom_scale = fabs(mass_wave * (s - sm)) + bn2;
    if (!resolved(denom, denom_scale)) {
      out.ok = false;
      return out;
    }

    const Real delta = (mass_wave * (sm - vx)) / denom;
    const Real fac = (bn * (sm - vx)) / denom;
    const Real vy_jump = product_sum2(b0.b0y, -fac, U.by, -fac);
    const Real vz_jump = product_sum2(b0.b0z, -fac, U.bz, -fac);
    const Real vys = product_sum2(vy, Real{1}, vy_jump, Real{1});
    const Real vzs = product_sum2(vz, Real{1}, vz_jump, Real{1});
    // b_t* = b_t + delta*b_t + delta*B0_t.  Do not round 1+delta first:
    // delta*b_t can be the finite remainder after the background term cancels.
    const Real by_jump = product_sum2(U.by, delta, b0.b0y, delta);
    const Real bz_jump = product_sum2(U.bz, delta, b0.b0z, delta);
    const Real bys = product_sum2(U.by, Real{1}, by_jump, Real{1});
    const Real bzs = product_sum2(U.bz, Real{1}, bz_jump, Real{1});

    MhdState us;
    us.rho = rhos;
    us.mx = rhos * sm;
    us.my = rhos * vys;
    us.mz = rhos * vzs;
    us.bx = bn_pert;
    us.by = bys;
    us.bz = bzs;

    const Real dot_a[6] = {vx, vy, vz, sm, vys, vzs};
    const Real dot_b[6] = {U.bx, U.by, U.bz,
                           -bn_pert, -bys, -bzs};
    const Real vdotb_difference = scaled_product_sum(dot_a, dot_b, 6);

    // Compute the jump E'*-E' directly.  Forming E'* and subtracting E' again
    // in the Rankine--Hugoniot flux lets one ulp of star-state rounding be
    // amplified by an O(|B0|) outer wave speed.  Algebraically,
    //
    // E'*-E' = [(SM-vx)(E'+q_side) + SM(q*-q_side)
    //           + Bn(v.b-v*.b*)] / (S-SM),
    //
    // q*-qL = wR(qR-qL) + wL*AL*(vxR-vxL),
    // q*-qR = wL(qL-qR) + wL*AL*(vxR-vxL).
    //
    // This form is exactly zero for a resolved contact without relying on the
    // rounded identity wL+wR==1.  Its 24 products are reduced only once.
    ScaledQuaternaryAccumulator energy_jump_sum;
    append_scaled_quaternary_product(
        energy_jump_sum, sm - vx, U.energy, Real{1}, Real{1});
    append_split_pressure_terms(
        U, b0, gamma, sm - vx, Real{1}, energy_jump_sum);
    if (left_side) {
      append_split_pressure_terms(L, b0, gamma, sm, -wR, energy_jump_sum);
      append_split_pressure_terms(R, b0, gamma, sm, wR, energy_jump_sum);
    } else {
      append_split_pressure_terms(L, b0, gamma, sm, wL, energy_jump_sum);
      append_split_pressure_terms(R, b0, gamma, sm, -wL, energy_jump_sum);
    }
    append_scaled_quaternary_product(
        energy_jump_sum, sm, wL, al, vxR - vxL);
    append_scaled_quaternary_product(
        energy_jump_sum, bn, vdotb_difference, Real{1}, Real{1});
    const Real energy_jump =
        finish_scaled_quaternary_sum(energy_jump_sum) / (s - sm);
    us.energy = product_sum2(U.energy, Real{1}, energy_jump, Real{1});

    // Store every Rankine--Hugoniot jump directly.  Reconstructing a rounded
    // star state and subtracting U later can turn a sub-ulp state jump into a
    // one-ulp error that is then amplified by the outer wave speed.
    MhdState jump{};
    jump.rho = (U.rho * (sm - vx)) / (s - sm);
    jump.mx = product_sum2(jump.rho, sm, U.rho, sm - vx);
    jump.my = product_sum2(jump.rho, vys, U.rho, vy_jump);
    jump.mz = product_sum2(jump.rho, vzs, U.rho, vz_jump);
    jump.energy = energy_jump;
    jump.by = by_jump;
    jump.bz = bz_jump;

    out.u = us;
    out.jump = jump;
    out.vy = vys;
    out.vz = vzs;
    out.by = bys;
    out.bz = bzs;
    out.ok = isfinite(us.energy) && isfinite(energy_jump) && isfinite(vys) &&
             isfinite(vzs) && isfinite(bys) && isfinite(bzs) &&
             isfinite(jump.rho) && isfinite(jump.mx) && isfinite(jump.my) &&
             isfinite(jump.mz) && isfinite(jump.by) && isfinite(jump.bz);
    return out;
  };

  const SplitStar sL = star_state(L, sl, vxL, vyL, vzL, rhoLs, true);
  const SplitStar sR = star_state(R, sr, vxR, vyR, vzR, rhoRs, false);
  if (!sL.ok || !sR.ok) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }
  const MhdMomentumFluxParts fl = physical_flux_split_parts_x(L, b0, gamma);
  const MhdMomentumFluxParts fr = physical_flux_split_parts_x(R, b0, gamma);

  auto flux_from_star = [&](const MhdMomentumFluxParts& fU,
                            const SplitStar& star, Real s) {
    MhdMomentumFluxParts f = fU;
    f.material.rho = product_sum2(
        fU.material.rho, Real{1}, s, star.jump.rho);
    f.wave.mx = s * star.jump.mx;
    f.wave.my = s * star.jump.my;
    f.wave.mz = s * star.jump.mz;
    f.material.energy = product_sum2(
        fU.material.energy, Real{1}, s, star.jump.energy);
    f.material.bx = Real{0};
    f.material.by = product_sum2(
        fU.material.by, Real{1}, s, star.jump.by);
    f.material.bz = product_sum2(
        fU.material.bz, Real{1}, s, star.jump.bz);
    return f;
  };

  const Real sqrt_rhoLs = sqrt(rhoLs);
  const Real sqrt_rhoRs = sqrt(rhoRs);
  const Real saL = sm - fabs(bn) / sqrt_rhoLs;
  const Real saR = sm + fabs(bn) / sqrt_rhoRs;

  // Test Bn against the total-field scale, not q (whose |B0|^2 baseline was
  // intentionally removed).  A tiny normal field in a huge tangential guide
  // field must still take the degenerate/HLL branch.
  const Real fieldL = scaled_norm3(bn, b0.b0y + L.by, b0.b0z + L.bz);
  const Real fieldR = scaled_norm3(bn, b0.b0y + R.by, b0.b0z + R.bz);
  const Real thermal_field = std::fmax(
      (pL > Real{0}) ? sqrt(pL) : Real{0},
      (pR > Real{0}) ? sqrt(pR) : Real{0});
  const Real total_field_scale =
      std::fmax(fieldL, std::fmax(fieldR, thermal_field));
  const bool bn_negligible =
      fabs(bn) <= sqrt(rel_tol) *
                      (total_field_scale + std::numeric_limits<Real>::min());

  MhdMomentumFluxParts f;
  if (bn_negligible) {
    if (sl <= Real{0} && Real{0} <= sm) {
      f = flux_from_star(fl, sL, sl);
    } else {
      f = flux_from_star(fr, sR, sr);
    }
    if (!isfinite(f.material.rho) || !isfinite(f.material.mx) ||
        !isfinite(f.material.my) || !isfinite(f.material.mz) ||
        !isfinite(f.wave.mx) || !isfinite(f.wave.my) ||
        !isfinite(f.wave.mz) || !isfinite(f.material.energy) ||
        !isfinite(f.material.by) || !isfinite(f.material.bz) ||
        !isfinite(f.cross_b.b0x) || !isfinite(f.cross_b.b0y) ||
        !isfinite(f.cross_b.b0z)) {
      return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
    }
    return f;
  }

  const Real sqrt_sum = sqrt_rhoLs + sqrt_rhoRs;
  if (!(sqrt_sum > Real{0}) || !isfinite(sqrt_sum)) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }
  const Real inv_sqrt_sum = Real{1} / sqrt_sum;
  const Real sqrt_weightL = sqrt_rhoLs * inv_sqrt_sum;
  const Real sqrt_weightR = sqrt_rhoRs * inv_sqrt_sum;
  const Real sgn_bn = (bn >= Real{0}) ? Real{1} : Real{-1};
  const Real dvy_star = product_sum2(sR.vy, Real{1}, sL.vy, Real{-1});
  const Real dvz_star = product_sum2(sR.vz, Real{1}, sL.vz, Real{-1});
  const Real dby_star = product_sum2(sR.by, Real{1}, sL.by, Real{-1});
  const Real dbz_star = product_sum2(sR.bz, Real{1}, sL.bz, Real{-1});

  // Form the common double-star values as corrections to the left star state.
  // Equal tangential star states then remain bit-exact even when the two star
  // densities (and hence their square-root weights) differ across a contact.
  const Real vyss_correction = product_sum2(
      sqrt_weightR, dvy_star, sgn_bn * inv_sqrt_sum, dby_star);
  const Real vzss_correction = product_sum2(
      sqrt_weightR, dvz_star, sgn_bn * inv_sqrt_sum, dbz_star);
  const Real vyss = product_sum2(sL.vy, Real{1}, vyss_correction, Real{1});
  const Real vzss = product_sum2(sL.vz, Real{1}, vzss_correction, Real{1});
  // These are perturbation fields directly.  The shared B0 contribution to the
  // ordinary total-field formula cancels against its denominator exactly.
  const Real byss_correction = product_sum2(
      sqrt_weightL, dby_star,
      sgn_bn * (sqrt_rhoLs * sqrt_weightR), dvy_star);
  const Real bzss_correction = product_sum2(
      sqrt_weightL, dbz_star,
      sgn_bn * (sqrt_rhoLs * sqrt_weightR), dvz_star);
  const Real byss = product_sum2(sL.by, Real{1}, byss_correction, Real{1});
  const Real bzss = product_sum2(sL.bz, Real{1}, bzss_correction, Real{1});

  struct SplitDoubleStar {
    MhdState u;
    MhdState jump;
    bool ok;
  };
  auto double_star = [&](const SplitStar& s, Real sqrt_rhos,
                         Real side_sign, bool left_side) -> SplitDoubleStar {
    SplitDoubleStar out{};
    const Real vy_jump = left_side
                             ? vyss_correction
                             : product_sum2(vyss_correction, Real{1},
                                            dvy_star, Real{-1});
    const Real vz_jump = left_side
                             ? vzss_correction
                             : product_sum2(vzss_correction, Real{1},
                                            dvz_star, Real{-1});
    const Real by_jump = left_side
                             ? byss_correction
                             : product_sum2(byss_correction, Real{1},
                                            dby_star, Real{-1});
    const Real bz_jump = left_side
                             ? bzss_correction
                             : product_sum2(bzss_correction, Real{1},
                                            dbz_star, Real{-1});
    MhdState uss;
    uss.rho = s.u.rho;
    uss.mx = uss.rho * sm;
    uss.my = product_sum2(s.u.my, Real{1}, uss.rho, vy_jump);
    uss.mz = product_sum2(s.u.mz, Real{1}, uss.rho, vz_jump);
    uss.bx = bn_pert;
    uss.by = product_sum2(s.u.by, Real{1}, by_jump, Real{1});
    uss.bz = product_sum2(s.u.bz, Real{1}, bz_jump, Real{1});
    const Real dot_a[6] = {sm, s.vy, s.vz, sm, vyss, vzss};
    const Real dot_b[6] = {bn_pert, s.by, s.bz,
                           -bn_pert, -byss, -bzss};
    const Real vdotb_difference = scaled_product_sum(dot_a, dot_b, 6);
    const Real energy_rotation =
        ((side_sign * sqrt_rhos) * sgn_bn) * vdotb_difference;
    uss.energy = product_sum2(
        s.u.energy, Real{1}, energy_rotation, Real{1});
    out.u = uss;
    out.jump = MhdState{};
    out.jump.my = uss.rho * vy_jump;
    out.jump.mz = uss.rho * vz_jump;
    out.jump.energy = energy_rotation;
    out.jump.by = by_jump;
    out.jump.bz = bz_jump;
    out.ok = isfinite(energy_rotation) && isfinite(uss.energy) &&
             isfinite(out.jump.my) && isfinite(out.jump.mz) &&
             isfinite(by_jump) && isfinite(bz_jump);
    return out;
  };

  const SplitDoubleStar ssL =
      double_star(sL, sqrt_rhoLs, Real{-1}, true);
  const SplitDoubleStar ssR =
      double_star(sR, sqrt_rhoRs, Real{+1}, false);
  if (!ssL.ok || !ssR.ok) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }
  if (Real{0} <= saL) {
    f = flux_from_star(fl, sL, sl);
  } else if (Real{0} <= sm) {
    f = fl;
    f.material.rho = product_sum3(fl.material.rho, Real{1},
                                  saL, ssL.jump.rho, sl, sL.jump.rho);
    f.wave.mx = product_sum2(saL, ssL.jump.mx, sl, sL.jump.mx);
    f.wave.my = product_sum2(saL, ssL.jump.my, sl, sL.jump.my);
    f.wave.mz = product_sum2(saL, ssL.jump.mz, sl, sL.jump.mz);
    f.material.energy = product_sum3(
        fl.material.energy, Real{1}, saL, ssL.jump.energy,
        sl, sL.jump.energy);
    f.material.bx = Real{0};
    f.material.by = product_sum3(
        fl.material.by, Real{1}, saL, ssL.jump.by, sl, sL.jump.by);
    f.material.bz = product_sum3(
        fl.material.bz, Real{1}, saL, ssL.jump.bz, sl, sL.jump.bz);
  } else if (Real{0} <= saR) {
    f = fr;
    f.material.rho = product_sum3(fr.material.rho, Real{1},
                                  saR, ssR.jump.rho, sr, sR.jump.rho);
    f.wave.mx = product_sum2(saR, ssR.jump.mx, sr, sR.jump.mx);
    f.wave.my = product_sum2(saR, ssR.jump.my, sr, sR.jump.my);
    f.wave.mz = product_sum2(saR, ssR.jump.mz, sr, sR.jump.mz);
    f.material.energy = product_sum3(
        fr.material.energy, Real{1}, saR, ssR.jump.energy,
        sr, sR.jump.energy);
    f.material.bx = Real{0};
    f.material.by = product_sum3(
        fr.material.by, Real{1}, saR, ssR.jump.by, sr, sR.jump.by);
    f.material.bz = product_sum3(
        fr.material.bz, Real{1}, saR, ssR.jump.bz, sr, sR.jump.bz);
  } else {
    f = flux_from_star(fr, sR, sr);
  }

  if (!isfinite(f.material.rho) || !isfinite(f.material.mx) ||
      !isfinite(f.material.my) || !isfinite(f.material.mz) ||
      !isfinite(f.wave.mx) || !isfinite(f.wave.my) ||
      !isfinite(f.wave.mz) || !isfinite(f.material.energy) ||
      !isfinite(f.material.by) || !isfinite(f.material.bz) ||
      !isfinite(f.cross_b.b0x) || !isfinite(f.cross_b.b0y) ||
      !isfinite(f.cross_b.b0z)) {
    return hll_flux_split_parts_x(L, R, b0, sl, sr, gamma);
  }
  return f;
}

QUASAR_HOST_DEVICE QUASAR_HLLD_DEVICE_NOINLINE inline MhdFlux hlld_flux_split_x(
    const MhdState& Lin, const MhdState& Rin, const MhdBackground& b0,
    Real gamma) {
  return compose_momentum_flux_parts(
      hlld_flux_split_parts_x(Lin, Rin, b0, gamma), b0);
}

#undef QUASAR_HLLD_DEVICE_NOINLINE

}  // namespace quasar::numerics::hlld
