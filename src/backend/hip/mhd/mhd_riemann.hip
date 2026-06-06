// Device Riemann flux for the ideal-MHD solver: full HLLD (Miyoshi & Kusano
// 2005) with a single-intermediate HLL fallback on degenerate fans.
//
// Inputs are the reconstructed LEFT/RIGHT conserved interface states at every
// interface normal to `dir`; the output is the numerical flux written into the
// 8 conserved-component buffers of an MhdField2D reused as flux storage.
//
// The flux is computed in a rotated frame where the interface normal is the
// local x-axis: for dir=1 (y-interfaces) the in-plane components are swapped
// (x<->y) before the solve and swapped back when storing, so a single normal-x
// HLLD routine serves both directions.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdFlux;
using quasar::numerics::MhdState;
using quasar::numerics::fast_magnetosonic_speed;
using quasar::numerics::pressure;

inline constexpr unsigned kBlock2D = 16;
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Rotate a conserved state into the frame where the interface normal is x.
// dir=0 is identity; dir=1 swaps the (x,y) momentum and (x,y) magnetic pairs.
__device__ inline MhdState rotate_in(const MhdState& s, int dir) {
  if (dir == 0) return s;
  MhdState r = s;
  r.mx = s.my; r.my = s.mx;
  r.bx = s.by; r.by = s.bx;
  return r;
}

// Rotate a flux computed in the normal-x frame back to the lab frame.
__device__ inline MhdFlux rotate_out(const MhdFlux& f, int dir) {
  if (dir == 0) return f;
  MhdFlux r = f;
  r.mx = f.my; r.my = f.mx;
  r.bx = f.by; r.by = f.bx;
  return r;
}

// Physical x-direction ideal-MHD flux of a conserved state (normal-x frame).
__device__ inline MhdFlux phys_flux_x(const MhdState& u, Real gamma) {
  const Real inv_rho = Real{1} / u.rho;
  const Real vx = u.mx * inv_rho;
  const Real vy = u.my * inv_rho;
  const Real vz = u.mz * inv_rho;
  const Real p = pressure(u, gamma);
  const Real b2 = u.bx * u.bx + u.by * u.by + u.bz * u.bz;
  const Real ptot = p + Real{0.5} * b2;
  const Real vdotb = vx * u.bx + vy * u.by + vz * u.bz;
  MhdFlux f;
  f.rho    = u.mx;
  f.mx     = u.mx * vx + ptot - u.bx * u.bx;
  f.my     = u.my * vx - u.bx * u.by;
  f.mz     = u.mz * vx - u.bx * u.bz;
  f.energy = (u.energy + ptot) * vx - u.bx * vdotb;
  f.bx     = Real{0};
  f.by     = u.by * vx - u.bx * vy;
  f.bz     = u.bz * vx - u.bx * vz;
  return f;
}

// HLL (single intermediate) flux in the normal-x frame -- robust fallback.
__device__ inline MhdFlux hll_flux_x(const MhdState& L, const MhdState& R,
                                     Real sl, Real sr, Real gamma) {
  const MhdFlux fl = phys_flux_x(L, gamma);
  const MhdFlux fr = phys_flux_x(R, gamma);
  if (sl >= Real{0}) return fl;
  if (sr <= Real{0}) return fr;
  const Real inv = Real{1} / (sr - sl);
  MhdFlux f;
  f.rho    = (sr * fl.rho    - sl * fr.rho    + sl * sr * (R.rho    - L.rho))    * inv;
  f.mx     = (sr * fl.mx     - sl * fr.mx     + sl * sr * (R.mx     - L.mx))     * inv;
  f.my     = (sr * fl.my     - sl * fr.my     + sl * sr * (R.my     - L.my))     * inv;
  f.mz     = (sr * fl.mz     - sl * fr.mz     + sl * sr * (R.mz     - L.mz))     * inv;
  f.energy = (sr * fl.energy - sl * fr.energy + sl * sr * (R.energy - L.energy)) * inv;
  f.bx     = (sr * fl.bx     - sl * fr.bx     + sl * sr * (R.bx     - L.bx))     * inv;
  f.by     = (sr * fl.by     - sl * fr.by     + sl * sr * (R.by     - L.by))     * inv;
  f.bz     = (sr * fl.bz     - sl * fr.bz     + sl * sr * (R.bz     - L.bz))     * inv;
  return f;
}

// Full HLLD flux in the normal-x frame, falling back to HLL on degeneracy.
__device__ inline MhdFlux hlld_flux_x(const MhdState& L, const MhdState& R, Real gamma) {
  const Real inv_rhoL = Real{1} / L.rho;
  const Real inv_rhoR = Real{1} / R.rho;
  const Real vxL = L.mx * inv_rhoL, vyL = L.my * inv_rhoL, vzL = L.mz * inv_rhoL;
  const Real vxR = R.mx * inv_rhoR, vyR = R.my * inv_rhoR, vzR = R.mz * inv_rhoR;
  const Real pL = pressure(L, gamma);
  const Real pR = pressure(R, gamma);
  // Normal B is taken continuous across the interface (CT keeps it so).
  const Real bx = Real{0.5} * (L.bx + R.bx);
  const Real bx2 = bx * bx;
  const Real ptL = pL + Real{0.5} * (L.bx * L.bx + L.by * L.by + L.bz * L.bz);
  const Real ptR = pR + Real{0.5} * (R.bx * R.bx + R.by * R.by + R.bz * R.bz);

  // Fastest signal speeds (Davis estimate with fast magnetosonic speeds).
  const Real cfL = fast_magnetosonic_speed(L, 0, gamma);
  const Real cfR = fast_magnetosonic_speed(R, 0, gamma);
  Real sl = fmin(vxL, vxR) - fmax(cfL, cfR);
  Real sr = fmax(vxL, vxR) + fmax(cfL, cfR);

  const MhdFlux fl = phys_flux_x(L, gamma);
  const MhdFlux fr = phys_flux_x(R, gamma);
  if (sl >= Real{0}) return fl;
  if (sr <= Real{0}) return fr;

  // Contact / entropy speed sm and the total pressure in the star region.
  const Real denom = (sr - vxR) * R.rho - (sl - vxL) * L.rho;
  if (fabs(denom) < Real{1e-30}) return hll_flux_x(L, R, sl, sr, gamma);
  const Real sm = ((sr - vxR) * R.rho * vxR - (sl - vxL) * L.rho * vxL - ptR + ptL) / denom;
  const Real pts = ((sr - vxR) * R.rho * ptL - (sl - vxL) * L.rho * ptR
                    + L.rho * R.rho * (sr - vxR) * (sl - vxL) * (vxR - vxL)) / denom;

  // Outer star (U*) densities.
  const Real rhoLs = L.rho * (sl - vxL) / (sl - sm);
  const Real rhoRs = R.rho * (sr - vxR) / (sr - sm);
  if (!(rhoLs > Real{0}) || !(rhoRs > Real{0})) return hll_flux_x(L, R, sl, sr, gamma);

  const Real sqrtRhoLs = sqrt(rhoLs);
  const Real sqrtRhoRs = sqrt(rhoRs);
  // Alfven speeds bounding the inner (U**) region.
  const Real sLs = sm - fabs(bx) / sqrtRhoLs;
  const Real sRs = sm + fabs(bx) / sqrtRhoRs;

  // --- Build U*_L --------------------------------------------------------
  auto build_star = [&](const MhdState& U, Real vx, Real vy, Real vz, Real pt,
                        Real S, Real rhoS, Real* vyS, Real* vzS,
                        Real* byS, Real* bzS, MhdState* out) -> bool {
    const Real sMinusV = S - vx;
    const Real factor = U.rho * sMinusV * (S - sm) - bx2;
    if (fabs(factor) < Real{1e-30}) return false;
    const Real invF = Real{1} / factor;
    const Real bcoef = bx * (sm - vx) * invF;
    *vyS = vy - U.by * bcoef;
    *vzS = vz - U.bz * bcoef;
    const Real bnum = U.rho * sMinusV * sMinusV - bx2;
    const Real bfac = bnum * invF;
    *byS = U.by * bfac;
    *bzS = U.bz * bfac;
    out->rho = rhoS;
    out->mx  = rhoS * sm;
    out->my  = rhoS * (*vyS);
    out->mz  = rhoS * (*vzS);
    out->bx  = bx;
    out->by  = *byS;
    out->bz  = *bzS;
    const Real vdotb  = vx * U.bx + vy * U.by + vz * U.bz;
    const Real vdotbS = sm * bx + (*vyS) * (*byS) + (*vzS) * (*bzS);
    out->energy = (sMinusV * U.energy - pt * vx + pts * sm
                   + bx * (vdotb - vdotbS)) / (S - sm);
    return true;
  };

  MhdState ULs, URs;
  Real vyLs, vzLs, byLs, bzLs, vyRs, vzRs, byRs, bzRs;
  if (!build_star(L, vxL, vyL, vzL, ptL, sl, rhoLs, &vyLs, &vzLs, &byLs, &bzLs, &ULs) ||
      !build_star(R, vxR, vyR, vzR, ptR, sr, rhoRs, &vyRs, &vzRs, &byRs, &bzRs, &URs)) {
    return hll_flux_x(L, R, sl, sr, gamma);
  }

  const MhdFlux fls{}; (void)fls;  // silence unused warnings on some toolchains

  if (sm >= Real{0}) {
    // Left side of the contact.
    if (sLs >= Real{0}) {
      // U*_L flux: F_L + sl (U*_L - U_L).
      MhdFlux f = fl;
      f.rho    += sl * (ULs.rho    - L.rho);
      f.mx     += sl * (ULs.mx     - L.mx);
      f.my     += sl * (ULs.my     - L.my);
      f.mz     += sl * (ULs.mz     - L.mz);
      f.energy += sl * (ULs.energy - L.energy);
      f.by     += sl * (ULs.by     - L.by);
      f.bz     += sl * (ULs.bz     - L.bz);
      return f;
    }
    // U**_L region (between sLs and sm). Build the rotational intermediate.
    const Real sgn = (bx >= Real{0}) ? Real{1} : Real{-1};
    const Real inv = Real{1} / (sqrtRhoLs + sqrtRhoRs);
    const Real vyss = (sqrtRhoLs * vyLs + sqrtRhoRs * vyRs
                       + (byRs - byLs) * sgn) * inv;
    const Real vzss = (sqrtRhoLs * vzLs + sqrtRhoRs * vzRs
                       + (bzRs - bzLs) * sgn) * inv;
    const Real byss = (sqrtRhoLs * byRs + sqrtRhoRs * byLs
                       + sqrtRhoLs * sqrtRhoRs * (vyRs - vyLs) * sgn) * inv;
    const Real bzss = (sqrtRhoLs * bzRs + sqrtRhoRs * bzLs
                       + sqrtRhoLs * sqrtRhoRs * (vzRs - vzLs) * sgn) * inv;
    MhdState Uss;
    Uss.rho = rhoLs;
    Uss.mx  = rhoLs * sm;
    Uss.my  = rhoLs * vyss;
    Uss.mz  = rhoLs * vzss;
    Uss.bx  = bx;
    Uss.by  = byss;
    Uss.bz  = bzss;
    const Real vdotbSS = sm * bx + vyss * byss + vzss * bzss;
    Uss.energy = ULs.energy - sgn * sqrtRhoLs
                 * ((sm * bx + vyLs * byLs + vzLs * bzLs) - vdotbSS);
    // F = F_L + sLs (U**_L - U*_L) + sl (U*_L - U_L).
    MhdFlux f = fl;
    f.rho    += sl * (ULs.rho - L.rho)       + sLs * (Uss.rho - ULs.rho);
    f.mx     += sl * (ULs.mx - L.mx)         + sLs * (Uss.mx - ULs.mx);
    f.my     += sl * (ULs.my - L.my)         + sLs * (Uss.my - ULs.my);
    f.mz     += sl * (ULs.mz - L.mz)         + sLs * (Uss.mz - ULs.mz);
    f.energy += sl * (ULs.energy - L.energy) + sLs * (Uss.energy - ULs.energy);
    f.by     += sl * (ULs.by - L.by)         + sLs * (Uss.by - ULs.by);
    f.bz     += sl * (ULs.bz - L.bz)         + sLs * (Uss.bz - ULs.bz);
    return f;
  } else {
    // Right side of the contact.
    if (sRs <= Real{0}) {
      MhdFlux f = fr;
      f.rho    += sr * (URs.rho    - R.rho);
      f.mx     += sr * (URs.mx     - R.mx);
      f.my     += sr * (URs.my     - R.my);
      f.mz     += sr * (URs.mz     - R.mz);
      f.energy += sr * (URs.energy - R.energy);
      f.by     += sr * (URs.by     - R.by);
      f.bz     += sr * (URs.bz     - R.bz);
      return f;
    }
    const Real sgn = (bx >= Real{0}) ? Real{1} : Real{-1};
    const Real inv = Real{1} / (sqrtRhoLs + sqrtRhoRs);
    const Real vyss = (sqrtRhoLs * vyLs + sqrtRhoRs * vyRs
                       + (byRs - byLs) * sgn) * inv;
    const Real vzss = (sqrtRhoLs * vzLs + sqrtRhoRs * vzRs
                       + (bzRs - bzLs) * sgn) * inv;
    const Real byss = (sqrtRhoLs * byRs + sqrtRhoRs * byLs
                       + sqrtRhoLs * sqrtRhoRs * (vyRs - vyLs) * sgn) * inv;
    const Real bzss = (sqrtRhoLs * bzRs + sqrtRhoRs * bzLs
                       + sqrtRhoLs * sqrtRhoRs * (vzRs - vzLs) * sgn) * inv;
    MhdState Uss;
    Uss.rho = rhoRs;
    Uss.mx  = rhoRs * sm;
    Uss.my  = rhoRs * vyss;
    Uss.mz  = rhoRs * vzss;
    Uss.bx  = bx;
    Uss.by  = byss;
    Uss.bz  = bzss;
    const Real vdotbSS = sm * bx + vyss * byss + vzss * bzss;
    Uss.energy = URs.energy + sgn * sqrtRhoRs
                 * ((sm * bx + vyRs * byRs + vzRs * bzRs) - vdotbSS);
    MhdFlux f = fr;
    f.rho    += sr * (URs.rho - R.rho)       + sRs * (Uss.rho - URs.rho);
    f.mx     += sr * (URs.mx - R.mx)         + sRs * (Uss.mx - URs.mx);
    f.my     += sr * (URs.my - R.my)         + sRs * (Uss.my - URs.my);
    f.mz     += sr * (URs.mz - R.mz)         + sRs * (Uss.mz - URs.mz);
    f.energy += sr * (URs.energy - R.energy) + sRs * (Uss.energy - URs.energy);
    f.by     += sr * (URs.by - R.by)         + sRs * (Uss.by - URs.by);
    f.bz     += sr * (URs.bz - R.bz)         + sRs * (Uss.bz - URs.bz);
    return f;
  }
}

__device__ inline MhdState load_state(const double* rho, const double* mx,
                                      const double* my, const double* mz,
                                      const double* energy, const double* bx,
                                      const double* by, const double* bz,
                                      std::size_t k) {
  MhdState s;
  s.rho = rho[k]; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
  s.energy = energy[k]; s.bx = bx[k]; s.by = by[k]; s.bz = bz[k];
  return s;
}

__global__ void hlld_kernel(
    quasar::Grid2D g, int dir, Real gamma,
    const double* __restrict__ Lrho, const double* __restrict__ Lmx,
    const double* __restrict__ Lmy, const double* __restrict__ Lmz,
    const double* __restrict__ Lenergy, const double* __restrict__ Lbx,
    const double* __restrict__ Lby, const double* __restrict__ Lbz,
    const double* __restrict__ Rrho, const double* __restrict__ Rmx,
    const double* __restrict__ Rmy, const double* __restrict__ Rmz,
    const double* __restrict__ Renergy, const double* __restrict__ Rbx,
    const double* __restrict__ Rby, const double* __restrict__ Rbz,
    double* __restrict__ Frho, double* __restrict__ Fmx, double* __restrict__ Fmy,
    double* __restrict__ Fmz, double* __restrict__ Fenergy, double* __restrict__ Fbx,
    double* __restrict__ Fby, double* __restrict__ Fbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);

  const MhdState L = load_state(Lrho, Lmx, Lmy, Lmz, Lenergy, Lbx, Lby, Lbz, k);
  const MhdState R = load_state(Rrho, Rmx, Rmy, Rmz, Renergy, Rbx, Rby, Rbz, k);

  const MhdState Lr = rotate_in(L, dir);
  const MhdState Rr = rotate_in(R, dir);
  const MhdFlux fr = hlld_flux_x(Lr, Rr, gamma);
  const MhdFlux f = rotate_out(fr, dir);

  Frho[k] = f.rho; Fmx[k] = f.mx; Fmy[k] = f.my; Fmz[k] = f.mz;
  Fenergy[k] = f.energy; Fbx[k] = f.bx; Fby[k] = f.by; Fbz[k] = f.bz;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface, int dir,
                          MhdField2D<Real>& flux_out, Real gamma, stream_t stream) {
  const Grid2D g = iface.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(hlld_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, dir, gamma,
                     iface.Lrho.device_ptr(), iface.Lmx.device_ptr(), iface.Lmy.device_ptr(),
                     iface.Lmz.device_ptr(), iface.Lenergy.device_ptr(), iface.Lbx.device_ptr(),
                     iface.Lby.device_ptr(), iface.Lbz.device_ptr(),
                     iface.Rrho.device_ptr(), iface.Rmx.device_ptr(), iface.Rmy.device_ptr(),
                     iface.Rmz.device_ptr(), iface.Renergy.device_ptr(), iface.Rbx.device_ptr(),
                     iface.Rby.device_ptr(), iface.Rbz.device_ptr(),
                     flux_out.rho.device_ptr(), flux_out.mx.device_ptr(), flux_out.my.device_ptr(),
                     flux_out.mz.device_ptr(), flux_out.energy.device_ptr(),
                     flux_out.bx_face.device_ptr(), flux_out.by_face.device_ptr(),
                     flux_out.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
