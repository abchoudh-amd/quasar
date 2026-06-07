#pragma once

// Scalar monotonicity-preserving (MP) limiter helpers, device-callable.
//
// These are the building blocks of the MP5/MP7 high-order reconstruction of
// Suresh & Huynh (1997), "Accurate Monotonicity-Preserving Schemes with
// Runge-Kutta Time Stepping", J. Comput. Phys. 136, 83-99. They originated in an
// anonymous namespace inside src/numerics/flux_reconstruction.cpp (the host
// registry path) and are exposed here as `QUASAR_HOST_DEVICE inline` so the
// device reconstruct kernel (src/backend/hip/mhd/mhd_reconstruct.hip) and the
// host registry classes call ONE shared implementation.
//
// The Suresh-Huynh MP limiter machinery (minmod2/minmod4/median3/mp_limit) is
// the unchanged historical math. The base interface INTERPOLATION coefficients
// in mp5_interp / mp7_interp, however, were CORRECTED during the device port:
// this is a Shu-Osher POINT-VALUE finite-difference scheme (the stored cell
// values are point samples, and the reconstructed face value is a point value),
// so mp5_interp / mp7_interp use the point-value Lagrange weights
// ((3,-20,90,60,-5)/128 and (-5,42,-175,700,525,-70,7)/1024) rather than the
// finite-VOLUME cell-average->face weights the original host code carried. The
// FV weights were a latent bug: fed point data they were only 2nd-order, so the
// pre-port host "MP5/MP7" silently ran ~2nd order. With the corrected point-value
// weights MP5/MP7 achieve their design 5th/7th order. Consequence: results are
// NOT bit-identical to any pre-port host MP5/MP7 output.
//
// All host-only `std::` math (`std::abs`, `std::min`, `std::max`) is replaced by
// the device-callable `fabs`/`fmin`/`fmax` so every helper compiles in both host
// and __device__ contexts.

#include "quasar/core/types.hpp"  // Real, QUASAR_HOST_DEVICE

#include <cmath>

namespace quasar::numerics {

// --- scalar limiter helpers --------------------------------------------------

QUASAR_HOST_DEVICE inline Real minmod2(Real a, Real b) {
  if (a * b <= Real{0}) return Real{0};
  return (fabs(a) < fabs(b)) ? a : b;
}

// 4-argument minmod (Suresh-Huynh Eq. 2.9): zero unless all four share a sign,
// otherwise the smallest magnitude.
QUASAR_HOST_DEVICE inline Real minmod4(Real a, Real b, Real c, Real d) {
  const Real sa = (a > Real{0}) ? Real{1} : ((a < Real{0}) ? Real{-1} : Real{0});
  const Real sb = (b > Real{0}) ? Real{1} : ((b < Real{0}) ? Real{-1} : Real{0});
  const Real sc = (c > Real{0}) ? Real{1} : ((c < Real{0}) ? Real{-1} : Real{0});
  const Real sd = (d > Real{0}) ? Real{1} : ((d < Real{0}) ? Real{-1} : Real{0});
  if (!(sa == sb && sb == sc && sc == sd) || sa == Real{0}) {
    return Real{0};
  }
  const Real mag = fmin(fmin(fabs(a), fabs(b)),
                        fmin(fabs(c), fabs(d)));
  return sa * mag;
}

QUASAR_HOST_DEVICE inline Real median3(Real a, Real b, Real c) {
  // median(a,b,c) = a + minmod(b-a, c-a) (Suresh-Huynh Eq. 2.7).
  return a + minmod2(b - a, c - a);
}

// MP monotonicity-preserving limiter applied to an unlimited candidate `vl`
// (the high-order interpolation of the LEFT-biased interface value at the right
// face of the central cell v[0]). Stencil: v[-2..+2] for MP5 / v[-3..+3] for MP7,
// but the MP bound only needs v[-1], v[0], v[+1] plus the curvature from the
// neighbours. We pass the five-point window {vm2,vm1,v0,vp1,vp2} since both MP5
// and MP7 use the same MP machinery on the central five points.
// Suresh-Huynh 1997, Section 2.2.
QUASAR_HOST_DEVICE inline Real mp_limit(Real vl, Real vm2, Real vm1, Real v0,
                                        Real vp1, Real vp2) {
  constexpr Real alpha = Real{4};
  constexpr Real eps   = static_cast<Real>(1e-10);

  // mp = v0 + minmod(d+, alpha*d-), the monotonicity-preserving guess (Eq. 2.6).
  const Real d_minus = v0 - vm1;
  const Real d_plus  = vp1 - v0;
  const Real vmp = v0 + minmod2(d_plus, alpha * d_minus);

  // Cheap early accept: if the candidate already lies in [v0, vmp] (Eq. 2.30
  // condition), it is monotonicity preserving and no further work is needed.
  if ((vl - v0) * (vl - vmp) <= eps) {
    return vl;
  }

  // Second differences (curvature), Eq. 2.19.
  const Real dm = vm2 - Real{2} * vm1 + v0;   // d_{j-1}
  const Real d0 = vm1 - Real{2} * v0 + vp1;   // d_j
  const Real dp = v0 - Real{2} * vp1 + vp2;   // d_{j+1}

  // Curvature-limited differences (Eq. 2.27): 4-arg minmod blends.
  const Real d_m4_jph = minmod4(Real{4} * d0 - dp, Real{4} * dp - d0, d0, dp);
  const Real d_m4_jmh = minmod4(Real{4} * d0 - dm, Real{4} * dm - d0, d0, dm);

  // Candidate bounds (Eqs. 2.24-2.26).
  const Real v_ul = v0 + alpha * d_minus;                          // upper-limit
  const Real v_av = Real{0.5} * (v0 + vp1);                        // average
  const Real v_md = v_av - Real{0.5} * d_m4_jph;                   // median
  const Real v_lc = v0 + Real{0.5} * d_minus + (Real{4} / Real{3}) * d_m4_jmh;  // large-curv.

  // Allowed interval [vmin, vmax] (Eqs. 2.24a/b).
  Real vmin = fmax(fmin(fmin(v0, vp1), v_md),
                   fmin(fmin(v0, v_ul), v_lc));
  Real vmax = fmin(fmax(fmax(v0, vp1), v_md),
                   fmax(fmax(v0, v_ul), v_lc));

  // Clamp the candidate into the interval via the median construction (Eq. 2.23).
  return median3(vl, vmin, vmax);
}

// 5th-order LEFT-biased interface interpolation at the right face of cell v0.
// Point-value (Lagrange) interpolation: the stored cell values are POINT samples
// at the cell centers x_{i-2..i+2}, and this returns the interpolated point value
// at the face x_{i+1/2}. Finite-VOLUME (cell-average -> face) coefficients are
// only 2nd-order accurate when fed point samples; these point-value weights are
// the correct high-order interpolant for the point-value reconstruction the
// Riemann solver consumes. Coefficients (sum = 128): (3, -20, 90, 60, -5)/128.
QUASAR_HOST_DEVICE inline Real mp5_interp(Real vm2, Real vm1, Real v0,
                                          Real vp1, Real vp2) {
  return (Real{3} * vm2 - Real{20} * vm1 + Real{90} * v0
          + Real{60} * vp1 - Real{5} * vp2) / Real{128};
}

// 7th-order LEFT-biased interface interpolation at the right face of cell v0.
// Point-value (Lagrange) interpolation on the cell-center point samples
// x_{i-3..i+3}, evaluated at the face x_{i+1/2}. NOT finite-volume cell-average
// coefficients. Coefficients (sum = 1024): (-5, 42, -175, 700, 525, -70, 7)/1024.
QUASAR_HOST_DEVICE inline Real mp7_interp(Real vm3, Real vm2, Real vm1, Real v0,
                                          Real vp1, Real vp2, Real vp3) {
  return (-Real{5} * vm3 + Real{42} * vm2 - Real{175} * vm1 + Real{700} * v0
          + Real{525} * vp1 - Real{70} * vp2 + Real{7} * vp3) / Real{1024};
}

// MP5 reconstruction: left-biased interface value at the right face of v[0] from
// the 5-point stencil, with the MP monotonicity-preserving limiter applied.
QUASAR_HOST_DEVICE inline Real mp5_reconstruct(Real vm2, Real vm1, Real v0,
                                               Real vp1, Real vp2) {
  const Real vl = mp5_interp(vm2, vm1, v0, vp1, vp2);
  return mp_limit(vl, vm2, vm1, v0, vp1, vp2);
}

// MP7 reconstruction: 7th-order base interpolation + the same MP limiter applied
// on the central five points.
QUASAR_HOST_DEVICE inline Real mp7_reconstruct(Real vm3, Real vm2, Real vm1,
                                               Real v0, Real vp1, Real vp2,
                                               Real vp3) {
  const Real vl = mp7_interp(vm3, vm2, vm1, v0, vp1, vp2, vp3);
  return mp_limit(vl, vm2, vm1, v0, vp1, vp2);
}

}  // namespace quasar::numerics
