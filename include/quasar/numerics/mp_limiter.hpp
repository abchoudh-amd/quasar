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
// The evolved MHD state is a finite-volume cell average and the residual is a
// conservative difference of face fluxes.  Accordingly mp5_interp/mp7_interp
// use the finite-volume cell-average-to-face coefficients.  Point-sample
// Lagrange interpolation at x_{i+1/2} is not interchangeable here: when fed
// cell averages it leaves an O(dx^2) defect in the conservative residual even
// though a point-interpolation-only test can appear high order.
//
// All host-only `std::` math (`std::abs`, `std::min`, `std::max`) is replaced by
// the device-callable `fabs`/`fmin`/`fmax` so every helper compiles in both host
// and __device__ contexts.

#include "quasar/core/types.hpp"  // Real, QUASAR_HOST_DEVICE

#include <cmath>

namespace quasar::numerics {

// --- scalar limiter helpers --------------------------------------------------

QUASAR_HOST_DEVICE inline Real minmod2(Real a, Real b) {
  // Do not infer the sign from a*b: two same-sign, non-zero slopes can have a
  // product that underflows to zero, incorrectly flattening a perfectly valid
  // monotone profile.  Direct comparisons are also safe when the product would
  // overflow for very large slopes.
  if (a == Real{0} || b == Real{0}
      || ((a < Real{0}) != (b < Real{0}))) {
    return Real{0};
  }
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
  const Real candidate_from_cell = vl - v0;
  const Real candidate_from_mp   = vl - vmp;
  const bool candidate_is_between =
      (candidate_from_cell <= Real{0} && candidate_from_mp >= Real{0})
      || (candidate_from_cell >= Real{0} && candidate_from_mp <= Real{0});

  // Suresh--Huynh's small early-accept tolerance is dimensionless.  Applying
  // it directly to the dimensional product makes the limiter depend on the
  // arbitrary amplitude scale: a sufficiently small copy of a discontinuity
  // would always be accepted and retain its overshoot.  Normalize by the local
  // variation before forming the product; this also prevents overflow and
  // underflow in the acceptance test.
  const Real variation = fmax(
      fmax(fabs(d_minus), fabs(d_plus)),
      fmax(fabs(candidate_from_cell), fabs(candidate_from_mp)));
  const bool within_roundoff = variation > Real{0}
      && (fabs(candidate_from_cell) / variation)
             * (fabs(candidate_from_mp) / variation) <= eps;
  if (candidate_is_between || within_roundoff) {
    return vl;
  }

  // Second differences (curvature), Eq. 2.19.
  const Real dm = (vm2 - vm1) - (vm1 - v0);   // d_{j-1}
  const Real d0 = (vm1 - v0) - (v0 - vp1);    // d_j
  const Real dp = (v0 - vp1) - (vp1 - vp2);   // d_{j+1}

  // Curvature-limited differences (Eq. 2.27): 4-arg minmod blends.
  const Real curvature_scale = fmax(fabs(dm), fmax(fabs(d0), fabs(dp)));
  Real d_m4_jph = Real{0};
  Real d_m4_jmh = Real{0};
  if (curvature_scale > Real{0}) {
    const Real dms = dm / curvature_scale;
    const Real d0s = d0 / curvature_scale;
    const Real dps = dp / curvature_scale;
    d_m4_jph = curvature_scale
        * minmod4(Real{4} * d0s - dps, Real{4} * dps - d0s, d0s, dps);
    d_m4_jmh = curvature_scale
        * minmod4(Real{4} * d0s - dms, Real{4} * dms - d0s, d0s, dms);
  }

  // Candidate bounds (Eqs. 2.24-2.26).
  const Real v_ul = v0 + alpha * d_minus;                          // upper-limit
  const Real v_av = Real{0.5} * v0 + Real{0.5} * vp1;              // average
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

// 5th-order LEFT-biased interface reconstruction at the right face of cell v0
// from finite-volume cell averages. Coefficients (sum = 60):
// (2, -13, 47, 27, -3)/60.
QUASAR_HOST_DEVICE inline Real mp5_interp(Real vm2, Real vm1, Real v0,
                                          Real vp1, Real vp2) {
  const Real scale = fmax(fmax(fabs(vm2), fabs(vm1)),
                          fmax(fabs(v0), fmax(fabs(vp1), fabs(vp2))));
  if (scale == Real{0}) return Real{0};
  const Real sum = Real{2} * (vm2 / scale) - Real{13} * (vm1 / scale)
                 + Real{47} * (v0 / scale) + Real{27} * (vp1 / scale)
                 - Real{3} * (vp2 / scale);
  return (sum / Real{60}) * scale;
}

// 7th-order LEFT-biased interface reconstruction at the right face of cell v0
// from finite-volume cell averages. Coefficients (sum = 420):
// (-3, 25, -101, 319, 214, -38, 4)/420.
QUASAR_HOST_DEVICE inline Real mp7_interp(Real vm3, Real vm2, Real vm1, Real v0,
                                          Real vp1, Real vp2, Real vp3) {
  const Real scale = fmax(fmax(fmax(fabs(vm3), fabs(vm2)),
                               fmax(fabs(vm1), fabs(v0))),
                          fmax(fabs(vp1), fmax(fabs(vp2), fabs(vp3))));
  if (scale == Real{0}) return Real{0};
  const Real sum = -Real{3} * (vm3 / scale) + Real{25} * (vm2 / scale)
                 - Real{101} * (vm1 / scale) + Real{319} * (v0 / scale)
                 + Real{214} * (vp1 / scale) - Real{38} * (vp2 / scale)
                 + Real{4} * (vp3 / scale);
  return (sum / Real{420}) * scale;
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
