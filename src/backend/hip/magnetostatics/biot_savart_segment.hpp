#pragma once

// Per-segment Biot-Savart device kernel. Included only from .hip translation
// units; guarded so accidental inclusion from plain C++ TUs is a no-op.
#ifdef __HIPCC__

#include <hip/hip_runtime.h>

#include "quasar/core/types.hpp"

namespace quasar::magnetostatics::detail {

// Closed-form B-field contribution from one straight filamentary segment a->b
// carrying current I, evaluated at observation point p, templated on the
// precision T (`float` or `double`).
//
// Derivation (Biot-Savart along a straight segment, then in closed form):
//   B(p) = (mu0 I / 4 pi) * (L x ra) * integral_0^1 dt / |ra - t L|^3
//        = (mu0 I / 4 pi) * (L x ra) * (Ra + Rb)
//                                    / ( Ra * Rb * (Ra*Rb + ra . rb) )
// with ra = p - a, rb = p - b, L = b - a, Ra = |ra|, Rb = |rb|.
//
// The (Ra*Rb + ra.rb) factor of the denominator vanishes only when p lies on
// the segment line (the physical singularity). `denom = RaRb * (RaRb + ra.rb)`
// scales as length^4, so a fixed absolute cutoff is dimensionally wrong and (in
// fp32) lets near-line points through after catastrophic cancellation in the
// sum. Guard each factor on its own scale instead: RaRb (length^2) against an
// absolute floor for the endpoint-coincident case, and the cancellation-prone
// sum relative to its RaRb magnitude.
template <class T>
__device__ __forceinline__
Vec3T<T> segment_B(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p, T I) {
  const Vec3T<T> ra   = p - a;
  const Vec3T<T> rb   = p - b;
  const T        Ra   = length(ra);
  const T        Rb   = length(rb);
  const Vec3T<T> L    = b - a;
  const T        RaRb = Ra * Rb;
  const T        sum  = RaRb + dot(ra, rb);  // >= 0 by Cauchy-Schwarz
  if (RaRb < kEps_v<T> || sum < kRelEps_v<T> * RaRb) {
    return Vec3T<T>{T{0}, T{0}, T{0}};
  }
  const T coeff = mu0_over_4pi_v<T> * I * (Ra + Rb) / (RaRb * sum);
  return coeff * cross(L, ra);
}

// Closed-form magnetic vector potential A (Coulomb gauge) of one straight
// filamentary segment a->b carrying current I, evaluated at observation point p,
// templated on precision T. This is the A such that B = curl A reproduces
// segment_B.
//
// Derivation (line integral of the current element along the straight segment):
//   A(p) = (mu0 I / 4 pi) * t_hat * ln( (Ra + Rb + L) / (Ra + Rb - L) )
// with ra = p - a, rb = p - b, L = b - a, Ra = |ra|, Rb = |rb|,
// Lmag = |L|, and t_hat = L / Lmag the segment unit tangent.
//
// Singularity: by the triangle inequality Ra + Rb >= Lmag with equality only on
// the segment itself, so the denominator (Ra + Rb - Lmag) vanishes exactly where
// the filament sits (the physical log divergence). Guard the segment length on an
// absolute floor (degenerate zero-length segment) and the cancellation-prone
// (Ra + Rb - Lmag) relative to (Ra + Rb), mirroring the scale-aware guards in
// segment_B.
template <class T>
__device__ __forceinline__
Vec3T<T> segment_A(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p, T I) {
  const Vec3T<T> ra   = p - a;
  const Vec3T<T> rb   = p - b;
  const T        Ra   = length(ra);
  const T        Rb   = length(rb);
  const Vec3T<T> L    = b - a;
  const T        Lmag = length(L);
  const T        sum  = Ra + Rb;
  const T        den  = sum - Lmag;  // >= 0 by triangle inequality
  if (Lmag < kEps_v<T> || den < kRelEps_v<T> * sum) {
    return Vec3T<T>{T{0}, T{0}, T{0}};
  }
  // (mu0 I / 4 pi) * ln((sum + Lmag)/(sum - Lmag)) / Lmag, then * L gives t_hat.
  const T coeff = mu0_over_4pi_v<T> * I * log((sum + Lmag) / den) / Lmag;
  return coeff * L;
}

// Closed-form Jacobian of segment_B with respect to the observation point p,
// templated on precision T. Returns the 3x3 matrix grad B with
// (grad B)_{ij} = d B_i / d p_j.
//
// Let u(p) = L x ra and  f(p) = (Ra + Rb) / ( Ra * Rb * (Ra*Rb + ra.rb) ).
// Then  B = (mu0 I / 4 pi) * u * f  and
//
//   d B_i / d p_j  =  (mu0 I / 4 pi) * ( f * (d u_i / d p_j) + u_i * (d f / d p_j) )
//
// with (d u_i / d p_j) = (L x e_j)_i, i.e. the cross-product matrix [L]_x.
//
// f = s / D where s = Ra + Rb and D = alpha * (alpha + beta) with
// alpha = Ra*Rb, beta = ra.rb. Then
//   grad s     = ra / Ra + rb / Rb
//   grad alpha = (Rb/Ra) * ra + (Ra/Rb) * rb
//   grad beta  = ra + rb
//   grad D     = (2 alpha + beta) * grad alpha + alpha * grad beta
//   grad f     = ( grad s * D - s * grad D ) / D^2.
template <class T>
__device__ __forceinline__
Mat3x3T<T> segment_gradB(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p, T I) {
  const Vec3T<T> ra    = p - a;
  const Vec3T<T> rb    = p - b;
  const T        Ra    = length(ra);
  const T        Rb    = length(rb);
  const Vec3T<T> L     = b - a;
  const T        RaRb  = Ra * Rb;
  const T        rarb  = dot(ra, rb);
  const T        sum   = RaRb + rarb;  // >= 0 by Cauchy-Schwarz
  // Geometry-scaled singularity guard, matching segment_B: RaRb (length^2)
  // against an absolute floor, and the cancellation-prone sum relative to RaRb.
  if (RaRb < kEps_v<T> || sum < kRelEps_v<T> * RaRb) {
    return Mat3x3T<T>{};
  }
  const T        D     = RaRb * sum;

  const T        s      = Ra + Rb;
  const T        f      = s / D;
  const Vec3T<T> u      = cross(L, ra);

  const T inv_Ra = T{1} / Ra;
  const T inv_Rb = T{1} / Rb;

  const Vec3T<T> grad_s     = ra * inv_Ra + rb * inv_Rb;
  const Vec3T<T> grad_alpha = (Rb * inv_Ra) * ra + (Ra * inv_Rb) * rb;
  const Vec3T<T> grad_beta  = ra + rb;
  const T        two_alpha_plus_beta = T{2} * RaRb + rarb;
  const Vec3T<T> grad_D     = grad_alpha * two_alpha_plus_beta + grad_beta * RaRb;

  const T        D2     = D * D;
  const Vec3T<T> grad_f = (grad_s * D - grad_D * s) / D2;

  const Vec3T<T> Lx_row0{T{0}, -L.z,  L.y};
  const Vec3T<T> Lx_row1{L.z,   T{0}, -L.x};
  const Vec3T<T> Lx_row2{-L.y,  L.x,  T{0}};

  const T coeff = mu0_over_4pi_v<T> * I;

  return Mat3x3T<T>{
      coeff * (Lx_row0 * f + grad_f * u.x),
      coeff * (Lx_row1 * f + grad_f * u.y),
      coeff * (Lx_row2 * f + grad_f * u.z),
  };
}

}  // namespace quasar::magnetostatics::detail

#endif  // __HIPCC__
