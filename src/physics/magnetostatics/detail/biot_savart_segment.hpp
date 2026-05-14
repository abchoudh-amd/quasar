#pragma once

// Per-segment Biot-Savart device kernel. Included only from .hip translation
// units; guarded so accidental inclusion from plain C++ TUs is a no-op.
#ifdef __HIPCC__

#include <hip/hip_runtime.h>

#include "quasar/core/types.hpp"

namespace quasar::magnetostatics::detail {

// Closed-form B-field contribution from one straight filamentary segment a->b
// carrying current I, evaluated at observation point p.
//
// Derivation (Biot-Savart along a straight segment, then in closed form):
//   B(p) = (mu0 I / 4 pi) * (L x ra) * integral_0^1 dt / |ra - t L|^3
//        = (mu0 I / 4 pi) * (L x ra) * (Ra + Rb)
//                                    / ( Ra * Rb * (Ra*Rb + ra . rb) )
// with ra = p - a, rb = p - b, L = b - a, Ra = |ra|, Rb = |rb|.
//
// The (Ra*Rb + ra.rb) denominator vanishes only when p lies on the segment
// itself (singular); we guard with kEps and return zero in that case.
__device__ __forceinline__ Vec3 segment_B(Vec3 a, Vec3 b, Vec3 p, Real I) {
  const Vec3 ra   = p - a;
  const Vec3 rb   = p - b;
  const Real Ra   = length(ra);
  const Real Rb   = length(rb);
  const Vec3 L    = b - a;
  const Real RaRb = Ra * Rb;
  const Real denom = RaRb * (RaRb + dot(ra, rb));
  if (denom < kEps) {
    return Vec3{Real{0}, Real{0}, Real{0}};
  }
  const Real coeff = mu0_over_4pi * I * (Ra + Rb) / denom;
  return coeff * cross(L, ra);
}

// Closed-form Jacobian of segment_B with respect to the observation point p.
// Returns the 3x3 matrix grad B with (grad B)_{ij} = d B_i / d p_j.
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
__device__ __forceinline__ Mat3x3 segment_gradB(Vec3 a, Vec3 b, Vec3 p, Real I) {
  const Vec3 ra    = p - a;
  const Vec3 rb    = p - b;
  const Real Ra    = length(ra);
  const Real Rb    = length(rb);
  const Vec3 L     = b - a;
  const Real RaRb  = Ra * Rb;
  const Real rarb  = dot(ra, rb);
  const Real D     = RaRb * (RaRb + rarb);
  if (D < kEps) {
    return Mat3x3{};
  }

  const Real s    = Ra + Rb;
  const Real f    = s / D;
  const Vec3 u    = cross(L, ra);

  const Real inv_Ra = Real{1} / Ra;
  const Real inv_Rb = Real{1} / Rb;

  const Vec3 grad_s     = ra * inv_Ra + rb * inv_Rb;
  const Vec3 grad_alpha = (Rb * inv_Ra) * ra + (Ra * inv_Rb) * rb;
  const Vec3 grad_beta  = ra + rb;
  const Real two_alpha_plus_beta = Real{2} * RaRb + rarb;
  const Vec3 grad_D     = grad_alpha * two_alpha_plus_beta + grad_beta * RaRb;

  const Real D2     = D * D;
  const Vec3 grad_f = (grad_s * D - grad_D * s) / D2;

  // Rows of [L]_x: ( L x e_x ) gives column 0; the matrix is antisymmetric.
  const Vec3 Lx_row0{Real{0}, -L.z,    L.y};
  const Vec3 Lx_row1{L.z,    Real{0}, -L.x};
  const Vec3 Lx_row2{-L.y,   L.x,    Real{0}};

  const Real coeff = mu0_over_4pi * I;

  return Mat3x3{
      coeff * (Lx_row0 * f + grad_f * u.x),
      coeff * (Lx_row1 * f + grad_f * u.y),
      coeff * (Lx_row2 * f + grad_f * u.z),
  };
}

}  // namespace quasar::magnetostatics::detail

#endif  // __HIPCC__
