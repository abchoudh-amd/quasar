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

}  // namespace quasar::magnetostatics::detail

#endif  // __HIPCC__
