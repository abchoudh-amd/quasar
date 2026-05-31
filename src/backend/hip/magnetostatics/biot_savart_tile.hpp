#pragma once

#include "quasar/core/types.hpp"

#include "physics/magnetostatics/detail/biot_savart_segment.hpp"

#include <hip/hip_runtime.h>

namespace quasar::magnetostatics::detail {

// Shared scaffold for the Biot-Savart device kernels (field and field-gradient).
// Both stream the conductor segments through a shared-memory tile and accumulate
// a per-observation-point quantity; they differ only in the accumulator and the
// per-segment body. This helper owns the tile load + cooperative copy + sync;
// the caller supplies `accum(a_seg, b_seg, I)`, invoked once per segment for the
// thread's own observation point `p` (captured by the functor).
template <class T, class SegFn>
__device__ inline void tiled_segment_reduce(
    const T* __restrict__ ax, const T* __restrict__ ay, const T* __restrict__ az,
    const T* __restrict__ bx, const T* __restrict__ by, const T* __restrict__ bz,
    const T* __restrict__ I_, int N, bool active, SegFn accum) {
  __shared__ T s_ax[kTileSegments];
  __shared__ T s_ay[kTileSegments];
  __shared__ T s_az[kTileSegments];
  __shared__ T s_bx[kTileSegments];
  __shared__ T s_by[kTileSegments];
  __shared__ T s_bz[kTileSegments];
  __shared__ T s_I [kTileSegments];

  for (int base = 0; base < N; base += kTileSegments) {
    const int lim = (N - base < kTileSegments) ? (N - base) : kTileSegments;

    for (int t = static_cast<int>(threadIdx.x); t < lim;
         t += static_cast<int>(blockDim.x)) {
      const int g = base + t;
      s_ax[t] = ax[g];
      s_ay[t] = ay[g];
      s_az[t] = az[g];
      s_bx[t] = bx[g];
      s_by[t] = by[g];
      s_bz[t] = bz[g];
      s_I [t] = I_[g];
    }
    __syncthreads();

    if (active) {
      for (int s = 0; s < lim; ++s) {
        accum(::quasar::Vec3T<T>{s_ax[s], s_ay[s], s_az[s]},
              ::quasar::Vec3T<T>{s_bx[s], s_by[s], s_bz[s]}, s_I[s]);
      }
    }

    __syncthreads();
  }
}

}  // namespace quasar::magnetostatics::detail
