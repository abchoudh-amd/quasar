#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"

#include "physics/magnetostatics/detail/biot_savart_segment.hpp"
#include "backend/hip/hip_check.hpp"
#include "backend/hip/magnetostatics/launch_params.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::magnetostatics::detail::kBlockSize;
using ::quasar::magnetostatics::detail::kTileSegments;
using ::quasar::magnetostatics::detail::segment_gradB;

// Output layout: G is (9, M) row-major. Component index c = 3*i + j encodes
// (grad B)_{ij} = dB_i / dp_j, and within each row threads write coalesced
// G[c * M + m].
template <class T>
__global__ void biot_savart_gradB_kernel(
    const T* __restrict__ ax,   const T* __restrict__ ay,   const T* __restrict__ az,
    const T* __restrict__ bx,   const T* __restrict__ by,   const T* __restrict__ bz,
    const T* __restrict__ I_,   int N,
    const T* __restrict__ px,   const T* __restrict__ py,   const T* __restrict__ pz,
    int M,
    T* __restrict__ G) {
  __shared__ T s_ax[kTileSegments];
  __shared__ T s_ay[kTileSegments];
  __shared__ T s_az[kTileSegments];
  __shared__ T s_bx[kTileSegments];
  __shared__ T s_by[kTileSegments];
  __shared__ T s_bz[kTileSegments];
  __shared__ T s_I [kTileSegments];

  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  ::quasar::Vec3T<T> p{T{0}, T{0}, T{0}};
  if (active) {
    p = ::quasar::Vec3T<T>{px[m], py[m], pz[m]};
  }

  T g[9] = {T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}};

  for (int base = 0; base < N; base += kTileSegments) {
    const int lim = (N - base < kTileSegments) ? (N - base) : kTileSegments;

    for (int t = static_cast<int>(threadIdx.x); t < lim;
         t += static_cast<int>(blockDim.x)) {
      const int gi = base + t;
      s_ax[t] = ax[gi];
      s_ay[t] = ay[gi];
      s_az[t] = az[gi];
      s_bx[t] = bx[gi];
      s_by[t] = by[gi];
      s_bz[t] = bz[gi];
      s_I [t] = I_[gi];
    }
    __syncthreads();

    if (active) {
      for (int s = 0; s < lim; ++s) {
        const ::quasar::Vec3T<T> a_seg{s_ax[s], s_ay[s], s_az[s]};
        const ::quasar::Vec3T<T> b_seg{s_bx[s], s_by[s], s_bz[s]};
        const ::quasar::Mat3x3T<T> dG = segment_gradB<T>(a_seg, b_seg, p, s_I[s]);
        g[0] += dG.r0.x;  g[1] += dG.r0.y;  g[2] += dG.r0.z;
        g[3] += dG.r1.x;  g[4] += dG.r1.y;  g[5] += dG.r1.z;
        g[6] += dG.r2.x;  g[7] += dG.r2.y;  g[8] += dG.r2.z;
      }
    }

    __syncthreads();
  }

  if (active) {
    #pragma unroll
    for (int c = 0; c < 9; ++c) {
      G[c * M + m] = g[c];
    }
  }
}

template <class T>
void launch_impl(
    const T* ax, const T* ay, const T* az,
    const T* bx, const T* by, const T* bz,
    const T* I_, int N,
    const T* px, const T* py, const T* pz, int M,
    T* G,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) return;
  const int block = kBlockSize;
  const int grid  = (M + block - 1) / block;
  biot_savart_gradB_kernel<T><<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, G);
}

}  // namespace

extern "C" void launch_biot_savart_gradB_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* G,
    ::quasar::backend::stream_t stream) {
  launch_impl<double>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, G,
                      static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_biot_savart_gradB_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* G,
    ::quasar::backend::stream_t stream) {
  launch_impl<float>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, G,
                     static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
