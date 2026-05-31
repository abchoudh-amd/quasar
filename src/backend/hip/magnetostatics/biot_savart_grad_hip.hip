#include "quasar/backend/device.hpp"
#include "quasar/backend/magnetostatics_kernels.hpp"
#include "quasar/core/types.hpp"

#include "physics/magnetostatics/detail/biot_savart_segment.hpp"
#include "backend/hip/hip_check.hpp"
#include "backend/hip/magnetostatics/launch_params.hpp"
#include "backend/hip/magnetostatics/biot_savart_tile.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::magnetostatics::detail::kBlockSize;
using ::quasar::magnetostatics::detail::kTileSegments;
using ::quasar::magnetostatics::detail::segment_gradB;
using ::quasar::magnetostatics::detail::tiled_segment_reduce;

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
  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  ::quasar::Vec3T<T> p{T{0}, T{0}, T{0}};
  if (active) {
    p = ::quasar::Vec3T<T>{px[m], py[m], pz[m]};
  }

  T g[9] = {T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}};
  tiled_segment_reduce<T>(ax, ay, az, bx, by, bz, I_, N, active,
      [&](const ::quasar::Vec3T<T>& a_seg, const ::quasar::Vec3T<T>& b_seg, T I) {
        const ::quasar::Mat3x3T<T> dG = segment_gradB<T>(a_seg, b_seg, p, I);
        g[0] += dG.r0.x;  g[1] += dG.r0.y;  g[2] += dG.r0.z;
        g[3] += dG.r1.x;  g[4] += dG.r1.y;  g[5] += dG.r1.z;
        g[6] += dG.r2.x;  g[7] += dG.r2.y;  g[8] += dG.r2.z;
      });

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
