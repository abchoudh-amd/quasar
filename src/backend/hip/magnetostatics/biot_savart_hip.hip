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
using ::quasar::magnetostatics::detail::segment_B;
using ::quasar::magnetostatics::detail::tiled_segment_reduce;

template <class T>
__global__ void biot_savart_B_kernel(
    const T* __restrict__ ax,   const T* __restrict__ ay,   const T* __restrict__ az,
    const T* __restrict__ bx,   const T* __restrict__ by,   const T* __restrict__ bz,
    const T* __restrict__ I_,   int N,
    const T* __restrict__ px,   const T* __restrict__ py,   const T* __restrict__ pz,
    int M,
    T* __restrict__ Bx,         T* __restrict__ By,         T* __restrict__ Bz) {
  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  ::quasar::Vec3T<T> p{T{0}, T{0}, T{0}};
  if (active) {
    p = ::quasar::Vec3T<T>{px[m], py[m], pz[m]};
  }

  ::quasar::Vec3T<T> B_acc{T{0}, T{0}, T{0}};
  tiled_segment_reduce<T>(ax, ay, az, bx, by, bz, I_, N, active,
      [&](const ::quasar::Vec3T<T>& a_seg, const ::quasar::Vec3T<T>& b_seg, T I) {
        const ::quasar::Vec3T<T> dB = segment_B<T>(a_seg, b_seg, p, I);
        B_acc.x += dB.x;
        B_acc.y += dB.y;
        B_acc.z += dB.z;
      });

  if (active) {
    Bx[m] = B_acc.x;
    By[m] = B_acc.y;
    Bz[m] = B_acc.z;
  }
}

template <class T>
void launch_impl(
    const T* ax, const T* ay, const T* az,
    const T* bx, const T* by, const T* bz,
    const T* I_, int N,
    const T* px, const T* py, const T* pz, int M,
    T* Bx, T* By, T* Bz,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) return;
  const int block = kBlockSize;
  const int grid  = (M + block - 1) / block;
  biot_savart_B_kernel<T><<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, Bx, By, Bz);
}

}  // namespace

extern "C" void launch_biot_savart_B_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* Bx, double* By, double* Bz,
    ::quasar::backend::stream_t stream) {
  launch_impl<double>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M,
                      Bx, By, Bz, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_biot_savart_B_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* Bx, float* By, float* Bz,
    ::quasar::backend::stream_t stream) {
  launch_impl<float>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M,
                     Bx, By, Bz, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
