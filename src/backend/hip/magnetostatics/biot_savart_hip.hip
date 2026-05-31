#include "quasar/backend/device.hpp"
#include "quasar/backend/magnetostatics_kernels.hpp"
#include "quasar/core/types.hpp"

#include "physics/magnetostatics/detail/biot_savart_segment.hpp"
#include "backend/hip/hip_check.hpp"
#include "backend/hip/magnetostatics/launch_params.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::magnetostatics::detail::kBlockSize;
using ::quasar::magnetostatics::detail::kTileSegments;
using ::quasar::magnetostatics::detail::segment_B;

template <class T>
__global__ void biot_savart_B_kernel(
    const T* __restrict__ ax,   const T* __restrict__ ay,   const T* __restrict__ az,
    const T* __restrict__ bx,   const T* __restrict__ by,   const T* __restrict__ bz,
    const T* __restrict__ I_,   int N,
    const T* __restrict__ px,   const T* __restrict__ py,   const T* __restrict__ pz,
    int M,
    T* __restrict__ Bx,         T* __restrict__ By,         T* __restrict__ Bz) {
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

  T Bx_acc = T{0};
  T By_acc = T{0};
  T Bz_acc = T{0};

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
        const ::quasar::Vec3T<T> a_seg{s_ax[s], s_ay[s], s_az[s]};
        const ::quasar::Vec3T<T> b_seg{s_bx[s], s_by[s], s_bz[s]};
        const ::quasar::Vec3T<T> dB = segment_B<T>(a_seg, b_seg, p, s_I[s]);
        Bx_acc += dB.x;
        By_acc += dB.y;
        Bz_acc += dB.z;
      }
    }

    __syncthreads();
  }

  if (active) {
    Bx[m] = Bx_acc;
    By[m] = By_acc;
    Bz[m] = Bz_acc;
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
