#include "quasar/backend/device.hpp"
#include "quasar/physics/magnetostatics/kernels.hpp"
#include "quasar/core/types.hpp"

#include "backend/hip/magnetostatics/biot_savart_segment.hpp"
#include "backend/hip/hip_check.hpp"
#include "backend/hip/magnetostatics/launch_params.hpp"
#include "backend/hip/magnetostatics/biot_savart_tile.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::magnetostatics::detail::kBlockSize;
using ::quasar::magnetostatics::detail::kTileSegments;
using ::quasar::magnetostatics::detail::ScaledAccumulator;
using ::quasar::magnetostatics::detail::segment_A;
using ::quasar::magnetostatics::detail::segment_B;
using ::quasar::magnetostatics::detail::tiled_segment_reduce;

template <class T>
__global__ void biot_savart_B_kernel(
    const T* __restrict__ ax,   const T* __restrict__ ay,   const T* __restrict__ az,
    const T* __restrict__ bx,   const T* __restrict__ by,   const T* __restrict__ bz,
    const T* __restrict__ I_,   int N,
    const T* __restrict__ px,   const T* __restrict__ py,   const T* __restrict__ pz,
    int M,
    T* __restrict__ Bx,         T* __restrict__ By,         T* __restrict__ Bz,
    int* __restrict__ status) {
  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  ::quasar::Vec3T<T> p{T{0}, T{0}, T{0}};
  if (active) {
    p = ::quasar::Vec3T<T>{px[m], py[m], pz[m]};
  }

  ScaledAccumulator<T> B_acc[3];
  bool singular_local = false;
  bool numeric_failure_local = false;
  tiled_segment_reduce<T>(ax, ay, az, bx, by, bz, I_, N, active,
      [&](const ::quasar::Vec3T<T>& a_seg, const ::quasar::Vec3T<T>& b_seg, T I) {
        const auto dB =
            segment_B<T>(a_seg, b_seg, p, I, singular_local,
                         numeric_failure_local);
        B_acc[0].add(dB.x, numeric_failure_local);
        B_acc[1].add(dB.y, numeric_failure_local);
        B_acc[2].add(dB.z, numeric_failure_local);
      });

  if (active) {
    const T bx = B_acc[0].value(numeric_failure_local);
    const T by = B_acc[1].value(numeric_failure_local);
    const T bz = B_acc[2].value(numeric_failure_local);
    Bx[m] = bx;
    By[m] = by;
    Bz[m] = bz;
    int status_local = singular_local ? 1 : 0;
    if (numeric_failure_local) status_local |= 2;
    if (!isfinite(bx) || !isfinite(by) || !isfinite(bz)) {
      status_local |= 2;
    }
    if (status_local != 0) atomicOr(status, status_local);
  }
}

// Vector-potential kernel: same tiled segment reduction as the B kernel, with
// segment_A as the per-segment body. The output SoA (Ax/Ay/Az) mirrors B so the
// host orchestrator and readback are shared.
template <class T>
__global__ void biot_savart_A_kernel(
    const T* __restrict__ ax,   const T* __restrict__ ay,   const T* __restrict__ az,
    const T* __restrict__ bx,   const T* __restrict__ by,   const T* __restrict__ bz,
    const T* __restrict__ I_,   int N,
    const T* __restrict__ px,   const T* __restrict__ py,   const T* __restrict__ pz,
    int M,
    T* __restrict__ Ax,         T* __restrict__ Ay,         T* __restrict__ Az,
    int* __restrict__ status) {
  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  ::quasar::Vec3T<T> p{T{0}, T{0}, T{0}};
  if (active) {
    p = ::quasar::Vec3T<T>{px[m], py[m], pz[m]};
  }

  ScaledAccumulator<T> A_acc[3];
  bool singular_local = false;
  bool numeric_failure_local = false;
  tiled_segment_reduce<T>(ax, ay, az, bx, by, bz, I_, N, active,
      [&](const ::quasar::Vec3T<T>& a_seg, const ::quasar::Vec3T<T>& b_seg, T I) {
        const auto dA =
            segment_A<T>(a_seg, b_seg, p, I, singular_local,
                         numeric_failure_local);
        A_acc[0].add(dA.x, numeric_failure_local);
        A_acc[1].add(dA.y, numeric_failure_local);
        A_acc[2].add(dA.z, numeric_failure_local);
      });

  if (active) {
    const T ax_value = A_acc[0].value(numeric_failure_local);
    const T ay_value = A_acc[1].value(numeric_failure_local);
    const T az_value = A_acc[2].value(numeric_failure_local);
    Ax[m] = ax_value;
    Ay[m] = ay_value;
    Az[m] = az_value;
    int status_local = singular_local ? 1 : 0;
    if (numeric_failure_local) status_local |= 2;
    if (!isfinite(ax_value) || !isfinite(ay_value) || !isfinite(az_value)) {
      status_local |= 2;
    }
    if (status_local != 0) atomicOr(status, status_local);
  }
}

template <class T>
void launch_impl(
    const T* ax, const T* ay, const T* az,
    const T* bx, const T* by, const T* bz,
    const T* I_, int N,
    const T* px, const T* py, const T* pz, int M,
    T* Bx, T* By, T* Bz, int* status,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) return;
  const int block = kBlockSize;
  const int grid  = 1 + (M - 1) / block;
  biot_savart_B_kernel<T><<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, Bx, By, Bz, status);
}

template <class T>
void launch_impl_A(
    const T* ax, const T* ay, const T* az,
    const T* bx, const T* by, const T* bz,
    const T* I_, int N,
    const T* px, const T* py, const T* pz, int M,
    T* Ax, T* Ay, T* Az, int* status,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) return;
  const int block = kBlockSize;
  const int grid  = 1 + (M - 1) / block;
  biot_savart_A_kernel<T><<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, Ax, Ay, Az, status);
}

}  // namespace

// The f64/f32 entry points differ only in their element type, so generate the
// extern "C" pair from one macro instead of hand-copying the cast-and-forward
// body. Both forward to launch_impl<T> and check for a launch error.
#define QUASAR_BS_DEFINE_B_LAUNCHER(NAME, T)                                  \
  extern "C" void NAME(                                                       \
      const T* ax, const T* ay, const T* az,                                 \
      const T* bx, const T* by, const T* bz,                                 \
      const T* I_, int N,                                                    \
      const T* px, const T* py, const T* pz, int M,                          \
      T* Bx, T* By, T* Bz, int* status,                                       \
      ::quasar::backend::stream_t stream) {                                   \
    launch_impl<T>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M,             \
                   Bx, By, Bz, status, static_cast<hipStream_t>(stream));    \
    QUASAR_HIP_CHECK(::hipGetLastError());                                    \
  }

QUASAR_BS_DEFINE_B_LAUNCHER(launch_biot_savart_B_f64, double)
QUASAR_BS_DEFINE_B_LAUNCHER(launch_biot_savart_B_f32, float)

#undef QUASAR_BS_DEFINE_B_LAUNCHER

// Vector-potential launchers, generated the same way (output SoA Ax/Ay/Az).
#define QUASAR_BS_DEFINE_A_LAUNCHER(NAME, T)                                  \
  extern "C" void NAME(                                                       \
      const T* ax, const T* ay, const T* az,                                 \
      const T* bx, const T* by, const T* bz,                                 \
      const T* I_, int N,                                                    \
      const T* px, const T* py, const T* pz, int M,                          \
      T* Ax, T* Ay, T* Az, int* status,                                       \
      ::quasar::backend::stream_t stream) {                                   \
    launch_impl_A<T>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M,           \
                     Ax, Ay, Az, status, static_cast<hipStream_t>(stream));  \
    QUASAR_HIP_CHECK(::hipGetLastError());                                    \
  }

QUASAR_BS_DEFINE_A_LAUNCHER(launch_biot_savart_A_f64, double)
QUASAR_BS_DEFINE_A_LAUNCHER(launch_biot_savart_A_f32, float)

#undef QUASAR_BS_DEFINE_A_LAUNCHER
