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
    T* __restrict__ G,
    int* __restrict__ status) {
  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  ::quasar::Vec3T<T> p{T{0}, T{0}, T{0}};
  if (active) {
    p = ::quasar::Vec3T<T>{px[m], py[m], pz[m]};
  }

  ScaledAccumulator<T> accumulators[9];
  T g[9] = {T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}, T{0}};
  bool singular_local = false;
  bool numeric_failure_local = false;
  tiled_segment_reduce<T>(ax, ay, az, bx, by, bz, I_, N, active,
      [&](const ::quasar::Vec3T<T>& a_seg, const ::quasar::Vec3T<T>& b_seg, T I) {
        const auto dG =
            segment_gradB<T>(a_seg, b_seg, p, I, singular_local,
                             numeric_failure_local);
        const decltype(dG.r0.x) values[9] = {
            dG.r0.x, dG.r0.y, dG.r0.z,
            dG.r1.x, dG.r1.y, dG.r1.z,
            dG.r2.x, dG.r2.y, dG.r2.z};
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
          accumulators[c].add(values[c], numeric_failure_local);
        }
      });

  if (active) {
    #pragma unroll
    for (int c = 0; c < 9; ++c) {
      g[c] = accumulators[c].value(numeric_failure_local);
      G[c * M + m] = g[c];
    }
    int status_local = singular_local ? 1 : 0;
    if (numeric_failure_local) status_local |= 2;
    for (int c = 0; c < 9; ++c) {
      if (!isfinite(g[c])) status_local |= 2;
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
    T* G, int* status,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) return;
  const int block = kBlockSize;
  const int grid  = 1 + (M - 1) / block;
  biot_savart_gradB_kernel<T><<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, G, status);
}

}  // namespace

// The f64/f32 entry points differ only in their element type, so generate the
// extern "C" pair from one macro instead of hand-copying the cast-and-forward
// body. Both forward to launch_impl<T> and check for a launch error.
#define QUASAR_BS_DEFINE_GRADB_LAUNCHER(NAME, T)                             \
  extern "C" void NAME(                                                      \
      const T* ax, const T* ay, const T* az,                                \
      const T* bx, const T* by, const T* bz,                                \
      const T* I_, int N,                                                   \
      const T* px, const T* py, const T* pz, int M,                         \
      T* G, int* status,                                                     \
      ::quasar::backend::stream_t stream) {                                  \
    launch_impl<T>(ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, G,         \
                   status, static_cast<hipStream_t>(stream));               \
    QUASAR_HIP_CHECK(::hipGetLastError());                                   \
  }

QUASAR_BS_DEFINE_GRADB_LAUNCHER(launch_biot_savart_gradB_f64, double)
QUASAR_BS_DEFINE_GRADB_LAUNCHER(launch_biot_savart_gradB_f32, float)

#undef QUASAR_BS_DEFINE_GRADB_LAUNCHER
