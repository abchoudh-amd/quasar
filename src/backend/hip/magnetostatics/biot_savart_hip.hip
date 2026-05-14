#include "quasar/core/types.hpp"

#include "physics/magnetostatics/detail/biot_savart_segment.hpp"
#include "backend/hip/magnetostatics/launch_params.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::detail::kBlockSize;
using ::quasar::magnetostatics::detail::kTileSegments;
using ::quasar::magnetostatics::detail::segment_B;

__global__ void biot_savart_B_kernel(
    const Real* __restrict__ ax,   const Real* __restrict__ ay,   const Real* __restrict__ az,
    const Real* __restrict__ bx,   const Real* __restrict__ by,   const Real* __restrict__ bz,
    const Real* __restrict__ I_,   int N,
    const Real* __restrict__ px,   const Real* __restrict__ py,   const Real* __restrict__ pz,
    int M,
    Real* __restrict__ Bx,         Real* __restrict__ By,         Real* __restrict__ Bz) {
  __shared__ Real s_ax[kTileSegments];
  __shared__ Real s_ay[kTileSegments];
  __shared__ Real s_az[kTileSegments];
  __shared__ Real s_bx[kTileSegments];
  __shared__ Real s_by[kTileSegments];
  __shared__ Real s_bz[kTileSegments];
  __shared__ Real s_I [kTileSegments];

  const int m      = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                     + static_cast<int>(threadIdx.x);
  const bool active = (m < M);

  Vec3 p{Real{0}, Real{0}, Real{0}};
  if (active) {
    p = Vec3{px[m], py[m], pz[m]};
  }

  Real Bx_acc = Real{0};
  Real By_acc = Real{0};
  Real Bz_acc = Real{0};

  for (int base = 0; base < N; base += kTileSegments) {
    const int lim = (N - base < kTileSegments) ? (N - base) : kTileSegments;

    // Cooperative tile load: every thread in the block strides through the
    // tile so we never depend on blockDim.x being >= kTileSegments.
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
        const Vec3 a_seg{s_ax[s], s_ay[s], s_az[s]};
        const Vec3 b_seg{s_bx[s], s_by[s], s_bz[s]};
        const Vec3 dB = segment_B(a_seg, b_seg, p, s_I[s]);
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

}  // namespace

extern "C" void launch_biot_savart_B(
    const Real* ax, const Real* ay, const Real* az,
    const Real* bx, const Real* by, const Real* bz,
    const Real* I_, int N,
    const Real* px, const Real* py, const Real* pz, int M,
    Real* Bx, Real* By, Real* Bz,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) {
    return;
  }
  const int block = kBlockSize;
  const int grid  = (M + block - 1) / block;
  biot_savart_B_kernel<<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, Bx, By, Bz);
}
