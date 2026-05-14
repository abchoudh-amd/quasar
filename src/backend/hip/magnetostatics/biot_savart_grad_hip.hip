#include "quasar/core/types.hpp"

#include "physics/magnetostatics/detail/biot_savart_segment.hpp"
#include "backend/hip/magnetostatics/launch_params.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::Mat3x3;
using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::detail::kBlockSize;
using ::quasar::magnetostatics::detail::kTileSegments;
using ::quasar::magnetostatics::detail::segment_gradB;

// Output layout: G is (9, M) row-major. Component index c = 3*i + j encodes
// (grad B)_{ij} = dB_i / dp_j, and within each row threads write coalesced
// G[c * M + m].
__global__ void biot_savart_gradB_kernel(
    const Real* __restrict__ ax,   const Real* __restrict__ ay,   const Real* __restrict__ az,
    const Real* __restrict__ bx,   const Real* __restrict__ by,   const Real* __restrict__ bz,
    const Real* __restrict__ I_,   int N,
    const Real* __restrict__ px,   const Real* __restrict__ py,   const Real* __restrict__ pz,
    int M,
    Real* __restrict__ G) {
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

  Real g[9] = {Real{0}, Real{0}, Real{0}, Real{0}, Real{0},
               Real{0}, Real{0}, Real{0}, Real{0}};

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
        const Vec3 a_seg{s_ax[s], s_ay[s], s_az[s]};
        const Vec3 b_seg{s_bx[s], s_by[s], s_bz[s]};
        const Mat3x3 dG = segment_gradB(a_seg, b_seg, p, s_I[s]);
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

}  // namespace

extern "C" void launch_biot_savart_gradB(
    const Real* ax, const Real* ay, const Real* az,
    const Real* bx, const Real* by, const Real* bz,
    const Real* I_, int N,
    const Real* px, const Real* py, const Real* pz, int M,
    Real* G,
    hipStream_t stream) {
  if (N <= 0 || M <= 0) {
    return;
  }
  const int block = kBlockSize;
  const int grid  = (M + block - 1) / block;
  biot_savart_gradB_kernel<<<dim3(grid), dim3(block), 0, stream>>>(
      ax, ay, az, bx, by, bz, I_, N, px, py, pz, M, G);
}
