#include "quasar/core/grid.hpp"
#include "quasar/numerics/stencil.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

template <int Order>
__global__ void fdtd_e_kernel(quasar::Grid2D g,
                              double* __restrict__ ex, double* __restrict__ ey,
                              double* __restrict__ ez,
                              const double* __restrict__ bx, const double* __restrict__ by,
                              const double* __restrict__ bz,
                              const double* __restrict__ jx, const double* __restrict__ jy,
                              const double* __restrict__ jz,
                              double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const quasar::Vec3 c = quasar::numerics::curl_b_at<Order>(bx, by, bz, g, i, j);
  ex[k] += dt * (c.x - jx[k]);
  ey[k] += dt * (c.y - jy[k]);
  ez[k] += dt * (c.z - jz[k]);
}

template <int Order>
void launch(quasar::Grid2D g, double* ex, double* ey, double* ez,
            const double* bx, const double* by, const double* bz,
            const double* jx, const double* jy, const double* jz,
            double dt, hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL((fdtd_e_kernel<Order>), grid, block, 0, stream,
                     g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_e_order2(const quasar::Grid2D& g,
                                         double* ex, double* ey, double* ez,
                                         const double* bx, const double* by, const double* bz,
                                         const double* jx, const double* jy, const double* jz,
                                         double dt, quasar_stream_t stream) {
  launch<2>(g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_fdtd_e_order4(const quasar::Grid2D& g,
                                         double* ex, double* ey, double* ez,
                                         const double* bx, const double* by, const double* bz,
                                         const double* jx, const double* jy, const double* jz,
                                         double dt, quasar_stream_t stream) {
  launch<4>(g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
