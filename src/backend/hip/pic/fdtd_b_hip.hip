#include "quasar/core/grid.hpp"
#include "quasar/numerics/stencil.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

template <int Order>
__global__ void fdtd_b_kernel(quasar::Grid2D g,
                              double* __restrict__ bx, double* __restrict__ by,
                              double* __restrict__ bz,
                              const double* __restrict__ ex, const double* __restrict__ ey,
                              const double* __restrict__ ez,
                              double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const quasar::Vec3 c = quasar::numerics::curl_e_at<Order>(ex, ey, ez, g, i, j);
  bx[k] -= dt * c.x;
  by[k] -= dt * c.y;
  bz[k] -= dt * c.z;
}

template <int Order>
void launch(quasar::Grid2D g, double* bx, double* by, double* bz,
            const double* ex, const double* ey, const double* ez, double dt,
            hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL((fdtd_b_kernel<Order>), grid, block, 0, stream,
                     g, bx, by, bz, ex, ey, ez, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_b_order2(const quasar::Grid2D& g,
                                         quasar::Real* bx, quasar::Real* by, quasar::Real* bz,
                                         const quasar::Real* ex, const quasar::Real* ey,
                                         const quasar::Real* ez,
                                         quasar::Real dt, quasar_stream_t stream) {
  launch<2>(g, bx, by, bz, ex, ey, ez, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_fdtd_b_order4(const quasar::Grid2D& g,
                                         quasar::Real* bx, quasar::Real* by, quasar::Real* bz,
                                         const quasar::Real* ex, const quasar::Real* ey,
                                         const quasar::Real* ez,
                                         quasar::Real dt, quasar_stream_t stream) {
  launch<4>(g, bx, by, bz, ex, ey, ez, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
