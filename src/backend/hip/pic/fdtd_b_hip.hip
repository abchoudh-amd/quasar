#include "quasar/core/grid.hpp"
#include "quasar/numerics/stencil.hpp"

#include "backend/hip/pic/launch.hpp"

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
  const dim3 block(16, 16);
  const dim3 grid((g.nx + block.x - 1) / block.x,
                  (g.ny + block.y - 1) / block.y);
  hipLaunchKernelGGL((fdtd_b_kernel<Order>), grid, block, 0, stream,
                     g, bx, by, bz, ex, ey, ez, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_b_order2(const quasar::Grid2D& g,
                                         double* bx, double* by, double* bz,
                                         const double* ex, const double* ey, const double* ez,
                                         double dt, hipStream_t stream) {
  launch<2>(g, bx, by, bz, ex, ey, ez, dt, stream);
}

extern "C" void launch_pic_fdtd_b_order4(const quasar::Grid2D& g,
                                         double* bx, double* by, double* bz,
                                         const double* ex, const double* ey, const double* ez,
                                         double dt, hipStream_t stream) {
  launch<4>(g, bx, by, bz, ex, ey, ez, dt, stream);
}
