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
  const auto current_x = quasar::numerics::detail::scaled_product_value(dt, jx[k]);
  const auto current_y = quasar::numerics::detail::scaled_product_value(dt, jy[k]);
  const auto current_z = quasar::numerics::detail::scaled_product_value(dt, jz[k]);
  ex[k] = quasar::numerics::detail::scaled_signed_sum(
      quasar::numerics::detail::scaled_value(ex[k]),
      quasar::numerics::ddy_staggered_increment_scaled<Order>(
          bz, g, i, j, dt),
      quasar::numerics::detail::negate(current_x));
  ey[k] = quasar::numerics::detail::scaled_signed_sum(
      quasar::numerics::detail::scaled_value(ey[k]),
      quasar::numerics::detail::negate(
          quasar::numerics::ddx_staggered_increment_scaled<Order>(
              bz, g, i, j, dt)),
      quasar::numerics::detail::negate(current_y));
  ez[k] = quasar::numerics::detail::scaled_signed_sum(
      quasar::numerics::detail::scaled_value(ez[k]),
      quasar::numerics::ddx_staggered_increment_scaled<Order>(
          by, g, i, j, dt),
      quasar::numerics::detail::negate(
          quasar::numerics::ddy_staggered_increment_scaled<Order>(
              bx, g, i, j, dt)),
      quasar::numerics::detail::negate(current_z));
}

// The charge grid has nx*ny cells, so its forward divergence also reads the
// upper normal faces Ex(nx,j) and Ey(i,ny).  Those faces live in the first halo
// and are not covered by the cell-shaped launch above.  Advancing them with the
// same Ampere curl is essential at a non-periodic wall: otherwise deposited
// boundary current removes rho while the matching electric flux stays stale.
template <int Order>
__global__ void fdtd_e_high_x_face_kernel(
    quasar::Grid2D g, double* ex, const double* bz, const double* jx,
    double dt) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny) return;
  const int i = g.nx;
  const auto k = g.index(i, j);
  ex[k] = quasar::numerics::detail::scaled_signed_sum(
      quasar::numerics::detail::scaled_value(ex[k]),
      quasar::numerics::ddy_staggered_increment_scaled<Order>(
          bz, g, i, j, dt),
      quasar::numerics::detail::negate(
          quasar::numerics::detail::scaled_product_value(dt, jx[k])));
}

template <int Order>
__global__ void fdtd_e_high_y_face_kernel(
    quasar::Grid2D g, double* ey, const double* bz, const double* jy,
    double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx) return;
  const int j = g.ny;
  const auto k = g.index(i, j);
  ey[k] = quasar::numerics::detail::scaled_signed_sum(
      quasar::numerics::detail::scaled_value(ey[k]),
      quasar::numerics::detail::negate(
          quasar::numerics::ddx_staggered_increment_scaled<Order>(
              bz, g, i, j, dt)),
      quasar::numerics::detail::negate(
          quasar::numerics::detail::scaled_product_value(dt, jy[k])));
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
  const dim3 face_block(quasar::backend::pic::kLaunchBlock);
  const dim3 x_face_grid = quasar::backend::pic::grid_1d(
      static_cast<std::size_t>(g.ny));
  const dim3 y_face_grid = quasar::backend::pic::grid_1d(
      static_cast<std::size_t>(g.nx));
  hipLaunchKernelGGL((fdtd_e_high_x_face_kernel<Order>), x_face_grid,
                     face_block, 0, stream, g, ex, bz, jx, dt);
  hipLaunchKernelGGL((fdtd_e_high_y_face_kernel<Order>), y_face_grid,
                     face_block, 0, stream, g, ey, bz, jy, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_e_order2(const quasar::Grid2D& g,
                                         quasar::Real* ex, quasar::Real* ey, quasar::Real* ez,
                                         const quasar::Real* bx, const quasar::Real* by,
                                         const quasar::Real* bz,
                                         const quasar::Real* jx, const quasar::Real* jy,
                                         const quasar::Real* jz,
                                         quasar::Real dt, quasar_stream_t stream) {
  launch<2>(g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_fdtd_e_order4(const quasar::Grid2D& g,
                                         quasar::Real* ex, quasar::Real* ey, quasar::Real* ez,
                                         const quasar::Real* bx, const quasar::Real* by,
                                         const quasar::Real* bz,
                                         const quasar::Real* jx, const quasar::Real* jy,
                                         const quasar::Real* jz,
                                         quasar::Real dt, quasar_stream_t stream) {
  launch<4>(g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
