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
  if (i > g.nx || j > g.ny) return;

  // Faraday lives on the three magnetic sub-lattices, not on a cell-shaped
  // nx*ny rectangle.  The high B edges and the high/high Bz corner are physical
  // Yee nodes.  Ampere's high Ex/Ey faces read them, so leaving them stale breaks
  // both the curl identity and mixed/non-periodic boundary updates.
  const std::size_t k = g.index(i, j);
  if (i < g.nx && j <= g.ny) {
    bx[k] = quasar::numerics::detail::scaled_signed_sum(
        quasar::numerics::detail::scaled_value(bx[k]),
        quasar::numerics::detail::negate(
            quasar::numerics::ddy_staggered_bwd_increment_scaled<Order>(
                ez, g, i, j, dt)));
  }
  if (i <= g.nx && j < g.ny) {
    by[k] = quasar::numerics::detail::scaled_signed_sum(
        quasar::numerics::detail::scaled_value(by[k]),
        quasar::numerics::ddx_staggered_bwd_increment_scaled<Order>(
            ez, g, i, j, dt));
  }
  if (i <= g.nx && j <= g.ny) {
    bz[k] = quasar::numerics::detail::scaled_signed_sum(
        quasar::numerics::detail::scaled_value(bz[k]),
        quasar::numerics::detail::negate(
            quasar::numerics::ddx_staggered_bwd_increment_scaled<Order>(
                ey, g, i, j, dt)),
        quasar::numerics::ddy_staggered_bwd_increment_scaled<Order>(
            ex, g, i, j, dt));
  }
}

template <int Order>
void launch(quasar::Grid2D g, double* bx, double* by, double* bz,
            const double* ex, const double* ey, const double* ez, double dt,
            hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid((static_cast<unsigned int>(g.nx + 1) + block.x - 1) / block.x,
                  (static_cast<unsigned int>(g.ny + 1) + block.y - 1) / block.y);
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

namespace {
__global__ void copy_b_kernel(std::size_t n, const double* bx, const double* by,
                              const double* bz, double* old_bx, double* old_by,
                              double* old_bz) {
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  old_bx[k] = bx[k];
  old_by[k] = by[k];
  old_bz[k] = bz[k];
}
}  // namespace

extern "C" void launch_pic_copy_b(const quasar::Grid2D& g,
                                   const quasar::YeeField2D<quasar::Real>& f,
                                   quasar::BField2D<quasar::Real>& old_b,
                                   quasar_stream_t stream) {
  const std::size_t n = g.storage_size();
  if (n == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(n);
  hipLaunchKernelGGL(copy_b_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     n, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                     old_b.bx.device_ptr(), old_b.by.device_ptr(), old_b.bz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}
