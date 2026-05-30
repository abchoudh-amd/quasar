#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void specular_kernel(quasar::Grid2D g, std::size_t n, int side,
                                double* x, double* y, double* x_prev, double* y_prev,
                                double* vx, double* vy, unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  // Reflect x_prev about the same wall as x so the charge-conserving deposit
  // (run after this BC, forming current from x_prev -> x) sees a consistent
  // sub-cell displacement on the reflected side of the wall.
  if (side == 0 && x[p] < g.origin_x) {
    x[p] = 2.0 * g.origin_x - x[p];
    x_prev[p] = 2.0 * g.origin_x - x_prev[p];
    vx[p] = -vx[p];
  } else if (side == 1 && x[p] > g.origin_x + g.lx) {
    const double w = g.origin_x + g.lx;
    x[p] = 2.0 * w - x[p];
    x_prev[p] = 2.0 * w - x_prev[p];
    vx[p] = -vx[p];
  } else if (side == 2 && y[p] < g.origin_y) {
    y[p] = 2.0 * g.origin_y - y[p];
    y_prev[p] = 2.0 * g.origin_y - y_prev[p];
    vy[p] = -vy[p];
  } else if (side == 3 && y[p] > g.origin_y + g.ly) {
    const double w = g.origin_y + g.ly;
    y[p] = 2.0 * w - y[p];
    y_prev[p] = 2.0 * w - y_prev[p];
    vy[p] = -vy[p];
  }
}

}  // namespace

extern "C" void launch_pic_boundary_specular_particles(const quasar::Grid2D& g,
                                                       quasar::pic::ParticleSpecies& s,
                                                       int side, quasar_stream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL(specular_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, s.size(), side, s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.alive());
  QUASAR_HIP_CHECK(::hipGetLastError());
}
