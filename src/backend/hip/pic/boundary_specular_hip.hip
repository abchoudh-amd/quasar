#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void specular_kernel(quasar::Grid2D g, std::size_t n, int side,
                                double* x, double* y, double* vx, double* vy,
                                unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  if (side == 0 && x[p] < g.origin_x) {
    x[p] = 2.0 * g.origin_x - x[p];
    vx[p] = -vx[p];
  } else if (side == 1 && x[p] > g.origin_x + g.lx) {
    x[p] = 2.0 * (g.origin_x + g.lx) - x[p];
    vx[p] = -vx[p];
  } else if (side == 2 && y[p] < g.origin_y) {
    y[p] = 2.0 * g.origin_y - y[p];
    vy[p] = -vy[p];
  } else if (side == 3 && y[p] > g.origin_y + g.ly) {
    y[p] = 2.0 * (g.origin_y + g.ly) - y[p];
    vy[p] = -vy[p];
  }
}

}  // namespace

extern "C" void launch_pic_boundary_specular_particles(const quasar::Grid2D& g,
                                                       quasar::pic::ParticleSpecies& s,
                                                       int side, hipStream_t stream) {
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL(specular_kernel, grid, block, 0, stream,
                     g, s.size(), side, s.x(), s.y(), s.vx(), s.vy(), s.alive());
}
