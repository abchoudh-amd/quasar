#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void absorb_kernel(quasar::Grid2D g, std::size_t n, int side,
                              const double* x, const double* y, unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  if ((side == 0 && x[p] < g.origin_x) ||
      (side == 1 && x[p] > g.origin_x + g.lx) ||
      (side == 2 && y[p] < g.origin_y) ||
      (side == 3 && y[p] > g.origin_y + g.ly)) {
    alive[p] = 0;
  }
}

}  // namespace

extern "C" void launch_pic_boundary_absorb_particles(const quasar::Grid2D& g,
                                                     quasar::pic::ParticleSpecies& s,
                                                     int side, hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL(absorb_kernel, grid, block, 0, stream,
                     g, s.size(), side, s.x(), s.y(), s.alive());
}
