#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void wrap_particles_kernel(quasar::Grid2D g, std::size_t n,
                                      double* x, double* y, unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  while (x[p] < g.origin_x) x[p] += g.lx;
  while (x[p] >= g.origin_x + g.lx) x[p] -= g.lx;
  while (y[p] < g.origin_y) y[p] += g.ly;
  while (y[p] >= g.origin_y + g.ly) y[p] -= g.ly;
}

}  // namespace

extern "C" void launch_pic_boundary_periodic_particles(const quasar::Grid2D& g,
                                                       quasar::pic::ParticleSpecies& s,
                                                       hipStream_t stream) {
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL(wrap_particles_kernel, grid, block, 0, stream,
                     g, s.size(), s.x(), s.y(), s.alive());
}

extern "C" void launch_pic_boundary_periodic_fields(const quasar::Grid2D&,
                                                    quasar::YeeField2D<double>&,
                                                    int, hipStream_t) {}
