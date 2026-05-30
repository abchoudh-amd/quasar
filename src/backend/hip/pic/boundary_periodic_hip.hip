#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void wrap_particles_kernel(quasar::Grid2D g, std::size_t n,
                                      double* x, double* y,
                                      double* x_prev, double* y_prev,
                                      unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  // Constant-time wrap. The same shift applied to x must also be applied to
  // x_prev so the charge-conserving deposit (which runs after this BC and forms
  // current from x_prev -> x) still sees a sub-cell displacement; otherwise a
  // seam-crossing particle records a ~whole-domain jump the deposit window
  // cannot span, injecting uncancelled current.
  const double sx = g.lx * floor((x[p] - g.origin_x) / g.lx);
  x[p] -= sx;
  x_prev[p] -= sx;
  const double sy = g.ly * floor((y[p] - g.origin_y) / g.ly);
  y[p] -= sy;
  y_prev[p] -= sy;
}

// Copies a single component's ghost layer on `side` from the opposite interior
// edge, reproducing periodic wrap. The Yee curl reads only along-axis
// neighbours, so corner ghosts are never used and are left untouched.
//   gh in [1, nghost]   line in [0, len)
__device__ inline void periodic_fill_component(const quasar::Grid2D& g, double* f,
                                               int side, int gh, int line) {
  switch (side) {
    case 0:  // x_lo: column -gh <- column nx-gh
      f[g.index(-gh, line)] = f[g.index(g.nx - gh, line)];
      break;
    case 1:  // x_hi: column nx-1+gh <- column gh-1
      f[g.index(g.nx - 1 + gh, line)] = f[g.index(gh - 1, line)];
      break;
    case 2:  // y_lo: row -gh <- row ny-gh
      f[g.index(line, -gh)] = f[g.index(line, g.ny - gh)];
      break;
    default:  // y_hi: row ny-1+gh <- row gh-1
      f[g.index(line, g.ny - 1 + gh)] = f[g.index(line, gh - 1)];
      break;
  }
}

__global__ void periodic_fields_kernel(quasar::Grid2D g, int side, int len,
                                       double* ex, double* ey, double* ez,
                                       double* bx, double* by, double* bz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // over nghost*len
  const int gh   = tid / len + 1;     // 1..nghost
  const int line = tid % len;         // 0..len-1
  if (gh > g.nghost || line >= len) return;
  periodic_fill_component(g, ex, side, gh, line);
  periodic_fill_component(g, ey, side, gh, line);
  periodic_fill_component(g, ez, side, gh, line);
  periodic_fill_component(g, bx, side, gh, line);
  periodic_fill_component(g, by, side, gh, line);
  periodic_fill_component(g, bz, side, gh, line);
}

}  // namespace

extern "C" void launch_pic_boundary_periodic_particles(const quasar::Grid2D& g,
                                                       quasar::pic::ParticleSpecies& s,
                                                       hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL(wrap_particles_kernel, grid, block, 0, stream,
                     g, s.size(), s.x(), s.y(), s.x_prev(), s.y_prev(), s.alive());
}

extern "C" void launch_pic_boundary_periodic_fields(const quasar::Grid2D& g,
                                                    quasar::YeeField2D<double>& f,
                                                    int side, hipStream_t stream) {
  if (g.nghost <= 0) return;
  const int len = (side == 0 || side == 1) ? g.ny : g.nx;
  const int total = g.nghost * len;
  if (total <= 0) return;
  const dim3 block(256);
  const dim3 grid((total + block.x - 1) / block.x);
  hipLaunchKernelGGL(periodic_fields_kernel, grid, block, 0, stream, g, side, len,
                     f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                     f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
}
