#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

// Wraps only particles that crossed the requested periodic side. The same shift
// applied to the current position is applied to the previous position so the
// charge-conserving deposit (which runs after this BC and forms current from prev
// -> current) still sees a sub-cell displacement; otherwise a seam-crossing
// particle records a ~whole-domain jump the deposit window cannot span. Keeping
// the side in the predicate also lets a half-periodic axis coexist with an
// absorbing/specular wall on the opposite side.
__global__ void wrap_particles_kernel(quasar::Grid2D g, std::size_t n, int side,
                                      double* x, double* y,
                                      double* x_prev, double* y_prev,
                                      unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  if (side == 0 && x[p] < g.origin_x) {
    const double periods = floor(quasar::detail::scaled_difference_quotient(
        x[p], g.origin_x, g.lx));
    x[p] = fma(-periods, g.lx, x[p]);
    x_prev[p] = fma(-periods, g.lx, x_prev[p]);
  } else if (side == 1 && x[p] >= g.origin_x + g.lx) {
    const double periods = floor(quasar::detail::scaled_difference_quotient(
        x[p], g.origin_x, g.lx));
    x[p] = fma(-periods, g.lx, x[p]);
    x_prev[p] = fma(-periods, g.lx, x_prev[p]);
  } else if (side == 2 && y[p] < g.origin_y) {
    const double periods = floor(quasar::detail::scaled_difference_quotient(
        y[p], g.origin_y, g.ly));
    y[p] = fma(-periods, g.ly, y[p]);
    y_prev[p] = fma(-periods, g.ly, y_prev[p]);
  } else if (side == 3 && y[p] >= g.origin_y + g.ly) {
    const double periods = floor(quasar::detail::scaled_difference_quotient(
        y[p], g.origin_y, g.ly));
    y[p] = fma(-periods, g.ly, y[p]);
    y_prev[p] = fma(-periods, g.ly, y_prev[p]);
  }
}

// Copies a single component's ghost layer on `side` from the opposite interior
// edge, reproducing periodic wrap.  A face component's high sample at n is the
// periodic duplicate of sample 0, so this same mapping is correct for both
// centre and face lattices.  `line` spans the *whole padded transverse axis*:
// high-face Ampere updates and full-lattice Faraday updates can read corner
// halos on mixed boundaries, and leaving those corners untouched makes their
// value depend on allocator history.
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
  const int line = tid % len - g.nghost;
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
                                                       int side, quasar_stream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(s.size());
  hipLaunchKernelGGL(wrap_particles_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, s.size(), side, s.x(), s.y(), s.x_prev(), s.y_prev(), s.alive());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_periodic_fields(const quasar::Grid2D& g,
                                                    quasar::YeeField2D<quasar::Real>& f,
                                                    int side, quasar_stream_t stream) {
  if (g.nghost <= 0) return;
  const int len = (side == 0 || side == 1) ? g.height() : g.pitch();
  const int total = g.nghost * len;
  if (total <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(static_cast<std::size_t>(total));
  hipLaunchKernelGGL(periodic_fields_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, side, len,
                     f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                     f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}
