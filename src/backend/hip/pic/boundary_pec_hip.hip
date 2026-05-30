#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

// Perfect-electric-conductor (PEC) wall as a ghost-cell fill.
//
// A PEC wall forces the tangential electric field and the normal magnetic field
// to vanish at the conductor surface. On the Yee mesh this is imposed by image
// (mirror) symmetry across the boundary plane: the tangential-E / normal-B
// components are reflected with a sign flip (odd), and the remaining components
// are reflected unchanged (even). After this fill the standard staggered curl
// reproduces a reflecting wall with no normal energy flux.
//
// For a wall on an x-face (sides 0/1): tangential E = {Ey, Ez}, normal B = Bx
// are odd; Ex, By, Bz are even. For a y-face (sides 2/3): tangential E =
// {Ex, Ez}, normal B = By are odd; Ey, Bx, Bz are even.

namespace {

// The Yee fields here are node-collocated (the curl uses forward differences, so
// the conductor surface sits ON the boundary interior node, not on a half-cell
// face). The mirror is therefore node-centred: ghost layer gh reflects the
// interior node gh cells inside the wall node. Odd components (tangential E,
// normal B) flip sign and are pinned to zero at the wall node itself (Dirichlet);
// even components reflect unchanged. A face-centred mirror (reflecting nx-gh
// instead of nx-1-gh) shifts the wall by half a cell relative to the field
// locations and drives an exponential boundary instability.
__device__ inline void mirror_x(const quasar::Grid2D& g, double* f, int side,
                                int gh, int j, double sign) {
  if (side == 0) {  // x_lo: wall node i=0
    f[g.index(-gh, j)] = sign * f[g.index(gh, j)];
    if (sign < 0.0 && gh == 1) f[g.index(0, j)] = 0.0;
  } else {          // x_hi: wall node i=nx-1
    f[g.index(g.nx - 1 + gh, j)] = sign * f[g.index(g.nx - 1 - gh, j)];
    if (sign < 0.0 && gh == 1) f[g.index(g.nx - 1, j)] = 0.0;
  }
}

__device__ inline void mirror_y(const quasar::Grid2D& g, double* f, int side,
                                int gh, int i, double sign) {
  if (side == 2) {  // y_lo: wall node j=0
    f[g.index(i, -gh)] = sign * f[g.index(i, gh)];
    if (sign < 0.0 && gh == 1) f[g.index(i, 0)] = 0.0;
  } else {          // y_hi: wall node j=ny-1
    f[g.index(i, g.ny - 1 + gh)] = sign * f[g.index(i, g.ny - 1 - gh)];
    if (sign < 0.0 && gh == 1) f[g.index(i, g.ny - 1)] = 0.0;
  }
}

__global__ void pec_x_kernel(quasar::Grid2D g, int side,
                             double* ex, double* ey, double* ez,
                             double* bx, double* by, double* bz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gh = tid / g.ny + 1;
  const int j  = tid % g.ny;
  if (gh > g.nghost) return;
  mirror_x(g, ex, side, gh, j, +1.0);  // normal E: even
  mirror_x(g, ey, side, gh, j, -1.0);  // tangential E: odd
  mirror_x(g, ez, side, gh, j, -1.0);  // tangential E: odd
  mirror_x(g, bx, side, gh, j, -1.0);  // normal B: odd
  mirror_x(g, by, side, gh, j, +1.0);
  mirror_x(g, bz, side, gh, j, +1.0);
}

__global__ void pec_y_kernel(quasar::Grid2D g, int side,
                             double* ex, double* ey, double* ez,
                             double* bx, double* by, double* bz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gh = tid / g.nx + 1;
  const int i  = tid % g.nx;
  if (gh > g.nghost) return;
  mirror_y(g, ex, side, gh, i, -1.0);  // tangential E: odd
  mirror_y(g, ey, side, gh, i, +1.0);  // normal E: even
  mirror_y(g, ez, side, gh, i, -1.0);  // tangential E: odd
  mirror_y(g, bx, side, gh, i, +1.0);
  mirror_y(g, by, side, gh, i, -1.0);  // normal B: odd
  mirror_y(g, bz, side, gh, i, +1.0);
}

}  // namespace

extern "C" void launch_pic_boundary_pec_fields(const quasar::Grid2D& g,
                                               quasar::YeeField2D<double>& f,
                                               int side, quasar_stream_t stream) {
  if (g.nghost <= 0) return;
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(256);
  if (side == 0 || side == 1) {
    const int total = g.nghost * g.ny;
    if (total <= 0) return;
    const dim3 grid((total + block.x - 1) / block.x);
    hipLaunchKernelGGL(pec_x_kernel, grid, block, 0, hip_stream, g, side,
                       f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                       f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
  } else {
    const int total = g.nghost * g.nx;
    if (total <= 0) return;
    const dim3 grid((total + block.x - 1) / block.x);
    hipLaunchKernelGGL(pec_y_kernel, grid, block, 0, hip_stream, g, side,
                       f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                       f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}
