#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/stencil.hpp"

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

// --------------------------------------------------------------------------
// One-sided characteristic-wall closure (replaces the mirror fill).
//
// Run AFTER the interior curl. The interior kernel updated the boundary row
// reading ghost cells; we correct it without ever filling ghosts:
//
//   correct_e: pin tangential E to 0 at the wall node (PEC Dirichlet). Normal E
//     uses only transverse derivatives, so the interior value is already right.
//     With tangential E = 0 on the wall plane the boundary Poynting flux is
//     identically zero, so the reflecting wall is exactly lossless.
//
//   correct_b: pin normal B to 0, and on the wall whose curl overran (the
//     backward B-curl overruns only the LOW wall), replace the normal-axis
//     derivative of each tangential-B component with a one-sided interior
//     difference. We apply the delta dt*(centered - one_sided) of just that
//     derivative; the transverse-derivative parts cancel. Because E is unchanged
//     between advance_b and this correction, recomputing the centered derivative
//     here reproduces exactly what the interior kernel used, so the delta is
//     exact regardless of the ghost contents.
// --------------------------------------------------------------------------
namespace {

namespace qn = quasar::numerics;

// correct_b for an x-face wall. tangential B = {by, bz}; normal B = bx.
template <int Order>
__global__ void wall_correct_b_x_kernel(quasar::Grid2D g, int side, double dt,
                                        double* bx, double* by, double* bz,
                                        const double* ex, const double* ey,
                                        const double* ez) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny) return;
  const int iw = (side == 0) ? 0 : g.nx - 1;
  bx[g.index(iw, j)] = 0.0;  // normal B pinned
  if (side != 0) return;     // only the low wall's backward curl overruns
  const std::size_t k = g.index(iw, j);
  // by uses curl.y = -d(ez)/dx ; bz uses curl.z including +d(ey)/dx.
  const double d_ez = qn::ddx_onesided_fwd1(ez, g, iw, j)
                    - qn::ddx_staggered_bwd<Order>(ez, g, iw, j);
  const double d_ey = qn::ddx_onesided_fwd1(ey, g, iw, j)
                    - qn::ddx_staggered_bwd<Order>(ey, g, iw, j);
  by[k] += dt * d_ez;
  bz[k] -= dt * d_ey;
}

// correct_b for a y-face wall. tangential B = {bx, bz}; normal B = by.
template <int Order>
__global__ void wall_correct_b_y_kernel(quasar::Grid2D g, int side, double dt,
                                        double* bx, double* by, double* bz,
                                        const double* ex, const double* ey,
                                        const double* ez) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx) return;
  const int jw = (side == 2) ? 0 : g.ny - 1;
  by[g.index(i, jw)] = 0.0;  // normal B pinned
  if (side != 2) return;     // only the low wall's backward curl overruns
  const std::size_t k = g.index(i, jw);
  // bx uses curl.x = +d(ez)/dy ; bz uses curl.z including -d(ex)/dy.
  const double d_ez = qn::ddy_onesided_fwd1(ez, g, i, jw)
                    - qn::ddy_staggered_bwd<Order>(ez, g, i, jw);
  const double d_ex = qn::ddy_onesided_fwd1(ex, g, i, jw)
                    - qn::ddy_staggered_bwd<Order>(ex, g, i, jw);
  bx[k] -= dt * d_ez;
  bz[k] += dt * d_ex;
}

__global__ void wall_correct_e_x_kernel(quasar::Grid2D g, int side,
                                        double* ey, double* ez) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny) return;
  const int iw = (side == 0) ? 0 : g.nx - 1;
  const std::size_t k = g.index(iw, j);
  ey[k] = 0.0;  // tangential E pinned (PEC Dirichlet)
  ez[k] = 0.0;
}

__global__ void wall_correct_e_y_kernel(quasar::Grid2D g, int side,
                                        double* ex, double* ez) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx) return;
  const int jw = (side == 2) ? 0 : g.ny - 1;
  const std::size_t k = g.index(i, jw);
  ex[k] = 0.0;  // tangential E pinned (PEC Dirichlet)
  ez[k] = 0.0;
}

}  // namespace

extern "C" void launch_pic_boundary_wall_correct_b_order2(const quasar::Grid2D& g,
                                                          quasar::YeeField2D<double>& f,
                                                          int side, double dt,
                                                          quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(256);
  if (side == 0 || side == 1) {
    if (g.ny <= 0) return;
    const dim3 grid((g.ny + block.x - 1) / block.x);
    hipLaunchKernelGGL((wall_correct_b_x_kernel<2>), grid, block, 0, hip_stream, g, side, dt,
                       f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                       f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr());
  } else {
    if (g.nx <= 0) return;
    const dim3 grid((g.nx + block.x - 1) / block.x);
    hipLaunchKernelGGL((wall_correct_b_y_kernel<2>), grid, block, 0, hip_stream, g, side, dt,
                       f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                       f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr());
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_wall_correct_e(const quasar::Grid2D& g,
                                                   quasar::YeeField2D<double>& f,
                                                   int side, quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(256);
  if (side == 0 || side == 1) {
    if (g.ny <= 0) return;
    const dim3 grid((g.ny + block.x - 1) / block.x);
    hipLaunchKernelGGL(wall_correct_e_x_kernel, grid, block, 0, hip_stream, g, side,
                       f.ey.device_ptr(), f.ez.device_ptr());
  } else {
    if (g.nx <= 0) return;
    const dim3 grid((g.nx + block.x - 1) / block.x);
    hipLaunchKernelGGL(wall_correct_e_y_kernel, grid, block, 0, hip_stream, g, side,
                       f.ex.device_ptr(), f.ez.device_ptr());
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

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
