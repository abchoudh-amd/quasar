#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/stencil.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Perfect-electric-conductor (PEC) reflecting wall as a one-sided closure.
//
// A PEC wall forces tangential E and normal B to vanish at the conductor
// surface. Rather than fill mirror ghost cells before the curl, the boundary
// nodes are corrected AFTER each interior curl (see the closure block below),
// so no ghost halo participates in the field BC. The Yee fields are
// node-collocated: the wall sits ON the boundary interior node.
//
// For a wall on an x-face (sides 0/1): tangential E = {Ey, Ez}, normal B = Bx.
// For a y-face (sides 2/3): tangential E = {Ex, Ez}, normal B = By.

// --------------------------------------------------------------------------
// One-sided characteristic-wall closure.
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

// --- 4th-order closures -----------------------------------------------------
// The order-4 staggered curls reach two cells past a wall, so the outer boundary
// layers are reduced to an interior-only stencil on the normal-axis derivative.
// The backward B-curl overruns the LOW wall at nodes {0,1} and the HIGH wall at
// node {N-1}; the forward E-curl overruns the HIGH wall's second interior layer
// {N-2} (the wall node itself is pinned). Each fix applies the delta
// dt*(interior_stencil - order4_stencil) of just that derivative; the order-4
// term reproduces what the interior kernel used (the field is unchanged since the
// curl), so the ghost contribution cancels exactly.

// Replace the x-derivative of tangential B (by,bz) at node (i,j) using `new_*`.
__device__ inline void b_x_delta(const quasar::Grid2D& g, double dt,
                                 double* by, double* bz, const double* ey,
                                 const double* ez, int i, int j,
                                 double new_ez, double new_ey) {
  const std::size_t k = g.index(i, j);
  by[k] += dt * (new_ez - qn::ddx_staggered_bwd<4>(ez, g, i, j));
  bz[k] -= dt * (new_ey - qn::ddx_staggered_bwd<4>(ey, g, i, j));
}

__device__ inline void b_y_delta(const quasar::Grid2D& g, double dt,
                                 double* bx, double* bz, const double* ex,
                                 const double* ez, int i, int j,
                                 double new_ez, double new_ex) {
  const std::size_t k = g.index(i, j);
  bx[k] -= dt * (new_ez - qn::ddy_staggered_bwd<4>(ez, g, i, j));
  bz[k] += dt * (new_ex - qn::ddy_staggered_bwd<4>(ex, g, i, j));
}

__global__ void wall_correct_b_x_kernel_o4(quasar::Grid2D g, int side, double dt,
                                           double* bx, double* by, double* bz,
                                           const double* ex, const double* ey,
                                           const double* ez) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny) return;
  if (side == 0) {  // x_lo: backward curl overruns nodes 0 and 1
    bx[g.index(0, j)] = 0.0;
    b_x_delta(g, dt, by, bz, ey, ez, 0, j,
              qn::ddx_onesided_fwd1(ez, g, 0, j),
              qn::ddx_onesided_fwd1(ey, g, 0, j));
    b_x_delta(g, dt, by, bz, ey, ez, 1, j,
              qn::ddx_staggered_bwd<2>(ez, g, 1, j),
              qn::ddx_staggered_bwd<2>(ey, g, 1, j));
  } else {          // x_hi: overruns node nx-1
    const int iw = g.nx - 1;
    bx[g.index(iw, j)] = 0.0;
    b_x_delta(g, dt, by, bz, ey, ez, iw, j,
              qn::ddx_staggered_bwd<2>(ez, g, iw, j),
              qn::ddx_staggered_bwd<2>(ey, g, iw, j));
  }
}

__global__ void wall_correct_b_y_kernel_o4(quasar::Grid2D g, int side, double dt,
                                           double* bx, double* by, double* bz,
                                           const double* ex, const double* ey,
                                           const double* ez) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx) return;
  if (side == 2) {  // y_lo: overruns nodes 0 and 1
    by[g.index(i, 0)] = 0.0;
    b_y_delta(g, dt, bx, bz, ex, ez, i, 0,
              qn::ddy_onesided_fwd1(ez, g, i, 0),
              qn::ddy_onesided_fwd1(ex, g, i, 0));
    b_y_delta(g, dt, bx, bz, ex, ez, i, 1,
              qn::ddy_staggered_bwd<2>(ez, g, i, 1),
              qn::ddy_staggered_bwd<2>(ex, g, i, 1));
  } else {          // y_hi: overruns node ny-1
    const int jw = g.ny - 1;
    by[g.index(i, jw)] = 0.0;
    b_y_delta(g, dt, bx, bz, ex, ez, i, jw,
              qn::ddy_staggered_bwd<2>(ez, g, i, jw),
              qn::ddy_staggered_bwd<2>(ex, g, i, jw));
  }
}

// E forward curl overruns only the HIGH wall's second interior layer (the wall
// node is pinned). Low walls need only the pin.
__global__ void wall_correct_e_x_kernel_o4(quasar::Grid2D g, int side, double dt,
                                           double* ey, double* ez,
                                           const double* bx, const double* by,
                                           const double* bz) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny) return;
  if (side == 0) {
    ey[g.index(0, j)] = 0.0;
    ez[g.index(0, j)] = 0.0;
  } else {
    const int iw = g.nx - 1;
    ey[g.index(iw, j)] = 0.0;
    ez[g.index(iw, j)] = 0.0;
    const int i2 = g.nx - 2;  // second layer overran by the forward curl
    const std::size_t k = g.index(i2, j);
    const double dby = qn::ddx_staggered<2>(by, g, i2, j)
                     - qn::ddx_staggered<4>(by, g, i2, j);
    const double dbz = qn::ddx_staggered<2>(bz, g, i2, j)
                     - qn::ddx_staggered<4>(bz, g, i2, j);
    ey[k] -= dt * dbz;  // ey += dt*(-d_bz_dx)
    ez[k] += dt * dby;  // ez += dt*( d_by_dx)
  }
}

__global__ void wall_correct_e_y_kernel_o4(quasar::Grid2D g, int side, double dt,
                                           double* ex, double* ez,
                                           const double* bx, const double* by,
                                           const double* bz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx) return;
  if (side == 2) {
    ex[g.index(i, 0)] = 0.0;
    ez[g.index(i, 0)] = 0.0;
  } else {
    const int jw = g.ny - 1;
    ex[g.index(i, jw)] = 0.0;
    ez[g.index(i, jw)] = 0.0;
    const int j2 = g.ny - 2;
    const std::size_t k = g.index(i, j2);
    const double dbx = qn::ddy_staggered<2>(bx, g, i, j2)
                     - qn::ddy_staggered<4>(bx, g, i, j2);
    const double dbz = qn::ddy_staggered<2>(bz, g, i, j2)
                     - qn::ddy_staggered<4>(bz, g, i, j2);
    ex[k] += dt * dbz;  // ex += dt*( d_bz_dy)
    ez[k] -= dt * dbx;  // ez += dt*(-d_bx_dy)
  }
}

}  // namespace

extern "C" void launch_pic_boundary_wall_correct_b_order2(const quasar::Grid2D& g,
                                                          quasar::YeeField2D<double>& f,
                                                          int side, double dt,
                                                          quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  quasar::backend::pic::launch_along_side(g, side,
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL((wall_correct_b_x_kernel<2>), grid, block, 0, hip_stream, g, side, dt,
                         f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                         f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr());
    },
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL((wall_correct_b_y_kernel<2>), grid, block, 0, hip_stream, g, side, dt,
                         f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                         f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr());
    });
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_wall_correct_e(const quasar::Grid2D& g,
                                                   quasar::YeeField2D<double>& f,
                                                   int side, quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  quasar::backend::pic::launch_along_side(g, side,
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL(wall_correct_e_x_kernel, grid, block, 0, hip_stream, g, side,
                         f.ey.device_ptr(), f.ez.device_ptr());
    },
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL(wall_correct_e_y_kernel, grid, block, 0, hip_stream, g, side,
                         f.ex.device_ptr(), f.ez.device_ptr());
    });
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_wall_correct_b_order4(const quasar::Grid2D& g,
                                                          quasar::YeeField2D<double>& f,
                                                          int side, double dt,
                                                          quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  quasar::backend::pic::launch_along_side(g, side,
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL(wall_correct_b_x_kernel_o4, grid, block, 0, hip_stream, g, side, dt,
                         f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                         f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr());
    },
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL(wall_correct_b_y_kernel_o4, grid, block, 0, hip_stream, g, side, dt,
                         f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                         f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr());
    });
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_wall_correct_e_order4(const quasar::Grid2D& g,
                                                          quasar::YeeField2D<double>& f,
                                                          int side, double dt,
                                                          quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  quasar::backend::pic::launch_along_side(g, side,
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL(wall_correct_e_x_kernel_o4, grid, block, 0, hip_stream, g, side, dt,
                         f.ey.device_ptr(), f.ez.device_ptr(),
                         f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
    },
    [&](dim3 grid, dim3 block) {
      hipLaunchKernelGGL(wall_correct_e_y_kernel_o4, grid, block, 0, hip_stream, g, side, dt,
                         f.ex.device_ptr(), f.ez.device_ptr(),
                         f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
    });
  QUASAR_HIP_CHECK(::hipGetLastError());
}
