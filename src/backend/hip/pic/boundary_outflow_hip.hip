#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/stencil.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

// First-order characteristic (Mur) OUTFLOW field wall.
//
// An open boundary that lets an outgoing wave leave with little reflection. The
// tangential-E components satisfy the 1-D one-way wave equation at the wall and
// are advanced with the node-collocated first-order Mur update
//
//   u0^{n+1} = u1^n + coeff * (u1^{n+1} - u0^n),   coeff = (c*dt - h)/(c*dt + h),
//
// where node 0 is the wall and node 1 the adjacent interior node, c = 1 in
// internal units, h is the wall-normal spacing. This needs the previous-step
// values u0^n, u1^n, kept in a small per-face history strip owned by the BC.
//
// The remaining components are closed like the wall: normal E uses only
// transverse derivatives (interior-correct, untouched); the B components whose
// backward curl overran the low wall get the same one-sided delta as PEC but are
// NOT pinned (an open wall pins nothing). At 2nd order only the low wall's
// backward B-curl overruns; the high wall is interior-correct.

namespace {

namespace qn = quasar::numerics;

// Mur history strip layout, per face of length `len`:
//   [a_u0(len), a_u1(len), b_u0(len), b_u1(len)]
// where a,b are the two tangential-E components for the face.
__device__ inline void mur_advance(double* comp, double* s_u0, double* s_u1,
                                   std::size_t kw, std::size_t ka,
                                   double coeff, bool init) {
  const double new_u1 = comp[ka];
  if (init) {
    *s_u0 = comp[kw];
    *s_u1 = new_u1;
  } else {
    const double new_u0 = *s_u1 + coeff * (new_u1 - *s_u0);
    comp[kw] = new_u0;
    *s_u0 = new_u0;
    *s_u1 = new_u1;
  }
}

__global__ void outflow_mur_x_kernel(quasar::Grid2D g, int side, double dt,
                                     double* ey, double* ez, double* mur, int init) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny) return;
  const int len = g.ny;
  const int iw = (side == 0) ? 0 : g.nx - 1;
  const int ia = (side == 0) ? 1 : g.nx - 2;
  const double coeff = (dt - g.dx()) / (dt + g.dx());
  const std::size_t kw = g.index(iw, j);
  const std::size_t ka = g.index(ia, j);
  mur_advance(ey, mur + 0 * len + j, mur + 1 * len + j, kw, ka, coeff, init != 0);
  mur_advance(ez, mur + 2 * len + j, mur + 3 * len + j, kw, ka, coeff, init != 0);
}

__global__ void outflow_mur_y_kernel(quasar::Grid2D g, int side, double dt,
                                     double* ex, double* ez, double* mur, int init) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx) return;
  const int len = g.nx;
  const int jw = (side == 2) ? 0 : g.ny - 1;
  const int ja = (side == 2) ? 1 : g.ny - 2;
  const double coeff = (dt - g.dy()) / (dt + g.dy());
  const std::size_t kw = g.index(i, jw);
  const std::size_t ka = g.index(i, ja);
  mur_advance(ex, mur + 0 * len + i, mur + 1 * len + i, kw, ka, coeff, init != 0);
  mur_advance(ez, mur + 2 * len + i, mur + 3 * len + i, kw, ka, coeff, init != 0);
}

// B closure (no pin). 2nd-order backward B-curl overruns only the low wall.
__global__ void outflow_correct_b_x_kernel(quasar::Grid2D g, int side, double dt,
                                           double* by, double* bz, const double* ey,
                                           const double* ez) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= g.ny || side != 0) return;
  const std::size_t k = g.index(0, j);
  by[k] += dt * (qn::ddx_onesided_fwd1(ez, g, 0, j) - qn::ddx_staggered_bwd<2>(ez, g, 0, j));
  bz[k] -= dt * (qn::ddx_onesided_fwd1(ey, g, 0, j) - qn::ddx_staggered_bwd<2>(ey, g, 0, j));
}

__global__ void outflow_correct_b_y_kernel(quasar::Grid2D g, int side, double dt,
                                           double* bx, double* bz, const double* ex,
                                           const double* ez) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.nx || side != 2) return;
  const std::size_t k = g.index(i, 0);
  bx[k] -= dt * (qn::ddy_onesided_fwd1(ez, g, i, 0) - qn::ddy_staggered_bwd<2>(ez, g, i, 0));
  bz[k] += dt * (qn::ddy_onesided_fwd1(ex, g, i, 0) - qn::ddy_staggered_bwd<2>(ex, g, i, 0));
}

}  // namespace

extern "C" void launch_pic_boundary_outflow_correct_b_order2(const quasar::Grid2D& g,
                                                             quasar::YeeField2D<double>& f,
                                                             int side, double dt,
                                                             quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(256);
  if (side == 0 || side == 1) {
    if (g.ny <= 0) return;
    const dim3 grid((g.ny + block.x - 1) / block.x);
    hipLaunchKernelGGL(outflow_correct_b_x_kernel, grid, block, 0, hip_stream, g, side, dt,
                       f.by.device_ptr(), f.bz.device_ptr(),
                       f.ey.device_ptr(), f.ez.device_ptr());
  } else {
    if (g.nx <= 0) return;
    const dim3 grid((g.nx + block.x - 1) / block.x);
    hipLaunchKernelGGL(outflow_correct_b_y_kernel, grid, block, 0, hip_stream, g, side, dt,
                       f.bx.device_ptr(), f.bz.device_ptr(),
                       f.ex.device_ptr(), f.ez.device_ptr());
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_outflow_correct_e_order2(const quasar::Grid2D& g,
                                                             quasar::YeeField2D<double>& f,
                                                             int side, double dt,
                                                             double* mur_strips, int init,
                                                             quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(256);
  if (side == 0 || side == 1) {
    if (g.ny <= 0) return;
    const dim3 grid((g.ny + block.x - 1) / block.x);
    hipLaunchKernelGGL(outflow_mur_x_kernel, grid, block, 0, hip_stream, g, side, dt,
                       f.ey.device_ptr(), f.ez.device_ptr(), mur_strips, init);
  } else {
    if (g.nx <= 0) return;
    const dim3 grid((g.nx + block.x - 1) / block.x);
    hipLaunchKernelGGL(outflow_mur_y_kernel, grid, block, 0, hip_stream, g, side, dt,
                       f.ex.device_ptr(), f.ez.device_ptr(), mur_strips, init);
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}
