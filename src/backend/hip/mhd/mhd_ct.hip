// Constrained-transport device kernels: corner-EMF construction and the
// face-staggered B update. Staggering convention (mhd_field.hpp / ct_scheme.hpp):
//   bx_face(i,j) = Bx on the x_lo (left)   face of cell (i,j)
//   by_face(i,j) = By on the y_lo (bottom) face of cell (i,j)
//   ez_edge(i,j) = Ez at the lower-left corner of cell (i,j)
//   divB(i,j) = (bx_face(i+1,j)-bx_face(i,j))/dx + (by_face(i,j+1)-by_face(i,j))/dy
// The face-B update curl below is chosen so this discrete div(B) telescopes to
// identically zero for any corner-Ez field (Evans & Hawley 1988).

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;

inline constexpr unsigned kBlock2D = 16;
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Cell-centered Ez = -(vx*By - vy*Bx) = vy*Bx - vx*By from the conserved state.
__device__ inline Real cell_ez(const double* rho, const double* mx, const double* my,
                               const double* bx, const double* by, std::size_t k) {
  const Real inv_rho = Real{1} / rho[k];
  const Real vx = mx[k] * inv_rho;
  const Real vy = my[k] * inv_rho;
  return vy * bx[k] - vx * by[k];
}

// Corner Ez at the lower-left corner of cell (i,j): the arithmetic average of
// the cell-centered Ez of the four cells meeting at that corner ((i-1,j-1),
// (i,j-1), (i-1,j), (i,j)). This is the simple (kinematic) CT averaging that
// keeps the curl-of-corner-Ez div(B) annihilation property. The interface
// states are accepted for ABI parity / future upwinded EMF; the v1 EMF is the
// cell-centered average so the build stays robust.
__global__ void ct_emf_kernel(
    quasar::Grid2D g,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ bx,
    const double* __restrict__ by,
    double* __restrict__ ez_edge, double* __restrict__ ex_edge,
    double* __restrict__ ey_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);

  const Real e00 = cell_ez(rho, mx, my, bx, by, g.index(i - 1, j - 1));
  const Real e10 = cell_ez(rho, mx, my, bx, by, g.index(i, j - 1));
  const Real e01 = cell_ez(rho, mx, my, bx, by, g.index(i - 1, j));
  const Real e11 = cell_ez(rho, mx, my, bx, by, g.index(i, j));
  ez_edge[k] = Real{0.25} * (e00 + e10 + e01 + e11);
  // In-plane EMF components are zero for the 2D m=0 induction update (no z
  // gradient); populated for parity with the EmfField2D layout.
  ex_edge[k] = Real{0};
  ey_edge[k] = Real{0};
}

// Face-B update: dB/dt = -curl E.
//   bx_face(i,j) -= dt * (ez(i,j+1) - ez(i,j)) / dy
//   by_face(i,j) += dt * (ez(i+1,j) - ez(i,j)) / dx
__global__ void face_b_update_kernel(
    quasar::Grid2D g, Real dt,
    double* __restrict__ bx_face, double* __restrict__ by_face,
    const double* __restrict__ ez_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const Real inv_dx = Real{1} / g.dx();
  const Real inv_dy = Real{1} / g.dy();
  bx_face[k] -= dt * (ez_edge[g.index(i, j + 1)] - ez_edge[k]) * inv_dy;
  by_face[k] += dt * (ez_edge[g.index(i + 1, j)] - ez_edge[k]) * inv_dx;
}

// CT EMF curl RATE written into the residual's face-B slots. This is the SAME
// stencil as face_b_update_kernel above with the dt factor removed (it is the
// dB/dt rate; rk_stage applies dt later) and assigned (=) instead of accumulated
// so it OVERWRITES the face-B contribution that flux_difference left in dudt.
// ONLY dudt.bx_face / dudt.by_face are touched -- every other dudt component is
// left exactly as flux_difference wrote it. The indexing, dx/dy, and interior
// loop bounds match face_b_update_kernel exactly so the telescoping div(B)
// cancellation stays bit-for-bit identical.
__global__ void emf_curl_rate_kernel(
    quasar::Grid2D g,
    double* __restrict__ dbx_face, double* __restrict__ dby_face,
    const double* __restrict__ ez_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const Real inv_dx = Real{1} / g.dx();
  const Real inv_dy = Real{1} / g.dy();
  dbx_face[k] = -(ez_edge[g.index(i, j + 1)] - ez_edge[k]) * inv_dy;
  dby_face[k] = +(ez_edge[g.index(i + 1, j)] - ez_edge[k]) * inv_dx;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_ct_emf(const MhdField2D<Real>& u,
                       const quasar::numerics::MhdInterfaceStates<Real>& /*ifx*/,
                       const quasar::numerics::MhdInterfaceStates<Real>& /*ify*/,
                       EmfField2D<Real>& emf, Real /*gamma*/, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(ct_emf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     emf.ez_edge.device_ptr(), emf.ex_edge.device_ptr(),
                     emf.ey_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_face_b_update(MhdField2D<Real>& u, const EmfField2D<Real>& emf,
                              Real dt, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(face_b_update_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, dt, u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     emf.ez_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_emf_curl_rate(const EmfField2D<Real>& emf, MhdField2D<Real>& dudt,
                              Grid2D grid, stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 launch_grid = mhd_grid_2d(grid, block);
  hipLaunchKernelGGL(emf_curl_rate_kernel, launch_grid, block, 0,
                     static_cast<hipStream_t>(stream), grid,
                     dudt.bx_face.device_ptr(), dudt.by_face.device_ptr(),
                     emf.ez_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
