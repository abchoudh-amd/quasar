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
// The in-plane field used here is the TOTAL field B = b + B0: `bx`/`by` carry the
// stored perturbation b and the cell-centered background B0 is sampled here by
// averaging the two bracketing faces of `b0x_face`/`b0y_face`. When B0 is
// inactive the background contribution is identically zero (the caller passes
// nullptr and `b0_active` is false), so this reduces bit-for-bit to the
// perturbation-only kinematic EMF.
__device__ inline Real cell_ez(const quasar::Grid2D& g,
                               const double* rho, const double* mx, const double* my,
                               const double* bx, const double* by,
                               bool b0_active, const double* b0x_face,
                               const double* b0y_face, int i, int j) {
  const std::size_t k = g.index(i, j);
  const Real inv_rho = Real{1} / rho[k];
  const Real vx = mx[k] * inv_rho;
  const Real vy = my[k] * inv_rho;
  Real Bx = bx[k];
  Real By = by[k];
  if (b0_active) {
    // Cell-centered B0 from the two face values bracketing the cell: B0x lives on
    // the x_lo face (i,j) and x_hi face (i+1,j); B0y on the y_lo face (i,j) and
    // y_hi face (i,j+1).
    Bx += Real{0.5} * (b0x_face[k] + b0x_face[g.index(i + 1, j)]);
    By += Real{0.5} * (b0y_face[k] + b0y_face[g.index(i, j + 1)]);
  }
  return vy * Bx - vx * By;
}

// Corner Ez at the lower-left corner of cell (i,j): the arithmetic average of
// the cell-centered (total-field) Ez of the cells meeting at that corner. For a
// periodic / interior corner this is the four cells ((i-1,j-1), (i,j-1),
// (i-1,j), (i,j)) -- the simple kinematic CT averaging that keeps the
// curl-of-corner-Ez div(B) annihilation property and is bit-identical to the
// original when B0 is inactive and no boundary flags are set.
//
// One-sided handling: `flags.side[k]` (0=x_lo,1=x_hi,2=y_lo,3=y_hi) is 1 for a
// non-periodic side. A corner that lies ON a non-periodic boundary edge must not
// reach across that boundary for its average; instead it averages only the
// interior neighbors on the inside of that edge. The lower-left corner of cell
// (i,j) sits on:
//   - the x_lo boundary when i == 0  (left column of cells (i-1,*) is the ghost)
//   - the x_hi boundary when i == nx (cells (i,*) are the ghost past the domain)
//   - the y_lo boundary when j == 0
//   - the y_hi boundary when j == ny
// At a non-periodic x_lo/x_hi edge we drop the outside column and average only
// the interior column; likewise for y. The interface states are accepted for ABI
// parity / future upwinded EMF; the v1 EMF is the cell-centered average so the
// build stays robust.
__global__ void ct_emf_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int b0_active,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ bx,
    const double* __restrict__ by,
    const double* __restrict__ b0x_face, const double* __restrict__ b0y_face,
    double* __restrict__ ez_edge, double* __restrict__ ex_edge,
    double* __restrict__ ey_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);

  // Decide which side columns/rows of the 2x2 corner stencil are valid. By
  // default both sides are used (the periodic two-sided average). At a
  // non-periodic boundary edge the column/row that would reach across the
  // boundary is dropped so the average is interior-only.
  bool use_left  = true;   // include the (i-1,*) column
  bool use_right = true;   // include the (i,*)   column
  bool use_below = true;   // include the (*,j-1) row
  bool use_above = true;   // include the (*,j)   row
  // One-sided corner closure at the LOW non-periodic edges (i==0 / j==0): drop
  // the column/row that reaches across the boundary into a ghost cell, so the
  // corner EMF is built from interior cells only and does not depend on the
  // ghost gradient.
  //
  // High-edge scope note: this kernel computes corner EMFs only over the interior
  // index range i in [0,nx), j in [0,ny) (the guard above). The x_hi / y_hi
  // *domain-edge* corner lives at index nx / ny and is NOT produced here; the
  // emf_curl_rate stencil reads that corner from the ghost slot, exactly as the
  // pre-existing periodic/halo path does. We deliberately do not special-case a
  // high-edge corner: there is no reachable interior corner whose 2x2 stencil
  // reaches a ghost on the high side (the last computed corner, i==nx-1, sits
  // between interior cells nx-2 and nx-1), so a one-sided drop there would
  // wrongly discard an interior cell. The high-edge corner handling is part of
  // the constrained-transport halo, shared with the periodic path, and is left
  // unchanged by this feature; only the low-edge one-sided closure is added.
  if (flags.side[0] && i == 0) use_left  = false;  // x_lo non-periodic
  if (flags.side[2] && j == 0) use_below = false;  // y_lo non-periodic

  const bool active = (b0_active != 0);
  Real sum = Real{0};
  Real wt  = Real{0};
  if (use_left && use_below) {
    sum += cell_ez(g, rho, mx, my, bx, by, active, b0x_face, b0y_face, i - 1, j - 1);
    wt  += Real{1};
  }
  if (use_right && use_below) {
    sum += cell_ez(g, rho, mx, my, bx, by, active, b0x_face, b0y_face, i, j - 1);
    wt  += Real{1};
  }
  if (use_left && use_above) {
    sum += cell_ez(g, rho, mx, my, bx, by, active, b0x_face, b0y_face, i - 1, j);
    wt  += Real{1};
  }
  if (use_right && use_above) {
    sum += cell_ez(g, rho, mx, my, bx, by, active, b0x_face, b0y_face, i, j);
    wt  += Real{1};
  }
  ez_edge[k] = (wt > Real{0}) ? (sum / wt) : Real{0};
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
                       const MhdBackgroundField<Real>& b0,
                       const quasar::numerics::MhdInterfaceStates<Real>& /*ifx*/,
                       const quasar::numerics::MhdInterfaceStates<Real>& /*ify*/,
                       BoundaryFlags4 flags, EmfField2D<Real>& emf, Real /*gamma*/,
                       stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  // When the background is inactive pass null device pointers; the kernel takes
  // the zero-background fast path and never dereferences them.
  const int b0_active = b0.active ? 1 : 0;
  const double* b0x = b0.active ? b0.b0x_face.device_ptr() : nullptr;
  const double* b0y = b0.active ? b0.b0y_face.device_ptr() : nullptr;
  hipLaunchKernelGGL(ct_emf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, flags, b0_active,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     b0x, b0y,
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
