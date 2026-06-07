// Device ghost-layer fill kernels for the 2D CT MHD boundary axis.
//
// These kernels replace the former host-staged ghost fills (see the prior
// src/physics/mhd/mhd_boundary.cpp): every conserved/CT buffer lives in
// DeviceBuffer<Real>, so the six registry BC classes are now thin launchers and
// the ghost-fill math runs here on device, operating on raw device pointers
// extracted on the host before launch (no copy_to_host / StagedComponent).
//
// The index maps reproduce the AUTHORITATIVE host reference rules exactly:
//
// Indexing (Grid2D::index(i,j)):
//   interior cells:  i in [0, nx),     j in [0, ny)
//   x ghost layers:  g in [1, nghost]  ->  i = -g (lo)  /  i = nx-1+g (hi)
//   y ghost layers:  g in [1, nghost]  ->  j = -g (lo)  /  j = ny-1+g (hi)
//
// THREE CRITICAL CORRECTNESS RULES (preserved bit-for-bit from the host):
//
//  (a) SHARED STORAGE EXTENT. Face arrays (bx_face/by_face) share the SAME
//      storage layout/extent as the cell arrays -- index(i,j) with valid
//      i in [-nghost, nx-1+nghost]. There is NO extra face slot at nx+nghost,
//      so the hi ghost target is gi = nx-1+layer (slots nx, nx+1, ...,
//      nx-1+nghost for layer 1..nghost), identical to the cell-centered hi
//      convention. Writing nx+layer would overflow by one column at
//      layer==nghost. Both the cell and the normal-face fill use this same hi
//      index; the normal face differs only in the symmetry SIGN, never the map.
//
//  (b) CORNER GHOST BLOCKS. y-side fills span the FULL storage width
//      i in [-ng, nx+ng) (not just the interior [0,nx)). The solver fills
//      x-sides BEFORE y-sides, so when a y-side runs the x-ghost columns are
//      already populated; copying them down/up into the y-ghost rows fills the
//      four CORNER ghost blocks (both i and j in the ghost range). The high-edge
//      CT corner-EMF reads these corners; an interior-only y-fill leaves them
//      stale and injects a div(B) leak. x-side fills iterate j over the interior
//      [0,ny) (the y-side picks up the corners).
//
//  (c) REFLECTING SIGN. The odd (sign-flip) symmetry applies ONLY to the
//      component normal to the wall -- the normal momentum (mx on an x-side,
//      my on a y-side) for the fluid fill, and the normal face-B (bx_face on an
//      x-side, by_face on a y-side) for the field fill. Every tangential
//      component and every scalar is copied even (sign +1).

#include "quasar/core/grid.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::Grid2D;

inline constexpr unsigned kBlock = 256;

// Resolve, for one ghost `layer` (1..nghost), the ghost target index `gi` on
// `side`'s axis and the matching source interior index `si` for the requested
// `mode` (0=periodic wrap, 1=outflow zero-gradient, 2=reflecting mirror). `n` is
// the axis extent (nx for an x-side, ny for a y-side) and `lo` is true for the
// low side. The map is identical for cell-centered and face-staggered storage
// because the face arrays share the cell storage extent (rule (a)).
__device__ inline void ghost_indices(int n, int layer, bool lo, int mode,
                                     int& gi, int& si) {
  gi = lo ? (-layer) : (n - 1 + layer);
  if (mode == 0) {
    si = lo ? (n - layer) : (layer - 1);   // periodic wrap
  } else if (mode == 1) {
    si = lo ? 0 : (n - 1);                  // outflow zero-gradient
  } else {
    si = lo ? (layer - 1) : (n - layer);   // reflecting mirror
  }
}

// ---------------------------------------------------------------------------
// Fluid ghost fill. One thread per (layer, line) slot. For an x-side the line is
// the interior column j in [0, ny); for a y-side the line is the FULL storage
// width i in [-ng, nx+ng) (rule (b), corner coverage). `s_normal` is the sign
// applied to the normal-momentum component (mx on x-side, my on y-side): +1 for
// periodic/outflow, -1 for a reflecting wall (rule (c)); every other component
// is even.
// ---------------------------------------------------------------------------
__global__ void fill_ghosts_fluid_kernel(
    Grid2D g, int side, int mode, Real s_normal,
    double* __restrict__ rho, double* __restrict__ mx, double* __restrict__ my,
    double* __restrict__ mz, double* __restrict__ energy) {
  const int nx = g.nx;
  const int ny = g.ny;
  const int ng = g.nghost;
  const bool x_side = (side == 0 || side == 1);  // 0=x_lo,1=x_hi,2=y_lo,3=y_hi
  const bool lo = (side == 0 || side == 2);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // line index range: x-side -> [0, ny); y-side -> [-ng, nx+ng)
  const int line_lo = x_side ? 0 : -ng;
  const int line_hi = x_side ? ny : (nx + ng);  // exclusive upper bound
  const int line_count = line_hi - line_lo;
  if (line_count <= 0) return;
  const int total = ng * line_count;
  if (tid >= total) return;

  const int layer = (tid / line_count) + 1;     // 1..nghost
  const int line = line_lo + (tid % line_count);

  if (x_side) {
    int gi, si;
    ghost_indices(nx, layer, lo, mode, gi, si);
    const int j = line;
    const std::size_t kd = g.index(gi, j);
    const std::size_t ks = g.index(si, j);
    rho[kd]    = rho[ks];
    mx[kd]     = s_normal * mx[ks];   // mx is normal on an x-side
    my[kd]     = my[ks];
    mz[kd]     = mz[ks];
    energy[kd] = energy[ks];
  } else {
    int gj, sj;
    ghost_indices(ny, layer, lo, mode, gj, sj);
    const int i = line;
    const std::size_t kd = g.index(i, gj);
    const std::size_t ks = g.index(i, sj);
    rho[kd]    = rho[ks];
    mx[kd]     = mx[ks];
    my[kd]     = s_normal * my[ks];   // my is normal on a y-side
    mz[kd]     = mz[ks];
    energy[kd] = energy[ks];
  }
}

// ---------------------------------------------------------------------------
// Field ghost fill. Same launch/line geometry as the fluid kernel. The normal
// face-B (bx_face on an x-side, by_face on a y-side) takes `s_normal` (odd at a
// reflecting wall); the tangential face-B and the cell-centered bz_cell are
// always even. The hi ghost target uses the cell-identical index (rule (a)).
// ---------------------------------------------------------------------------
__global__ void fill_ghosts_field_kernel(
    Grid2D g, int side, int mode, Real s_normal,
    double* __restrict__ bx_face, double* __restrict__ by_face,
    double* __restrict__ bz_cell) {
  const int nx = g.nx;
  const int ny = g.ny;
  const int ng = g.nghost;
  const bool x_side = (side == 0 || side == 1);
  const bool lo = (side == 0 || side == 2);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int line_lo = x_side ? 0 : -ng;
  const int line_hi = x_side ? ny : (nx + ng);
  const int line_count = line_hi - line_lo;
  if (line_count <= 0) return;
  const int total = ng * line_count;
  if (tid >= total) return;

  const int layer = (tid / line_count) + 1;
  const int line = line_lo + (tid % line_count);

  if (x_side) {
    int gi, si;
    ghost_indices(nx, layer, lo, mode, gi, si);
    const int j = line;
    const std::size_t kd = g.index(gi, j);
    const std::size_t ks = g.index(si, j);
    bx_face[kd] = s_normal * bx_face[ks];  // bx_face is normal on an x-side
    by_face[kd] = by_face[ks];             // tangential -> even
    bz_cell[kd] = bz_cell[ks];             // cell-centered -> even
  } else {
    int gj, sj;
    ghost_indices(ny, layer, lo, mode, gj, sj);
    const int i = line;
    const std::size_t kd = g.index(i, gj);
    const std::size_t ks = g.index(i, sj);
    bx_face[kd] = bx_face[ks];             // tangential -> even
    by_face[kd] = s_normal * by_face[ks];  // by_face is normal on a y-side
    bz_cell[kd] = bz_cell[ks];             // cell-centered -> even
  }
}

// Total thread count covering (layer 1..nghost) x (line range). The line range
// is the interior [0,ny)/[0,nx) on an x-side and the full storage width
// [-ng, nx+ng) on a y-side (corner coverage, rule (b)).
inline unsigned fill_total_threads(const Grid2D& g, int side) {
  const bool x_side = (side == 0 || side == 1);
  const int line_count = x_side ? g.ny : (g.nx + 2 * g.nghost);
  const long total = static_cast<long>(g.nghost) * static_cast<long>(line_count);
  return total > 0 ? static_cast<unsigned>(total) : 0u;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_fill_ghosts_fluid(MhdField2D<Real>& u, int side, int mode,
                                  stream_t stream) {
  const Grid2D g = u.grid;
  const unsigned total = fill_total_threads(g, side);
  if (total == 0) return;
  // Normal momentum (mx on an x-side, my on a y-side) gets the odd sign only on
  // a reflecting wall (mode 2); everything else is even.
  const bool x_side = (side == 0 || side == 1);
  const Real s_normal = (mode == 2) ? Real{-1} : Real{1};
  (void)x_side;  // selection of which component is normal happens in the kernel
  const dim3 block(kBlock);
  const dim3 grid((total + block.x - 1) / block.x);
  hipLaunchKernelGGL(fill_ghosts_fluid_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, side, mode, s_normal,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_fill_ghosts_field(MhdField2D<Real>& u, int side, int mode,
                                  stream_t stream) {
  const Grid2D g = u.grid;
  const unsigned total = fill_total_threads(g, side);
  if (total == 0) return;
  const Real s_normal = (mode == 2) ? Real{-1} : Real{1};
  const dim3 block(kBlock);
  const dim3 grid((total + block.x - 1) / block.x);
  hipLaunchKernelGGL(fill_ghosts_field_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, side, mode, s_normal,
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     u.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
