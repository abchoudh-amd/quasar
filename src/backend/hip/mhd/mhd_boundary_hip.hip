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
//      allocation. A normal face is nevertheless half a cell offset: the
//      physical faces are 0..n, so slot n is the high boundary face rather than
//      a derived ghost. Periodic closure identifies it with face 0; wall closure
//      pins it; outflow leaves its CT-evolved value intact and fills only slots
//      beyond it. The normal-face map therefore differs from cell-centred data.
//
//  (b) CORNER GHOST BLOCKS. Both axis fills span the FULL padded transverse
//      range. In a serial solver the later y-side pass still deterministically
//      overwrites all four corner blocks. In a distributed tile one axis may be
//      an exchanged internal no-op; the orthogonal physical/periodic pass must
//      then close the already-exchanged halo all the way through its corners.
//      Restricting x-side work to interior y rows leaves stale RK scratch at a
//      physical-x/internal-y corner and poisons transverse reconstruction.
//
//  (c) WALL SIGN. The odd (sign-flip) symmetry applies ONLY to the
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
// `mode` (0=periodic wrap, 1=outflow zero-gradient, 2=wall mirror). `n` is
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
    si = lo ? (layer - 1) : (n - layer);   // wall mirror
  }
}

// ---------------------------------------------------------------------------
// Fluid ghost fill. One thread per (layer, line) slot. For an x-side the line is
// the full padded y range; for a y-side the line is the full padded x range
// (rule (b), corner coverage). `s_normal` is the sign
// applied to the normal-momentum component (mx on x-side, my on y-side): +1 for
// periodic/outflow, -1 for a wall (rule (c)); every other component
// is even.
// ---------------------------------------------------------------------------
__global__ void QUASAR_LAUNCH_BOUNDS_256 fill_ghosts_fluid_kernel(
    Grid2D g, int side, int mode, Real s_normal,
    double* __restrict__ rho, double* __restrict__ mx, double* __restrict__ my,
    double* __restrict__ mz, double* __restrict__ energy) {
  const int nx = g.nx;
  const int ny = g.ny;
  const int ng = g.nghost;
  const bool x_side = (side == 0 || side == 1);  // 0=x_lo,1=x_hi,2=y_lo,3=y_hi
  const bool lo = (side == 0 || side == 2);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int line_lo = -ng;
  const int line_hi = x_side ? (ny + ng) : (nx + ng);
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
    mz[kd]     = (mode == 3 ? Real{-1} : Real{1}) * mz[ks];
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
// face-B (bx_face on an x-side, by_face on a y-side) is odd about its physical
// wall face (which is pinned to zero); the tangential face-B and cell-centered
// bz_cell are even. At a high outflow boundary, layer 1 is physical face n: CT
// owns it, so it must not be overwritten as if it were a cell ghost. Layers 2+
// extrapolate that face value into the actual face ghosts.
// ---------------------------------------------------------------------------
__global__ void QUASAR_LAUNCH_BOUNDS_256 fill_ghosts_field_kernel(
    Grid2D g, int side, int mode, Real s_normal,
    double* __restrict__ bx_face, double* __restrict__ by_face,
    double* __restrict__ bz_cell) {
  const int nx = g.nx;
  const int ny = g.ny;
  const int ng = g.nghost;
  const bool x_side = (side == 0 || side == 1);
  const bool lo = (side == 0 || side == 2);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int line_lo = -ng;
  const int line_hi = x_side ? (ny + ng) : (nx + ng);
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
    by_face[kd] = by_face[ks];             // tangential -> even
    bz_cell[kd] = (mode == 3 ? Real{-1} : Real{1}) * bz_cell[ks];
    if (mode == 2 || mode == 3) {
      // Normal face data mirror about the boundary FACE, not about the first
      // cell centre.  Pin Bn on the wall/axis itself and antisymmetrically fill
      // the remaining face ghosts.
      if (lo) {
        bx_face[g.index(0, j)] = Real{0};
        bx_face[g.index(-layer, j)] = -bx_face[g.index(layer, j)];
      } else if (layer == 1) {
        bx_face[g.index(nx, j)] = Real{0};
      } else {
        const int offset = layer - 1;
        bx_face[g.index(nx + offset, j)] = -bx_face[g.index(nx - offset, j)];
      }
    } else if (mode == 1 && !lo) {
      // gi==nx at layer 1 is the physical high-x face and was evolved by CT.
      // Preserve it; only nx+1, nx+2, ... are outflow ghost faces.
      if (layer > 1) {
        bx_face[g.index(nx - 1 + layer, j)] = bx_face[g.index(nx, j)];
      }
    } else {
      bx_face[kd] = s_normal * bx_face[ks];
    }
  } else {
    int gj, sj;
    ghost_indices(ny, layer, lo, mode, gj, sj);
    const int i = line;
    const std::size_t kd = g.index(i, gj);
    const std::size_t ks = g.index(i, sj);
    bx_face[kd] = bx_face[ks];             // tangential -> even
    bz_cell[kd] = bz_cell[ks];             // cell-centered -> even
    if (mode == 2) {
      if (lo) {
        by_face[g.index(i, 0)] = Real{0};
        by_face[g.index(i, -layer)] = -by_face[g.index(i, layer)];
      } else if (layer == 1) {
        by_face[g.index(i, ny)] = Real{0};
      } else {
        const int offset = layer - 1;
        by_face[g.index(i, ny + offset)] = -by_face[g.index(i, ny - offset)];
      }
    } else if (mode == 1 && !lo) {
      // gj==ny at layer 1 is the physical high-y face and was evolved by CT.
      if (layer > 1) {
        by_face[g.index(i, ny - 1 + layer)] = by_face[g.index(i, ny)];
      }
    } else {
      by_face[kd] = s_normal * by_face[ks];
    }
  }
}

// Total thread count covering (layer 1..nghost) x the full padded transverse
// range (corner coverage, rule (b)).
inline unsigned fill_total_threads(const Grid2D& g, int side) {
  const bool x_side = (side == 0 || side == 1);
  const int line_count = (x_side ? g.ny : g.nx) + 2 * g.nghost;
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
  // a wall (mode 2); everything else is even.
  const bool x_side = (side == 0 || side == 1);
  const Real s_normal = (mode == 2 || mode == 3) ? Real{-1} : Real{1};
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
  const bool x_side = (side == 0 || side == 1);
  const int line_count = (x_side ? g.ny : g.nx) + 2 * g.nghost;
  const long total_long =
      static_cast<long>(g.nghost) * static_cast<long>(line_count);
  const unsigned total = total_long > 0
      ? static_cast<unsigned>(total_long) : 0u;
  if (total == 0) return;
  const Real s_normal = (mode == 2 || mode == 3) ? Real{-1} : Real{1};
  const dim3 block(kBlock);
  const dim3 grid((total + block.x - 1) / block.x);
  hipLaunchKernelGGL(fill_ghosts_field_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, side, mode, s_normal,
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     u.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_fill_ghosts_background(MhdBackgroundField<Real>& b0, int side,
                                       int mode, stream_t stream) {
  const Grid2D g = b0.grid;
  const unsigned total = fill_total_threads(g, side);
  if (total == 0) return;
  const Real s_normal = (mode == 2 || mode == 3) ? Real{-1} : Real{1};
  const dim3 block(kBlock);
  const dim3 grid((total + block.x - 1) / block.x);
  hipLaunchKernelGGL(fill_ghosts_field_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, side, mode, s_normal,
                     b0.b0x_face.device_ptr(), b0.b0y_face.device_ptr(),
                     b0.b0z_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
