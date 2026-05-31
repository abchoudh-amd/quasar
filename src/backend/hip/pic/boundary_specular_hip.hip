#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void specular_kernel(quasar::Grid2D g, std::size_t n, int side,
                                double* x, double* y, double* x_prev, double* y_prev,
                                double* vx, double* vy, unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  // Reflect the post-move position and the normal velocity about the wall. The
  // pre-move position (x_prev) is deliberately LEFT inside the domain: the
  // charge-conserving deposit then forms current from the true in-domain start to
  // the just-crossed end, and the part of the deposit stencil that spills past the
  // wall lands in ghost cells, which boundary_specular_foldback reflects back into
  // the interior (image-charge current). Mirroring x_prev too (the old behaviour)
  // put the start point outside the domain and the deposit teleported current to
  // the opposite edge via the periodic wrap.
  if (side == 0 && x[p] < g.origin_x) {
    x[p] = 2.0 * g.origin_x - x[p];
    vx[p] = -vx[p];
  } else if (side == 1 && x[p] > g.origin_x + g.lx) {
    const double w = g.origin_x + g.lx;
    x[p] = 2.0 * w - x[p];
    vx[p] = -vx[p];
  } else if (side == 2 && y[p] < g.origin_y) {
    y[p] = 2.0 * g.origin_y - y[p];
    vy[p] = -vy[p];
  } else if (side == 3 && y[p] > g.origin_y + g.ly) {
    const double w = g.origin_y + g.ly;
    y[p] = 2.0 * w - y[p];
    vy[p] = -vy[p];
  }
}

// Folds current deposited into the ghost cells of one reflecting side back into
// the interior as image-charge current, conserving charge at the wall. The image
// of ghost node (relative offset gh past the boundary) is the interior node gh-1
// short of it; the deposit's continuity stencil places the boundary-crossing
// current there. The normal component (Jx for an x-wall, Jy for a y-wall) is
// added with a sign flip so net normal current at the wall vanishes (a perfect
// reflector carries no current through itself); tangential and out-of-plane
// components fold with the same sign. After folding, ghost cells are zeroed so a
// subsequent (interior-only) read never double-counts them.
__global__ void foldback_kernel(quasar::Grid2D g, int side,
                                double* jx, double* jy, double* jz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int len = (side < 2) ? g.ny : g.nx;
  const int total = g.nghost * len;
  if (tid >= total) return;
  const int gh = tid / len + 1;     // ghost layer, 1..nghost
  const int line = tid % len;       // index along the wall

  int gi, gj, ii, ij;               // ghost node and its interior image
  bool normal_is_x;
  if (side == 0) {                  // x_lo: ghost col -gh, image col gh-1
    gi = -gh; gj = line; ii = gh - 1; ij = line; normal_is_x = true;
  } else if (side == 1) {           // x_hi: ghost col nx-1+gh, image col nx-gh
    gi = g.nx - 1 + gh; gj = line; ii = g.nx - gh; ij = line; normal_is_x = true;
  } else if (side == 2) {           // y_lo
    gi = line; gj = -gh; ii = line; ij = gh - 1; normal_is_x = false;
  } else {                          // y_hi
    gi = line; gj = g.ny - 1 + gh; ii = line; ij = g.ny - gh; normal_is_x = false;
  }

  const std::size_t kg = g.index(gi, gj);
  const std::size_t ki = g.index(ii, ij);
  const double gx = jx[kg], gy = jy[kg], gz = jz[kg];
  // Normal component flips sign; tangential + Jz keep sign.
  jx[ki] += normal_is_x ? -gx : gx;
  jy[ki] += normal_is_x ? gy : -gy;
  jz[ki] += gz;
  jx[kg] = 0.0; jy[kg] = 0.0; jz[kg] = 0.0;
}

}  // namespace

extern "C" void launch_pic_boundary_specular_particles(const quasar::Grid2D& g,
                                                       quasar::pic::ParticleSpecies& s,
                                                       int side, quasar_stream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(s.size());
  hipLaunchKernelGGL(specular_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, s.size(), side, s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.alive());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_specular_foldback(const quasar::Grid2D& g,
                                                      quasar::JField2D<double>& j,
                                                      int side, quasar_stream_t stream) {
  const int len = (side < 2) ? g.ny : g.nx;
  const int total = g.nghost * len;
  if (total <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(static_cast<std::size_t>(total));
  hipLaunchKernelGGL(foldback_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, side, j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}
