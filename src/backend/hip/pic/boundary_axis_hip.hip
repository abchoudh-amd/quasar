#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Axis (r = 0) boundary for the axisymmetric (m=0) cylindrical PIC.
//
// The grid axis i indexes the radius r with r = 0 at i = 0 (x-axis); j indexes
// the axial coordinate z (y-axis). Stored field components map (x,y,z) ->
// (r,z,phi):  ex->Er, ey->Ez, ez->Ephi, bx->Br, by->Bz, bz->Bphi.
//
// Under the reflection r -> -r the m=0 fields have definite parity:
//   ODD  (vanish on axis, mirror antisymmetrically): Er, Ephi, Br, Bphi
//   EVEN (mirror symmetrically into the ghost):       Ez, Bz
// The Yee fields are node-collocated with node 0 at r = dr/2, so the cell-centred
// mirror of ghost column -gh is interior column gh-1:
//   f(-gh, j) = parity * f(gh-1, j),  gh in [1, nghost].
// The odd components are additionally pinned to zero ON the axis node (i=0); the
// FDTD axis closure already does this, and re-applying it here is idempotent and
// keeps the parity self-consistent if any caller skips the closure.
//
// launch_pic_boundary_axis_particles folds a particle that has been pushed to
// r < 0 back across the axis: r -> -r and vr -> -vr (vz, vphi unchanged). x holds
// r, y holds z; vx holds vr.

namespace {

// Fill one component's axis ghost column from its interior mirror with the given
// parity, and (for odd parity) pin the on-axis node to zero.
//   parity = +1 : even (Ez, Bz)
//   parity = -1 : odd  (Er, Ephi, Br, Bphi)
__device__ inline void axis_fill_component(const quasar::Grid2D& g, double* f,
                                           double parity, int gh, int j) {
  f[g.index(-gh, j)] = parity * f[g.index(gh - 1, j)];
  if (parity < 0.0 && gh == 1) {
    f[g.index(0, j)] = 0.0;  // odd component vanishes on the axis node
  }
}

__global__ void axis_fields_kernel(quasar::Grid2D g, int len,
                                   double* ex, double* ey, double* ez,
                                   double* bx, double* by, double* bz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // over nghost*len
  const int gh = tid / len + 1;     // 1..nghost
  const int j = tid % len;          // 0..ny-1
  if (gh > g.nghost || j >= len) return;
  // Even parity: Ez (ey), Bz (by).
  axis_fill_component(g, ey, +1.0, gh, j);
  axis_fill_component(g, by, +1.0, gh, j);
  // Odd parity: Er (ex), Ephi (ez), Br (bx), Bphi (bz).
  axis_fill_component(g, ex, -1.0, gh, j);
  axis_fill_component(g, ez, -1.0, gh, j);
  axis_fill_component(g, bx, -1.0, gh, j);
  axis_fill_component(g, bz, -1.0, gh, j);
}

__global__ void axis_particles_kernel(quasar::Grid2D g, std::size_t n,
                                      double* x, double* x_prev,
                                      double* vx, unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  // Fold a particle that crossed the axis (r < 0) back into the physical domain.
  // The radial velocity flips; axial / azimuthal velocities are untouched. The
  // pre-move radial position is left as-is so the charge-conserving deposit forms
  // current from the true start to the folded end (the deposit's r-weighting and
  // the axis ghost columns absorb the crossing, mirroring the specular wall).
  if (x[p] < g.origin_x) {
    x[p] = 2.0 * g.origin_x - x[p];  // origin_x is r = 0
    vx[p] = -vx[p];
  }
  (void)x_prev;
}

}  // namespace

extern "C" void launch_pic_boundary_axis_fields(const quasar::Grid2D& g,
                                                quasar::YeeField2D<quasar::Real>& f,
                                                quasar_stream_t stream) {
  if (g.nghost <= 0) return;
  const int len = g.ny;
  const int total = g.nghost * len;
  if (total <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(static_cast<std::size_t>(total));
  hipLaunchKernelGGL(axis_fields_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, len,
                     f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                     f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_axis_particles(const quasar::Grid2D& g,
                                                   quasar::pic::ParticleSpecies& s,
                                                   quasar_stream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(s.size());
  hipLaunchKernelGGL(axis_particles_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, s.size(), s.x(), s.x_prev(), s.vx(), s.alive());
  QUASAR_HIP_CHECK(::hipGetLastError());
}
