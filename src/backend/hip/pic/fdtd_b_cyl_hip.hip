#include "quasar/core/grid.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/fdtd_cyl_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Axisymmetric (m=0) cylindrical Faraday update of B from E: dB/dt = -curl E.
//
// Staggered Yee layout / component mapping (x,y,z)->(r,z,phi) and axis convention
// (i=radius, r=0 at i=0, j=axial) consistent with the cylindrical E-update. The
// even components (Ez, Bz) live on radial NODES r_c(i); the odd components (Er,
// Ephi, Br, Bphi) live on radial FACES r_e(i), stored at column i = lower face.
//
// The Faraday radial divergence for the even Bz component is the SAME natural
// forward finite-volume flux the E-update uses for Ez:
//   (1/r_c(i)) * ( r_e(i+1) f(i+1,j) - r_e(i) f(i,j) ) / dr,
// which is axis-regular at i=0 (the r_e(0)=0 inner face zeroes the singular term),
// rather than a backward flux that divides by r_e(i) and blows up on the axis. The
// purely axial / plain radial derivatives (used by Br and Bphi, the odd faces) are
// backward differences, the adjoint partners of the E-update's forward reads. The
// odd components Br and Bphi vanish on the axis face (r=0) by parity and are
// pinned to zero. The earlier "4*Ephi/dr" axis hack for Bz injected the same
// spurious resolution-independent eigenvalue shift the E-update closure did and is
// removed.

namespace {

// radial_flux_fwd, ddz_fwd, ddz_bwd, ddr_bwd are the shared cylindrical FDTD
// operators (fdtd_cyl_common.hpp), used by both the Ampere and Faraday (here)
// updates so the adjoint operator pair cannot drift. The even Bz reads the odd
// Ephi through the forward face-to-node flux radial_flux_fwd (axis-regular); the
// odd Br/Bphi use the backward differences ddz_bwd/ddr_bwd.
using namespace quasar::backend::pic;

__global__ void fdtd_b_cyl_kernel(quasar::Grid2D g,
                                  double* __restrict__ bx, double* __restrict__ by,
                                  double* __restrict__ bz,
                                  const double* __restrict__ ex, const double* __restrict__ ey,
                                  const double* __restrict__ ez,
                                  double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);

  if (i == 0) {
    // On-axis node. Br (radial) and Bphi (azimuthal) are odd in r and vanish on
    // the axis face (r_e(0)=0) -- pin them. Bz (even) lives on the axis node; its
    // radial flux uses the natural zero-radius inner face (radial_flux_fwd reduces
    // to r_e(1)*Ephi(1)/(r_c(0)*dr) here), the axis-regular closure. The previous
    // "4*Ephi/dr" factor injected the same spurious eigenvalue shift the E-update
    // closure did and is removed.
    const double curl_z = radial_flux_fwd(ez, g, 0, j);  // (1/r) d(r Ephi)/dr, axis-regular
    bx[k] = 0.0;
    bz[k] = 0.0;  // Bphi component (stored in bz slot)
    by[k] -= dt * curl_z;
    return;
  }

  // Interior: cylindrical curl of E. The even Bz reads the odd Ephi via the
  // forward face-to-node finite-volume flux (the adjoint of the E-update's
  // backward ddr_bwd(Bz) in the Ephi equation, and axis-regular); the odd Br/Bphi
  // use plain backward differences (adjoints of the E-update's forward reads).
  const double curl_r = -ddz_bwd(ez, g, i, j);                      // -dEphi/dz
  const double curl_z = radial_flux_fwd(ez, g, i, j);              // (1/r) d(r Ephi)/dr
  const double curl_phi = ddz_bwd(ex, g, i, j) - ddr_bwd(ey, g, i, j);  // dEr/dz - dEz/dr

  bx[k] -= dt * curl_r;    // Br
  by[k] -= dt * curl_z;    // Bz
  bz[k] -= dt * curl_phi;  // Bphi
}

void launch(quasar::Grid2D g, double* bx, double* by, double* bz,
            const double* ex, const double* ey, const double* ez, double dt,
            hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL(fdtd_b_cyl_kernel, grid, block, 0, stream,
                     g, bx, by, bz, ex, ey, ez, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_b_cyl_order2(const quasar::Grid2D& g,
                                             quasar::Real* bx, quasar::Real* by,
                                             quasar::Real* bz,
                                             const quasar::Real* ex, const quasar::Real* ey,
                                             const quasar::Real* ez,
                                             quasar::Real dt, quasar_stream_t stream) {
  launch(g, bx, by, bz, ex, ey, ez, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
