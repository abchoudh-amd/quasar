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

// The timestep-weighted radial-flux and plain staggered increments are shared
// with Ampere through fdtd_cyl_common.hpp so the adjoint operator pair cannot
// drift. The even Bz reads odd Ephi through the axis-regular forward flux; odd
// Br/Bphi use the backward axial/radial increments.
using namespace quasar::backend::pic;

template <int Order>
__global__ void fdtd_b_cyl_kernel(quasar::Grid2D g,
                                  double* __restrict__ bx, double* __restrict__ by,
                                  double* __restrict__ bz,
                                  const double* __restrict__ ex, const double* __restrict__ ey,
                                  const double* __restrict__ ez,
                                  double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);

  if (i == 0 && g.origin_x == 0.0) {
    // On-axis node. Br (radial) and Bphi (azimuthal) are odd in r and vanish on
    // the axis face (r_e(0)=0) -- pin them. Bz (even) lives on the axis node; its
    // radial flux uses the natural zero-radius inner face (radial_flux_fwd reduces
    // to r_e(1)*Ephi(1)/(r_c(0)*dr) here), the axis-regular closure. The previous
    // "4*Ephi/dr" factor injected the same spurious eigenvalue shift the E-update
    // closure did and is removed.
    // Br/Bphi include the full axial-face row j=ny and vanish there too. Bz is
    // cell-centred in both axes and is advanced only for j<ny.
    bx[k] = 0.0;
    bz[k] = 0.0;  // Bphi component (stored in bz slot)
    if (j < g.ny) {
      by[k] = quasar::numerics::detail::scaled_signed_sum(
          quasar::numerics::detail::scaled_value(by[k]),
          quasar::numerics::detail::negate(
              radial_flux_fwd_increment_scaled<Order>(ez, g, 0, j, dt)));
    }
    return;
  }

  // Interior: cylindrical curl of E. The even Bz reads the odd Ephi via the
  // forward face-to-node finite-volume flux (the adjoint of the E-update's
  // backward ddr_bwd(Bz) in the Ephi equation, and axis-regular); the odd Br/Bphi
  // use plain backward differences (adjoints of the E-update's forward reads).
  if (i <= g.nx && j <= g.ny) {
    bx[k] = quasar::numerics::detail::scaled_signed_sum(
        quasar::numerics::detail::scaled_value(bx[k]),
        ddz_bwd_increment_scaled<Order>(ez, g, i, j, dt));  // Br
    bz[k] = quasar::numerics::detail::scaled_signed_sum(
        quasar::numerics::detail::scaled_value(bz[k]),
        quasar::numerics::detail::negate(
            ddz_bwd_increment_scaled<Order>(ex, g, i, j, dt)),
        ddr_bwd_increment_scaled<Order>(ey, g, i, j, dt));  // Bphi
  }
  if (i < g.nx && j < g.ny) {
    by[k] = quasar::numerics::detail::scaled_signed_sum(
        quasar::numerics::detail::scaled_value(by[k]),
        quasar::numerics::detail::negate(
            radial_flux_fwd_increment_scaled<Order>(ez, g, i, j, dt)));  // Bz
  }
}

template <int Order>
void launch(quasar::Grid2D g, double* bx, double* by, double* bz,
            const double* ex, const double* ey, const double* ez, double dt,
            hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid((static_cast<unsigned int>(g.nx + 1) + block.x - 1) / block.x,
                  (static_cast<unsigned int>(g.ny + 1) + block.y - 1) / block.y);
  hipLaunchKernelGGL((fdtd_b_cyl_kernel<Order>), grid, block, 0, stream,
                     g, bx, by, bz, ex, ey, ez, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_b_cyl_order2(const quasar::Grid2D& g,
                                             quasar::Real* bx, quasar::Real* by,
                                             quasar::Real* bz,
                                             const quasar::Real* ex, const quasar::Real* ey,
                                             const quasar::Real* ez,
                                             quasar::Real dt, quasar_stream_t stream) {
  launch<2>(g, bx, by, bz, ex, ey, ez, dt,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_fdtd_b_cyl_order4(const quasar::Grid2D& g,
                                             quasar::Real* bx, quasar::Real* by,
                                             quasar::Real* bz,
                                             const quasar::Real* ex, const quasar::Real* ey,
                                             const quasar::Real* ez,
                                             quasar::Real dt, quasar_stream_t stream) {
  launch<4>(g, bx, by, bz, ex, ey, ez, dt,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
