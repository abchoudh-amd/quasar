#include "quasar/core/grid.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/fdtd_cyl_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Axisymmetric (m=0) cylindrical Ampere update of E from B and J.
//
// The Yee fields are node-collocated; the grid axis i indexes the radius r
// (x-axis) with r=0 at i=0, and j indexes the axial coordinate z (y-axis). The
// stored vector components map (x,y,z) -> (r,z,phi):
//   ex -> Er,  ey -> Ez,  ez -> Ephi   (likewise bx->Br, by->Bz, bz->Bphi).
//
// The axisymmetric curl of a field A = (A_r, A_z, A_phi) is
//   (curl A)_r   = -dA_phi/dz
//   (curl A)_z   = (1/r) d(r A_phi)/dr
//   (curl A)_phi =  dA_r/dz - dA_z/dr
// (all d/dphi terms vanish for m=0). The radial-flux term (1/r) d(r A_phi)/dr is
// discretised in finite-volume form so the discrete operator is the adjoint of
// the backward-difference radial divergence used by the cylindrical deposit and
// the Faraday update -- this keeps the scheme stable and charge-consistent.
//
// This is a staggered Yee scheme: Bphi (and the other odd components) live on the
// radial FACES, stored so that the array slot at column i carries the value at the
// lower face r_e(i) = i*dr. The E-update at node i then reads the lower face (i)
// and the upper face (i+1) -- a FORWARD radial read matching the Cartesian
// ddx_staggered -- while the Faraday update writes a face from a BACKWARD Ez
// difference, the adjoint pair that makes the composed operator the correct 2nd-
// order cylindrical Bessel operator. The radial finite-volume forward flux at
// radial node i, axial node j is
//   (1/r_c(i)) * ( r_e(i+1) * Bphi(i+1,j) - r_e(i) * Bphi(i,j) ) / dr,
// where r_c(i) = r_at_cell_center(i) is the node radius and r_e(i) =
// r_at_edge(i) is the lower-face radius.
//
// On the axis (i=0) the lower face sits at r_e(0) = 0, so the inner flux term
// r_e(0)*Bphi(0,j) vanishes IDENTICALLY -- this IS the regularised on-axis
// closure for Ez (and Bz), no l'Hopital "4*value/dr" factor is needed. The
// earlier 4*Bphi(0)/dr closure injected a resolution-independent eigenvalue
// shift (~10-14%) that broke convergence to the analytic TM0n0 spectrum; the
// natural zero-radius-face flux below converges to j0n at 2nd order. The odd
// components Er and Ephi still vanish on the axis node by parity and are pinned.

namespace {

// radial_flux_fwd, ddz_fwd, ddz_bwd, ddr_bwd are the shared cylindrical FDTD
// operators (fdtd_cyl_common.hpp), used by both the Ampere (here) and Faraday
// updates so the adjoint operator pair cannot drift. The Ephi update's d(Bz)/dr
// term uses the BACKWARD radial difference ddr_bwd (the adjoint of the forward
// face-to-node flux radial_flux_fwd the Bz Faraday update uses to read Ephi).
using namespace quasar::backend::pic;

__global__ void fdtd_e_cyl_kernel(quasar::Grid2D g,
                                  double* __restrict__ ex, double* __restrict__ ey,
                                  double* __restrict__ ez,
                                  const double* __restrict__ bx, const double* __restrict__ by,
                                  const double* __restrict__ bz,
                                  const double* __restrict__ jx, const double* __restrict__ jy,
                                  const double* __restrict__ jz,
                                  double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);

  if (i == 0) {
    // On-axis node. Er (radial) and Ephi (azimuthal) are odd in r and vanish
    // identically on the axis -- pin them. Ez (and Bz in the Faraday update) are
    // even and DO live on the axis; their radial flux uses the natural zero-radius
    // inner face: radial_flux_fwd already evaluates to r_e(1)*Bphi(1)/(r_c(0)*dr)
    // here because r_e(0)=0, so no special "4/dr" factor is applied. This natural
    // closure is what converges to the analytic Bessel spectrum.
    const double curl_z = radial_flux_fwd(bz, g, 0, j);  // (1/r) d(r Bphi)/dr, axis-regular
    ex[k] = 0.0;
    ez[k] = 0.0;  // Ephi component (stored in ez slot)
    ey[k] += dt * (curl_z - jy[k]);
    return;
  }

  // Interior: cylindrical curl of B with finite-volume radial flux.
  const double curl_r = -ddz_fwd(bz, g, i, j);                 // -dBphi/dz
  const double curl_z = radial_flux_fwd(bz, g, i, j);          // (1/r) d(r Bphi)/dr
  const double curl_phi = ddz_fwd(bx, g, i, j) - ddr_bwd(by, g, i, j);  // dBr/dz - dBz/dr

  ex[k] += dt * (curl_r - jx[k]);    // Er
  ey[k] += dt * (curl_z - jy[k]);    // Ez
  ez[k] += dt * (curl_phi - jz[k]);  // Ephi
}

void launch(quasar::Grid2D g, double* ex, double* ey, double* ez,
            const double* bx, const double* by, const double* bz,
            const double* jx, const double* jy, const double* jz,
            double dt, hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL(fdtd_e_cyl_kernel, grid, block, 0, stream,
                     g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt);
}

}  // namespace

extern "C" void launch_pic_fdtd_e_cyl_order2(const quasar::Grid2D& g,
                                             quasar::Real* ex, quasar::Real* ey,
                                             quasar::Real* ez,
                                             const quasar::Real* bx, const quasar::Real* by,
                                             const quasar::Real* bz,
                                             const quasar::Real* jx, const quasar::Real* jy,
                                             const quasar::Real* jz,
                                             quasar::Real dt, quasar_stream_t stream) {
  launch(g, ex, ey, ez, bx, by, bz, jx, jy, jz, dt, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
