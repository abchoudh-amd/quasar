// Device flux reconstruction for the conservative-FD ideal-MHD solver.
//
// Implements a 2nd-order MUSCL-minmod reconstruction in PRIMITIVE variables on
// every interface normal to `dir`. For dir=0, face (i,j) sits between cells
// (i-1,j) and (i,j): the LEFT interface state is the right-biased extrapolation
// from cell (i-1) and the RIGHT interface state is the left-biased
// extrapolation from cell (i). The eight conserved components are reconstructed
// after converting cell-centered conserved -> primitive (so the minmod limiter
// acts on rho/v/p/B), then converted back to conserved for the L/R buffers.
//
// SIMPLIFICATION (flagged for review): scheme_order 5/7 (host "mp5"/"mp7") are
// not ported to device -- the host MP limiters are not exposed as
// QUASAR_HOST_DEVICE helpers. All scheme_order values resolve to this 2nd-order
// MUSCL path on device; the host registry keeps the full high-order schemes.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdPrim;
using quasar::numerics::MhdState;

// File-local 2-D launch tiling (the MHD backend has no shared launch header; the
// per-cell kernels here all use the same square block). Kept tiny and internal
// so each .hip is self-contained.
inline constexpr unsigned kBlock2D = 16;
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Minmod limiter on two slopes (returns the smaller-magnitude same-sign slope,
// zero on a sign change). Device-callable, matches the host muscl_minmod logic.
__device__ inline Real minmod(Real a, Real b) {
  if (a * b <= Real{0}) return Real{0};
  return (fabs(a) < fabs(b)) ? a : b;
}

// Load the cell-centered conserved state at storage index k into a MhdState.
__device__ inline MhdState load_state(const double* rho, const double* mx,
                                      const double* my, const double* mz,
                                      const double* energy, const double* bx,
                                      const double* by, const double* bz,
                                      std::size_t k) {
  MhdState s;
  s.rho = rho[k]; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
  s.energy = energy[k]; s.bx = bx[k]; s.by = by[k]; s.bz = bz[k];
  return s;
}

// Componentwise minmod-limited interface primitive: given the primitive at the
// cell (wc), its lower neighbor (wm) and upper neighbor (wp), return the value
// extrapolated to the +side (sign=+1, right face of the cell) or -side
// (sign=-1, left face of the cell): w +/- 0.5 * minmod(wc-wm, wp-wc).
__device__ inline MhdPrim muscl_face(const MhdPrim& wm, const MhdPrim& wc,
                                     const MhdPrim& wp, Real sign) {
  const Real h = Real{0.5} * sign;
  MhdPrim f;
  f.rho = wc.rho + h * minmod(wc.rho - wm.rho, wp.rho - wc.rho);
  f.vx  = wc.vx  + h * minmod(wc.vx  - wm.vx,  wp.vx  - wc.vx);
  f.vy  = wc.vy  + h * minmod(wc.vy  - wm.vy,  wp.vy  - wc.vy);
  f.vz  = wc.vz  + h * minmod(wc.vz  - wm.vz,  wp.vz  - wc.vz);
  f.p   = wc.p   + h * minmod(wc.p   - wm.p,   wp.p   - wc.p);
  f.bx  = wc.bx  + h * minmod(wc.bx  - wm.bx,  wp.bx  - wc.bx);
  f.by  = wc.by  + h * minmod(wc.by  - wm.by,  wp.by  - wc.by);
  f.bz  = wc.bz  + h * minmod(wc.bz  - wm.bz,  wp.bz  - wc.bz);
  return f;
}

__global__ void reconstruct_kernel(
    quasar::Grid2D g, int dir, Real gamma,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    double* __restrict__ Lrho, double* __restrict__ Lmx, double* __restrict__ Lmy,
    double* __restrict__ Lmz, double* __restrict__ Lenergy, double* __restrict__ Lbx,
    double* __restrict__ Lby, double* __restrict__ Lbz,
    double* __restrict__ Rrho, double* __restrict__ Rmx, double* __restrict__ Rmy,
    double* __restrict__ Rmz, double* __restrict__ Renergy, double* __restrict__ Rbx,
    double* __restrict__ Rby, double* __restrict__ Rbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;

  // Neighbor offsets along the reconstruction direction.
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;

  // Face (i,j) lies between the "minus" cell (i-di, j-dj) and the "plus" cell
  // (i,j). The LEFT state extrapolates from the minus cell to its +side; the
  // RIGHT state extrapolates from the plus cell to its -side.
  const int im = i - di, jm = j - dj;       // minus cell (owns the L state)
  const int imm = i - 2 * di, jmm = j - 2 * dj;
  const int ip = i + di, jp = j + dj;       // plus-cell upper neighbor

  // Minus-cell stencil: (imm) (im) (i)  -> face value at +side of (im).
  const MhdState sm_lo = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(imm, jmm));
  const MhdState sm_c  = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(im, jm));
  const MhdState sm_hi = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(i, j));
  // Plus-cell stencil: (im) (i) (ip) -> face value at -side of (i).
  const MhdState sp_lo = sm_c;
  const MhdState sp_c  = sm_hi;
  const MhdState sp_hi = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(ip, jp));

  const MhdPrim wm_lo = to_primitive(sm_lo, gamma);
  const MhdPrim wm_c  = to_primitive(sm_c, gamma);
  const MhdPrim wm_hi = to_primitive(sm_hi, gamma);
  const MhdPrim wp_hi = to_primitive(sp_hi, gamma);

  const MhdPrim wl = muscl_face(wm_lo, wm_c, wm_hi, Real{+1});  // +side of minus cell
  const MhdPrim wr = muscl_face(wm_c, wm_hi, wp_hi, Real{-1});  // -side of plus cell

  const MhdState ul = to_conserved(wl, gamma);
  const MhdState ur = to_conserved(wr, gamma);

  const std::size_t k = g.index(i, j);
  Lrho[k] = ul.rho; Lmx[k] = ul.mx; Lmy[k] = ul.my; Lmz[k] = ul.mz;
  Lenergy[k] = ul.energy; Lbx[k] = ul.bx; Lby[k] = ul.by; Lbz[k] = ul.bz;
  Rrho[k] = ur.rho; Rmx[k] = ur.mx; Rmy[k] = ur.my; Rmz[k] = ur.mz;
  Renergy[k] = ur.energy; Rbx[k] = ur.bx; Rby[k] = ur.by; Rbz[k] = ur.bz;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_reconstruct(const MhdField2D<Real>& u, int dir,
                            quasar::numerics::MhdInterfaceStates<Real>& out,
                            int /*scheme_order*/, Real gamma, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(reconstruct_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, dir, gamma,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr(),
                     out.Lrho.device_ptr(), out.Lmx.device_ptr(), out.Lmy.device_ptr(),
                     out.Lmz.device_ptr(), out.Lenergy.device_ptr(), out.Lbx.device_ptr(),
                     out.Lby.device_ptr(), out.Lbz.device_ptr(),
                     out.Rrho.device_ptr(), out.Rmx.device_ptr(), out.Rmy.device_ptr(),
                     out.Rmz.device_ptr(), out.Renergy.device_ptr(), out.Rbx.device_ptr(),
                     out.Rby.device_ptr(), out.Rbz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
