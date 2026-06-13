// Device Riemann flux for the ideal-MHD solver: full HLLD (Miyoshi & Kusano
// 2005) with a single-intermediate HLL fallback on degenerate fans.
//
// Inputs are the reconstructed LEFT/RIGHT conserved interface states at every
// interface normal to `dir`; the output is the numerical flux written into the
// 8 conserved-component buffers of an MhdField2D reused as flux storage.
//
// The flux is computed in a rotated frame where the interface normal is the
// local x-axis: for dir=1 (y-interfaces) the in-plane components are swapped
// (x<->y) before the solve and swapped back when storing, so a single normal-x
// HLLD routine serves both directions.
//
// FIELD-SPLIT (B = B0 + b): the interface states store the PERTURBATION b
// (bx/by/bz) and the perturbation-only energy E = E_gas + 0.5 rho|v|^2 +
// 0.5|b|^2. The PHYSICS (wave speeds, Maxwell stress, magnetic pressure) needs
// the TOTAL field B = B0 + b. We sample the static background b0 at the
// interface index, build a fully-consistent TOTAL-field conserved state
// (bx/by/bz = b + B0, with the energy bumped so the gas pressure is preserved),
// run the EXISTING unmodified HLLD on the total-field state (so the wave speeds
// and fluxes see total B), then convert the resulting flux back to the
// perturbation flux. rho/momentum/induction fluxes are identical between b and
// B (B0 is a static constant); only the energy flux needs the constant cross
// term subtracted: f_E_pert = f_E_tot - B0 . f_B. With b0 inactive (B0 == 0)
// the total state equals the perturbation state and the correction vanishes, so
// the result is bit-for-bit identical to the original body.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/hlld_core.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdFlux;
using quasar::numerics::MhdState;

inline constexpr unsigned kBlock2D = 16;

// Static background magnetic field B0 = (b0x, b0y, b0z) sampled at one interface
// index. Carried alongside the rotated state so the field-split flux can build
// the total field and apply the energy back-correction.
struct Bg3 { Real b0x; Real b0y; Real b0z; };

// Rotate B0 into the normal-x frame the same way hlld::rotate_in() rotates the
// state (dir==1 swaps the in-plane x<->y magnetic components). b0z is the
// out-of-plane component and is invariant under the in-plane swap.
__device__ inline Bg3 rotate_bg(const Bg3& b, int dir) {
  if (dir == 0) return b;
  return Bg3{b.b0y, b.b0x, b.b0z};
}

// Build the TOTAL-field conserved state from a perturbation conserved state and
// the (rotated) background B0. The magnetic components become B = b + B0; the
// energy is bumped by (b . B0 + 0.5|B0|^2) so that pressure(total, gamma)
// recovers the same gas pressure as the field-split EOS (perturbation-only
// magnetic energy). With B0 == 0 this returns u unchanged (bit-identical path).
__device__ inline MhdState to_total(const MhdState& u, const Bg3& b0) {
  MhdState t = u;
  t.bx = u.bx + b0.b0x;
  t.by = u.by + b0.b0y;
  t.bz = u.bz + b0.b0z;
  const Real cross = u.bx * b0.b0x + u.by * b0.b0y + u.bz * b0.b0z;
  const Real bg2   = b0.b0x * b0.b0x + b0.b0y * b0.b0y + b0.b0z * b0.b0z;
  t.energy = u.energy + cross + Real{0.5} * bg2;
  return t;
}

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

__global__ void hlld_kernel(
    quasar::Grid2D g, int dir, Real gamma,
    // Field-split background B0 (perturbation-only state evolved; physics uses
    // total B = B0 + b). `bg_active`==0 takes the zero-background fast path and
    // ignores the (possibly null) b0* buffers.
    int bg_active,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    const double* __restrict__ Lrho, const double* __restrict__ Lmx,
    const double* __restrict__ Lmy, const double* __restrict__ Lmz,
    const double* __restrict__ Lenergy, const double* __restrict__ Lbx,
    const double* __restrict__ Lby, const double* __restrict__ Lbz,
    const double* __restrict__ Rrho, const double* __restrict__ Rmx,
    const double* __restrict__ Rmy, const double* __restrict__ Rmz,
    const double* __restrict__ Renergy, const double* __restrict__ Rbx,
    const double* __restrict__ Rby, const double* __restrict__ Rbz,
    double* __restrict__ Frho, double* __restrict__ Fmx, double* __restrict__ Fmy,
    double* __restrict__ Fmz, double* __restrict__ Fenergy, double* __restrict__ Fbx,
    double* __restrict__ Fby, double* __restrict__ Fbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // Compute the flux over the interior PLUS the high-edge face layer (index
  // nx/ny): flux_difference at the last interior cell (i==nx-1) reads the flux
  // at face nx, so it must be a real HLLD flux, not stale buffer data. The
  // reconstruction kernel writes its L/R interface states over the same extended
  // range, and index nx/ny is an in-bounds ghost slot (nghost>=2).
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);

  MhdState L = load_state(Lrho, Lmx, Lmy, Lmz, Lenergy, Lbx, Lby, Lbz, k);
  MhdState R = load_state(Rrho, Rmx, Rmy, Rmz, Renergy, Rbx, Rby, Rbz, k);

  // Sample the static background at this interface index (lab frame). The
  // in-plane b0x/b0y use the same staggering as the state's B; b0z is
  // cell-centered. Zero when inactive -> total state == perturbation state.
  Bg3 bg{Real{0}, Real{0}, Real{0}};
  if (bg_active != 0) {
    bg = Bg3{b0x[k], b0y[k], b0z[k]};
  }

  // Promote the interface states to the TOTAL field (B = b + B0) so the HLLD
  // wave speeds / Maxwell stress see the physical field, then rotate into the
  // normal-x frame. With bg == 0 to_total() is the identity. The seven-wave
  // algebra is the shared host/device core (numerics/hlld_core.hpp), so device
  // and host fluxes use identical guards and intermediate-state formulas.
  const Bg3 bg_rot = rotate_bg(bg, dir);
  const MhdState Lr = to_total(quasar::numerics::hlld::rotate_in(L, dir), bg_rot);
  const MhdState Rr = to_total(quasar::numerics::hlld::rotate_in(R, dir), bg_rot);

  const MhdFlux fr = quasar::numerics::hlld::hlld_flux_x(Lr, Rr, gamma);
  const MhdFlux f = quasar::numerics::hlld::rotate_out(fr, dir);

  // The rho / momentum / induction fluxes advance the perturbation b unchanged
  // (B0 is a static constant). The energy flux of the perturbation energy is the
  // total-field energy flux minus the constant cross term B0 . F_B (a rotation-
  // invariant dot product, computed here in the lab frame). Zero when inactive.
  const Real e_corr = bg.b0x * f.bx + bg.b0y * f.by + bg.b0z * f.bz;

  Frho[k] = f.rho; Fmx[k] = f.mx; Fmy[k] = f.my; Fmz[k] = f.mz;
  Fenergy[k] = f.energy - e_corr; Fbx[k] = f.bx; Fby[k] = f.by; Fbz[k] = f.bz;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface,
                          const MhdBackgroundField<Real>& b0, int dir,
                          MhdField2D<Real>& flux_out, Real gamma, stream_t stream) {
  const Grid2D g = iface.grid;
  const dim3 block(kBlock2D, kBlock2D);
  // Launch over nx+1 x ny+1 so the high-edge face flux (index nx/ny) consumed by
  // flux_difference at the last interior cell is produced, not left stale.
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  // Zero-background fast path: pass null b0 pointers and bg_active=0 so the
  // kernel never dereferences them and the flux is bit-identical to the
  // perturbation-only body.
  const int bg_active = b0.active ? 1 : 0;
  const double* b0x = b0.active ? b0.b0x_face.device_ptr() : nullptr;
  const double* b0y = b0.active ? b0.b0y_face.device_ptr() : nullptr;
  const double* b0z = b0.active ? b0.b0z_cell.device_ptr() : nullptr;
  hipLaunchKernelGGL(hlld_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, dir, gamma,
                     bg_active, b0x, b0y, b0z,
                     iface.Lrho.device_ptr(), iface.Lmx.device_ptr(), iface.Lmy.device_ptr(),
                     iface.Lmz.device_ptr(), iface.Lenergy.device_ptr(), iface.Lbx.device_ptr(),
                     iface.Lby.device_ptr(), iface.Lbz.device_ptr(),
                     iface.Rrho.device_ptr(), iface.Rmx.device_ptr(), iface.Rmy.device_ptr(),
                     iface.Rmz.device_ptr(), iface.Renergy.device_ptr(), iface.Rbx.device_ptr(),
                     iface.Rby.device_ptr(), iface.Rbz.device_ptr(),
                     flux_out.rho.device_ptr(), flux_out.mx.device_ptr(), flux_out.my.device_ptr(),
                     flux_out.mz.device_ptr(), flux_out.energy.device_ptr(),
                     flux_out.bx_face.device_ptr(), flux_out.by_face.device_ptr(),
                     flux_out.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
