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
// (bx/by/bz) and
//   E'=rho*e+|m|^2/(2*rho)+0.5|b|^2.
// The PHYSICS (wave speeds, Maxwell stress, magnetic pressure) needs
// the TOTAL field B = B0 + b. We sample B0 at the interface and run the
// split-energy HLLD algebra directly. It emits momentum with the static
// background Maxwell stress removed and energy F_E-B0.F_B. The update kernel
// combines the decomposed momentum stresses and directly discretizes the split
// energy source v.[(curl B0)x(B0+b)], reducing curl(B0) before multiplication.
// No E'+|B0|^2/2 state or O(B0^2) numerical flux is ever formed.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/hlld_core.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdFlux;
using quasar::numerics::hlld::MhdMomentumFluxParts;
using quasar::numerics::MhdState;

inline constexpr unsigned kBlock2D = 16;

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
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int dir, int hll_only, Real gamma,
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
    double* __restrict__ Fby, double* __restrict__ Fbz,
    int decompose_momentum,
    double* __restrict__ Wmx, double* __restrict__ Wmy,
    double* __restrict__ Wmz, double* __restrict__ Cbx,
    double* __restrict__ Cby, double* __restrict__ Cbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // Compute the flux over the interior PLUS the high-edge face layer (index
  // nx/ny): flux_difference at the last interior cell (i==nx-1) reads the flux
  // at face nx, so it must be a real HLLD flux, not stale buffer data. The
  // reconstruction kernel writes its L/R interface states over the same extended
  // range, and index nx/ny is an in-bounds ghost slot (nghost>=2).
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);
  int sample_i = i;
  int sample_j = j;
  if (dir == 0 && i == g.nx && flags.side[0] == 0 && flags.side[1] == 0) {
    sample_i = 0;
  }
  if (dir == 1 && j == g.ny && flags.side[2] == 0 && flags.side[3] == 0) {
    sample_j = 0;
  }
  const std::size_t ks = g.index(sample_i, sample_j);

  MhdState L = load_state(Lrho, Lmx, Lmy, Lmz, Lenergy, Lbx, Lby, Lbz, ks);
  MhdState R = load_state(Rrho, Rmx, Rmy, Rmz, Renergy, Rbx, Rby, Rbz, ks);

  // Sample the static background at this interface index (lab frame). The
  // in-plane b0x/b0y use the same staggering as the state's B; b0z is
  // cell-centered. Zero when inactive -> total state == perturbation state.
  quasar::numerics::MhdBackground bg{};
  if (bg_active != 0) {
    const auto sampled =
        quasar::mhd::load_interface_background(
            g, dir, b0x, b0y, b0z, sample_i, sample_j,
            hll_only != 0 ? 1 : 0);
    bg = sampled;
  }

  // Rotate both the perturbation state and prescribed field into the normal-x
  // frame. The shared core performs the affine split transform symbolically.
  const auto bg_rot = quasar::numerics::hlld::rotate_in(bg, dir);
  const MhdState Lr = quasar::numerics::hlld::rotate_in(L, dir);
  const MhdState Rr = quasar::numerics::hlld::rotate_in(R, dir);

  MhdFlux f;
  if (decompose_momentum != 0) {
    const MhdMomentumFluxParts fr = hll_only != 0
        ? quasar::numerics::hlld::lax_friedrichs_flux_split_parts_x(
              Lr, Rr, bg_rot, gamma)
        : quasar::numerics::hlld::hlld_flux_split_parts_x(
              Lr, Rr, bg_rot, gamma);
    const MhdMomentumFluxParts parts =
        quasar::numerics::hlld::rotate_out(fr, dir);
    f = parts.material;
    Wmx[k] = parts.wave.mx;
    Wmy[k] = parts.wave.my;
    Wmz[k] = parts.wave.mz;
    Cbx[k] = parts.cross_b.b0x;
    Cby[k] = parts.cross_b.b0y;
    Cbz[k] = parts.cross_b.b0z;
  } else {
    const MhdFlux fr = hll_only != 0
        ? quasar::numerics::hlld::lax_friedrichs_flux_split_x(
              Lr, Rr, bg_rot, gamma)
        : quasar::numerics::hlld::hlld_flux_split_x(
              Lr, Rr, bg_rot, gamma);
    f = quasar::numerics::hlld::rotate_out(fr, dir);
  }

  Frho[k] = f.rho; Fmx[k] = f.mx; Fmy[k] = f.my; Fmz[k] = f.mz;
  Fenergy[k] = f.energy; Fbx[k] = f.bx; Fby[k] = f.by; Fbz[k] = f.bz;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_hlld_flux(const quasar::numerics::MhdInterfaceStates<Real>& iface,
                          const MhdBackgroundField<Real>& b0, int dir,
                          MhdField2D<Real>& flux_out, BoundaryFlags4 flags,
                          Real gamma, stream_t stream,
                          bool hll_only,
                          MhdMomentumFluxParts2D<Real>* momentum_parts) {
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
  const int decompose_momentum =
      (b0.active && momentum_parts != nullptr) ? 1 : 0;
  hipLaunchKernelGGL(hlld_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, flags, dir, hll_only ? 1 : 0, gamma,
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
                     flux_out.bz_cell.device_ptr(), decompose_momentum,
                     decompose_momentum != 0 ? momentum_parts->wave_x.device_ptr() : nullptr,
                     decompose_momentum != 0 ? momentum_parts->wave_y.device_ptr() : nullptr,
                     decompose_momentum != 0 ? momentum_parts->wave_z.device_ptr() : nullptr,
                     decompose_momentum != 0 ? momentum_parts->cross_bx.device_ptr() : nullptr,
                     decompose_momentum != 0 ? momentum_parts->cross_by.device_ptr() : nullptr,
                     decompose_momentum != 0 ? momentum_parts->cross_bz.device_ptr() : nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
