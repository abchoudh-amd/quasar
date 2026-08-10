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
// combines the decomposed momentum stresses and applies the CT-consistent
// energy change of variables using the retained induction flux and final CT
// rate. No E'+|B0|^2/2 state or O(B0^2) numerical flux is ever formed.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/hlld_core.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"
#include "mhd_face_quadrature.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdFlux;
using quasar::mhd::detail::FaceBackgroundView;
using quasar::mhd::detail::FaceInterfaceView;
using quasar::mhd::detail::FaceMomentumFluxParts;

inline constexpr unsigned kBlock2D = 16;

struct CrossBPointOutputView {
  double* x[4]{};
  double* y[4]{};
  double* z[4]{};
  int* quadrature_valid{};
  quasar::numerics::ScaledValue* b0_induction_covariance{};
};

__global__ void QUASAR_LAUNCH_BOUNDS_256 hlld_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    quasar::mhd::FaceOwnershipFlags4 ownership,
    int dir, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables,
    int hll_only, Real gamma,
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
    double* __restrict__ Wmz, CrossBPointOutputView cross_b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // Compute the flux over the interior PLUS the high-edge face layer (index
  // nx/ny): flux_difference at the last interior cell (i==nx-1) reads the flux
  // at face nx, so it must be a real HLLD flux, not stale buffer data. The
  // reconstruction kernel writes its L/R interface states over the same extended
  // range, and index nx/ny is an in-bounds ghost slot (nghost>=2).
  if (i > g.nx || j > g.ny) return;
  // A shared face is evaluated only on its canonical tile. The non-owner's
  // output slot deliberately remains untouched until the runtime broadcasts
  // the complete owner record before launching any divergence/update kernel.
  if ((dir == 0 && ((i == 0 && ownership.side[0] == 0) ||
                    (i == g.nx && ownership.side[1] == 0))) ||
      (dir == 1 && ((j == 0 && ownership.side[2] == 0) ||
                    (j == g.ny && ownership.side[3] == 0)))) {
    return;
  }
  const std::size_t k = g.index(i, j);
  int sample_i = i;
  int sample_j = j;
  if (dir == 0 && i == g.nx && flags.side[0] == 0 && flags.side[1] == 0) {
    sample_i = 0;
  }
  if (dir == 1 && j == g.ny && flags.side[2] == 0 && flags.side[3] == 0) {
    sample_j = 0;
  }
  const FaceInterfaceView interface{
      Lrho, Lmx, Lmy, Lmz, Lenergy, Lbx, Lby, Lbz,
      Rrho, Rmx, Rmy, Rmz, Renergy, Rbx, Rby, Rbz};
  const FaceBackgroundView background{bg_active, b0x, b0y, b0z};

  MhdFlux f;
  if (decompose_momentum != 0) {
    const FaceMomentumFluxParts parts =
        quasar::mhd::detail::average_face_flux_parts(
            g, dir, sample_i, sample_j, scheme_order, hll_only, gamma,
            interface, background, radial_tables);
    f = parts.flux.material;
    Wmx[k] = parts.flux.wave.mx;
    Wmy[k] = parts.flux.wave.my;
    Wmz[k] = parts.flux.wave.mz;
    for (int q = 0; q < 4; ++q) {
      cross_b.x[q][k] = parts.cross_b_point[q].b0x;
      cross_b.y[q][k] = parts.cross_b_point[q].b0y;
      cross_b.z[q][k] = parts.cross_b_point[q].b0z;
    }
    cross_b.quadrature_valid[k] = parts.quadrature_valid;
    cross_b.b0_induction_covariance[k] =
        parts.b0_induction_covariance;
  } else {
    f = quasar::mhd::detail::average_face_flux(
        g, dir, sample_i, sample_j, scheme_order, hll_only, gamma,
        interface, background, radial_tables);
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
                          MhdMomentumFluxParts2D<Real>* momentum_parts,
                          int scheme_order,
                          FaceOwnershipFlags4 ownership,
                          quasar::numerics::RadialTablesView radial_tables) {
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
  CrossBPointOutputView cross_b;
  if (decompose_momentum != 0) {
    for (int q = 0; q < 4; ++q) {
      cross_b.x[q] = momentum_parts->cross_b_point[q].x.device_ptr();
      cross_b.y[q] = momentum_parts->cross_b_point[q].y.device_ptr();
      cross_b.z[q] = momentum_parts->cross_b_point[q].z.device_ptr();
    }
    cross_b.quadrature_valid = momentum_parts->quadrature_valid.device_ptr();
    cross_b.b0_induction_covariance =
        momentum_parts->b0_induction_covariance.device_ptr();
  }
  hipLaunchKernelGGL(hlld_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, flags, ownership, dir, scheme_order, radial_tables,
                     hll_only ? 1 : 0, gamma,
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
                     cross_b);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
