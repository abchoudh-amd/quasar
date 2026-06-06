// Device update kernels for the ideal-MHD solver: conservative flux difference,
// SSP-RK stage combine, positivity floors, and the cylindrical geometric source.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

inline constexpr unsigned kBlock2D = 16;
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// dudt += -(F_{i+1/2} - F_{i-1/2}) / d(dir). The flux at face (i,j) is the
// x_lo/y_lo face of cell (i,j); the i+1/2 face of cell (i,j) is therefore the
// stored flux of cell (i+di, j+dj). Accumulates so the caller can sum both
// directions into a single dudt (zeroed before the first direction).
__global__ void flux_difference_kernel(
    quasar::Grid2D g, int dir, Real inv_d,
    const double* __restrict__ Frho, const double* __restrict__ Fmx,
    const double* __restrict__ Fmy, const double* __restrict__ Fmz,
    const double* __restrict__ Fenergy, const double* __restrict__ Fbx,
    const double* __restrict__ Fby, const double* __restrict__ Fbz,
    double* __restrict__ drho, double* __restrict__ dmx, double* __restrict__ dmy,
    double* __restrict__ dmz, double* __restrict__ denergy, double* __restrict__ dbx,
    double* __restrict__ dby, double* __restrict__ dbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;
  const std::size_t k  = g.index(i, j);            // F at i-1/2 face
  const std::size_t kp = g.index(i + di, j + dj);  // F at i+1/2 face
  drho[k]    += -(Frho[kp]    - Frho[k])    * inv_d;
  dmx[k]     += -(Fmx[kp]     - Fmx[k])     * inv_d;
  dmy[k]     += -(Fmy[kp]     - Fmy[k])     * inv_d;
  dmz[k]     += -(Fmz[kp]     - Fmz[k])     * inv_d;
  denergy[k] += -(Fenergy[kp] - Fenergy[k]) * inv_d;
  dbx[k]     += -(Fbx[kp]     - Fbx[k])     * inv_d;
  dby[k]     += -(Fby[kp]     - Fby[k])     * inv_d;
  dbz[k]     += -(Fbz[kp]     - Fbz[k])     * inv_d;
}

// out = a*un + b*ustage + c_dt*dudt, componentwise over all 8 conserved fields.
__global__ void rk_stage_kernel(
    quasar::Grid2D g, Real a, Real b, Real c_dt,
    const double* __restrict__ un_rho, const double* __restrict__ un_mx,
    const double* __restrict__ un_my, const double* __restrict__ un_mz,
    const double* __restrict__ un_energy, const double* __restrict__ un_bx,
    const double* __restrict__ un_by, const double* __restrict__ un_bz,
    const double* __restrict__ us_rho, const double* __restrict__ us_mx,
    const double* __restrict__ us_my, const double* __restrict__ us_mz,
    const double* __restrict__ us_energy, const double* __restrict__ us_bx,
    const double* __restrict__ us_by, const double* __restrict__ us_bz,
    const double* __restrict__ d_rho, const double* __restrict__ d_mx,
    const double* __restrict__ d_my, const double* __restrict__ d_mz,
    const double* __restrict__ d_energy, const double* __restrict__ d_bx,
    const double* __restrict__ d_by, const double* __restrict__ d_bz,
    double* __restrict__ o_rho, double* __restrict__ o_mx, double* __restrict__ o_my,
    double* __restrict__ o_mz, double* __restrict__ o_energy, double* __restrict__ o_bx,
    double* __restrict__ o_by, double* __restrict__ o_bz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  o_rho[k]    = a * un_rho[k]    + b * us_rho[k]    + c_dt * d_rho[k];
  o_mx[k]     = a * un_mx[k]     + b * us_mx[k]     + c_dt * d_mx[k];
  o_my[k]     = a * un_my[k]     + b * us_my[k]     + c_dt * d_my[k];
  o_mz[k]     = a * un_mz[k]     + b * us_mz[k]     + c_dt * d_mz[k];
  o_energy[k] = a * un_energy[k] + b * us_energy[k] + c_dt * d_energy[k];
  o_bx[k]     = a * un_bx[k]     + b * us_bx[k]     + c_dt * d_bx[k];
  o_by[k]     = a * un_by[k]     + b * us_by[k]     + c_dt * d_by[k];
  o_bz[k]     = a * un_bz[k]     + b * us_bz[k]     + c_dt * d_bz[k];
}

// Clamp rho >= rho_floor and gas pressure >= p_floor, re-deriving total energy
// from the floored pressure (momentum and B held fixed).
//
// Field-split convention (B = B0 + b): the stored conserved state carries the
// perturbation b in bx/by/bz and the perturbation-only magnetic energy 0.5|b|^2.
// Under this convention the gas pressure has NO B0 cross term -- the background
// is B0-independent for the floor -- so the floored energy is re-derived with the
// perturbation-only magnetic energy 0.5|b|^2 and the perturbation-only gas
// pressure. The `MhdBackground` argument to pressure() exists only to share the
// EOS call surface; it is intentionally unread (pressure(s, b0, gamma) is
// bit-for-bit pressure(s, gamma)), so with B0 inactive this is bit-identical to
// the original body. `b0_active` is accepted for ABI/signature parity; the floor
// re-derivation does not depend on it.
__global__ void apply_floors_kernel(
    quasar::Grid2D g, int /*b0_active*/, Real rho_floor, Real p_floor, Real gamma,
    double* __restrict__ rho, double* __restrict__ mx, double* __restrict__ my,
    double* __restrict__ mz, double* __restrict__ energy, double* __restrict__ bx,
    double* __restrict__ by, double* __restrict__ bz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  if (rho[k] < rho_floor) rho[k] = rho_floor;
  MhdState s;
  s.rho = rho[k]; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
  s.energy = energy[k]; s.bx = bx[k]; s.by = by[k]; s.bz = bz[k];
  // Perturbation-only gas pressure via the background-aware overload (B0 is
  // unread under the field-split EOS, so this equals pressure(s, gamma)).
  const Real p = pressure(s, MhdBackground{}, gamma);
  if (p < p_floor) {
    const Real inv_rho = Real{1} / s.rho;
    const Real kinetic = Real{0.5} * (s.mx * s.mx + s.my * s.my + s.mz * s.mz) * inv_rho;
    // Stored magnetic energy stays the perturbation-only 0.5|b|^2 (b = bx/by/bz).
    const Real magnetic = Real{0.5} * (s.bx * s.bx + s.by * s.by + s.bz * s.bz);
    energy[k] = p_floor / (gamma - Real{1}) + kinetic + magnetic;
  }
}

// Axisymmetric (r,z) geometric source for the m=0 cylindrical update. With the
// x-axis as the radius r, the standard divergence-form axisymmetric source adds
// to the radial-momentum equation the pressure-like term (p_total - B_r^2 +
// rho*v_phi^2 - B_phi^2 stays grouped; here v_phi = vz slot, B_phi = bz slot):
//   d(rho v_r)/dt += (p_gas + rho*v_phi^2 - B_phi^2) / r
// and the azimuthal-momentum / toroidal-field equations carry the curvature
// (-rho v_r v_phi / r + B_r B_phi / r) and (-(v_r B_phi - v_phi B_r)/r) terms.
// An on-axis guard skips cells with r below half a cell width.
__global__ void geometric_source_kernel(
    quasar::Grid2D g, Real /*gamma*/,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    double* __restrict__ dmr, double* __restrict__ dmz,
    double* __restrict__ dbphi) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const Real r = g.r_at_cell_center(i);
  const Real r_floor = Real{0.5} * g.dx();
  if (r <= r_floor) return;  // on-axis guard: source is regular / vanishes
  const std::size_t k = g.index(i, j);
  const Real inv_rho = Real{1} / rho[k];
  // Slot mapping: mx = radial momentum (m_r), my = axial (m_z), mz = azimuthal
  // (m_phi); bx = B_r, by = B_z, bz = B_phi.
  const Real vr   = mx[k] * inv_rho;
  const Real vphi = mz[k] * inv_rho;
  const Real br   = bx[k];
  const Real bphi = bz[k];
  const Real inv_r = Real{1} / r;
  // Radial-momentum geometric source (axisymmetric, divergence-form residual):
  // the centrifugal term plus the toroidal magnetic-tension term that the planar
  // flux divergence omits.  S_{m_r} = (rho*v_phi^2 - B_phi^2)/r.
  dmr[k]   += (rho[k] * vphi * vphi - bphi * bphi) * inv_r;
  // Axial momentum carries no geometric source in axisymmetry.
  (void)dmz; (void)my; (void)by; (void)energy;
  // Toroidal-field induction curvature term: -(v_r B_phi - v_phi B_r)/r.
  dbphi[k] += -(vr * bphi - vphi * br) * inv_r;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_flux_difference(const MhdField2D<Real>& flux, int dir,
                                MhdField2D<Real>& dudt, stream_t stream) {
  const Grid2D g = flux.grid;
  const Real inv_d = (dir == 0) ? (Real{1} / g.dx()) : (Real{1} / g.dy());
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(flux_difference_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, dir, inv_d,
                     flux.rho.device_ptr(), flux.mx.device_ptr(), flux.my.device_ptr(),
                     flux.mz.device_ptr(), flux.energy.device_ptr(), flux.bx_face.device_ptr(),
                     flux.by_face.device_ptr(), flux.bz_cell.device_ptr(),
                     dudt.rho.device_ptr(), dudt.mx.device_ptr(), dudt.my.device_ptr(),
                     dudt.mz.device_ptr(), dudt.energy.device_ptr(), dudt.bx_face.device_ptr(),
                     dudt.by_face.device_ptr(), dudt.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_rk_stage(MhdField2D<Real>& out, const MhdField2D<Real>& un,
                         const MhdField2D<Real>& ustage, const MhdField2D<Real>& dudt,
                         Real a, Real b, Real c_dt, stream_t stream) {
  const Grid2D g = out.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(rk_stage_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, a, b, c_dt,
                     un.rho.device_ptr(), un.mx.device_ptr(), un.my.device_ptr(),
                     un.mz.device_ptr(), un.energy.device_ptr(), un.bx_face.device_ptr(),
                     un.by_face.device_ptr(), un.bz_cell.device_ptr(),
                     ustage.rho.device_ptr(), ustage.mx.device_ptr(), ustage.my.device_ptr(),
                     ustage.mz.device_ptr(), ustage.energy.device_ptr(), ustage.bx_face.device_ptr(),
                     ustage.by_face.device_ptr(), ustage.bz_cell.device_ptr(),
                     dudt.rho.device_ptr(), dudt.mx.device_ptr(), dudt.my.device_ptr(),
                     dudt.mz.device_ptr(), dudt.energy.device_ptr(), dudt.bx_face.device_ptr(),
                     dudt.by_face.device_ptr(), dudt.bz_cell.device_ptr(),
                     out.rho.device_ptr(), out.mx.device_ptr(), out.my.device_ptr(),
                     out.mz.device_ptr(), out.energy.device_ptr(), out.bx_face.device_ptr(),
                     out.by_face.device_ptr(), out.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_apply_floors(MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                             Real rho_floor, Real p_floor, Real gamma, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  // The floor re-derivation is B0-independent under the field-split EOS; the
  // background-active flag is threaded only for ABI parity with the field-split
  // kernel family. No B0 buffers are read.
  const int b0_active = b0.active ? 1 : 0;
  hipLaunchKernelGGL(apply_floors_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, b0_active, rho_floor, p_floor, gamma,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 Grid2D grid, Real gamma, stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 launch_grid = mhd_grid_2d(grid, block);
  hipLaunchKernelGGL(geometric_source_kernel, launch_grid, block, 0,
                     static_cast<hipStream_t>(stream), grid, gamma,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr(),
                     dudt.mx.device_ptr(), dudt.my.device_ptr(), dudt.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
