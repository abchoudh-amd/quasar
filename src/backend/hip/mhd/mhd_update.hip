// Direct state-update kernels for the ideal-MHD solver: conservative flux
// difference, SSP-RK stage combination, and explicit positivity repair.

#include "mhd_update_detail.hpp"

#include "backend/hip/hip_check.hpp"

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

using namespace quasar::mhd::update_detail;

// dudt += -(F_{i+1/2} - F_{i-1/2}) / d(dir). The flux at face (i,j) is the
// x_lo/y_lo face of cell (i,j); the i+1/2 face of cell (i,j) is therefore the
// stored flux of cell (i+di, j+dj). Accumulates so the caller can sum both
// directions into a single dudt (zeroed before the first direction).
__global__ void QUASAR_LAUNCH_BOUNDS_256 flux_difference_kernel(
    quasar::Grid2D g, int dir, int cylindrical,
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
  const Real spacing_value = (dir == 0) ? g.dx() : g.dy();
  const auto difference_quotient = [&](Real flo, Real fhi) {
    const Real a[2] = {fhi, -flo};
    const Real one[2] = {Real{1}, Real{1}};
    const Real spacing[2] = {spacing_value, spacing_value};
    return quasar::numerics::scaled_product_quotient_sum(
        a, one, spacing, one, 2);
  };
  if (cylindrical != 0 && dir == 0) {
    // Ring-volume finite-volume divergence. On a uniform radial mesh,
    //   [r_hi F_hi-r_lo F_lo]/int(r dr)
    // = (F_hi-F_lo)/dr + (F_hi/2+F_lo/2)/r_c.
    // The factored form avoids differences of squared radii and overflowing
    // coordinate*flux products, and is regular in the r_lo=0 axis cell.
    const Real dr = g.dx();
    const Real r_center = g.r_at_cell_center(i);
    const auto annular_div = [&](Real flo, Real fhi) {
      const Real a[4] = {fhi, -flo, fhi, flo};
      const Real b[4] = {Real{1}, Real{1}, Real{1}, Real{1}};
      const Real d0[4] = {dr, dr, r_center, r_center};
      const Real d1[4] = {Real{1}, Real{1}, Real{2}, Real{2}};
      return quasar::numerics::scaled_product_quotient_sum(a, b, d0, d1, 4);
    };
    drho[k]    += -annular_div(Frho[k], Frho[kp]);
    // Cylindrical radial momentum is deliberately omitted here. Its ordinary
    // tensor derivatives, pressure-free curvature difference, and optional
    // static-background stress are rounded together by the dedicated kernel.
    dmy[k]     += -annular_div(Fmy[k], Fmy[kp]);
    // m_phi is stored as an angular-momentum-weighted cell average under
    // r^2 dr. Use that same exact cell moment rather than an annular divergence
    // plus a pointwise -F/r source. With q=dr/(2*r_c),
    //
    //   D_J = [(1+q)^2 F_hi - (1-q)^2 F_lo]
    //         / [dr*(1+q^2/3)].
    //
    // This normalized form never constructs r^2 or W=int(r^2 dr), remains
    // regular at the axis (q=1), and telescopes exactly when weighted by W.
    const CylindricalAngularMetric angular =
        cylindrical_angular_metric(dr, r_center);
    const Real angular_a[2] = {Fmz[kp], -Fmz[k]};
    const Real angular_b[2] = {
        angular.hi_weight, angular.lo_weight};
    const Real angular_d0[2] = {dr, dr};
    const Real angular_d1[2] = {
        angular.moment_norm, angular.moment_norm};
    dmz[k] += -quasar::numerics::scaled_product_quotient_sum(
        angular_a, angular_b, angular_d0, angular_d1, 2);
    denergy[k] += -annular_div(Fenergy[k], Fenergy[kp]);

    // B_phi obeys the metric-free point equation
    // d_t B_phi + d_r F_{Bphi}=0, so use the ordinary radial face difference.
    const Real a_bphi[2] = {-Fbz[kp], Fbz[k]};
    const Real one[2] = {Real{1}, Real{1}};
    const Real spacing[2] = {dr, dr};
    dbz[k] += quasar::numerics::scaled_product_quotient_sum(
        a_bphi, one, spacing, one, 2);
  } else {
    drho[k]    += -difference_quotient(Frho[k], Frho[kp]);
    if (cylindrical == 0) {
      dmx[k] += -difference_quotient(Fmx[k], Fmx[kp]);
    }
    dmy[k]     += -difference_quotient(Fmy[k], Fmy[kp]);
    dmz[k]     += -difference_quotient(Fmz[k], Fmz[kp]);
    denergy[k] += -difference_quotient(Fenergy[k], Fenergy[kp]);
    dbz[k]     += -difference_quotient(Fbz[k], Fbz[kp]);
  }
  dbx[k] += -difference_quotient(Fbx[k], Fbx[kp]);
  dby[k] += -difference_quotient(Fby[k], Fby[kp]);
}

// Evaluate the Shu--Osher linear combination without reading an operand whose
// coefficient is exactly zero.  This is not merely an optimization: after a
// rejected stage a scratch register may legitimately contain NaN, and the
// mathematical zero-weight term must not become IEEE 0*NaN and poison the
// retry.  Nonzero terms still share the exponent-scaled reduction, preserving
// finite survivors when individually overflowing products cancel.
__device__ inline Real rk_weighted_sum(Real a, Real un, Real b, Real ustage,
                                       Real c_dt, Real dudt) {
  const bool use_a = a != Real{0};
  const bool use_b = b != Real{0};
  const bool use_c = c_dt != Real{0};
  if (use_a) {
    if (use_b) {
      return use_c
          ? quasar::numerics::product_sum3(
                a, un, b, ustage, c_dt, dudt)
          : quasar::numerics::product_sum2(a, un, b, ustage);
    }
    return use_c
        ? quasar::numerics::product_sum2(a, un, c_dt, dudt)
        : a * un;
  }
  if (use_b) {
    return use_c
        ? quasar::numerics::product_sum2(b, ustage, c_dt, dudt)
        : b * ustage;
  }
  return use_c ? c_dt * dudt : Real{0};
}

// out = a*un + b*ustage + c_dt*dudt. Cell-centred components update over the
// nx-by-ny cell interior. The staggered poloidal components also update their
// physical high-face layer: bx on i=nx and by on j=ny. Those faces participate
// in the divergence of the last cell and must receive the same SSP-RK combine as
// the other CT faces; only face slots beyond them are derived ghosts.
__global__ void QUASAR_LAUNCH_BOUNDS_256 rk_stage_kernel(
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
  if (i > g.nx || j > g.ny) return;
  const bool cell = i < g.nx && j < g.ny;
  const bool bx_physical = i <= g.nx && j < g.ny;
  const bool by_physical = i < g.nx && j <= g.ny;
  if (!(cell || bx_physical || by_physical)) return;
  const std::size_t k = g.index(i, j);
  // copy_state() deliberately reuses this launcher with (a,b,c)=(1,0,0).
  // Take an exact copy path rather than evaluating 0*dudt: a rejected stage may
  // have left NaN in its residual, and IEEE 0*NaN is NaN rather than zero. This
  // branch gives rollback a strong bitwise state-restoration guarantee.
  if (a == Real{1} && b == Real{0} && c_dt == Real{0}) {
    if (cell) {
      o_rho[k] = un_rho[k];
      o_mx[k] = un_mx[k];
      o_my[k] = un_my[k];
      o_mz[k] = un_mz[k];
      o_energy[k] = un_energy[k];
      o_bz[k] = un_bz[k];
    }
    if (bx_physical) o_bx[k] = un_bx[k];
    if (by_physical) o_by[k] = un_by[k];
    return;
  }
  if (cell) {
    o_rho[k] = rk_weighted_sum(
        a, un_rho[k], b, us_rho[k], c_dt, d_rho[k]);
    o_mx[k] = rk_weighted_sum(
        a, un_mx[k], b, us_mx[k], c_dt, d_mx[k]);
    o_my[k] = rk_weighted_sum(
        a, un_my[k], b, us_my[k], c_dt, d_my[k]);
    o_mz[k] = rk_weighted_sum(
        a, un_mz[k], b, us_mz[k], c_dt, d_mz[k]);
    o_energy[k] = rk_weighted_sum(
        a, un_energy[k], b, us_energy[k], c_dt, d_energy[k]);
    o_bz[k] = rk_weighted_sum(
        a, un_bz[k], b, us_bz[k], c_dt, d_bz[k]);
  }
  if (bx_physical) {
    o_bx[k] = rk_weighted_sum(
        a, un_bx[k], b, us_bx[k], c_dt, d_bx[k]);
  }
  if (by_physical) {
    o_by[k] = rk_weighted_sum(
        a, un_by[k], b, us_by[k], c_dt, d_by[k]);
  }
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
__global__ void QUASAR_LAUNCH_BOUNDS_256 apply_floors_kernel(
    quasar::Grid2D g, int /*b0_active*/, Real rho_floor, Real p_floor, Real gamma,
    int collocation_order,
    quasar::numerics::RadialTablesView radial_tables,
    double* __restrict__ rho, double* __restrict__ mx, double* __restrict__ my,
    double* __restrict__ mz, double* __restrict__ energy, double* __restrict__ bx,
    double* __restrict__ by, double* __restrict__ bz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  if (!isfinite(rho[k]) || rho[k] < rho_floor) rho[k] = rho_floor;
  if (!isfinite(mx[k])) mx[k] = Real{0};
  if (!isfinite(my[k])) my[k] = Real{0};
  if (!isfinite(mz[k])) mz[k] = Real{0};
  MhdState s = quasar::mhd::load_cell_state(
      g, rho, mx, my, mz, energy, bx, by, bz, i, j, radial_tables,
      collocation_order);
  // Perturbation-only gas pressure via the background-aware overload (B0 is
  // unread under the field-split EOS, so this equals pressure(s, gamma)).
  const Real p = pressure(s, MhdBackground{}, gamma);
  if (!isfinite(p) || p < p_floor) {
    const Real kinetic = quasar::numerics::kinetic_from_momentum(
        s.mx, s.my, s.mz, s.rho);
    // Stored magnetic energy stays the perturbation-only 0.5|b|^2 (b = bx/by/bz).
    const Real magnetic = quasar::numerics::half_squared_norm3(s.bx, s.by, s.bz);
    energy[k] = p_floor / (gamma - Real{1}) + kinetic + magnetic;
  }
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_flux_difference(const MhdField2D<Real>& flux, int dir,
                                MhdField2D<Real>& dudt, stream_t stream,
                                bool cylindrical) {
  const Grid2D g = flux.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(flux_difference_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, dir, cylindrical ? 1 : 0,
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
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
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
                             Real rho_floor, Real p_floor, Real gamma,
                             stream_t stream, int collocation_order,
                             quasar::numerics::RadialTablesView radial_tables) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  // The floor re-derivation is B0-independent under the field-split EOS; the
  // background-active flag is threaded only for ABI parity with the field-split
  // kernel family. No B0 buffers are read.
  const int b0_active = b0.active ? 1 : 0;
  hipLaunchKernelGGL(apply_floors_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, b0_active, rho_floor, p_floor, gamma,
                     collocation_order, radial_tables,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
