#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/deposit_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Charge-conserving (Esirkepov density-decomposition) current deposition in
// axisymmetric (m=0) cylindrical (r,z) coordinates.
//
// The grid axis i indexes the radius r (x-axis, r=0 at i=0) and j indexes the
// axial coordinate z (y-axis). The stored current components map (x,y,z) ->
// (r,z,phi): jx -> Jr, jy -> Jz, jz -> Jphi.
//
// In cylindrical form the discrete continuity that must hold exactly is
//   d rho/dt + (1/r_c) d(r Jr)/dr + d(Jz)/dz = 0,
// where rho = q w S_r S_z / V_cell and V_cell(i) = cell_volume(i) is the
// cylindrical (ring) cell volume at radial node i. The radial term carries the
// metric factor (1/r) d(r .)/dr, so the discrete radial divergence telescopes in
// the RING-WEIGHTED flux r * Jr rather than in Jr itself. We therefore build the
// Esirkepov telescoping prefix sum in the charge-flux variable and convert it to
// a radial CURRENT DENSITY by dividing the ring flux by the face radius r_e:
//
//   r_e(face) * Jr(face) = -(q w)/(dt) * prefix(dS_r) * Sz_mid / dz_ring,
//
// chosen so that the FORWARD ring-flux divergence used by Ampere/Gauss
//   ( r_e(i+1) Jr(i+1) - r_e(i) Jr(i) )/(r_c dr)
// exactly cancels the radial part of d rho/dt with rho normalised by V_cell.
// The prefix associated with charge cell i is stored on its upper face i+1.
// Consecutive faces therefore telescope under the forward divergence; at i=0
// this naturally retains the outward flux while the r=0 inner flux vanishes.
// The axial current Jz telescopes as in Cartesian but is normalised by the same
// cylindrical V_cell so its divergence matches. The axis cell (i=0) uses
// cell_volume(0), the wedge volume of the on-axis ring, so the on-axis charge is
// weighted correctly.
//
// Jphi enters no continuity relation and is NOT a charge-centred density. It is
// collocated with Ephi on the radial faces, so it is deposited with the
// time-centred shape on the (offset 0, 1/2) lattice and normalised by that
// face's radial dual annulus -- not by the charge cell's V_cell.
//
// The per-axis periodic/clamp indexing and the deposition-window overflow guard
// are identical to the Cartesian deposit; only the r-weighting and the V_cell
// normalisation differ.

namespace {

// kWindowFor, deposit_axis_x/y, weight_1d, and the sum-to-1 deposition-window
// guard are shared with the Cartesian deposit and live in deposit_common.hpp.
// Only the r-weighting / V_cell current normalisation below is cylindrical.
using namespace quasar::backend::pic;

template <int ShapeOrder>
__global__ void QUASAR_LAUNCH_BOUNDS_256 deposit_cyl_kernel(quasar::Grid2D g, std::size_t n, double q, double dt,
                                   bool periodic_x, bool periodic_y,
                                   const double* __restrict__ x, const double* __restrict__ y,
                                   const double* __restrict__ x_prev,
                                   const double* __restrict__ y_prev,
                                   const double* __restrict__ vz,
                                   const double* __restrict__ vphi_deposit,
                                   const double* __restrict__ weight,
                                   const unsigned char* __restrict__ alive,
                                   double* __restrict__ jx, double* __restrict__ jy,
                                   double* __restrict__ jz,
                                   unsigned int* __restrict__ overflow) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  // Reduced (cell-centred) coordinates: gr along radius, gz along axis.
  const double gr0 = quasar::detail::scaled_difference_quotient(
      x_prev[p], g.origin_x, g.dx()) - 0.5;
  const double gz0 = quasar::detail::scaled_difference_quotient(
      y_prev[p], g.origin_y, g.dy()) - 0.5;
  const double gr1 = quasar::detail::scaled_difference_quotient(
      x[p], g.origin_x, g.dx()) - 0.5;
  const double gz1 = quasar::detail::scaled_difference_quotient(
      y[p], g.origin_y, g.dy()) - 0.5;

  if (!reduced_coordinate_representable(gr0)
      || !reduced_coordinate_representable(gr1)
      || !reduced_coordinate_representable(gz0)
      || !reduced_coordinate_representable(gz1)) {
    mark_deposit_error(overflow);
    return;
  }

  constexpr int kWindow = kWindowFor<ShapeOrder>;
  const int br = static_cast<int>(floor(gr0 < gr1 ? gr0 : gr1)) - ShapeOrder;
  const int bz = static_cast<int>(floor(gz0 < gz1 ? gz0 : gz1)) - ShapeOrder;

  double Sr0[kWindow], Sr1[kWindow], Sz0[kWindow], Sz1[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    const int ir = br + a;
    const int iz = bz + a;
    Sr0[a] = weight_1d<ShapeOrder>(gr0 - ir);
    Sr1[a] = weight_1d<ShapeOrder>(gr1 - ir);
    Sz0[a] = weight_1d<ShapeOrder>(gz0 - iz);
    Sz1[a] = weight_1d<ShapeOrder>(gz1 - iz);
  }

  // Scalar charge is even across r=0. Fold any shape support on negative-radius
  // cell centres into its physical mirror (-i-1) before constructing the radial
  // flux prefix. Dropping those weights (the old inv_vol<=0 path) changed the
  // represented charge whenever a particle entered the axis cell and made an
  // exact axis continuity balance impossible.
  if (g.origin_x == 0.0) {
    for (int a = 0; a < kWindow; ++a) {
      const int ir = br + a;
      if (ir >= 0) continue;
      const int mirror = -ir - 1;
      const int ma = mirror - br;
      if (ma >= 0 && ma < kWindow) {
        Sr0[ma] += Sr0[a];
        Sr1[ma] += Sr1[a];
        Sr0[a] = 0.0;
        Sr1[a] = 0.0;
      }
    }
  }

  if (!shape_strip_sums_to_one(Sr0, kWindow) || !shape_strip_sums_to_one(Sr1, kWindow) ||
      !shape_strip_sums_to_one(Sz0, kWindow) || !shape_strip_sums_to_one(Sz1, kWindow)) {
    mark_deposit_error(overflow);
    return;
  }

  const int pitch = g.pitch();
  // Charge-centred indices, used by the Jr/Jz prefix deposits below. Jphi is not
  // charge-centred: it builds its own indices on the radial-face lattice.
  int colr[kWindow];        // radial charge cell index
  int colr_face[kWindow];   // Jr upper-face index
  int rowz[kWindow];        // axial charge cell row
  int rowz_face[kWindow];   // Jz upper-face row
  int gnode[kWindow];  // un-wrapped radial node, for the r-weight metric
  bool colr_valid[kWindow];
  bool colr_face_valid[kWindow];
  bool rowz_valid[kWindow];
  bool rowz_face_valid[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    const int ir = br + a;
    colr_valid[a] = periodic_x
        || (ir >= -g.nghost && ir <= g.nx - 1 + g.nghost);
    colr[a] = (periodic_x ? g.wrap_i(ir)
                          : deposit_clamp_axis(ir, -g.nghost,
                                               g.nx - 1 + g.nghost))
            + g.nghost;
    const int fr = br + a + 1;
    colr_face_valid[a] = periodic_x
        || (fr >= -g.nghost && fr <= g.nx + g.nghost - 1);
    colr_face[a] = (periodic_x ? g.wrap_i(fr)
                               : deposit_clamp_axis(fr, -g.nghost,
                                                    g.nx + g.nghost - 1))
                   + g.nghost;
    const int iz = bz + a;
    rowz_valid[a] = periodic_y
        || (iz >= -g.nghost && iz <= g.ny - 1 + g.nghost);
    rowz[a] = pitch * ((periodic_y ? g.wrap_j(iz)
                                   : deposit_clamp_axis(iz, -g.nghost,
                                                        g.ny - 1 + g.nghost))
                       + g.nghost);
    const int fz = bz + a + 1;
    rowz_face_valid[a] = periodic_y
        || (fz >= -g.nghost && fz <= g.ny + g.nghost - 1);
    rowz_face[a] = pitch * ((periodic_y ? g.wrap_j(fz)
                                        : deposit_clamp_axis(
                                              fz, -g.nghost,
                                              g.ny + g.nghost - 1))
                            + g.nghost);
    gnode[a] = br + a;
  }

  const double dr = g.dx();
  const double dz = g.dy();

  // The two continuity-carrying components (Jr, Jz) are normalised by the SAME
  // cylindrical cell volume V(i) = cell_volume(i) so that, with
  // rho(i,j) = qw*Sr(i)*Sz(j)/V(i), the discrete cylindrical continuity
  // (solver-compatible forward divergence)
  //   (rho1-rho0)/dt + ( r_e(i+1) Jr(i+1) - r_e(i) Jr(i) )/(r_c dr)
  //                  + ( Jz(j+1) - Jz(j) )/dz = 0
  // holds exactly term-by-term, independent of any constant (e.g. 2*pi) factor
  // baked into cell_volume(): the factor cancels between rho and J.

  // Precompute radii only. The ring-volume factors are kept factored in each
  // scaled ratio below: V=2*pi*r_c*dr*dz. This avoids materialising either V or
  // 1/V, both of which can overflow/underflow on a thin large-radius annulus
  // even when the final density/current is representable.
  double r_c[kWindow], r_e_upper[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    r_c[a] = g.r_at_cell_center(gnode[a]);
    r_e_upper[a] = g.r_at_edge(gnode[a] + 1);
  }

  // Jr (radial): the radial divergence telescopes in the RING flux r_e * Jr, so
  // we form the prefix sum of charge flux and divide by the face radius r_e. The
  // node weight is r_c*dr/V(a) (= 1/(K*r_e*dz) for V=K*r_c*dr*dz), times the
  // running shape-flux and the time-centred axial shape.
  for (int b = 0; b < kWindow; ++b) {
    const double sz_mid = Sz0[b] + 0.5 * (Sz1[b] - Sz0[b]);
    double run = 0.0;
    for (int a = 0; a < kWindow; ++a) {
      const double dsr = Sr1[a] - Sr0[a];
      run += dsr;  // charge-flux fraction through cell a's upper radial face
      const double ar = r_e_upper[a] < 0.0 ? -r_e_upper[a] : r_e_upper[a];
      // A zero-radius face carries no ring current. The physical axis is cell
      // zero's lower face, not the upper face written by its prefix.
      if (ar == 0.0) continue;
      // r_c*dr/V cancels analytically to 1/(2*pi*dz), leaving a
      // well-conditioned conservative ring-flux normalization.
      ScaledProduct jr_scaled;
      scaled_multiply(jr_scaled, -q);
      scaled_multiply(jr_scaled, weight[p]);
      scaled_multiply(jr_scaled, run);
      scaled_multiply(jr_scaled, sz_mid);
      scaled_divide(jr_scaled, dt);
      scaled_divide(jr_scaled, 2.0);
      scaled_divide(jr_scaled, quasar::pi);
      scaled_divide(jr_scaled, dz);
      scaled_divide(jr_scaled, r_e_upper[a]);
      const double jr = scaled_value(jr_scaled);
      if (!scaled_value_representable(jr_scaled, jr)) {
        mark_deposit_error(overflow);
        return;
      }
      if (jr != 0.0 && colr_face_valid[a] && rowz_valid[b]) {
        checked_atomic_add(jx + colr_face[a] + rowz[b], jr, overflow);
      }
    }
  }

  // Jz (axial): telescoping prefix sum along z for each radial column. The node
  // weight dz/V(a) makes the discrete d(Jz)/dz cancel the axial part of d rho/dt.
  for (int a = 0; a < kWindow; ++a) {
    const double sr_mid = Sr0[a] + 0.5 * (Sr1[a] - Sr0[a]);
    if (!(r_c[a] > 0.0)) continue;
    // dz/V = 1/(2*pi*r_c*dr), so dz never needs to be formed or cancelled.
    ScaledProduct kz_scaled;
    scaled_multiply(kz_scaled, -q);
    scaled_multiply(kz_scaled, weight[p]);
    scaled_divide(kz_scaled, dt);
    scaled_divide(kz_scaled, 2.0);
    scaled_divide(kz_scaled, quasar::pi);
    scaled_divide(kz_scaled, r_c[a]);
    scaled_divide(kz_scaled, dr);
    double run = 0.0;
    for (int b = 0; b < kWindow; ++b) {
      const double dsz = Sz1[b] - Sz0[b];
      ScaledProduct increment_scaled = kz_scaled;
      scaled_multiply(increment_scaled, dsz);
      scaled_multiply(increment_scaled, sr_mid);
      const double increment = scaled_value(increment_scaled);
      if (!scaled_value_representable(increment_scaled, increment)) {
        mark_deposit_error(overflow);
        return;
      }
      run += increment;
      if (!isfinite(run)) {
        mark_deposit_error(overflow);
        return;
      }
      if (run != 0.0 && colr_valid[a] && rowz_face_valid[b]) {
        checked_atomic_add(jy + colr[a] + rowz_face[b], run, overflow);
      }
    }
  }

  // Jphi is collocated with Ephi on radial faces, not on the charge centres.
  // Deposit each endpoint shape with half weight on the (offset 0,1/2) lattice;
  // this is the exact adjoint of the Ephi gather used by the pusher. Normalize by
  // the face's radial dual annulus rather than a neighbouring cell volume.
  const auto phi0 = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x_prev[p], y_prev[p], g, 0.0, 0.5);
  const auto phi1 = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.0, 0.5);
  const auto deposit_phi_endpoint = [&](const auto& phi, double vphi) {
    for (int b = 0; b < phi.ny; ++b) {
      int iz = phi.iy[b];
      if (periodic_y) {
        iz = g.wrap_j(iz);
      } else if (iz < -g.nghost || iz > g.ny - 1 + g.nghost) {
        continue;
      }
      for (int a = 0; a < phi.nx; ++a) {
        int ir = phi.ix[a];
        double radial_parity = 1.0;
        // At r=0, fold negative-radius support onto its positive face with odd
        // Jphi parity. Ephi uses the same odd axis continuation in the gather, so
        // this minus sign is required for the particle/grid work identity. Adding
        // the image weight would turn a linear odd field into an even one for the
        // deposit. The axis face itself is identically zero by regularity.
        if (g.origin_x == 0.0 && ir < 0) {
          ir = -ir;
          radial_parity = -1.0;
        }
        if (g.origin_x == 0.0 && ir == 0) continue;
        bool ir_valid = true;
        if (periodic_x) {
          ir = g.wrap_i(ir);
        } else {
          ir_valid = ir >= -g.nghost && ir <= g.nx + g.nghost - 1;
        }
        if (!ir_valid) continue;

        const double rf = g.r_at_edge(ir);
        const double width = (ir == 0 || ir == g.nx) ? 0.5 * dr : dr;
        const double midpoint = ir == 0 ? rf + 0.5 * width
                              : (ir == g.nx ? rf - 0.5 * width : rf);
        if (!(width > 0.0 && midpoint > 0.0)) continue;
        const double shape = radial_parity * phi.wx[a] * phi.wy[b];
        ScaledProduct value_scaled;
        scaled_multiply(value_scaled, 0.5);
        scaled_multiply(value_scaled, q);
        scaled_multiply(value_scaled, weight[p]);
        scaled_multiply(value_scaled, vphi);
        scaled_multiply(value_scaled, shape);
        // dual_volume = pi * width * (2*midpoint) * dz.
        scaled_divide(value_scaled, quasar::pi);
        scaled_divide(value_scaled, width);
        scaled_divide(value_scaled, 2.0);
        scaled_divide(value_scaled, midpoint);
        scaled_divide(value_scaled, dz);
        const double value = scaled_value(value_scaled);
        if (!scaled_value_representable(value_scaled, value)) {
          mark_deposit_error(overflow);
          return;
        }
        if (value != 0.0) {
          checked_atomic_add(jz + g.index(ir, iz), value, overflow);
        }
      }
    }
  };
  deposit_phi_endpoint(phi0, vphi_deposit[p]);
  deposit_phi_endpoint(phi1, vz[p]);
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, const quasar::pic::ParticleSpecies& s,
            quasar::JField2D<double>& j, double dt, bool periodic_x, bool periodic_y,
            hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL((deposit_cyl_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), dt, periodic_x, periodic_y,
                     s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vz(), s.vphi_deposit(), s.weight(), s.alive(),
                     j.jx.device_ptr(), j.jy.device_ptr(),
                     j.jz.device_ptr(), s.deposit_overflow());
  // The launch error is checked once in the extern-C wrapper below, matching the
  // single-check convention of the fdtd_b / gather_push launchers.
}

template <int ShapeOrder>
__global__ void QUASAR_LAUNCH_BOUNDS_256 charge_cyl_kernel(quasar::Grid2D g, std::size_t n, double q,
                                  bool periodic_y,
                                  const double* __restrict__ r,
                                  const double* __restrict__ z,
                                  const double* __restrict__ weight,
                                  const unsigned char* __restrict__ alive,
                                  double* __restrict__ rho,
                                  unsigned int* __restrict__ overflow) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  const double gr = quasar::detail::scaled_difference_quotient(
      r[p], g.origin_x, g.dx()) - 0.5;
  const double gz = quasar::detail::scaled_difference_quotient(
      z[p], g.origin_y, g.dy()) - 0.5;
  if (!reduced_coordinate_representable(gr)
      || !reduced_coordinate_representable(gz)) {
    mark_deposit_error(overflow);
    return;
  }
  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(r[p], z[p], g);
  for (int b = 0; b < w.ny; ++b) {
    int jj = w.iy[b];
    if (periodic_y) jj = g.wrap_j(jj);
    else if (jj < -g.nghost || jj > g.ny - 1 + g.nghost) continue;
    for (int a = 0; a < w.nx; ++a) {
      int ii = w.ix[a];
      if (ii < 0 && g.origin_x == 0.0) {
        ii = -ii - 1;  // even scalar-charge parity at the physical axis
      }
      if (ii < -g.nghost || ii > g.nx - 1 + g.nghost) continue;
      const double rc = g.r_at_cell_center(ii);
      if (!(rc > 0.0)) continue;
      ScaledProduct value_scaled;
      scaled_multiply(value_scaled, q);
      scaled_multiply(value_scaled, weight[p]);
      scaled_multiply(value_scaled, w.wx[a]);
      scaled_multiply(value_scaled, w.wy[b]);
      scaled_divide(value_scaled, 2.0);
      scaled_divide(value_scaled, quasar::pi);
      scaled_divide(value_scaled, rc);
      scaled_divide(value_scaled, g.dx());
      scaled_divide(value_scaled, g.dy());
      const double value = scaled_value(value_scaled);
      if (!scaled_value_representable(value_scaled, value)) {
        mark_deposit_error(overflow);
        return;
      }
      if (value != 0.0) {
        checked_atomic_add(rho + g.index(ii, jj), value, overflow);
      }
    }
  }
}

template <int ShapeOrder>
void launch_charge(const quasar::Grid2D& g, const quasar::pic::ParticleSpecies& s,
                   quasar::ScalarGrid2D<double>& rho, bool periodic_y,
                   hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(s.size());
  hipLaunchKernelGGL((charge_cyl_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), periodic_y, s.x(), s.y(),
                     s.weight(), s.alive(), rho.values.device_ptr(),
                     s.deposit_overflow());
}

}  // namespace

QUASAR_PIC_DEPOSIT_ABI(launch_pic_deposit_cyl_shape1, 1)
QUASAR_PIC_DEPOSIT_ABI(launch_pic_deposit_cyl_shape2, 2)

#define QUASAR_PIC_CHARGE_CYL_ABI(fn_name, shape_order)                          \
  extern "C" void fn_name(const quasar::Grid2D& g,                              \
                          const quasar::pic::ParticleSpecies& s,                 \
                          quasar::ScalarGrid2D<quasar::Real>& rho,                \
                          int periodic_x, int periodic_y,                         \
                          quasar_stream_t stream) {                               \
    (void)periodic_x;                                                              \
    launch_charge<shape_order>(g, s, rho, periodic_y != 0,                        \
                               static_cast<hipStream_t>(stream));                 \
    QUASAR_HIP_CHECK(::hipGetLastError());                                        \
  }

QUASAR_PIC_CHARGE_CYL_ABI(launch_pic_charge_cyl_shape1, 1)
QUASAR_PIC_CHARGE_CYL_ABI(launch_pic_charge_cyl_shape2, 2)
#undef QUASAR_PIC_CHARGE_CYL_ABI

namespace {

__device__ inline int cyl_correction_z(int j, int n) {
  int r = j % n;
  return r < 0 ? r + n : r;
}

// Initialise the compact solve in Jr variables. The operator is still applied
// to q=r_face*Jr, but each row is analytically divided by its own radius below.
// This avoids ever materialising r*J, which can overflow on a large-radius
// annulus even when J and the radius ratios in the final solve are representable.
__global__ void QUASAR_LAUNCH_BOUNDS_256 cyl_current_correct_init_kernel(
    quasar::Grid2D g, const double* j_r, const double* j_z,
    double* rhs_r, double* rhs_z, double* iter_r, double* iter_z) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  if (i <= g.nx && j < g.ny) {
    const std::size_t k = g.index(i, j);
    rhs_r[k] = j_r[k];
    iter_r[k] = j_r[k];
  }
  if (i < g.nx && j <= g.ny) {
    const std::size_t k = g.index(i, j);
    rhs_z[k] = j_z[k];
    iter_z[k] = j_z[k];
  }
}

// Jacobi step for the cylindrical compact inverse.  For axial current this is
// the Cartesian S inverse.  Radially, q(0)=0 and q(-dr)=q(dr).  Factoring the
// axis-cell fourth-order divergence gives
//   (7/6) q_1 - q_2/24 = q_raw,1,
// while rows i>1 contain the shared S(q)_0=-q_1/12 axis flux:
//   13 q_i/12 - (q_{i-1}+q_{i+1})/24 + q_1/12 = q_raw,i.
// These are precisely D4+(q)=D2+(q_raw) after cumulative summation from r=0.
__global__ void QUASAR_LAUNCH_BOUNDS_256 cyl_current_correct_step_kernel(
    quasar::Grid2D g, const double* rhs_r, const double* rhs_z,
    const double* in_r, const double* in_z,
    double* out_r, double* out_z, int r_lo_mode, int r_hi_mode,
    int z_lo_mode, int z_hi_mode) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  const bool on_axis = g.origin_x == 0.0;

  if (i <= g.nx && j < g.ny) {
    const std::size_t k = g.index(i, j);
    if (on_axis && i == 0) {
      out_r[k] = 0.0;
    } else if (on_axis && i == 1) {
      if (g.nx == 1) {
        // The first nonzero radial face is also the outer wall. With PEC-even
        // continuation its outer ghost mirrors the zero axis face; with linear
        // outflow continuation the one-cell D4 equation reduces to identity.
        out_r[k] = r_hi_mode == 2 ? rhs_r[k]
                                      : (6.0 / 7.0) * rhs_r[k];
      } else {
        const double radius_ratio = g.r_at_edge(2) / g.r_at_edge(1);
        out_r[k] = (6.0 / 7.0) *
                   (rhs_r[k]
                    + radius_ratio * in_r[g.index(2, j)] / 24.0);
      }
    } else {
      const double ri = g.r_at_edge(i);
      if (!on_axis && i == 0) {
        if (r_lo_mode == 2) {
          // Linear E ghost: E_-1=2E_0-E_1. Since q=rE, the metric leaves a
          // small dr/r correction in the endpoint row.
          const double delta = g.dx() / ri;
          out_r[k] = (12.0 * rhs_r[k]
                      + delta * in_r[g.index(1, j)]) / (12.0 + delta);
        } else {
          // PEC-even E ghost. r_-1+r_1=2r_0, so the radius factors cancel.
          out_r[k] = (12.0 / 13.0) * rhs_r[k]
                   + (1.0 / 13.0) * in_r[g.index(1, j)];
        }
      } else if (i == g.nx) {
        const double delta = g.dx() / ri;
        const double inner = in_r[g.index(i - 1, j)];
        const double axis_term = on_axis
            ? (g.r_at_edge(1) / ri) * in_r[g.index(1, j)]
            : 0.0;
        if (r_hi_mode == 2) {
          out_r[k] = (12.0 * rhs_r[k] - delta * inner - axis_term)
                   / (12.0 - delta);
        } else {
          out_r[k] = (12.0 / 13.0)
                   * (rhs_r[k] + inner / 12.0 - axis_term / 12.0);
        }
      } else {
        const double rm_ratio = g.r_at_edge(i - 1) / ri;
        const double rp_ratio = g.r_at_edge(i + 1) / ri;
        const double axis_term = on_axis
            ? (g.r_at_edge(1) / ri) * in_r[g.index(1, j)] / 12.0
            : 0.0;
        out_r[k] = (12.0 / 13.0)
                 * (rhs_r[k]
                    + (rm_ratio * in_r[g.index(i - 1, j)]
                       + rp_ratio * in_r[g.index(i + 1, j)]) / 24.0
                    - axis_term);
      }
    }
  }

  const bool periodic_z = z_lo_mode == 0 && z_hi_mode == 0;
  const int z_faces = periodic_z ? g.ny : g.ny + 1;
  if (i < g.nx && j < z_faces) {
    const std::size_t k = g.index(i, j);
    if (periodic_z) {
      const int jm = cyl_correction_z(j - 1, z_faces);
      const int jp = cyl_correction_z(j + 1, z_faces);
      out_z[k] = (12.0 / 13.0) * rhs_z[k]
               + (1.0 / 26.0) *
                 (in_z[g.index(i, jm)] + in_z[g.index(i, jp)]);
    } else if (j == 0) {
      out_z[k] = z_lo_mode == 2
          ? rhs_z[k]
          : (12.0 / 13.0) * rhs_z[k]
              + (1.0 / 13.0) * in_z[g.index(i, 1)];
    } else if (j == z_faces - 1) {
      out_z[k] = z_hi_mode == 2
          ? rhs_z[k]
          : (12.0 / 13.0) * rhs_z[k]
              + (1.0 / 13.0) * in_z[g.index(i, z_faces - 2)];
    } else {
      out_z[k] = (12.0 / 13.0) * rhs_z[k]
               + (1.0 / 26.0) *
                 (in_z[g.index(i, j - 1)] + in_z[g.index(i, j + 1)]);
    }
  }
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 cyl_current_correct_finish_kernel(
    quasar::Grid2D g, const double* corrected_r, double* j_r,
    const double* corrected_z, double* j_z, bool periodic_z) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  if (i <= g.nx && j < g.ny) {
    const std::size_t k = g.index(i, j);
    j_r[k] = corrected_r[k];
  }
  if (i < g.nx && j < (periodic_z ? g.ny : g.ny + 1)) {
    j_z[g.index(i, j)] = corrected_z[g.index(i, j)];
  }
  if (periodic_z && i < g.nx && j == 0) {
    j_z[g.index(i, g.ny)] = corrected_z[g.index(i, 0)];
  }
}

}  // namespace

extern "C" void launch_pic_current_correct_cyl_order4(
    const quasar::Grid2D& g, quasar::JField2D<quasar::Real>& j,
    quasar::Real* rhs_r, quasar::Real* rhs_z,
    quasar::Real* iter_r, quasar::Real* iter_z,
    int r_lo_mode, int r_hi_mode, int z_lo_mode, int z_hi_mode,
    quasar_stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid((static_cast<unsigned int>(g.nx + 1) + block.x - 1) / block.x,
                  (static_cast<unsigned int>(g.ny + 1) + block.y - 1) / block.y);
  hipLaunchKernelGGL(cyl_current_correct_init_kernel, grid, block, 0, hip_stream,
                     g, j.jx.device_ptr(), j.jy.device_ptr(), rhs_r, rhs_z,
                     iter_r, iter_z);

  constexpr int kIterations = 20;
  const bool periodic_z = z_lo_mode == 0 && z_hi_mode == 0;
  for (int it = 0; it < kIterations; ++it) {
    const bool even = (it & 1) == 0;
    const double* in_r = even ? iter_r : j.jx.device_ptr();
    const double* in_z = even ? iter_z : j.jy.device_ptr();
    double* out_r = even ? j.jx.device_ptr() : iter_r;
    double* out_z = even ? j.jy.device_ptr() : iter_z;
    hipLaunchKernelGGL(cyl_current_correct_step_kernel, grid, block, 0,
                       hip_stream, g, rhs_r, rhs_z, in_r, in_z, out_r, out_z,
                       r_lo_mode, r_hi_mode, z_lo_mode, z_hi_mode);
  }
  // Twenty iterations leave the final radial/axial solutions in iter_r/iter_z.
  hipLaunchKernelGGL(cyl_current_correct_finish_kernel, grid, block, 0,
                     hip_stream, g, iter_r, j.jx.device_ptr(), iter_z,
                     j.jy.device_ptr(), periodic_z);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
