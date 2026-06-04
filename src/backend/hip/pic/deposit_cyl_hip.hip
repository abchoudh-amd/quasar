#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/deposit_common.hpp"
#include "backend/hip/pic/launch.hpp"

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
// chosen so that the BACKWARD ring-flux divergence (the adjoint of the FDTD
// forward flux)
//   ( r_e(i) Jr(i) - r_e(i-1) Jr(i-1) )/(r_c dr)
// exactly cancels the radial part of d rho/dt with rho normalised by V_cell.
// (The running prefix sum below makes r_e(i)*Jr(i) proportional to the partial
// charge-flux through node i's lower face, so consecutive nodes telescope as a
// backward difference, not a forward/centred one.)
// The axial current Jz telescopes as in Cartesian but is normalised by the same
// cylindrical V_cell so its divergence matches. Jphi (out-of-plane, no grid axis)
// is deposited with the time-centred shape as a density over V_cell. The axis
// cell (i=0) uses cell_volume(0), the wedge volume of the on-axis ring, so the
// on-axis charge is weighted correctly.
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
__global__ void deposit_cyl_kernel(quasar::Grid2D g, std::size_t n, double q, double dt,
                                   bool periodic_x, bool periodic_y,
                                   const double* __restrict__ x, const double* __restrict__ y,
                                   const double* __restrict__ x_prev,
                                   const double* __restrict__ y_prev,
                                   const double* __restrict__ vz,
                                   const double* __restrict__ weight,
                                   const unsigned char* __restrict__ alive,
                                   double* __restrict__ jx, double* __restrict__ jy,
                                   double* __restrict__ jz,
                                   unsigned int* __restrict__ overflow) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  const double inv_dr = 1.0 / g.dx();
  const double inv_dz = 1.0 / g.dy();

  // Reduced (cell-centred) coordinates: gr along radius, gz along axis.
  const double gr0 = (x_prev[p] - g.origin_x) * inv_dr - 0.5;
  const double gz0 = (y_prev[p] - g.origin_y) * inv_dz - 0.5;
  const double gr1 = (x[p] - g.origin_x) * inv_dr - 0.5;
  const double gz1 = (y[p] - g.origin_y) * inv_dz - 0.5;

  if (!isfinite(gr0) || !isfinite(gr1) || !isfinite(gz0) || !isfinite(gz1)) return;

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

  if (!shape_strip_sums_to_one(Sr0, kWindow) || !shape_strip_sums_to_one(Sr1, kWindow) ||
      !shape_strip_sums_to_one(Sz0, kWindow) || !shape_strip_sums_to_one(Sz1, kWindow)) {
    atomicAdd(overflow, 1u);
    return;
  }

  const int pitch = g.pitch();
  int colr[kWindow];   // wrapped/clamped radial node index (i + nghost)
  int rowz[kWindow];   // pitch * (wrapped/clamped axial node index + nghost)
  int gnode[kWindow];  // un-wrapped radial node, for the r-weight metric
  for (int a = 0; a < kWindow; ++a) {
    colr[a] = deposit_axis_x(g, br + a, periodic_x) + g.nghost;
    rowz[a] = pitch * (deposit_axis_y(g, bz + a, periodic_y) + g.nghost);
    gnode[a] = br + a;
  }

  const double qw = q * weight[p];
  const double inv_dt = 1.0 / dt;
  const double dr = g.dx();
  const double dz = g.dy();

  // All three components are normalised by the SAME cylindrical cell volume
  // V(i) = cell_volume(i) so that, with rho(i,j) = qw*Sr(i)*Sz(j)/V(i), the
  // discrete cylindrical continuity (backward ring-flux divergence)
  //   (rho1-rho0)/dt + ( r_e(i) Jr(i) - r_e(i-1) Jr(i-1) )/(r_c dr)
  //                  + ( Jz(j) - Jz(j-1) )/dz = 0
  // holds exactly term-by-term, independent of any constant (e.g. 2*pi) factor
  // baked into cell_volume(): the factor cancels between rho and J.

  // Precompute per-radial-node metrics: cell volume, node radius, lower-face
  // radius (the ring through which Jr at that node flows).
  double inv_vol[kWindow], r_c[kWindow], r_e[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    const double vol = g.cell_volume(gnode[a]);
    inv_vol[a] = (vol > 0.0) ? (1.0 / vol) : 0.0;
    r_c[a] = g.r_at_cell_center(gnode[a]);
    r_e[a] = g.r_at_edge(gnode[a]);
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
      run += dsr;  // charge-flux fraction crossing node a's lower radial face
      const double ar = r_e[a] < 0.0 ? -r_e[a] : r_e[a];
      // The on-axis (and inside-axis ghost) lower face has zero/negative radius
      // and carries no ring current; skip it -- the on-axis radial flux vanishes.
      if (ar < 1.0e-300) continue;
      const double jr = -qw * inv_dt * run * sz_mid * (r_c[a] * dr * inv_vol[a]) / r_e[a];
      if (jr != 0.0) {
        atomicAdd(jx + colr[a] + rowz[b], jr);
      }
    }
  }

  // Jz (axial): telescoping prefix sum along z for each radial column. The node
  // weight dz/V(a) makes the discrete d(Jz)/dz cancel the axial part of d rho/dt.
  for (int a = 0; a < kWindow; ++a) {
    const double sr_mid = Sr0[a] + 0.5 * (Sr1[a] - Sr0[a]);
    const double kz = -qw * inv_dt * dz * inv_vol[a];
    double run = 0.0;
    for (int b = 0; b < kWindow; ++b) {
      const double dsz = Sz1[b] - Sz0[b];
      run += kz * dsz * sr_mid;
      if (run != 0.0) {
        atomicAdd(jy + colr[a] + rowz[b], run);
      }
    }
  }

  // Jphi (out-of-plane): time-centred shape, deposited as a density over V(a).
  for (int b = 0; b < kWindow; ++b) {
    for (int a = 0; a < kWindow; ++a) {
      if (inv_vol[a] == 0.0) continue;
      const double smid = 0.5 * (Sr0[a] * Sz0[b] + Sr1[a] * Sz1[b]);
      const double jphi = qw * vz[p] * smid * inv_vol[a];
      if (jphi != 0.0) {
        atomicAdd(jz + colr[a] + rowz[b], jphi);
      }
    }
  }
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
                     s.vz(), s.weight(), s.alive(), j.jx.device_ptr(), j.jy.device_ptr(),
                     j.jz.device_ptr(), s.deposit_overflow());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace

extern "C" void launch_pic_deposit_cyl_shape1(const quasar::Grid2D& g,
                                              const quasar::pic::ParticleSpecies& s,
                                              quasar::JField2D<quasar::Real>& j,
                                              quasar::Real dt, int periodic_x, int periodic_y,
                                              quasar_stream_t stream) {
  launch<1>(g, s, j, dt, periodic_x != 0, periodic_y != 0,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_deposit_cyl_shape2(const quasar::Grid2D& g,
                                              const quasar::pic::ParticleSpecies& s,
                                              quasar::JField2D<quasar::Real>& j,
                                              quasar::Real dt, int periodic_x, int periodic_y,
                                              quasar_stream_t stream) {
  launch<2>(g, s, j, dt, periodic_x != 0, periodic_y != 0,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
